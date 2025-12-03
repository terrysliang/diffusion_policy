import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.spatial.transform as st

from pylibfranka import Robot, ControllerMode, CartesianPose

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty
)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer
)
from diffusion_policy.common.pose_trajectory_interpolator import (
    PoseTrajectoryInterpolator
)

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


def o_t_ee_to_6d(O_T_EE):
    """
    Convert Franka's O_T_EE (16-element COLUMN-major 4x4) to 6D [x,y,z,rx,ry,rz].
    """
    # Franka stores O_T_EE in column-major order, so use order='F'
    T = np.array(O_T_EE, dtype=float).reshape(4, 4, order='F')
    p = T[:3, 3]
    R = T[:3, :3]
    rotvec = st.Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([p, rotvec])


def pose6d_to_matrix(pose6d):
    """
    Convert 6D [x,y,z,rx,ry,rz] pose to 4x4 homogeneous matrix.
    """
    pose6d = np.asarray(pose6d, dtype=float)
    p = pose6d[:3]
    r = pose6d[3:]
    R = st.Rotation.from_rotvec(r).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

class FrankaInterpolationController(mp.Process):
    """
    Franka version of RTDEInterpolationController.

    Exposes the same API:
      - servoL(pose, duration)
      - schedule_waypoint(pose, target_time)
      - get_state(), get_all_state()

    Internally uses pylibfranka external cartesian pose control with
    an interpolated 6D pose trajectory.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        frequency=1000,           # Franka control loop is 1 kHz
        max_pos_speed=0.05,
        max_rot_speed=0.3,
        max_pos_acc=0.5,          # m/s^2, conservative
        max_rot_acc=1.0,          # rad/s^2, conservative
        launch_timeout=3.0,
        tcp_offset_pose=None,     # ignored or used with set_EE / set_load if you want
        payload_mass=None,        # can be wired into robot.set_load()
        payload_cog=None,
        joints_init=None,         # can be used to move to a start joint config via a separate script
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
    ):
        """
        frequency: nominal external loop frequency (used for ring buffer timing).
        max_pos_speed: m/s (for interpolator)
        max_rot_speed: rad/s (for interpolator)
        tcp_offset_pose: 6D pose, not directly applied here (Franka uses set_EE / set_load).
        payload_mass: float
        payload_cog: 3D CoG (if used with set_load)
        """
        assert frequency > 0
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose, dtype=float)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert payload_mass >= 0
        if payload_cog is not None:
            payload_cog = np.array(payload_cog, dtype=float)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None

        super().__init__(name="FrankaInterpolationController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.max_pos_acc = max_pos_acc
        self.max_rot_acc = max_rot_acc
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # ===== build input queue =====
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # ===== build ring buffer =====
        # emulate UR-style receive_keys if not provided
        if receive_keys is None:
            # We'll map O_T_EE -> ActualTCPPose (6D),
            # ActualQ, ActualQd from RobotState.q, dq.
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]

        # For examples, we need a temporary Robot connection
        tmp_robot = Robot(robot_ip)
        tmp_state = tmp_robot.read_once()
        dof = 7  # or: dof = len(tmp_state.q) if you keep the tmp_robot code

        example_state = {}
        for key in receive_keys:
            if key in ('ActualTCPPose', 'TargetTCPPose'):
                # 6D pose [x,y,z,rx,ry,rz]
                example_state[key] = np.zeros((6,), dtype=float)
            elif key in ('ActualTCPSpeed', 'TargetTCPSpeed'):
                # 6D twist [vx,vy,vz,wx,wy,wz]
                example_state[key] = np.zeros((6,), dtype=float)
            elif key in ('ActualQ', 'ActualQd', 'TargetQ', 'TargetQd'):
                # joint positions / velocities
                example_state[key] = np.zeros((dof,), dtype=float)
            else:
                # scalar or unknown – 1D placeholder
                example_state[key] = np.zeros((1,), dtype=float)

        example_state['robot_receive_timestamp'] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose (seconds, in world/base frame).
        pose: 6D [x,y,z,rx,ry,rz] (rotation vector) in base frame.
        """
        assert self.is_alive()
        pose = np.array(pose, dtype=float)
        assert pose.shape == (6,)
        assert duration > 0.0

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': float(duration)
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        """
        Schedule a pose to be reached at a specific wall-clock time (like UR version).
        """
        assert target_time > time.time()
        pose = np.array(pose, dtype=float)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': float(target_time)
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def duration_to_sec(self, duration):
    # pylibfranka changed naming once; be robust
        if hasattr(duration, "to_sec"):
            return duration.to_sec()
        elif hasattr(duration, "toSec"):
            return duration.toSec()
        else:
            return 0.001  # fallback 1 kHz

    # ========= main loop in process ============
    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                if self.verbose:
                    print("Soft real time not enabled:", e)

        robot_ip = self.robot_ip
        outer_hz = float(self.frequency)   # state logging / DP polling rate
        outer_dt = 1.0 / outer_hz

        # --- connect robot and start cartesian pose control (like official example) ---
        robot = Robot(robot_ip)

        # Start Cartesian pose control
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        # First state under active control
        robot_state, duration = active_control.readOnce()
        dt = self.duration_to_sec(duration)
        if dt <= 0.0:
            dt = 1.0 / 1000.0

        # Initial commanded pose = current pose
        O_T_EE = np.array(robot_state.O_T_EE, dtype=float)
        T_cmd = O_T_EE.reshape(4, 4, order='F')   # column-major → 4x4
        R_cmd = T_cmd[:3, :3].copy()
        p_cmd = T_cmd[:3, 3].copy()

        # Target pose (6D) starts as current pose
        pose_ref = o_t_ee_to_6d(robot_state.O_T_EE)  # [x,y,z,rx,ry,rz]

        # Velocities to be kept continuous
        v_curr = np.zeros(3)   # linear velocity [m/s]
        w_curr = np.zeros(3)   # angular velocity [rad/s]

        # Publish initial state to ring buffer
        state0 = dict()
        state0['ActualTCPPose']   = pose_ref.copy()
        state0['ActualQ']         = np.array(robot_state.q, dtype=float)
        state0['ActualQd']        = np.array(robot_state.dq, dtype=float)
        state0['TargetTCPPose']   = pose_ref.copy()
        state0['TargetTCPSpeed']  = np.zeros((6,))
        state0['TargetQ']         = np.zeros((7,))
        state0['TargetQd']        = np.zeros((7,))
        state0['robot_receive_timestamp'] = time.time()
        self.ring_buffer.put(state0)
        self.ready_event.set()

        keep_running = True
        iter_idx = 0
        next_outer_poll = time.monotonic() + outer_dt
        last_state_log = time.monotonic()

        # Simple time constants for "servo-like" behavior
        # Smaller → more aggressive (but closer to accel limits)
        tau_pos = 0.2   # seconds
        tau_rot = 0.2

        if self.verbose:
            print(f"[FrankaInterpolationController] Started control loop on {robot_ip}")

        try:
            while keep_running:
                # (1) FCI tick: read state and get true dt
                robot_state, duration = active_control.readOnce()
                dt = self.duration_to_sec(duration)
                if dt <= 0.0:
                    dt = 1.0 / 1000.0
                now = time.monotonic()

                # (2) Poll latest command (if any)
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                if n_cmd > 0:
                    # Only keep the latest command
                    command = {k: v[n_cmd - 1] for k, v in commands.items()}
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False

                    elif cmd == Command.SERVOL.value:
                        # New target 6D pose; we treat it as a setpoint for a velocity PD
                        target_pose = np.array(command['target_pose'], dtype=float)
                        assert target_pose.shape == (6,)
                        pose_ref = target_pose  # update setpoint
                        # duration is available as command['duration'] if you want
                        # to adapt tau_pos / tau_rot based on it.

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # Treat same as SERVOL for now: new setpoint
                        target_pose = np.array(command['target_pose'], dtype=float)
                        assert target_pose.shape == (6,)
                        pose_ref = target_pose

                    else:
                        keep_running = False

                # (3) Pose error between current commanded pose and reference pose
                #     Build T_ref from pose_ref
                T_ref = pose6d_to_matrix(pose_ref)
                p_ref = T_ref[:3, 3]
                R_ref = T_ref[:3, :3]

                # position error
                e_p = p_ref - p_cmd

                # orientation error on SO(3): R_err = R_ref * R_cmd^T, convert to rotvec
                R_err = R_ref @ R_cmd.T
                rot_err = st.Rotation.from_matrix(R_err).as_rotvec()

                # (4) Desired velocities from a simple first-order "servo"
                #     v_des ≈ e / tau, limited by max speed
                v_des = e_p / max(tau_pos, 1e-3)
                w_des = rot_err / max(tau_rot, 1e-3)

                # limit speeds
                v_norm = np.linalg.norm(v_des)
                if v_norm > self.max_pos_speed:
                    v_des *= self.max_pos_speed / max(v_norm, 1e-9)

                w_norm = np.linalg.norm(w_des)
                if w_norm > self.max_rot_speed:
                    w_des *= self.max_rot_speed / max(w_norm, 1e-9)

                # (5) Acceleration limiting: keep v_curr/w_curr continuous
                #     dv / dt <= max_pos_acc, dw / dt <= max_rot_acc
                # linear
                dv = v_des - v_curr
                dv_norm = np.linalg.norm(dv)
                dv_max = self.max_pos_acc * dt
                if dv_norm > dv_max:
                    v_curr += dv * (dv_max / max(dv_norm, 1e-9))
                else:
                    v_curr = v_des

                # angular
                dw = w_des - w_curr
                dw_norm = np.linalg.norm(dw)
                dw_max = self.max_rot_acc * dt
                if dw_norm > dw_max:
                    w_curr += dw * (dw_max / max(dw_norm, 1e-9))
                else:
                    w_curr = w_des

                # (6) Integrate commanded pose in SE(3)
                p_cmd = p_cmd + v_curr * dt
                if np.linalg.norm(w_curr) > 1e-9:
                    dR = st.Rotation.from_rotvec(w_curr * dt).as_matrix()
                    R_cmd = dR @ R_cmd

                T_cmd[:3, 3] = p_cmd
                T_cmd[:3, :3] = R_cmd

                # (7) Send command to robot (column-major 4x4 like example)
                cmd_pose = CartesianPose(T_cmd.reshape(-1, order='F').tolist())
                active_control.writeOnce(cmd_pose)

                # (8) Log state to ring buffer at outer_hz
                if now >= next_outer_poll:
                    target_rotvec = st.Rotation.from_matrix(R_cmd).as_rotvec()
                    target_pose6d = np.concatenate([p_cmd, target_rotvec])
                    target_twist = np.concatenate([v_curr, w_curr])

                    state = dict()
                    state['ActualTCPPose']   = o_t_ee_to_6d(robot_state.O_T_EE)
                    state['ActualQ']         = np.array(robot_state.q, dtype=float)
                    state['ActualQd']        = np.array(robot_state.dq, dtype=float)
                    state['TargetTCPPose']   = target_pose6d
                    state['TargetTCPSpeed']  = target_twist
                    state['TargetQ']         = np.zeros((7,))
                    state['TargetQd']        = np.zeros((7,))
                    state['robot_receive_timestamp'] = time.time()
                    self.ring_buffer.put(state)

                    next_outer_poll = now + outer_dt

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            # graceful stop like official example
            try:
                # send final pose with motion_finished=True
                cmd_pose = CartesianPose(T_cmd.reshape(-1, order='F').tolist())
                cmd_pose.motion_finished = True
                active_control.writeOnce(cmd_pose)
            except Exception:
                pass
            try:
                robot.stop()
            except Exception:
                pass
            self.ready_event.set()
            if self.verbose:
                print(f"[FrankaInterpolationController] Disconnected from robot: {robot_ip}")
