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
    Convert Franka's O_T_EE (length-16 row-major 4x4) to 6D [x,y,z,rx,ry,rz]
    using rotation vector (axis-angle).
    """
    T = np.array(O_T_EE, dtype=float).reshape(4, 4)
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
        max_pos_speed=0.25,
        max_rot_speed=0.16,
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
                'ActualQ',
                'ActualQd',
                # Optionally you can add more keys later and fill them from RobotState.
            ]

        # For examples, we need a temporary Robot connection
        tmp_robot = Robot(robot_ip)
        tmp_state = tmp_robot.read_once()
        example_state = {}

        for key in receive_keys:
            if key == 'ActualTCPPose':
                example_state[key] = o_t_ee_to_6d(tmp_state.O_T_EE)
            elif key == 'ActualQ':
                example_state[key] = np.array(tmp_state.q, dtype=float)
            elif key == 'ActualQd':
                example_state[key] = np.array(tmp_state.dq, dtype=float)
            else:
                # Unknown key – user can extend this mapping as needed
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
        outer_hz = float(self.frequency)        # e.g. 10 Hz teleop / command rate
        outer_dt = 1.0 / outer_hz

        # --- connect Franka ---
        robot = Robot(robot_ip)
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        # initial state and pose
        robot_state, duration = active_control.readOnce()
        t0 = time.monotonic()
        curr_pose6d = o_t_ee_to_6d(robot_state.O_T_EE)  # [x,y,z,rx,ry,rz], like Aubo’s rotvec pose
        pose_interp = PoseTrajectoryInterpolator(times=[t0], poses=[curr_pose6d])

        # prepublish one state to ring buffer
        state0 = dict()
        state0['ActualTCPPose']  = curr_pose6d
        state0['ActualQ']        = np.array(robot_state.q, dtype=float)
        state0['ActualQd']       = np.array(robot_state.dq, dtype=float)
        state0['TargetTCPPose']  = curr_pose6d.copy()
        state0['TargetTCPSpeed'] = np.zeros((6,))
        state0['TargetQ']        = np.zeros((7,))   # 7-DOF
        state0['TargetQd']       = np.zeros((7,))
        state0['robot_receive_timestamp'] = time.time()
        self.ring_buffer.put(state0)
        self.ready_event.set()

        next_outer_poll = t0 + outer_dt
        keep_running = True
        iter_idx = 0

        try:
            while keep_running:
                # === (1) FCI tick: read state, get dt ===
                robot_state, duration = active_control.readOnce()
                now = time.monotonic()

                # === (2) Evaluate interpolated pose at 'now' ===
                pose_cmd = pose_interp(now)  # 6D [x,y,z,rx,ry,rz]
                T_cmd = pose6d_to_matrix(pose_cmd)
                cartesian_pose = CartesianPose(T_cmd.reshape(-1).tolist())

                # send command for this control cycle
                active_control.writeOnce(cartesian_pose)

                # === (3) Occasionally handle commands & log state (outer_hz) ===
                if now >= next_outer_poll:
                    # poll commands (take latest)
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    if n_cmd > 0:
                        command = {k: v[n_cmd - 1] for k, v in commands.items()}
                        cmd = command['cmd']

                        if cmd == Command.STOP.value:
                            keep_running = False

                        elif cmd == Command.SERVOL.value:
                            target_pose = np.array(command['target_pose'], dtype=float)
                            duration_cmd = float(command['duration'])
                            curr_time = now

                            # get current pose from interpolator (anchor)
                            curr_pose_now = pose_interp(curr_time)
                            # reset interpolator anchor to avoid discontinuity
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose_now]
                            )

                            # simple duration selection (you can copy your Aubo t_req logic here)
                            t_insert = curr_time + duration_cmd
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed
                            )

                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pose = np.array(command['target_pose'], dtype=float)
                            target_time = float(command['target_time'])

                            mono_offset = getattr(self, "_mono_offset_cached", None)
                            if mono_offset is None:
                                mono_offset = time.monotonic() - time.time()
                                self._mono_offset_cached = mono_offset
                            target_time_mono = mono_offset + target_time

                            curr_time = now
                            curr_pose_now = pose_interp(curr_time)
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose_now]
                            )
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time_mono,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=curr_time
                            )

                        else:
                            keep_running = False

                    # log state at outer rate
                    state = dict()
                    state['ActualTCPPose']   = o_t_ee_to_6d(robot_state.O_T_EE)
                    state['ActualQ']         = np.array(robot_state.q, dtype=float)
                    state['ActualQd']        = np.array(robot_state.dq, dtype=float)
                    state['TargetTCPPose']   = pose_cmd
                    state['TargetTCPSpeed']  = np.zeros((6,))
                    state['TargetQ']         = np.zeros((7,))
                    state['TargetQd']        = np.zeros((7,))
                    state['robot_receive_timestamp'] = time.time()
                    self.ring_buffer.put(state)

                    next_outer_poll = now + outer_dt

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            # graceful stop
            try:
                # send a final motion_finished= True
                pose6d = o_t_ee_to_6d(robot_state.O_T_EE)
                T_final = pose6d_to_matrix(pose6d)
                cmd = CartesianPose(T_final.reshape(-1).tolist())
                cmd.motion_finished = True
                active_control.writeOnce(cmd)
            except Exception:
                pass
            try:
                robot.stop()
            except Exception:
                pass
            self.ready_event.set()
            if self.verbose:
                print(f"[FrankaInterpolationController] Disconnected from robot: {robot_ip}")
