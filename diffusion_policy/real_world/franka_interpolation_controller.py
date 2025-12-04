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
        frequency=100,
        max_pos_speed=0.01,       # [m/s]
        max_rot_speed=0.05,        # [rad/s]
        max_pos_acc=0.05,          # [m/s^2]   <- NEW
        max_rot_acc=0.2,          # [rad/s^2] <- NEW
        launch_timeout=3.0,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
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
        assert 0 < max_pos_acc
        assert 0 < max_rot_acc

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
        self.Kp_pos = 4.0
        self.Kp_rot = 4.0

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
        outer_hz = float(self.frequency)   # for queue polling + logging only
        outer_dt = 1.0 / outer_hz

        # --- connect Franka (no external control yet) ---
        robot = Robot(robot_ip)

        # --- start external cartesian pose control ---
        active_control = robot.start_cartesian_pose_control(
            ControllerMode.JointImpedance
        )

        # FIRST state inside pose-control mode, like official example
        robot_state, duration = active_control.readOnce()
        t0 = time.monotonic()

        # Initial 6D pose
        curr_pose6d = o_t_ee_to_6d(robot_state.O_T_EE)

        # Internal command state
        cmd_pose  = curr_pose6d.copy()            # what we command
        cmd_vel   = np.zeros(6, dtype=float)      # [vx,vy,vz,wx,wy,wz]
        goal_pose = curr_pose6d.copy()            # latest requested target
        goal_arrival_time = t0                    # “already at goal”

        # Prepublish one state so env.is_ready becomes True quickly
        state0 = dict()
        state0['ActualTCPPose']   = curr_pose6d.copy()
        state0['ActualQ']         = np.array(robot_state.q, dtype=float)
        state0['ActualQd']        = np.array(robot_state.dq, dtype=float)
        state0['TargetTCPPose']   = cmd_pose.copy()
        state0['TargetTCPSpeed']  = cmd_vel.copy()
        state0['TargetQ']         = np.zeros((7,), dtype=float)
        state0['TargetQd']        = np.zeros((7,), dtype=float)
        state0['robot_receive_timestamp'] = time.time()
        self.ring_buffer.put(state0)
        self.ready_event.set()

        next_outer_poll = t0 + outer_dt
        keep_running = True
        iter_idx = 0

        # Local copies of limits
        max_pos_speed = float(self.max_pos_speed)
        max_rot_speed = float(self.max_rot_speed)
        max_pos_acc   = float(self.max_pos_acc)
        max_rot_acc   = float(self.max_rot_acc)

        # for schedule_waypoint wall→mono
        mono_offset = None
        last_time = t0

        if self.verbose:
            print("[FrankaInterpolationController] External cartesian pose control started.")
            print(f"Initial pose6d: {curr_pose6d}")

        try:
            while keep_running:
                # === (1) FCI tick: read state, get dt ===
                robot_state, duration = active_control.readOnce()
                now = time.monotonic()
                dt = duration.to_sec()
                if dt <= 0.0:
                    dt = max(now - last_time, 1e-4)
                last_time = now

                # === (2) Outer-rate work: commands & state logging ===
                if now >= next_outer_poll:
                    # ---- poll commands (take latest only) ----
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    if n_cmd > 0:
                        command = {k: v[n_cmd - 1] for k, v in commands.items()}
                        cmd_type = command['cmd']

                        if cmd_type == Command.STOP.value:
                            keep_running = False

                        elif cmd_type == Command.SERVOL.value:
                            target_pose = np.array(command['target_pose'], dtype=float)
                            duration_cmd = float(command['duration'])
                            if duration_cmd <= 0.0:
                                duration_cmd = outer_dt
                            goal_pose = target_pose
                            goal_arrival_time = now + duration_cmd

                            if self.verbose:
                                print("[FrankaInterpolationController] SERVOL to",
                                      target_pose, "over", duration_cmd, "s")

                        elif cmd_type == Command.SCHEDULE_WAYPOINT.value:
                            target_pose = np.array(command['target_pose'], dtype=float)
                            target_time_wall = float(command['target_time'])
                            if mono_offset is None:
                                mono_offset = time.monotonic() - time.time()
                            target_time_mono = mono_offset + target_time_wall

                            duration_cmd = max(target_time_mono - now, outer_dt)
                            goal_pose = target_pose
                            goal_arrival_time = now + duration_cmd

                            if self.verbose:
                                print("[FrankaInterpolationController] SCHEDULE_WAYPOINT to",
                                      target_pose, "arrive in", duration_cmd, "s")

                        else:
                            keep_running = False

                    # ---- log state at outer rate ----
                    log_state = dict()
                    log_state['ActualTCPPose'] = o_t_ee_to_6d(robot_state.O_T_EE)
                    log_state['ActualQ']       = np.array(robot_state.q, dtype=float)
                    log_state['ActualQd']      = np.array(robot_state.dq, dtype=float)
                    log_state['TargetTCPPose'] = cmd_pose.copy()
                    log_state['TargetTCPSpeed'] = cmd_vel.copy()
                    log_state['TargetQ']       = np.zeros((7,), dtype=float)
                    log_state['TargetQd']      = np.zeros((7,), dtype=float)
                    log_state['robot_receive_timestamp'] = time.time()
                    self.ring_buffer.put(log_state)

                    next_outer_poll = now + outer_dt

                # === (3) Velocity / acceleration limited servo towards goal_pose ===

                # Position error
                pos_err = goal_pose[:3] - cmd_pose[:3]

                # Orientation error via relative rotation:
                R_cmd = st.Rotation.from_rotvec(cmd_pose[3:])
                R_goal = st.Rotation.from_rotvec(goal_pose[3:])
                R_err = R_goal * R_cmd.inv()
                rot_err = R_err.as_rotvec()      # “shortest” rotation from cmd→goal

                # If we have time_to_goal>0, aim to arrive in that time.
                time_to_goal = goal_arrival_time - now
                if time_to_goal <= 0.0:
                    vel_des = np.zeros(6, dtype=float)
                else:
                    v_pos_des = pos_err / time_to_goal
                    v_rot_des = rot_err / time_to_goal
                    vel_des = np.concatenate([v_pos_des, v_rot_des])

                # --- clamp velocity ---
                v_pos_des = vel_des[:3]
                v_rot_des = vel_des[3:]

                pos_speed = np.linalg.norm(v_pos_des)
                if pos_speed > max_pos_speed > 0.0:
                    v_pos_des *= (max_pos_speed / pos_speed)

                rot_speed = np.linalg.norm(v_rot_des)
                if rot_speed > max_rot_speed > 0.0:
                    v_rot_des *= (max_rot_speed / rot_speed)

                vel_des[:3] = v_pos_des
                vel_des[3:] = v_rot_des

                # --- acceleration limiting (change from cmd_vel → vel_des) ---
                dv = vel_des - cmd_vel

                dv_pos = dv[:3]
                dv_rot = dv[3:]

                max_dv_pos = max_pos_acc * dt
                if max_dv_pos > 0.0:
                    dv_pos_norm = np.linalg.norm(dv_pos)
                    if dv_pos_norm > max_dv_pos:
                        dv_pos *= (max_dv_pos / dv_pos_norm)

                max_dv_rot = max_rot_acc * dt
                if max_dv_rot > 0.0:
                    dv_rot_norm = np.linalg.norm(dv_rot)
                    if dv_rot_norm > max_dv_rot:
                        dv_rot *= (max_dv_rot / dv_rot_norm)

                cmd_vel[:3] += dv_pos
                cmd_vel[3:] += dv_rot

                # --- integrate pose command ---
                # position:
                cmd_pose[:3] += cmd_vel[:3] * dt

                # orientation via incremental rotation
                w_inc = cmd_vel[3:] * dt
                w_norm = np.linalg.norm(w_inc)
                if w_norm > 1e-9:
                    R_inc = st.Rotation.from_rotvec(w_inc)
                    R_new = R_inc * R_cmd
                    cmd_pose[3:] = R_new.as_rotvec()

                # === (4) Send command for this control cycle ===
                T_cmd = pose6d_to_matrix(cmd_pose)
                cartesian_pose = CartesianPose(T_cmd.reshape(-1, order='F').tolist())
                active_control.writeOnce(cartesian_pose)

                if self.verbose and (iter_idx % 500 == 0):
                    print(f"[FrankaInterpolationController] dt={dt:.6f}, "
                          f"|v_pos|={np.linalg.norm(cmd_vel[:3]):.4f}, "
                          f"|v_rot|={np.linalg.norm(cmd_vel[3:]):.4f}")

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            # graceful stop
            try:
                pose6d = o_t_ee_to_6d(robot_state.O_T_EE)
                T_final = pose6d_to_matrix(pose6d)
                cmd = CartesianPose(T_final.reshape(-1, order='F').tolist())
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
