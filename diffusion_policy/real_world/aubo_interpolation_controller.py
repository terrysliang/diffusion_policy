import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R
import numpy as np
import pyaubo_sdk

from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

M_PI = 3.14159265358979323846

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class AuboInterpolationController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip,
            frequency=50,
            max_pos_speed=0.50,
            max_rot_speed=1.00,
            launch_timeout=3,
            tcp_offset_pose=None,
            joints_init=None,
            joints_init_speed=0.7,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        assert 0 < frequency <= 200   # for safety, Aubo real-time control is not 500Hz!
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)
        super().__init__(name="AuboInterpolationController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
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

        # build ring buffer for robot state
        if receive_keys is None:
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
        example = {key: np.zeros((6,), dtype=np.float64) for key in receive_keys}
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[AuboInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                print("Soft real time not enabled:", e)

        robot_ip = self.robot_ip
        # outer/planner rate (command polling / logging), can be 10, 30, 50, etc.
        outer_hz = float(self.frequency)
        outer_dt = 1.0 / outer_hz

        # fixed servo streaming period (Aubo ~125 Hz)
        servo_dt = getattr(self, "servo_internal_dt", 1.0/100.0)

        # --- connect & init (unchanged) ---
        rpc_client = pyaubo_sdk.RpcClient()
        rpc_client.setRequestTimeout(1000)
        ret = rpc_client.connect(robot_ip, 30004)
        assert ret == 0, f"Failed to connect, error code: {ret}"
        if rpc_client.hasConnected():
            rpc_client.login("aubo", "123456")
            if self.verbose:
                print("[AuboInterpolationController] RPC client connected and logged in.")
        robot_name = rpc_client.getRobotNames()[0]
        robot_interface = rpc_client.getRobotInterface(robot_name)
        mc = robot_interface.getMotionControl()

        if self.joints_init is not None:
            mc.moveJoint(self.joints_init.tolist(), self.joints_init_speed, self.joints_init_speed, 0., 0.)
            time.sleep(3.0)

        mc.setServoMode(True)
        i = 0
        while not mc.isServoModeEnabled():
            i += 1
            if i > 5:
                print("Failed to start servo mode, current state: ", mc.isServoModeEnabled())
                return -1
            time.sleep(0.005)

        # ---- build time-parameterized interpolator ----
        tcp0_euler = robot_interface.getRobotState().getTcpPose()
        curr_pose = np.array(tcp0_euler[:3] + R.from_euler('xyz', tcp0_euler[3:]).as_rotvec().tolist())
        t0 = time.monotonic()
        pose_interp = PoseTrajectoryInterpolator(times=[t0], poses=[curr_pose])

        # helpers
        def unwrap_euler(prev, cur):
            if prev is None:
                return cur
            out = cur.copy()
            for k in range(3):
                d = out[k] - prev[k]
                out[k] -= 2*np.pi * np.round(d / (2*np.pi))
            return out

        last_euler = None
        keep_running = True
        iter_idx = 0

        # decouple: servo at 125 Hz, poll commands/log at outer_hz
        next_outer_poll = t0 + outer_dt
        next_state_log  = t0  # throttle state logging if you like

        try:
            while keep_running:
                tick_start = time.monotonic()

                # === (1) Evaluate trajectory at current time ===
                pose_cmd = pose_interp(tick_start)  # rotvec orientation
                curr_euler = R.from_rotvec(pose_cmd[3:]).as_euler('xyz')
                pose_euler = np.concatenate([pose_cmd[:3], unwrap_euler(last_euler, curr_euler)])
                last_euler = pose_euler[3:].copy()

                # === (2) Stream one servo tick ===
                if not mc.isServoModeEnabled():
                    mc.setServoMode(True)
                    j = 0
                    while not mc.isServoModeEnabled():
                        j += 1
                        if j > 5:
                            print("Failed to re-enable servo mode.")
                            return -1
                        time.sleep(0.005)

                ret = mc.servoCartesian(pose_euler.tolist(), 0, 0, servo_dt, 0, 0)
                if ret != 0 and self.verbose:
                    print(f"[servoCartesian] ret={ret}")

                # === (3) Occasionally poll commands (10–50 Hz) and update interpolator ===
                now = tick_start
                if now >= next_outer_poll:
                    # handle queued commands
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    for k in range(n_cmd):
                        command = {kk: vv[k] for kk, vv in commands.items()}
                        cmd = command['cmd']
                        if cmd == Command.STOP.value:
                            keep_running = False
                            break
                        elif cmd == Command.SERVOL.value:
                            target_pose = command['target_pose']
                            duration = float(command['duration'])
                            curr_time = now
                            t_insert  = curr_time + duration
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed
                            )
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pose = command['target_pose']
                            target_time = float(command['target_time'])
                            # convert wall time to monotonic
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=None  # or track if you use it
                            )
                        else:
                            keep_running = False
                            break

                    next_outer_poll = now + outer_dt

                    # (optional) log state at outer rate (or slower)
                    state = dict()
                    tcp_pose = np.array(robot_interface.getRobotState().getTcpPose())
                    actual_pose_rotvec = np.concatenate([tcp_pose[:3], R.from_euler('xyz', tcp_pose[3:]).as_rotvec()])
                    state['ActualTCPPose']   = actual_pose_rotvec
                    state['ActualTCPSpeed']  = np.array(robot_interface.getRobotState().getTcpSpeed())
                    state['ActualQ']         = np.array(robot_interface.getRobotState().getJointPositions())
                    state['ActualQd']        = np.array(robot_interface.getRobotState().getJointSpeeds())
                    state['TargetTCPPose']   = np.array(pose_cmd)
                    state['TargetTCPSpeed']  = np.zeros((6,))
                    state['TargetQ']         = np.zeros((6,))
                    state['TargetQd']        = np.zeros((6,))
                    state['robot_receive_timestamp'] = time.time()
                    self.ring_buffer.put(state)

                # === (4) sleep to maintain servo period ===
                elapsed = time.monotonic() - tick_start
                time.sleep(max(0.0, servo_dt - elapsed))

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            mc.setServoMode(False)
            time.sleep(0.1)
            rpc_client.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[AuboInterpolationController] Disconnected from robot: {robot_ip}")


if __name__ == '__main__':
    robot_ip = "192.168.100.77" 
    test_pose = np.array([-0.46, 0.14, 0.70, 90 * M_PI / 180, 0, -90 * M_PI / 180])  # (x,y,z,Rx,Ry,Rz), meters/radians

    with SharedMemoryManager() as shm_manager:
        ctrl = AuboInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,
            verbose=True,
        )
        ctrl.start()
        print("Controller started and ready:", ctrl.is_ready)

        # Send a single servoL command to a test pose
        ctrl.servoL(test_pose, duration=1.0)
        print("Sent servoL command:", test_pose)

        # Print 10 robot states
        for i in range(10):
            state = ctrl.get_state()
            print(f"Iteration {i} state keys:", state.keys())
            print("  ActualTCPPose:", state['ActualTCPPose'])
            time.sleep(0.1)

        # Stop the controller process
        ctrl.stop()
        print("Controller stopped.")
