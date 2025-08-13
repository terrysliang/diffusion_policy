import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_aubo import RealEnv
from diffusion_policy.real_world.gripper_controller import GripperController
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

LOCK_STATES = ["None", "Position", "Orientation"]

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="Robot's IP address (e.g. 192.168.100.77)")
@click.option('--vis_camera_idx', default=0, type=int, help="Camera index to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Initialize robot joint configuration.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency (s) between SpaceMouse command and robot execution.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1 / frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                obs_image_resolution=(640,480),
                video_capture_resolution=(640,480),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                device_ids=[9, 16, 0]
            ) as env, \
            GripperController("/dev/ttyUSB0") as gripper:

            cv2.setNumThreads(1)

            # Optionally set exposure/wb on all cameras
            for cam in env.multi_camera.cameras.values():
                try:
                    cam.set_exposure(exposure=120, gain=0)
                    cam.set_white_balance(white_balance=5900)
                except AttributeError:
                    print(f"Camera {cam} does not support exposure/wb.")

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose'].copy()  # Avoid mutating original
            
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            # Initialize states for locking and gripper
            lock_state = 0  # 0: none, 1: position locked, 2: orientation locked
            last_lock_button = False
            gripper_closed = False
            last_gripper_button = False
            gripper.set_closed(gripper_closed)

            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()
                obs['robot_gripper_qpos'] = np.array([float(gripper_closed)], dtype=np.float32)

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    # elif key_stroke == KeyCode(char='i'):
                    #     env.robot  @TODO: map a key that move the robot to its initial pose
                    #     key_counter.clear()
                    #     print('Moving to init pose.')
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False

                stage = key_counter[Key.space]

                # ===== SpaceMouse Button Logic =====
                current_gripper_button = sm.is_button_pressed(0)
                current_lock_button = sm.is_button_pressed(1)

                # Gripper toggle on rising edge
                if current_gripper_button and not last_gripper_button:
                    gripper_closed = not gripper_closed
                    gripper.set_closed(gripper_closed)  # Actual command for your gripper

                # Cycle lock state on rising edge
                if current_lock_button and not last_lock_button:
                    lock_state = (lock_state + 1) % 3

                last_gripper_button = current_gripper_button
                last_lock_button = current_lock_button

                # ===== SpaceMouse Motion Logic with Locking =====
                sm_state = sm.get_motion_state_transformed()
                spacemouse_scale = 0.5

                dpos = sm_state[:3] * spacemouse_scale * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * spacemouse_scale * (env.max_rot_speed / frequency)
                if lock_state == 0:  # Nothing locked
                    pass
                elif lock_state == 1:  # Position locked
                    dpos[:] = 0
                elif lock_state == 2:  # Orientation locked
                    drot_xyz[:] = 0

                target_pose[:3] += dpos

                if not np.allclose(drot_xyz, 0):
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                # print(f"[Teleop Debug] dpos: {dpos}, drot_xyz: {drot_xyz}, target_pose: {target_pose}")

                # ===== Visualization =====
                vis_img = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                lock_str = LOCK_STATES[lock_state]
                grip_str = "Closed" if gripper_closed else "Open"
                text = f'Episode: {episode_id}, Stage: {stage}, Lock: {lock_str}, Gripper: {grip_str}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)

                # ===== Execute Teleop Command =====
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=[stage]
                )
                precise_wait(t_cycle_end)
                iter_idx += 1

if __name__ == '__main__':
    main()
