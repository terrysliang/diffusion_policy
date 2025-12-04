"""
Franka FR3 teleop demo with SpaceMouse and DH gripper.

Usage:
(robodiff)$ python demo_real_franka.py -o <demo_save_dir> --robot_ip <ip_of_franka>

Controls
--------
SpaceMouse:
  - 6 DOF motion:
      translation: x,y,z
      rotation:    rx,ry,rz (applied as rotvec increments)
  - Button 0: toggle gripper open/closed
  - Button 1: cycle lock state:
        0: "None"        -> full 6D motion
        1: "Lock Rot"    -> only translation
        2: "Lock Trans"  -> only rotation
Keyboard (focus the OpenCV window):
  - 'C' : start recording
  - 'S' : stop recording
  - 'Q' : quit
  - Backspace : drop last episode
"""

import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from diffusion_policy.real_world.real_env_franka import RealEnv
from diffusion_policy.real_world.gripper_controller import GripperController
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

LOCK_STATES = ["None", "Lock Rotation", "Lock Translation"]


@click.command()
@click.option('--output', '-o', required=True,
              help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True,
              help="Franka FR3 IP address, e.g. 172.16.0.3")
@click.option('--vis_camera_idx', default=0, type=int,
              help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float,
              help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float,
              help="Latency between receiving SpaceMouse command and executing on Robot in seconds.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1.0 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
             Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                 output_dir=output,
                 robot_ip=robot_ip,
                 # observation / recording resolution
                 obs_image_resolution=(1280, 720),
                 frequency=frequency,
                 init_joints=init_joints,
                 enable_multi_cam_vis=True,
                 record_raw_video=True,
                 # number of threads per camera view for video recording (H.264)
                 thread_per_video=3,
                 # video quality (lower is better quality, but slower)
                 video_crf=21,
                 shm_manager=shm_manager
             ) as env, \
             GripperController("/dev/ttyUSB0") as gripper:

            cv2.setNumThreads(1)

            # Optional: Realsense exposure & white balance
            # Adjust these to taste or comment them out if you prefer auto.
            # try:
            #     env.realsense.set_exposure(exposure=800, gain=0)
            #     env.realsense.set_white_balance(white_balance=5500)
            # except AttributeError:
            #     # If your RealEnv wraps MultiRealsense or different API
            #     print("[Warning] env.realsense does not support exposure / white balance API.")

            time.sleep(1.0)
            print("Ready!")

            # Initial robot state and target pose (rotvec 6D pose)
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose'].copy()

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            # ===== Initialize lock + gripper state =====
            lock_state = 0
            last_lock_button = False

            gripper_closed = False
            last_gripper_button = False
            gripper.set_closed(gripper_closed)

            # Scale for SpaceMouse sensitivity
            spacemouse_scale = 0.5

            while not stop:
                # Timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # Pump obs
                obs = env.get_obs()
                # Add gripper qpos to obs (same pattern as Aubo script)
                obs['robot_gripper_qpos'] = np.array(
                    [float(gripper_closed)], dtype=np.float32
                )

                # ===== Handle keyboard presses =====
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True

                    elif key_stroke == KeyCode(char='c'):
                        # Start recording from a little bit in the future
                        start_time = (
                            t_start
                            + (iter_idx + 2) * dt
                            - time.monotonic()
                            + time.time()
                        )
                        env.start_episode(start_time)
                        key_counter.clear()
                        is_recording = True
                        print("Recording!")

                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print("Stopped.")

                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                            print("Dropped last episode.")

                stage = key_counter[Key.space]

                # ===== SpaceMouse Button Logic (same semantics as Aubo) =====
                current_gripper_button = sm.is_button_pressed(0)
                current_lock_button = sm.is_button_pressed(1)

                # Gripper toggle on rising edge of button 0
                if current_gripper_button and not last_gripper_button:
                    gripper_closed = not gripper_closed
                    gripper.set_closed(gripper_closed)

                # Cycle lock state on rising edge of button 1
                if current_lock_button and not last_lock_button:
                    lock_state = (lock_state + 1) % 3
                    print(f"[Teleop] Lock state -> {LOCK_STATES[lock_state]}")

                last_gripper_button = current_gripper_button
                last_lock_button = current_lock_button

                # ===== SpaceMouse Motion Logic with Locking =====
                sm_state = sm.get_motion_state_transformed()
                # scale by env.*_speed, same idea as Aubo script
                dpos = sm_state[:3] * spacemouse_scale * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * spacemouse_scale * (env.max_rot_speed / frequency)

                if lock_state == 0:
                    # Nothing locked: full 6D motion
                    pass
                elif lock_state == 1:
                    # Lock rotation
                    drot_xyz[:] = 0.0
                elif lock_state == 2:
                    # Lock translation
                    dpos[:] = 0.0

                # Apply translation
                target_pose[:3] += dpos

                # Apply rotation (rotvec update) if any
                if not np.allclose(drot_xyz, 0.0):
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[3:] = (
                        drot * st.Rotation.from_rotvec(target_pose[3:])
                    ).as_rotvec()

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
                    color=(255, 255, 255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                # ===== Timing: sampling & action execution =====
                precise_wait(t_sample)

                env.exec_actions(
                    actions=[target_pose],
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=[stage]
                )

                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()
