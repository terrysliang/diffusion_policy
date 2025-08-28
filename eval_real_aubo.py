"""
Usage:
(robodiff)$ python eval_real_aubo.py \
    -i <ckpt_path> -o <save_dir> -ri <ip_of_aubo> \
    [--vis_camera_idx 0] [--frequency 10]

================ Human in control ==============
SpaceMouse:
- Button 0: Toggle gripper (open/close)
- Button 1: Cycle lock state: None -> Position-locked -> Orientation-locked
- Axes: 6-DoF motion; scaling matches dataset collection

Keyboard:
- Click OpenCV window to focus
- C: start evaluation (hand control to policy)
- S: stop evaluation (back to human control)
- Q: quit
- Space: stage counter (included in logs)
- Backspace: drop the current episode (with prompt)

================ Policy in control ==============
- Policy outputs absolute 6D poses [x,y,z, rx,ry,rz] in rotvec
- Space/gripper buttons still work while policy runs
- Keep hardware E-stop within reach
"""

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st

from diffusion_policy.real_world.real_env_aubo import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.real_world.gripper_controller import GripperController
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

OmegaConf.register_new_resolver("eval", eval, replace=True)

LOCK_STATES = ["None", "Lock Rx+Ry", "Only Rz"]

def lift_action_to_pose6(action: np.ndarray, template_pose: np.ndarray) -> np.ndarray:
    """
    Lift policy action to 6D target poses.
    Supports:
      - 2D:  [x, y]
      - 4D:  [x, y, z, rz]     (rx, ry held from template)
      - 6D:  [x, y, z, rx, ry, rz]
    """
    a = np.atleast_2d(action)
    targets = np.tile(template_pose, (a.shape[0], 1)).astype(np.float64)

    adim = a.shape[1]
    if   adim == 2:
        targets[:, [0, 1]] = a
        # z, rx, ry, rz remain whatever template_pose has
    elif adim == 4:
        targets[:, [0, 1, 2, 5]] = a
        # keep rx, ry from template (i.e., locked)
        targets[:, 3] = template_pose[3]
        targets[:, 4] = template_pose[4]
    elif adim == 6:
        targets[:, :6] = a
    else:
        raise ValueError(f"Unsupported action dim {adim}; expected 2, 4, or 6.")
    return targets

def _clip_workspace(poses, env):
    # XY clip (same as dataset collection)
    poses[:, :2] = np.clip(poses[:, :2], [0.25, -0.45], [0.77, 0.40])
    # Optional Z bounds if your env exposes them
    z_bounds = getattr(env, "z_bounds", None)
    if z_bounds is not None:
        poses[:, 2] = np.clip(poses[:, 2], z_bounds[0], z_bounds[1])
    return poses

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save evaluation logs/videos')
@click.option('--robot_ip', '-ri', required=True, help="Aubo controller IP, e.g. 192.168.100.77")
@click.option('--match_dataset', '-m', default=None, help='Dataset for frame overlay (alignment aid)')
@click.option('--match_episode', '-me', default=None, type=int, help='Specific episode index for overlay')
@click.option('--vis_camera_idx', default=0, type=int, help="Camera index to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Initialize robot joint configuration at start.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon per inference.")
@click.option('--max_duration', '-md', default=60, type=float, help='Per-episode max duration (sec).')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency (Hz).")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency compensation (sec).")
@click.option('--device_ids', default=None, type=str, help='Comma-separated video device IDs, e.g. "8,16,0"')
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints,
         steps_per_inference, max_duration,
         frequency, command_latency,
         device_ids):
    # -------- optional overlay frames (first frames of episodes) --------
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # -------- load checkpoint/policy --------
    payload = torch.load(open(input, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset = 0
    delta_action = False  # you said it's OFF
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.num_inference_steps = 16
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    elif 'robomimic' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)
    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096
        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # -------- experiment setup --------
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps:", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)


    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
             Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=obs_res,
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                device_ids=[8, 16, 6]
            ) as env, \
            GripperController("/dev/ttyUSB0") as gripper:

            cv2.setNumThreads(1)

            # Optional camera properties (safe-guarded)
            for cam in getattr(env.multi_camera, "cameras", {}).values():
                try:
                    cam.set_exposure(exposure=120, gain=0)
                    cam.set_white_balance(white_balance=5900)
                except Exception:
                    pass

            # initial states
            time.sleep(1.0)
            print("Warming up policy inference")
            obs = env.get_obs()
            # after: obs = env.get_obs()  (and any policy warm-up)
            if 'robot_eef_pose' in obs:
                target_pose = obs['robot_eef_pose'][-1].copy()
            else:
                s = env.get_robot_state()
                target_pose = s.get('ActualTCPPose', s['TargetTCPPose']).copy()

            gripper_closed = False
            gripper.set_closed(gripper_closed)
            # inject gripper channel if used by model
            # obs['robot_gripper_qpos'] = np.array([float(gripper_closed)], dtype=np.float32)

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                _warmup_action = result['action'][0].detach().to('cpu').numpy()
                print("Policy action shape:", _warmup_action.shape)
                del result

            print('Ready!')

            # Teleop state (matches your dataset script)
            # state = env.get_robot_state()
            # target_pose = state['TargetTCPPose'].copy()
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_eval_running = False  # policy in control
            lock_state = 0  # 0: none, 1: position locked, 2: orientation locked
            last_lock_button = False
            last_gripper_button = False
            
            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # --- key handling (works in both human/policy phases) ---
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # end everything
                        if is_eval_running:
                            env.end_episode()
                        stop = True
                    elif key_stroke == KeyCode(char='c') and not is_eval_running:
                        # start evaluation
                        start_time = t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()
                        env.start_episode(start_time)
                        key_counter.clear()
                        is_eval_running = True
                        print('Evaluation started.')
                    elif key_stroke == KeyCode(char='s') and is_eval_running:
                        # stop evaluation
                        env.end_episode()
                        key_counter.clear()
                        is_eval_running = False
                        print('Evaluation stopped.')
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop the current episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_eval_running = False

                # stage counter from Space key (same as dataset)
                stage = key_counter[Key.space]

                # --- SpaceMouse buttons: gripper + lock cycle ---
                current_gripper_button = sm.is_button_pressed(0)
                current_lock_button = sm.is_button_pressed(1)

                if current_gripper_button and not last_gripper_button:
                    gripper_closed = not gripper_closed
                    gripper.set_closed(gripper_closed)

                if current_lock_button and not last_lock_button:
                    lock_state = (lock_state + 1) % 3

                last_gripper_button = current_gripper_button
                last_lock_button = current_lock_button

                # --- pump obs; inject gripper channel if present in model ---
                obs = env.get_obs()
                # obs['robot_gripper_qpos'] = np.array([float(gripper_closed)], dtype=np.float32)

                # --- visualization (overlay optional) ---
                vis_img = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                lock_str = LOCK_STATES[lock_state]
                grip_str = "Closed" if gripper_closed else "Open"
                text = f'Episode: {episode_id}, Stage: {stage}, Lock: {lock_str}, Gripper: {grip_str}'
                if is_eval_running:
                    text += ', Policy'
                cv2.putText(vis_img, text, (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, thickness=2, color=(255, 255, 255))

                # optional alignment overlay
                match_episode_id = match_episode if match_episode is not None else episode_id
                if match_episode_id in episode_first_frame_map:
                    match_img = episode_first_frame_map[match_episode_id]
                    ih, iw, _ = match_img.shape
                    oh, ow, _ = vis_img.shape
                    tf = get_image_transform((iw, ih), (ow, oh), bgr_to_rgb=False)
                    match_img = tf(match_img).astype(np.float32) / 255
                    # blend with min (same trick as original)
                    vis_img = np.minimum(vis_img.astype(np.float32)/255, match_img).astype(np.float32)
                    vis_img = (vis_img * 255).astype(np.uint8)

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                # --- timing anchor before control ---
                precise_wait(t_sample)

                if not is_eval_running:
                    # ==================== HUMAN-IN-CONTROL (dataset-matched) ====================
                    sm_state = sm.get_motion_state_transformed()
                    spacemouse_scale = 0.5
                    dpos = sm_state[:3] * spacemouse_scale * (env.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * spacemouse_scale * (env.max_rot_speed / frequency)

                    if lock_state == 0:  # Nothing locked
                        pass
                    elif lock_state == 1:  # Lock rx and ry, leave rz + position free
                        drot_xyz[0] = 0.0  # rx
                        drot_xyz[1] = 0.0  # ry
                    elif lock_state == 2:  # Only rz allowed; lock position + rx + ry
                        dpos[:] = 0.0
                        drot_xyz[0] = 0.0  # rx
                        drot_xyz[1] = 0.0  # ry

                    target_pose[:3] += dpos
                    if not np.allclose(drot_xyz, 0):
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                    # clip workspace same as dataset
                    # target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40])

                    env.exec_actions(
                        actions=[target_pose],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        stages=[stage]
                    )

                else:
                    # ==================== POLICY-IN-CONTROL ====================
                    try:
                        # build obs_dict (with gripper channel if present)
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0].detach().to('cpu').numpy()
                            # Lift 2D/4D/6D -> 6D target poses
                            this_target_poses = lift_action_to_pose6(action, template_pose=target_pose)
                            # Optional safety clamp
                            # this_target_poses = _clip_workspace(this_target_poses, env)
                            print('Inference latency:', time.time() - s)

                        # schedule
                        obs_timestamps = obs['timestamp']
                        action_timestamps = (np.arange(this_target_poses.shape[0], dtype=np.float64) + action_offset
                                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)

                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - (t_start + dt)) / dt))
                            action_timestamps = np.array([t_start + next_step_idx * dt + (time.time() - time.monotonic())])
                            print('Over budget', action_timestamps[0] - curr_time)
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            stages=[stage] * len(action_timestamps)
                        )
                        print(f"Submitted {len(this_target_poses)} step(s).")

                    except KeyboardInterrupt:
                        print("Interrupted! Ending episode.")
                        env.end_episode()
                        is_eval_running = False

                precise_wait(t_cycle_end)
                iter_idx += 1

    print("Done.")

if __name__ == '__main__':
    main()
