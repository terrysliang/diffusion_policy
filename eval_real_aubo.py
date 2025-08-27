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

# -------------------- helpers --------------------
def _expected_obs_shapes(shape_meta, n_obs_steps):
    exp = {}
    for k, attr in shape_meta['obs'].items():
        typ = attr.get('type', 'low_dim')
        shape = tuple(attr['shape'])
        if typ == 'rgb':
            c, h, w = shape
            exp[k] = ('rgb', (n_obs_steps, c, h, w))
        else:
            # low-dim: shape like (D,)
            d = shape[0]
            exp[k] = ('low', (n_obs_steps, d))
    return exp

def _ensure_T(x, T):
    """Make sure leading time dim is T (pad by repeating last, or trim)."""
    if x.ndim == 3:          # HWC or CHW -> add T
        x = x[None, ...]
    if x.shape[0] < T:
        last = x[-1:]
        x = np.concatenate([x] + [last] * (T - x.shape[0]), axis=0)
    elif x.shape[0] > T:
        x = x[-T:]
    return x

def _resize_frame_hw(frame_hw_c, H, W):
    """Resize a single HWC frame to HxW."""
    return cv2.resize(frame_hw_c, (W, H), interpolation=cv2.INTER_AREA)

def coerce_obs_shapes(obs_dict_np: dict, shape_meta: dict, n_obs_steps: int) -> dict:
    """
    Make obs tensors match (T,C,H,W) for rgb and (T,D) for low-dim
    as specified by shape_meta. Also converts *pose* keys to the
    expected dimensionality (e.g., 6D -> 4D [x,y,z,rz]).
    """
    exp = _expected_obs_shapes(shape_meta, n_obs_steps)
    out = {}
    for k, (kind, exp_shape) in exp.items():
        if k not in obs_dict_np:
            continue
        x = np.asarray(obs_dict_np[k])

        if kind == 'rgb':
            T, C, H, W = exp_shape
            x = _ensure_T(x, T)

            if x.ndim == 4 and x.shape[-1] == 3:     # THWC
                if (x.shape[1], x.shape[2]) != (H, W):
                    x = np.stack([_resize_frame_hw(f, H, W) for f in x], axis=0)
                x = x.transpose(0, 3, 1, 2)           # THWC -> TCHW
            elif x.ndim == 4 and x.shape[1] == 3:     # TCHW
                if (x.shape[2], x.shape[3]) != (H, W):
                    frames = []
                    for f in x:                        # f: CHW
                        f_hw_c = np.moveaxis(f, 0, -1)
                        f_hw_c = _resize_frame_hw(f_hw_c, H, W)
                        frames.append(np.moveaxis(f_hw_c, -1, 0))
                    x = np.stack(frames, 0)
            else:
                raise ValueError(f"Unexpected rgb shape for key '{k}': {x.shape}")

            if x.dtype == np.uint8:
                x = x.astype(np.float32) / 255.0
            else:
                x = x.astype(np.float32)
            out[k] = x

        else:
            # ----- low-dim -----
            T, D = exp_shape
            x = _ensure_T(x, T)

            # unify to (T, Dim)
            if x.ndim == 1:
                x = x[None, ...]
            if x.ndim > 2:
                x = x.reshape(x.shape[0], -1)

            # Special handling for pose keys
            # If model expects 4D pose but runtime has 6D rotvec,
            # keep [x,y,z,rz] -> indices [0,1,2,5].
            if ('pose' in k) and (D == 4) and (x.shape[1] >= 6):
                x = x[:, [0, 1, 2, 5]]

            # If still wrong dim, last resort: trim or pad with last value
            if x.shape[1] != D:
                if x.shape[1] > D:
                    x = x[:, :D]
                else:
                    pad = np.repeat(x[:, -1:], D - x.shape[1], axis=1)
                    x = np.concatenate([x, pad], axis=1)

            out[k] = x.astype(np.float32)

    return out

def _strip_module_prefix(sd: dict) -> dict:
    """Remove leading 'module.' added by DDP wrapping."""
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _maybe_inject_gripper(obs: dict, shape_meta: dict, closed: bool):
    obs_meta = shape_meta.get('obs', {})
    if 'robot_gripper_qpos' in obs_meta:
        obs['robot_gripper_qpos'] = np.array([float(closed)], dtype=np.float32)

def _parse_device_ids(s: str):
    if not s:
        return None
    return [int(x) for x in s.split(',') if x.strip() != ""]

def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    # must have mostly string keys
    num_keys = len(d)
    num_str_keys = sum(isinstance(k, str) for k in d.keys())
    if num_str_keys < max(1, int(0.7 * num_keys)):
        return False

    # either many tensor-ish values OR many dotted string keys
    has_tensors = any(isinstance(v, torch.Tensor) for v in d.values())
    has_familiar_names = any(
        isinstance(k, str) and (
            "obs_encoder" in k or "down_modules" in k or "up_modules" in k or
            "normalizer" in k or k.startswith("model.") or k.startswith("module.")
        )
        for k in d.keys()
    )
    # also allow typical param/buffer shape (float/ndarray) as weak signal
    has_arrayish = any(
        isinstance(v, (np.ndarray,)) for v in d.values()
    )
    return has_tensors or has_familiar_names or has_arrayish

def _find_state_dict(payload: dict) -> dict:
    # 1) common top-level locations
    for k in ["ema_model", "model", "state_dict", "model_state_dict", "weights"]:
        v = payload.get(k, None)
        if isinstance(v, dict) and _looks_like_state_dict(v):
            return v

    # 2) shallow nested dicts
    for outer in ["payload", "state", "states", "checkpoints"]:
        v = payload.get(outer, None)
        if isinstance(v, dict):
            for _, vv in v.items():
                if isinstance(vv, dict) and _looks_like_state_dict(vv):
                    return vv

    # 3) bounded recursive scan to avoid weird structures
    seen = set()
    def _scan(obj, depth=0, max_depth=4):
        if depth > max_depth:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        if isinstance(obj, dict):
            if _looks_like_state_dict(obj):
                return obj
            for vv in obj.values():
                hit = _scan(vv, depth+1, max_depth)
                if hit is not None:
                    return hit
        elif isinstance(obj, (list, tuple)):
            for vv in obj:
                hit = _scan(vv, depth+1, max_depth)
                if hit is not None:
                    return hit
        return None

    hit = _scan(payload)
    if hit is None:
        raise RuntimeError("Could not locate a model state_dict in checkpoint.")
    return hit

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
    elif adim == 4:
        targets[:, [0, 1, 2, 5]] = a
        targets[:, 3] = template_pose[3]
        targets[:, 4] = template_pose[4]
    elif adim == 6:
        targets[:, :6] = a
    else:
        raise ValueError(f"Unsupported action dim {adim}; expected 2, 4, or 6.")
    return targets

def _clip_workspace(poses, env):
    poses[:, :2] = np.clip(poses[:, :2], [0.25, -0.45], [0.77, 0.40])
    z_bounds = getattr(env, "z_bounds", None)
    if z_bounds is not None:
        poses[:, 2] = np.clip(poses[:, 2], z_bounds[0], z_bounds[1])
    return poses
# ------------------------------------------------

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

    # -------- load checkpoint/policy (robust / DDP-safe) --------
    payload = torch.load(open(input, 'rb'), pickle_module=dill, map_location='cpu')

    cfg = payload.get('cfg', None)
    if cfg is None:
        raise RuntimeError("Checkpoint missing 'cfg'. Make sure this is a training checkpoint.")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)  # build the policy class & config

    # find and prep the state_dict
    state_dict = _find_state_dict(payload)
    state_dict = _strip_module_prefix(state_dict)

    # pick the model object to load into (prefer EMA if config says so and it exists)
    use_ema = bool(cfg.get('training', {}).get('use_ema', False))
    target_model = workspace.ema_model if (use_ema and getattr(workspace, "ema_model", None) is not None) else workspace.model

    missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:    print("  (first few missing)", missing[:8])
    if unexpected: print("  (first few unexpected)", unexpected[:8])

    # pick policy and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy: BaseImagePolicy = target_model
    policy.eval().to(device)

    # typical diffusion-policy inference knobs (safe if they exist)
    if hasattr(policy, "horizon") and hasattr(policy, "n_obs_steps"):
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = max(1, getattr(policy, "num_inference_steps", 16))


    # -------- policy selection & basic setup --------
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.eval().to(device)
        # for image policy: infer steps / horizon
        if hasattr(policy, "horizon") and hasattr(policy, "n_obs_steps"):
            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        policy.num_inference_steps = getattr(policy, "num_inference_steps", 16)
    elif 'robomimic' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)
    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.eval().to(device)
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096
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

    cam_device_ids = _parse_device_ids(device_ids)

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
            if 'robot_eef_pose' in obs:
                target_pose = obs['robot_eef_pose'][-1].copy()
            else:
                s = env.get_robot_state()
                target_pose = s.get('ActualTCPPose', s['TargetTCPPose']).copy()

            gripper_closed = False
            gripper.set_closed(gripper_closed)

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict_np = coerce_obs_shapes(obs_dict_np, cfg.task.shape_meta, n_obs_steps)
                if not hasattr(coerce_obs_shapes, "_printed"):
                    exp = _expected_obs_shapes(cfg.task.shape_meta, n_obs_steps)
                    for k, (_, shp) in exp.items():
                        if k in obs_dict_np:
                            print(f"[debug] {k} -> got {obs_dict_np[k].shape}, expect {shp}")
                    coerce_obs_shapes._printed = True

                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                _warmup_action = result['action'][0].detach().to('cpu').numpy()
                print("Policy action shape:", _warmup_action.shape)
                del result

            print('Ready!')

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

                n_sched_this_iter = 1   # default for human mode

                # --- key handling (works in both human/policy phases) ---
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        if is_eval_running:
                            env.end_episode()
                        stop = True
                    elif key_stroke == KeyCode(char='c') and not is_eval_running:
                        # Start evaluation with a clean timebase
                        start_delay = 1.0
                        eval_start_wall = time.time() + start_delay
                        env.start_episode(eval_start_wall)

                        # Reset the loop's monotonic anchor and iteration index
                        t_start = time.monotonic() + start_delay
                        iter_idx = 0

                        # brief settle to reduce initial camera latency
                        time.sleep(max(0.0, start_delay - 1.0/30.0))

                        key_counter.clear()
                        is_eval_running = True
                        print('Evaluation started.')

                    elif key_stroke == KeyCode(char='s') and is_eval_running:
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

                # --- pump obs; inject gripper if present in model ---
                obs = env.get_obs()
                _maybe_inject_gripper(obs, cfg.task.shape_meta, gripper_closed)

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

                match_episode_id = match_episode if match_episode is not None else episode_id
                if match_episode_id in episode_first_frame_map:
                    match_img = episode_first_frame_map[match_episode_id]
                    ih, iw, _ = match_img.shape
                    oh, ow, _ = vis_img.shape
                    tf = get_image_transform((iw, ih), (ow, oh), bgr_to_rgb=False)
                    match_img = tf(match_img).astype(np.float32) / 255
                    vis_img = np.minimum(vis_img.astype(np.float32)/255, match_img).astype(np.float32)
                    vis_img = (vis_img * 255).astype(np.uint8)

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                # --- timing anchor before control ---
                precise_wait(t_sample)

                if not is_eval_running:
                    # ==================== HUMAN-IN-CONTROL ====================
                    sm_state = sm.get_motion_state_transformed()
                    spacemouse_scale = 0.5
                    dpos = sm_state[:3] * spacemouse_scale * (env.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * spacemouse_scale * (env.max_rot_speed / frequency)

                    if lock_state == 1:  # Lock rx and ry, leave rz + position free
                        drot_xyz[0] = 0.0
                        drot_xyz[1] = 0.0
                    elif lock_state == 2:  # Only rz allowed; lock position + rx + ry
                        dpos[:] = 0.0
                        drot_xyz[0] = 0.0
                        drot_xyz[1] = 0.0

                    target_pose[:3] += dpos
                    if not np.allclose(drot_xyz, 0):
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                    env.exec_actions(
                        actions=[target_pose],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        stages=[stage]
                    )

                else:
                    # ==================== POLICY-IN-CONTROL ====================
                    try:
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict_np = coerce_obs_shapes(obs_dict_np, cfg.task.shape_meta, n_obs_steps)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0].detach().to('cpu').numpy()

                            # Lift 2D/4D/6D -> 6D target poses
                            this_target_poses = lift_action_to_pose6(action, template_pose=target_pose)
                            # Optional clamp:
                            # this_target_poses = _clip_workspace(this_target_poses, env)

                            print('Inference latency:', time.time() - s)

                        # Build timestamps for this burst
                        obs_timestamps = obs['timestamp']
                        action_timestamps = (np.arange(this_target_poses.shape[0], dtype=np.float64) + action_offset
                                            ) * dt + obs_timestamps[-1]

                        exec_cushion = 0.01
                        now = time.time()
                        is_new = action_timestamps > (now + exec_cushion)

                        if np.sum(is_new) == 0:
                            # Over budget: schedule a single step slightly in the future
                            this_target_poses = this_target_poses[[-1]]
                            action_timestamps = np.array([now + 2*dt])
                            n_sched_this_iter = 1
                            print('Over budget, scheduling 1 step at', action_timestamps[0] - now, 's ahead')
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]
                            n_sched_this_iter = len(action_timestamps)

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            stages=[stage] * n_sched_this_iter
                        )
                        print(f"Submitted {n_sched_this_iter} step(s).")

                        # Keep the template pose in sync for the next lift
                        target_pose = this_target_poses[-1]

                    except KeyboardInterrupt:
                        print("Interrupted! Ending episode.")
                        env.end_episode()
                        is_eval_running = False
                        n_sched_this_iter = 1

                precise_wait(t_cycle_end)
                if is_eval_running:
                    iter_idx += max(1, n_sched_this_iter)
                else:
                    iter_idx += 1

    print("Done.")

if __name__ == '__main__':
    main()
