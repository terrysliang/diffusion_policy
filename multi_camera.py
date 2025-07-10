from typing import List, Optional, Union, Dict, Callable
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from diffusion_policy.real_world.single_camera import SingleCamera
from diffusion_policy.real_world.video_recorder import VideoRecorder

def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

class MultiCamera:
    def __init__(self,
        device_ids: Optional[List[int]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        get_max_k=30,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose=False
    ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if device_ids is None:
            # Default to first two cameras (adjust as needed or auto-detect)
            device_ids = [0, 1]
        n_cameras = len(device_ids)

        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)

        cameras = dict()
        for i, dev_id in enumerate(device_ids):
            cameras[dev_id] = SingleCamera(
                shm_manager=shm_manager,
                device_id=dev_id,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                get_max_k=get_max_k,
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )
        
        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        return all(camera.is_ready for camera in self.cameras.values())
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None and results:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for camera in self.cameras.values():
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

if __name__ == '__main__':
    from multiprocessing.managers import SharedMemoryManager
    with SharedMemoryManager() as shm_mgr:
        cam_sys = MultiCamera(
            device_ids=[0, 2],  # or whatever indices your cameras are
            shm_manager=shm_mgr,
            resolution=(640, 480),
            capture_fps=30
        )
        cam_sys.start()
        cam_sys.start_wait()
        print('All cameras ready:', cam_sys.is_ready)
        frames = cam_sys.get(k=1)
        for cam_idx, data in frames.items():
            print(f'Cam {cam_idx} latest ts:', data['timestamp'][-1])
        cam_sys.stop()
