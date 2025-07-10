from typing import Optional, Callable, Dict
import enum
import time
import numpy as np
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleCamera(mp.Process):
    MAX_PATH_LENGTH = 4096  # for video_path

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            device_id=0,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,         # ignored for OpenCV, but kept for API compatibility
            enable_infrared=False,      # ignored for OpenCV, but kept for API compatibility
            get_max_k=30,
            advanced_mode_config=None,  # ignored, placeholder for API
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False
        ):
        super().__init__()
        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps
        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=1
            )

        shape = tuple(resolution[::-1])  # H, W
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )
        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        command_examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': 0,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=command_examples,
            buffer_size=128
        )

        # dummy intrinsics array for compatibility (not used for OpenCV)
        self.intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(7,),
            dtype=np.float64)
        self.intrinsics_array.get()[:] = 0

        self.device_id = device_id
        self.resolution = tuple(resolution)
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def get_intrinsics(self):
        # No real calibration for OpenCV cam, return zeros/identity
        assert self.ready_event.is_set()
        return np.eye(3)

    def get_depth_scale(self):
        # Not supported for OpenCV cam, return dummy
        assert self.ready_event.is_set()
        return 1.0

    # --- The following API keeps compatibility, but only record/put/restart are actually supported ---

    def set_color_option(self, option, value):
        # Not supported, for API compatibility only
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option,
            'option_value': value
        })

    def set_exposure(self, exposure=None, gain=None):
        self.set_color_option(100, exposure if exposure else 0)  # 100 dummy

    def set_white_balance(self, white_balance=None):
        self.set_color_option(101, white_balance if white_balance else 0)  # 101 dummy

    def start_recording(self, video_path: str, start_time: float = -1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })

    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)

        cap = cv2.VideoCapture(self.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        put_idx = None
        put_start_time = self.put_start_time
        if put_start_time is None:
            put_start_time = time.time()

        iter_idx = 0
        t_start = time.time()
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[SingleCamera] Frame grab failed.")
                continue
            receive_time = time.time()
            data = dict()
            if self.enable_color:
                data['color'] = frame
            data['camera_receive_timestamp'] = receive_time
            data['camera_capture_timestamp'] = receive_time  # No HW timestamp
            if self.enable_depth:
                data['depth'] = np.zeros(self.resolution[::-1], dtype=np.uint16)  # dummy
            if self.enable_infrared:
                data['infrared'] = np.zeros(self.resolution[::-1], dtype=np.uint8)  # dummy

            # Apply transform
            put_data = data
            if self.transform is not None:
                put_data = self.transform(dict(data))

            # put_downsample logic (like in single_realsense)
            if self.put_downsample:
                local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                    timestamps=[receive_time],
                    start_time=put_start_time,
                    dt=1/self.put_fps,
                    next_global_idx=put_idx,
                    allow_negative=True
                )
                for step_idx in global_idxs:
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
            else:
                step_idx = int((receive_time - put_start_time) * self.put_fps)
                put_data['step_idx'] = step_idx
                put_data['timestamp'] = receive_time
                self.ring_buffer.put(put_data, wait=False)

            # Signal ready
            if iter_idx == 0:
                self.ready_event.set()

            # vis buffer
            vis_data = data
            if self.vis_transform == self.transform:
                vis_data = put_data
            elif self.vis_transform is not None:
                vis_data = self.vis_transform(dict(data))
            self.vis_ring_buffer.put(vis_data, wait=False)

            # record frame
            rec_data = data
            if self.recording_transform == self.transform:
                rec_data = put_data
            elif self.recording_transform is not None:
                rec_data = self.recording_transform(dict(data))
            if self.video_recorder.is_ready():
                self.video_recorder.write_frame(rec_data['color'], frame_time=receive_time)

            # Perf print (optional)
            t_end = time.time()
            if self.verbose:
                print(f'[SingleCamera {self.device_id}] FPS {np.round(1/(t_end-t_start),1)}')
            t_start = t_end

            # fetch commands
            try:
                commands = self.command_queue.get_all()
                n_cmd = len(commands['cmd'])
            except Empty:
                n_cmd = 0

            for i in range(n_cmd):
                command = dict()
                for key, value in commands.items():
                    command[key] = value[i]
                cmd = command['cmd']
                if cmd == Command.SET_COLOR_OPTION.value:
                    pass  # Not implemented
                elif cmd == Command.SET_DEPTH_OPTION.value:
                    pass  # Not implemented
                elif cmd == Command.START_RECORDING.value:
                    video_path = str(command['video_path'])
                    start_time = command['recording_start_time']
                    if start_time < 0:
                        start_time = None
                    self.video_recorder.start(video_path, start_time=start_time)
                elif cmd == Command.STOP_RECORDING.value:
                    self.video_recorder.stop()
                    put_idx = None
                elif cmd == Command.RESTART_PUT.value:
                    put_idx = None
                    put_start_time = command['put_start_time']

            iter_idx += 1

        self.video_recorder.stop()
        cap.release()
        self.ready_event.set()
        if self.verbose:
            print(f'[SingleCamera {self.device_id}] Exiting worker process.')

# Example usage
if __name__ == '__main__':
    with SharedMemoryManager() as shm_mgr:
        cam = SingleCamera(shm_mgr, device_id=0, resolution=(640,480))
        cam.start()
        cam.start_wait()
        print('Camera ready:', cam.is_ready)
        for i in range(10):
            frame = cam.get(k=1)
            print('Frame timestamp:', frame['timestamp'][-1])
            time.sleep(0.1)
        cam.stop()
