import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from pymodbus.client.sync import ModbusSerialClient as ModbusClient

class GripperController(mp.Process):
    def __init__(self, shm_manager, serial_port, baud_rate=115200, parity='N', data_bits=8, stop_bits=1):
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.parity = parity
        self.data_bits = data_bits
        self.stop_bits = stop_bits
        self.slave_id = 1
        self._client = None

        # Make a SharedMemoryQueue for commands
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={'closed': False},  # just a bool: True=close, False=open
            buffer_size=16
        )

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    def __enter__(self):
        self.start()
        self.ready_event.wait()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.join()

    # MAIN PROCESS API
    def set_closed(self, closed=True):
        self.command_queue.put({'closed': closed})

    def open(self):
        self.set_closed(False)

    def close(self):
        self.set_closed(True)

    # The actual process loop
    def run(self):
        self._client = ModbusClient(
            method='rtu',
            port=self.serial_port,
            baudrate=self.baud_rate,
            parity=self.parity,
            stopbits=self.stop_bits,
            bytesize=self.data_bits,
            timeout=1
        )
        if not self._client.connect():
            print(f"Failed to connect to gripper at {self.serial_port}")
            return
        print(f"[GripperController] Connected to {self.serial_port}")
        self.ready_event.set()
        while not self.stop_event.is_set():
            try:
                # get_all returns a dict with arrays, so process all commands in the queue
                commands = self.command_queue.get_all()
                n = len(commands['closed'])
                for i in range(n):
                    closed = commands['closed'][i]
                    print(f"[GripperController] set_closed({closed}) called")
                    self._set_closed_impl(closed)
            except Empty:
                time.sleep(0.01)
        self._client.close()
        print(f"[GripperController] Exiting process.")

    def _set_closed_impl(self, closed):
        pos = 50 if closed else 500
        self.set_gripper_position(pos)

    def set_gripper_position(self, position_value):
        POSITION_REGISTER = 0x0103
        GRIPPER_STATE_REGISTER = 0x0201
        ACTUAL_POSITION_REGISTER = 0x0202

        result = self._client.write_register(POSITION_REGISTER, position_value, unit=self.slave_id)
        if result.isError():
            print(f"Failed to write position: {result}")
            return False
        time.sleep(0.05)
        max_attempts = 50
        for i in range(max_attempts):
            state = self._client.read_holding_registers(GRIPPER_STATE_REGISTER, 1, unit=self.slave_id)
            if state.isError():
                print(f"Failed to read gripper state: {state}")
                continue
            status = state.registers[0]
            if status in [1, 2, 3]:
                break
            time.sleep(0.1)
        pos = self._client.read_holding_registers(ACTUAL_POSITION_REGISTER, 1, unit=self.slave_id)
        if pos.isError():
            print(f"Actual gripper position: {pos.registers[0]}")
        return True

# Example usage
if __name__ == '__main__':
    with SharedMemoryManager() as shm_mgr:
        with GripperController(shm_mgr, '/dev/ttyUSB0') as gripper:
            print("Opening gripper...")
            gripper.open()
            time.sleep(2)
            print("Closing gripper...")
            gripper.close()
            time.sleep(2)
