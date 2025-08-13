import time
from pymodbus.client.sync import ModbusSerialClient as ModbusClient

class GripperController:
    def __init__(self, serial_port, baud_rate=115200, parity='N', data_bits=8, stop_bits=1):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.parity = parity
        self.data_bits = data_bits
        self.stop_bits = stop_bits
        self.client = None
        self.connected = False
        self.slave_id = 1  # Default Modbus ID for DH grippers

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        self.client = ModbusClient(
            method='rtu',
            port=self.serial_port,
            baudrate=self.baud_rate,
            parity=self.parity,
            stopbits=self.stop_bits,
            bytesize=self.data_bits,
            timeout=1
        )
        if not self.client.connect():
            raise RuntimeError(f"Failed to connect to gripper at {self.serial_port}")
        self.connected = True

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
        self.connected = False

    def activate_gripper(self):
        # 0x0100: Initialization
        INIT_REGISTER = 0x0100
        result = self.client.write_register(INIT_REGISTER, 0x01, unit=self.slave_id)
        if result.isError():
            print(f"Failed to activate gripper on port: {self.serial_port}, reason: {result}")
            return False
        time.sleep(0.05)

        # Set default force and speed using correct registers
        FORCE_REGISTER = 0x0101  # 20–100 (%)
        SPEED_REGISTER = 0x0104  # 1–100 (%)
        self.client.write_register(FORCE_REGISTER, 80, unit=self.slave_id)
        time.sleep(0.05)
        self.client.write_register(SPEED_REGISTER, 80, unit=self.slave_id)
        time.sleep(0.05)

        return True

    def set_gripper_position(self, position_value):
        POSITION_REGISTER = 0x0103
        GRIPPER_STATE_REGISTER = 0x0201
        ACTUAL_POSITION_REGISTER = 0x0202

        # Write position (0–1000)
        result = self.client.write_register(POSITION_REGISTER, position_value, unit=self.slave_id)
        if result.isError():
            print(f"Failed to write position: {result}")
            return False
        time.sleep(0.05)

        # Wait until motion complete (gripper state becomes 1, 2, or 3)
        max_attempts = 50
        for i in range(max_attempts):
            state = self.client.read_holding_registers(GRIPPER_STATE_REGISTER, 1, unit=self.slave_id)
            if state.isError():
                print(f"Failed to read gripper state: {state}")
                continue
            status = state.registers[0]
            # print(f"Gripper state: {status}")
            if status in [1, 2, 3]:  # 1: reached position, 2: object caught, 3: dropped
                break
            time.sleep(0.1)

        # Read actual gripper position
        pos = self.client.read_holding_registers(ACTUAL_POSITION_REGISTER, 1, unit=self.slave_id)
        if pos.isError():
            print(f"Actual gripper position: {pos.registers[0]}")

        return True

    def set_closed(self, closed=True):
        pos = 50 if closed else 500
        return self.set_gripper_position(pos)

    def open(self):
        return self.set_closed(False)

    def close(self):
        return self.set_closed(True)



if __name__ == '__main__':
    # Example usage
    with GripperController('/dev/ttyUSB0') as gripper:
        print("Activating gripper...")
        # gripper.activate_gripper()

        print("Opening gripper...")
        gripper.open()
        time.sleep(2)

        print("Closing gripper...")
        gripper.close()
