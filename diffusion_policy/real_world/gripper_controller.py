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

    def __enter__(self):
        self.connect(slave_id=1)  # Default slave_id; override if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self, slave_id=1):
        # Pymodbus parity: 'N' (none), 'E' (even), 'O' (odd)
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
        self.slave_id = slave_id
        self.connected = True

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
        self.connected = False

    def activate_gripper(self, activation_register):
        # Send activation command (0x0001)
        command = 0x0001
        result = self.client.write_register(
            activation_register, command, unit=self.slave_id)
        if result.isError():
            print(f"Failed to activate gripper on port: {self.serial_port}")
            return False
        return True

    def set_gripper_position(self, position_value):
        position_register = 0x0103
        status_register = 0x0201
        max_attempts = 50
        delay_sec = 0.1

        for attempt in range(max_attempts):
            # Write target position
            result = self.client.write_register(
                position_register, position_value, unit=self.slave_id)
            if result.isError():
                print(f"Failed to write position to gripper on port: {self.serial_port}, reason: {result}")
                continue

            # Read gripper status
            rr = self.client.read_holding_registers(
                status_register, 1, unit=self.slave_id)
            if rr.isError():
                print(f"Failed to read status register on port: {self.serial_port}, reason: {rr}")
                continue

            status = rr.registers[0]
            # status == 1 or 2 means action complete
            if status == 1 or status == 2:
                print(f"Gripper action complete (status: {status})")
                return True

            time.sleep(delay_sec)

        print("Gripper did not reach the target position within the expected time.")
        return False

    def set_closed(self, closed=True):
        # Example: closed=True => position_value=1000; open => 0
        pos = 1000 if closed else 0
        return self.set_gripper_position(pos)

    def open(self):
        return self.set_closed(False)

    def close(self):
        return self.set_closed(True)

if __name__ == '__main__':
    # Usage example (replace with your serial port):
    with GripperController('/dev/ttyUSB0') as gripper:
        print("Activating gripper...")
        gripper.activate_gripper(activation_register=0x0100)  # Use your actual register
        print("Opening gripper...")
        gripper.open()
        time.sleep(1)
        print("Closing gripper...")
        gripper.close()
