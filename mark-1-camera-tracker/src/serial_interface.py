import time
import threading
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False


class GimbalSerialController:
    """
    High-speed USB serial interface between Mark 1 Tracker and Arduino Nano + PCA9685.
    Features auto-port discovery, framed packet transmission (<P,xxx,T,yyy>),
    rate-limited non-blocking streaming, and graceful Virtual Mock fallback.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        update_rate_hz: float = 50.0,
        timeout: float = 0.05,
    ):
        self.port = port
        self.baudrate = baudrate
        self.min_interval = 1.0 / update_rate_hz
        self.timeout = timeout

        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.last_send_time = 0.0
        self.last_pan = 90.0
        self.last_tilt = 90.0

        self._lock = threading.Lock()
        self._init_connection()

    def _auto_detect_port(self) -> Optional[str]:
        """Scans USB ports for common Arduino / USB-to-UART chipsets."""
        if not HAS_PYSERIAL:
            return None

        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            # Common Arduino Nano clones use CH340, FTDI, or CP2102
            if any(k in desc or k in hwid for k in ["ch340", "arduino", "ftdi", "cp210", "usb serial"]):
                return p.device

        # If only one COM port is available (other than COM1), use it
        usb_ports = [p.device for p in ports if p.device != "COM1"]
        return usb_ports[0] if usb_ports else None

    def _init_connection(self):
        """Attempts to open real serial port; falls back to Virtual Mock if unavailable."""
        if not HAS_PYSERIAL:
            print("[SERIAL] 'pyserial' not installed. Operating in Virtual Mock mode.")
            self.is_connected = False
            return

        target_port = self.port or self._auto_detect_port()

        if target_port:
            try:
                self.serial_conn = serial.Serial(
                    port=target_port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=0.05,
                )
                time.sleep(1.8)  # Allow Arduino DTR auto-reset cycle to complete
                self.is_connected = True
                self.port = target_port
                print(f"[SERIAL] Connected to Arduino Nano on {target_port} @ {self.baudrate} baud.")
                return
            except (serial.SerialException, OSError) as e:
                print(f"[SERIAL] Failed to open {target_port}: {e}")

        self.is_connected = False
        print("[SERIAL] No physical Arduino detected. Operating in Virtual Serial Mock mode.")

    def send_angles(self, pan: float, tilt: float) -> bool:
        """
        Transmits target angles to the Arduino Nano using the framed protocol:
        Packet format: <P,xxx.xx,T,yyy.yy>\n
        Enforces a 50 Hz rate limit to match the physical servo refresh frequency.
        """
        now = time.perf_counter()
        if (now - self.last_send_time) < self.min_interval:
            return False  # Rate limit to avoid flooding USB buffer

        self.last_send_time = now
        self.last_pan = pan
        self.last_tilt = tilt

        packet = f"<P,{pan:06.2f},T,{tilt:06.2f}>\n"

        if not self.is_connected or self.serial_conn is None:
            # Virtual mock mode: packet valid but physical transmission skipped
            return True

        try:
            with self._lock:
                self.serial_conn.write(packet.encode("ascii"))
                # Non-blocking check for ACK
                if self.serial_conn.in_waiting > 0:
                    ack = self.serial_conn.readline().decode("ascii", errors="ignore").strip()
            return True
        except Exception as e:
            print(f"[SERIAL ERROR] Write failed: {e}")
            self.is_connected = False
            return False

    def close(self):
        """Safely closes serial connection and returns servos to neutral."""
        if self.is_connected and self.serial_conn:
            try:
                self.send_angles(90.0, 90.0)
                time.sleep(0.05)
                self.serial_conn.close()
            except Exception:
                pass
            self.is_connected = False
            print("[SERIAL] Connection closed.")


# ==========================================
# STANDALONE VERIFICATION BENCHMARK
# ==========================================
if __name__ == "__main__":
    print("[SERIAL TEST] Initializing Gimbal Serial Controller...")
    ctrl = GimbalSerialController()

    print(f"[SERIAL TEST] Mode: {'HARDWARE' if ctrl.is_connected else 'VIRTUAL MOCK'}")
    print("[SERIAL TEST] Simulating 10 tracking angle transmissions at 50 Hz...")

    for i in range(10):
        test_pan = 90.0 + i * 2.5
        test_tilt = 90.0 - i * 1.5
        success = ctrl.send_angles(test_pan, test_tilt)
        print(f"Frame {i:02d} -> Commanded PAN: {test_pan:.1f} | TLT: {test_tilt:.1f} | Sent: {success}")
        time.sleep(0.02)

    ctrl.close()
    print("[SERIAL TEST] Verification complete.")
