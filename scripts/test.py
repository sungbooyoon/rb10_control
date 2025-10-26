# scan_robotiq.py
import minimalmodbus, time

PORT = '/dev/ttyUSB0'
BAUDS = [115200, 57600, 38400, 19200, 9600]
IDS   = list(range(1, 17))  # 1~16 스캔

def try_read(instr, reg):
    try:
        return instr.read_register(reg, 0)  # holding register
    except Exception:
        return None

for b in BAUDS:
    for sid in IDS:
        try:
            g = minimalmodbus.Instrument(PORT, sid)
            g.serial.baudrate = b
            g.serial.bytesize = 8
            g.serial.parity   = 'N'
            g.serial.stopbits = 1
            g.serial.timeout  = 0.2
            # Robotiq에서 자주 쓰는 레지스터 후보 두 개를 시도
            for reg in (1000, 2000):  # 0x03E8, 0x07D0
                val = try_read(g, reg)
                if val is not None:
                    print(f"[HIT] baud={b}, id={sid}, reg={reg}, val={val}")
                    raise SystemExit(0)
        except Exception:
            pass
print("No response on any baud/id candidate.")
