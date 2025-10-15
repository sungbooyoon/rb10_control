import time
import rbpodo as rb
import rclpy

IP = "10.0.2.7"
HZ = 30.0
PERIOD = 1.0 / HZ
FIRST_TIMEOUT = 1.0
LOOP_TIMEOUT = 0.05

def read_once(ch, timeout):
    try:
        return ch.request_data(timeout)
    except Exception as e:
        print(f"[WARN] request_data error: {e}")
        return None

def as_list(a, n): return [float(a[i]) for i in range(n)]
def diff_arr(prev, curr, dt):
    if prev is None or curr is None or dt is None or dt <= 0: return None
    return [(curr[i] - prev[i]) / dt for i in range(len(curr))]

def main():
    ch = rb.CobotData(IP)

    # 첫 패킷 대기
    deadline = time.time() + 5.0
    data = None
    while time.time() < deadline:
        data = read_once(ch, FIRST_TIMEOUT)
        if data is not None and getattr(data, "sdata", None) is not None:
            break
        print("[INFO] waiting first packet...")
    if data is None or getattr(data, "sdata", None) is None:
        print("[ERROR] no data received (sdata is None).")
        return
    print("[OK] first packet received.")

    rclpy.init(args=None)
    from rclpy.node import Node
    node = Node('rb10_data_reader')
    rate = node.create_rate(HZ)

    # dt는 perf_counter 기반(안정), 스탬프는 time.time 기반(공통 기준)
    prev_pc = time.perf_counter()
    prev_stamp = time.time()  # 필요하면 저장
    prev_jnt_ang = as_list(data.sdata.jnt_ang, 6)
    prev_tcp_pos = as_list(data.sdata.tcp_pos, 6)

    while True:
        data = read_once(ch, LOOP_TIMEOUT)
        if data is None or getattr(data, "sdata", None) is None:
            print("[WARN] no packet this cycle.")
        else:
            s = data.sdata

            # 공통 기준 스탬프(RealSense 등과 맞출 용도)
            stamp = time.time()

            # 속도 계산용 안정적인 dt
            now_pc = time.perf_counter()
            dt = now_pc - prev_pc if prev_pc is not None else None

            # 값 추출
            jnt_ang = as_list(s.jnt_ang, 6)          # deg
            tcp_pos = as_list(s.tcp_pos, 6)          # [mm,mm,mm,deg,deg,deg]
            jnt_vel = diff_arr(prev_jnt_ang, jnt_ang, dt)
            tcp_vel = diff_arr(prev_tcp_pos, tcp_pos, dt)
            free_drive = int(getattr(s, "is_freedrive_mode", 0))

            eft_fx = getattr(s, "eft_fx", None)
            eft_fy = getattr(s, "eft_fy", None)
            eft_fz = getattr(s, "eft_fz", None)
            eft_mx = getattr(s, "eft_mx", None)
            eft_my = getattr(s, "eft_my", None)
            eft_mz = getattr(s, "eft_mz", None)

            def fmt(v):
                if v is None: return "None"
                return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"

            print(
                f"stamp={stamp:.6f} (time.time) | dt={None if dt is None else f'{dt:.6f}s'} | FreeDrive={free_drive}\n"
                f"  jnt_ang(deg)        = {fmt(jnt_ang)}\n"
                f"  jnt_vel(deg/s)      = {fmt(jnt_vel)}\n"
                f"  tcp_pos([mm,deg])   = {fmt(tcp_pos)}\n"
                f"  tcp_vel([mm/s,deg/s])= {fmt(tcp_vel)}"
            )
            if eft_fx is not None:
                print(f"  eft = {fmt([eft_fx, eft_fy, eft_fz, eft_mx, eft_my, eft_mz])}")

            # 업데이트
            prev_pc = now_pc
            prev_stamp = stamp
            prev_jnt_ang = jnt_ang
            prev_tcp_pos = tcp_pos

        # 30 Hz 주기 유지(선택)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    finally:
        node.destroy_node()
        rclpy.shutdown()
