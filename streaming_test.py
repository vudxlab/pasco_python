"""
Thử nghiệm STREAMING MODE cho cảm biến PASCO
— khai thác đường thông báo BLE sẵn có trong pasco_ble_device.py
và chỉ dùng lệnh streaming thiết bị‑đặc‑thù khi có (Rotary Motion)
"""

import sys
sys.path.append('src')

from pasco.pasco_ble_device import PASCOBLEDevice
import time

class PASCOStreamingDevice(PASCOBLEDevice):
    """
    Bắt dữ liệu dạng streaming qua BLE notifications.
    - Nếu thiết bị là "Rotary Motion": pasco_ble_device đã tự gửi WIRELESS_RMS_START
      khi connect, nên sẽ có dữ liệu định kỳ.
    - Thiết bị khác: nếu không có API streaming công khai, sẽ fallback sang fast polling.
    """

    def __init__(self):
        super().__init__()
        self.streaming_active = False
        self.streaming_data = []  # list of {timestamp, values: {measurement: value}}
        self._selected_measurements = []

    def set_selected_measurements(self, measurements):
        self._selected_measurements = list(measurements or [])

    def fast_polling_mode(self, measurements, duration=20):
        print("\nFAST POLLING (không cần streaming API)")
        print(f"  Measurements: {measurements}")
        print(f"  Duration: {duration}s\n")

        all_data = []
        start_time = time.time()
        last_progress_time = start_time
        last_count = 0

        try:
            while (time.time() - start_time) < duration:
                try:
                    data_point = self.read_data_list(measurements)
                    data_point['timestamp'] = time.time() - start_time
                    all_data.append(data_point)
                except Exception as e:
                    print(f"  Lỗi đọc: {e}")

                now = time.time()
                if (now - last_progress_time) >= 2.0:
                    elapsed = now - start_time
                    total = len(all_data)
                    inst = (total - last_count) / 2.0
                    avg = total / elapsed if elapsed > 0 else 0
                    print(f"  {elapsed:.1f}s: {total} mẫu | Tức thì: {inst:.1f} Hz | TB: {avg:.1f} Hz")
                    last_progress_time = now
                    last_count = total

            total_time = time.time() - start_time
            freq = len(all_data) / total_time if total_time > 0 else 0
            print(f"\nKết thúc. Tổng mẫu: {len(all_data)} | {freq:.2f} Hz")
            return all_data
        except Exception as e:
            print(f"\nLỗi: {e}")
            return all_data

    # Hook vào đường decode của lớp cha để thu mẫu theo từng notify
    def process_measurement_response(self, sensor_id, data):
        # Gọi xử lý gốc để cập nhật self._data_results
        super().process_measurement_response(sensor_id, data)

        # data[0] <= 0x1F: gói periodic từ thiết bị
        if self.streaming_active and (len(data) > 0 and data[0] <= 0x1F):
            snapshot = {}
            if self._selected_measurements:
                for m in self._selected_measurements:
                    try:
                        snapshot[m] = self.data_results().get(m)
                    except Exception:
                        snapshot[m] = None
            else:
                # nếu không chọn, lấy toàn bộ (có thể nhiều)
                snapshot = dict(self.data_results())

            self.streaming_data.append({
                'timestamp': time.time(),
                'values': snapshot,
            })

    def collect_streaming(self, measurements, duration=10):
        """
        Thu thập dữ liệu dựa trên BLE notifications trong 'duration' giây.
        Lưu ý: Yêu cầu thiết bị thực sự phát stream định kỳ (ví dụ Rotary Motion).
        """
        self.set_selected_measurements(measurements)
        self.streaming_data = []
        self.streaming_active = True

        print("\nSTREAMING (qua notifications)")
        print(f"  Measurements: {measurements}")
        print(f"  Duration: {duration}s\n")

        start = time.time()
        last_progress = start
        last_count = 0

        while (time.time() - start) < duration:
            time.sleep(0.01)  # nhường CPU, notifications vẫn chạy nền
            now = time.time()
            if (now - last_progress) >= 2.0:
                total = len(self.streaming_data)
                inst = (total - last_count) / 2.0
                avg = total / (now - start)
                print(f"  {now-start:.1f}s: {total} mẫu | Tức thì: {inst:.1f} Hz | TB: {avg:.1f} Hz")
                last_progress = now
                last_count = total

        self.streaming_active = False

        total = len(self.streaming_data)
        elapsed = time.time() - start
        freq = total / elapsed if elapsed > 0 else 0
        print(f"\nKết thúc streaming. Tổng mẫu: {total} | {freq:.2f} Hz")
        return self.streaming_data


def main():
    print("=" * 70)
    print("THỬ NGHIỆM STREAMING MODE - PASCO SENSORS")
    print("=" * 70)

    dev = PASCOStreamingDevice()

    # Scan & connect
    print("\n1) Đang quét cảm biến...")
    devices = dev.scan()
    if not devices:
        print("Không tìm thấy thiết bị")
        return
    print(f"  Tìm thấy {len(devices)} thiết bị:")
    for i, d in enumerate(devices):
        print(f"   {i}: {d.name}")

    print(f"\n2) Kết nối {devices[0].name}...")
    dev.connect(devices[0])
    print("  Đã kết nối!")

    # Measurement list
    print("\n3) Danh sách measurements:")
    all_ms = dev.get_measurement_list()
    for i, m in enumerate(all_ms):
        print(f"   {i}: {m}")

    # Chọn tối đa 3 measurement đầu
    sel_idx = [0, 1, 2][:len(all_ms)]
    selected = [all_ms[i] for i in sel_idx]
    print("\n4) Chọn measurements:")
    for m in selected:
        print(f"   • {m}")

    # Thử STREAMING nếu thiết bị hỗ trợ, else fallback
    print("\n" + "=" * 70)
    print("ƯU TIÊN STREAMING (nếu thiết bị phát periodic)")
    print("=" * 70)

    # Thiết bị "Rotary Motion" sẽ được bật RMS streaming tự động từ thư viện
    can_stream = getattr(dev, "_dev_type", "") == "Rotary Motion"

    if can_stream:
        input("Nhấn Enter để bắt đầu thu streaming 10s...")
        data = dev.collect_streaming(selected, duration=10)
        if not data:
            print("Không nhận được periodic notifications. Chuyển sang Fast Polling.")
            data = dev.fast_polling_mode(selected, duration=10)
    else:
        print("Thiết bị không có API streaming công khai trong thư viện hiện tại.")
        input("Nhấn Enter để chạy Fast Polling 20s...")
        data = dev.fast_polling_mode(selected, duration=20)

    # Tổng kết
    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    if data:
        # In 3 mẫu đầu (nếu streaming thì cấu trúc khác polling)
        print("\n3 mẫu đầu:")
        for i, s in enumerate(data[:3]):
            if 'values' in s:
                print(f"  {i+1}. t={s['timestamp']:.4f} | {s['values']}")
            else:
                print(f"  {i+1}. t={s['timestamp']:.4f} | { {k:v for k,v in s.items() if k!='timestamp'} }")

    dev.disconnect()
    print("\nĐã ngắt kết nối.")


if __name__ == "__main__":
    main()
