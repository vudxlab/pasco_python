"""
Tkinter GUI for 3 PASCO sensors (non‑blocking + synchronized 10 Hz sampling)

Layout by columns:
- Left: Connect controls and a live log of connection/progress
- Middle: Real‑time time‑domain visualization of selected signals
- Right: FFT magnitude spectra at fixed 10 Hz sampling

Features:
- Connect up to 3 sensors by 6‑digit ID, pick measurement per sensor
- Start/Stop recording (synchronized: all samples taken at the same timestamps)
- Overload alert per sensor (threshold); save combined CSV (time + A/B/C)
- Optional NumPy for faster FFT; fallback DFT if NumPy unavailable
"""

import sys
import time
import csv
import math
from datetime import datetime
import os

# Matplotlib embedding for better visuals
import matplotlib
# Use TkAgg for interactive embedding in Tkinter
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl

# Lightweight rendering tweaks
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5
mpl.rcParams['agg.path.chunksize'] = 2000

try:
    import numpy as np  # optional
except Exception:  # numpy not required
    np = None

try:
    import winsound  # Windows beep for overload
except Exception:
    winsound = None

sys.path.append('src')
from pasco.pasco_ble_device import PASCOBLEDevice  # noqa: E402

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class SensorClient:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.dev = PASCOBLEDevice()
        self.connected = False
        self.connecting = False
        self.measurements = []
        self.selected_measurement = None
        # synchronized buffers (time, value) at 10 Hz
        self.data = []
        self.threshold = 0.0
        # UI variables will be attached dynamically by App

    def connect_by_id(self, sensor_id: str):
        self.dev.connect_by_id(sensor_id)
        self.connected = True
        self.measurements = self.dev.get_measurement_list()
        if self.measurements and not self.selected_measurement:
            self.selected_measurement = self.measurements[0]
        self.data.clear()

    def disconnect(self):
        if self.connected:
            try:
                self.dev.disconnect()
            except Exception:
                pass
        self.connected = False

    def read_once(self):
        if not (self.connected and self.selected_measurement):
            return None
        return self.dev.read_data(self.selected_measurement)

    def save_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['time_s', self.selected_measurement or 'value'])
            for t, v in self.data:
                w.writerow([f"{t:.6f}", v])


def compute_fft(times_vals, fixed_fs=10.0):
    """
    Compute FFT magnitude spectrum from (t, v) samples.
    Returns (freqs, mags). If insufficient samples, returns ([], []).
    """
    if len(times_vals) < 4:
        return [], []
    # Uniformly re-sample if timestamps jitter mildly; approximate fs from median dt
    ts = [t for t, _ in times_vals]
    vs = [v for _, v in times_vals]
    # Use fixed sampling rate (10 Hz) as requested
    fs = float(fixed_fs)
    dt = 1.0 / fs
    N = len(vs)

    if np is not None:
        arr = np.array(vs, dtype=float)
        arr = arr - np.mean(arr)
        spec = np.fft.rfft(arr)
        mags = np.abs(spec)
        freqs = np.fft.rfftfreq(N, d=dt)
        return freqs.tolist(), mags.tolist()
    else:
        # naive DFT (O(N^2)), fine for small N
        N2 = N
        mags = []
        freqs = []
        for k in range(N2 // 2 + 1):
            re = 0.0
            im = 0.0
            for n, x in enumerate(vs):
                angle = 2 * math.pi * k * n / N2
                re += x * math.cos(angle)
                im -= x * math.sin(angle)
            mags.append(math.hypot(re, im))
            freqs.append(k * fs / N2)
        return freqs, mags


def find_modes(freqs, mags, top_n=3, min_freq=0.5):
    """
    Simple mode detection: pick top-N peaks above a minimal frequency.
    Returns list of (freq, magnitude) sorted by magnitude desc.
    """
    if not freqs or not mags:
        return []
    pairs = [(f, m) for f, m in zip(freqs, mags) if f >= min_freq]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Multi-Sensor GUI')
        self.geometry('1200x720')

        # Fixed 10 Hz sampling
        self.fs = 10.0
        self.dt = 1.0 / self.fs
        # Compatibility flags for older start/stop flow
        self.loop_running = False

        self.sensors = [
            SensorClient('A', '#d32f2f'),  # red
            SensorClient('B', '#388e3c'),  # green
            SensorClient('C', '#1976d2'),  # blue
        ]
        self.recording = False
        self.t0 = None  # recording start time
        self._tick_after_id = None

        self._build_ui()
        # schedule periodic plot refresh (lower rate for stability)
        self.plot_interval_ms = 500
        self.fft_interval_s = 2.0
        self.after(self.plot_interval_ms, self._refresh_plots)
        # CSV logging state
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.csv_dir = os.path.join(os.getcwd(), 'recordings')
        os.makedirs(self.csv_dir, exist_ok=True)

    def _build_ui(self):
        header = ttk.Label(self, text='Giải pháp đo dao động kết cấu nhịp cầu bằng hệ thống thiết bị đo dao động không dây', font=('Segoe UI', 12, 'bold'))
        header.grid(row=0, column=0, columnspan=3, pady=6)

        # Create narrow left column for connect/log; plots take most space
        self.left_min = 220
        self.columnconfigure(0, weight=0, minsize=self.left_min)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        # Allow the main row (plots + controls) to expand
        self.rowconfigure(1, weight=1)

        left = ttk.Frame(self)
        mid = ttk.Frame(self)
        right = ttk.Frame(self)
        left.grid(row=1, column=0, sticky='nsew', padx=6, pady=6)
        mid.grid(row=1, column=1, sticky='nsew', padx=6, pady=6)
        right.grid(row=1, column=2, sticky='nsew', padx=6, pady=6)

        # LEFT: connections + log
        self._build_left(left)
        # MIDDLE: time-domain plot
        self._build_middle(mid)
        # RIGHT: FFT plot
        self._build_right(right)

    def _build_left(self, left: ttk.Frame):
        # Per-sensor connection panels
        for s in self.sensors:
            fr = ttk.Labelframe(left, text=f'Sensor {s.name}')
            fr.pack(fill='x', padx=4, pady=4)

            ttk.Label(fr, text='ID (e.g. 123-456):').grid(row=0, column=0, sticky='w')
            id_var = tk.StringVar()
            s.id_var = id_var
            ttk.Entry(fr, textvariable=id_var, width=12).grid(row=0, column=1, sticky='w', padx=4)
            btn_connect = ttk.Button(fr, text='Connect', command=lambda ss=s: self.connect_sensor_async(ss))
            btn_disconnect = ttk.Button(fr, text='Disconnect', command=lambda ss=s: self.disconnect_sensor(ss))
            btn_connect.grid(row=0, column=2, padx=4)
            btn_disconnect.grid(row=0, column=3, padx=4)
            s.btn_connect = btn_connect

            ttk.Label(fr, text='Measurement:').grid(row=1, column=0, sticky='w')
            s.meas_var = tk.StringVar()
            s.meas_cb = ttk.Combobox(fr, textvariable=s.meas_var, width=24, state='readonly', values=[])
            s.meas_cb.grid(row=1, column=1, columnspan=3, sticky='we', padx=4)
            
            ttk.Label(fr, text='Overload:').grid(row=2, column=0, sticky='w')
            s.thr_var = tk.DoubleVar(value=0.0)
            ttk.Entry(fr, textvariable=s.thr_var, width=10).grid(row=2, column=1, sticky='w', padx=4)
            s.val_var = tk.StringVar(value='—')
            s.val_lbl = ttk.Label(fr, textvariable=s.val_var, width=14, foreground=s.color)
            s.val_lbl.grid(row=2, column=2, sticky='w', padx=4)
            ttk.Button(fr, text='Save CSV', command=lambda ss=s: self.save_csv(ss)).grid(row=2, column=3, padx=4)

        # Start/Stop buttons and combined Save
        ctl = ttk.Frame(left)
        ctl.pack(fill='x', padx=4, pady=6)
        ttk.Button(ctl, text='Start Recording', command=self.start_recording).pack(side='left', padx=3)
        ttk.Button(ctl, text='Stop', command=self.stop_recording).pack(side='left', padx=3)
        ttk.Button(ctl, text='Save All CSV', command=self.save_all_csv).pack(side='left', padx=3)

        # Log
        ttk.Label(left, text='Log').pack(anchor='w', padx=4)
        self.log_text = tk.Text(left, height=6)
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)
        # Progress label for compatibility with older flow
        self.progress = ttk.Label(left, text='Idle')
        self.progress.pack(anchor='w', padx=4, pady=2)

    def _build_middle(self, mid: ttk.Frame):
        ttk.Label(mid, text='Time-domain (Acceleration)').pack(anchor='w')
        # Matplotlib Figure with 3 stacked subplots (one per sensor)
        self.fig_time = Figure(figsize=(6, 5), dpi=90)
        self.ax_time = []
        self.time_lines = {}
        for i, s in enumerate(self.sensors):
            ax = self.fig_time.add_subplot(3, 1, i + 1)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 30)
            ax.set_ylim(-1, 1)
            ax.set_title(f'Sensor {s.name}', fontsize=9)
            if i == 2:
                ax.set_xlabel('Time [s]')
            ax.set_ylabel('Acceleration [m/s²]')
            line, = ax.plot([], [], color=s.color, lw=1.2, antialiased=False)
            self.ax_time.append(ax)
            self.time_lines[s.name] = line
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=mid)
        self.canvas_time.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        # visibility toggles
        vis = ttk.Frame(mid)
        vis.pack(anchor='w', padx=4)
        self.visible = {}
        for s in self.sensors:
            v = tk.BooleanVar(value=True)
            self.visible[s.name] = v
            ttk.Checkbutton(vis, text=f'Sensor {s.name}', variable=v).pack(side='left', padx=4)

    def _build_right(self, right: ttk.Frame):
        ttk.Label(right, text='FFT').pack(anchor='w')
        self.fig_fft = Figure(figsize=(6, 5), dpi=90)
        self.ax_fft = []
        self.fft_lines = {}
        for i, s in enumerate(self.sensors):
            ax = self.fig_fft.add_subplot(3, 1, i + 1)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.fs / 2.0)
            ax.set_ylim(0, 1)
            ax.set_title(f'Sensor {s.name}', fontsize=9)
            if i == 2:
                ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude')
            line, = ax.plot([], [], color=s.color, lw=1.0, antialiased=False)
            self.ax_fft.append(ax)
            self.fft_lines[s.name] = line
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=right)
        self.canvas_fft.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self.fft_info = ttk.Label(right, text='')
        self.fft_info.pack(anchor='w', padx=4)

    def log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log_text.insert('end', f'[{ts}] {msg}\n')
        self.log_text.see('end')

    def connect_sensor(self, sensor: SensorClient):
        sid = sensor.id_var.get().strip()
        if not sid:
            messagebox.showwarning('Connect', 'Please enter a 6-digit sensor ID (e.g. 123-456).')
            return
        self.log(f'Sensor {sensor.name}: connecting to {sid}...')
        try:
            sensor.connect_by_id(sid)
            # Show only acceleration-related measurements to match your requirement
            accel_names = [m for m in sensor.measurements if 'accel' in m.lower()]
            sensor.meas_cb['values'] = accel_names if accel_names else sensor.measurements
            # Prefer Accelerationx if available
            preferred = None
            for cand in ['Accelerationx', 'AccelerationX', 'accelerationx', 'Acceleration']:
                if cand in sensor.meas_cb['values']:
                    preferred = cand
                    break
            if preferred:
                sensor.selected_measurement = preferred
                sensor.meas_var.set(preferred)
            elif sensor.meas_cb['values']:
                sensor.selected_measurement = sensor.meas_cb['values'][0]
                sensor.meas_var.set(sensor.selected_measurement)
            self.log(f'Sensor {sensor.name}: connected.')
        except Exception as e:
            self.log(f'Sensor {sensor.name}: failed to connect: {e}')
            messagebox.showerror('Connect', f'Sensor {sensor.name}: {e}')

        def on_meas_change(*_):
            sensor.selected_measurement = sensor.meas_var.get()
        sensor.meas_var.trace_add('write', on_meas_change)

    def connect_sensor_async(self, sensor: SensorClient):
        # Run connect in a background thread to keep UI responsive
        sid = sensor.id_var.get().strip()
        if not sid:
            messagebox.showwarning('Connect', 'Please enter a 6-digit sensor ID (e.g. 123-456).')
            return
        if getattr(sensor, '_connecting', False):
            return
        sensor._connecting = True
        sensor.btn_connect['state'] = 'disabled'
        self.config(cursor='watch'); self.update_idletasks()
        self.log(f'Sensor {sensor.name}: connecting to {sid}...')

        def worker():
            ok = True
            err = None
            try:
                # Attempt blocking connect
                self.connect_sensor(sensor)
            except Exception as e:
                ok = False
                err = e
            finally:
                def finalize():
                    sensor._connecting = False
                    sensor.btn_connect['state'] = 'normal'
                    self.config(cursor='')
                    if not ok:
                        self.log(f'Sensor {sensor.name}: failed to connect: {err}')
                        messagebox.showerror('Connect', f'Sensor {sensor.name}: {err}')
                self.after(0, finalize)

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def disconnect_sensor(self, sensor: SensorClient):
        sensor.disconnect()
        sensor.meas_cb['values'] = []
        sensor.meas_var.set('')
        sensor.val_var.set('—')
        messagebox.showinfo('Disconnect', f'Disconnected Sensor {sensor.name}.')

    def start_recording(self):
        for s in self.sensors:
            if s.connected and s.meas_var.get():
                s.selected_measurement = s.meas_var.get()
                s.threshold = s.thr_var.get()
                s.begin_record()
        self.recording = True
        if not self.loop_running:
            self.loop_running = True
            self.after(1, self._sample_loop)
        self.progress.config(text='Recording...')

    def stop_recording(self):
        self.recording = False
        self.loop_running = False
        if self._tick_after_id is not None:
            try:
                self.after_cancel(self._tick_after_id)
            except Exception:
                pass
            self._tick_after_id = None
        self.log('Recording stopped.')
        # Close CSV log if open
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        finally:
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None

    def _tick(self):
        if not self.recording:
            return
        if self.t0 is None:
            self.t0 = time.perf_counter()
        now = time.perf_counter()
        stamp = now - self.t0
        window_max_len = int(30 * self.fs)
        row_vals = []
        any_read = False
        for s in self.sensors:
            if s.connected and s.selected_measurement:
                try:
                    val = s.read_once()
                except Exception:
                    val = None
                if isinstance(val, (int, float)):
                    any_read = True
                    s.data.append((stamp, val))
                    if len(s.data) > window_max_len:
                        s.data = s.data[-window_max_len:]
                    s.val_var.set(f"{val:.4f}")
                    thr = None
                    try:
                        thr = float(s.thr_var.get())
                    except Exception:
                        pass
                    if thr and abs(val) >= abs(thr):
                        s.val_lbl.configure(foreground='red')
                        if winsound and (time.perf_counter() - getattr(self, '_last_beep', 0)) > 0.5:
                            # Run beep on a background thread to avoid blocking UI
                            import threading
                            def _beep():
                                try:
                                    winsound.Beep(2000, 60)
                                except Exception:
                                    pass
                            threading.Thread(target=_beep, daemon=True).start()
                            self._last_beep = time.perf_counter()
                    else:
                        s.val_lbl.configure(foreground=s.color)
                    row_vals.append(val)
                else:
                    row_vals.append('')
            else:
                row_vals.append('')
        # Auto-append to CSV if any sensor produced a sample
        if any_read and self.csv_writer:
            try:
                self.csv_writer.writerow([f"{stamp:.6f}", *row_vals])
                self.csv_file.flush()
            except Exception:
                pass
        # draw
        self._refresh_plots()
        # schedule next tick at 10 Hz
        self._tick_after_id = self.after(int(self.dt * 1000), self._tick)

    # Backwards-compatible sampling loop used by older Start/Stop flow
    def _sample_loop(self):
        if not self.recording:
            self.loop_running = False
            return
        # Reuse the same sampling logic
        self._tick()
        if self.recording:
            self.loop_running = True
            self.after(int(self.dt * 1000), self._sample_loop)

    def save_csv(self, sensor: SensorClient):
        if not sensor.data:
            messagebox.showinfo('Save CSV', 'No data to save for this sensor.')
            return
        default = f"sensor_{sensor.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(title='Save CSV', defaultextension='.csv', initialfile=default, filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            sensor.save_csv(path)
            messagebox.showinfo('Save CSV', f'Saved: {path}')
        except Exception as e:
            messagebox.showerror('Save CSV', f'Failed: {e}')

    def save_all_csv(self):
        any_data = any(s.data for s in self.sensors)
        if not any_data:
            messagebox.showinfo('Save CSV', 'No data to save.')
            return
        folder = filedialog.askdirectory(title='Select output folder')
        if not folder:
            return
        # Save combined synchronized CSV (time, A, B, C)
        name = f"sensors_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = f"{folder}/{name}"
        try:
            # Build a common time vector based on 10 Hz clock
            max_len = max(len(s.data) for s in self.sensors)
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['time_s', 'A', 'B', 'C'])
                for i in range(max_len):
                    # time from any sensor that has this index; else i * dt
                    t = i * self.dt
                    vals = []
                    for s in self.sensors:
                        if i < len(s.data):
                            vals.append(s.data[i][1])
                        else:
                            vals.append('')
                    w.writerow([f"{t:.6f}", *vals])
            messagebox.showinfo('Save CSV', f'Saved: {path}')
        except Exception as e:
            messagebox.showerror('Save CSV', f'Failed: {e}')

    # Live plotting helpers -------------------------------------------------
    def _refresh_plots(self):
        self._draw_time_plot()
        # update FFT less frequently
        if not hasattr(self, '_last_fft_draw') or (time.time() - self._last_fft_draw) > self.fft_interval_s:
            self._draw_fft_plot()
            self._last_fft_draw = time.time()
        # Schedule next regular refresh in case sampler is not running
        self.after(self.plot_interval_ms, self._refresh_plots)

    def _get_visible_series(self):
        series = []
        for s in self.sensors:
            if self.visible.get(s.name, tk.BooleanVar(value=True)).get() and s.data:
                series.append(s)
        return series

    def _draw_time_plot(self):
        window_s = 30.0
        visible = {s.name: self.visible.get(s.name, tk.BooleanVar(value=True)).get() for s in self.sensors}
        # Per-sensor scaling
        for i, s in enumerate(self.sensors):
            ax = self.ax_time[i]
            line = self.time_lines[s.name]
            if not (visible.get(s.name) and s.data):
                line.set_data([], [])
                continue
            now_t = s.data[-1][0]
            xs = [t - (now_t - window_s) for t, _ in s.data if now_t - t <= window_s]
            ys = [v for t, v in s.data if now_t - t <= window_s]
            line.set_data(xs, ys)
            if ys:
                ymin, ymax = min(ys), max(ys)
                if ymin == ymax:
                    ymin -= 1.0; ymax += 1.0
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlim(0, window_s)
            # Update Y label with unit for clarity
            if s.connected and s.selected_measurement:
                try:
                    unit = s.dev.get_measurement_unit(s.selected_measurement)
                except Exception:
                    unit = ''
                ylabel = f'Acceleration [{unit}]' if unit else 'Acceleration'
                ax.set_ylabel(ylabel)
        # Use deferred drawing for responsiveness
        self.canvas_time.draw_idle()

    def _draw_fft_plot(self):
        max_freq = self.fs / 2.0
        best_peaks = []
        visible = {s.name: self.visible.get(s.name, tk.BooleanVar(value=True)).get() for s in self.sensors}
        for i, s in enumerate(self.sensors):
            ax = self.ax_fft[i]
            line = self.fft_lines[s.name]
            if not (visible.get(s.name) and len(s.data) >= 16):
                line.set_data([], [])
                continue
            freqs, mags = compute_fft(s.data, fixed_fs=self.fs)
            if not freqs:
                line.set_data([], [])
                continue
            # limit to Nyquist
            xs = []
            ys = []
            mmax = max(mags) or 1.0
            for f, m in zip(freqs, mags):
                if f > max_freq:
                    break
                xs.append(f)
                ys.append(m / mmax)
            line.set_data(xs, ys)
            ax.set_xlim(0, max_freq)
            ax.set_ylim(0, 1.05)
            peaks = find_modes(freqs, mags, top_n=1, min_freq=0.1)
            if peaks:
                best_peaks.append((s.name, peaks[0]))
        self.canvas_fft.draw_idle()
        if best_peaks:
            info = 'Dominant peaks: ' + ', '.join(f"{name}:{f:.2f}Hz" for name,(f,_ ) in best_peaks)
        else:
            info = 'Dominant peaks: —'
        self.fft_info.config(text=info)

    # Recording control -----------------------------------------------------
    def start_recording(self):
        # clear previous data and begin synchronized sampling
        for s in self.sensors:
            s.data.clear()
            try:
                s.threshold = float(s.thr_var.get())
            except Exception:
                s.threshold = 0.0
        self.recording = True
        self.log('Recording started (10 Hz).')
        self.t0 = None
        # Open combined CSV for auto logging
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = os.path.join(self.csv_dir, f'recording_{ts}.csv')
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            # Header with measurement names
            names = []
            for s in self.sensors:
                label = s.selected_measurement if s.selected_measurement else ''
                names.append(label)
            self.csv_writer.writerow(['time_s', 'A', 'B', 'C'])
            self.log(f'Auto CSV logging: {self.csv_path}')
        except Exception as e:
            self.log(f'Auto CSV open failed: {e}')
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None
        self._tick()


if __name__ == '__main__':
    app = App()
    app.mainloop()
