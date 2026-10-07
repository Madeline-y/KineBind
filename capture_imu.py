"""Capture KineBind IMU CSV and optionally display a live six-axis plot."""

import argparse
from collections import deque
import csv
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import queue
import sys
import threading
import time

from imu_protocol import AXES, WIRE_COLUMNS, LegacyFirmwareError, LineBuffer, ProtocolError, SampleClock, parse_line

SHANGHAI = timezone(timedelta(hours=8))
CSV_COLUMNS = ("received_at", "elapsed_s", *WIRE_COLUMNS)


def serial_lines(port, baud, stop):
    import serial
    from serial.tools import list_ports

    try:
        connection = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as error:
        ports = ", ".join(item.device for item in list_ports.comports()) or "none"
        raise RuntimeError(
            f"Cannot open {port}: {error}. Close Arduino Serial Monitor/Plotter and miniterm. "
            f"Available ports: {ports}"
        ) from error
    with connection:
        buffer = LineBuffer()
        while not stop.is_set():
            chunk = connection.read(min(connection.in_waiting or 1, 4096))
            if chunk:
                yield from buffer.feed(chunk)
            else:
                yield None  # Lets the collector enforce timeout and duration.


def simulated_lines(stop):
    """Deterministic motion at 50 Hz, explicitly labeled as synthetic."""
    sequence = 0
    deadline = time.monotonic()
    while not stop.is_set():
        t = sequence / 50.0
        values = (0.2 * math.sin(2*t), 0.15 * math.cos(3*t), 1 + 0.1 * math.sin(t),
                  35 * math.sin(2*t), 20 * math.cos(3*t), 10 * math.sin(t))
        yield f"{sequence*20},{sequence}," + ",".join(f"{value:.4f}" for value in values)
        sequence += 1
        deadline += 0.02
        stop.wait(max(0, deadline - time.monotonic()))


class Collector:
    def __init__(self, args, output):
        self.args = args
        self.output = output
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.plot_queue = queue.Queue(maxsize=1000)
        self.clock = SampleClock()
        self.count = 0
        self.bad_lines = 0
        self.error = None

    def run(self):
        started = last_valid = last_flush = time.monotonic()
        source = simulated_lines(self.stop) if self.args.simulate else serial_lines(self.args.port, self.args.baud, self.stop)
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            # Never replace an existing recording, even when --output is explicit.
            with self.output.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_COLUMNS)
                stream.flush()
                for line in source:
                    now = time.monotonic()
                    if self.stop.is_set() or (self.args.duration and now - started >= self.args.duration):
                        break
                    if now - last_valid > 10:
                        raise RuntimeError("No valid IMU record for 10 seconds. Check COM port and upload imu_stream.ino.")
                    if line is None:
                        continue
                    try:
                        sample = parse_line(line)
                    except LegacyFirmwareError:
                        raise
                    except ProtocolError:
                        self.bad_lines += 1
                        continue
                    if sample is None:
                        continue
                    elapsed = self.clock.advance(sample)
                    writer.writerow((datetime.now(SHANGHAI).isoformat(timespec="milliseconds"),
                                     f"{elapsed:.3f}", sample.timestamp_ms, sample.sequence, *sample.values))
                    self.count += 1
                    last_valid = now
                    if self.count == 1:
                        print("Receiving six-axis data. CSV recording started.", flush=True)
                    if now - last_flush >= 1:
                        stream.flush()
                        last_flush = now
                        if self.args.no_plot:
                            print(f"Samples: {self.count} | device time: {elapsed:.2f}s", flush=True)
                    if not self.args.no_plot:
                        # CSV is independent of plot speed; only visual updates can be dropped.
                        try:
                            self.plot_queue.put_nowait((elapsed, sample.values))
                        except queue.Full:
                            try:
                                self.plot_queue.get_nowait()
                            except queue.Empty:
                                pass  # The plotting thread already drained it.
                            self.plot_queue.put_nowait((elapsed, sample.values))
            if self.count == 0:
                raise RuntimeError("Recording stopped without any valid samples.")
        except Exception as error:
            self.error = str(error)
        finally:
            source.close()
            self.finished.set()


def show_plot(collector):
    import matplotlib.pyplot as plt

    fig, plots = plt.subplots(2, 1, sharex=True, figsize=(10, 7), layout="constrained")
    source = "SIMULATED" if collector.args.simulate else collector.args.port
    fig.canvas.manager.set_window_title(f"KineBind IMU - {source}")
    plots[0].set_ylabel("Acceleration (g, includes gravity)")
    plots[1].set_ylabel("Angular velocity (deg/s)")
    plots[1].set_xlabel("Device time since first sample (s)")
    lines = []
    for index, axis in enumerate(plots):
        for name, color in zip(AXES[index*3:index*3+3], ("#e45756", "#4c78a8", "#54a24b")):
            lines.append(axis.plot([], [], label=name, color=color)[0])
        axis.legend(loc="upper right")
        axis.grid(alpha=0.25)
    history = deque(maxlen=max(1000, int(collector.args.window * 100)))
    plt.show(block=False)
    try:
        while plt.fignum_exists(fig.number) and not collector.finished.is_set():
            while True:
                try:
                    history.append(collector.plot_queue.get_nowait())
                except queue.Empty:
                    break
            if history:
                right = history[-1][0]
                left = max(0, right - collector.args.window)
                while len(history) > 1 and history[0][0] < left:
                    history.popleft()
                times = [item[0] for item in history]
                for index, line in enumerate(lines):
                    line.set_data(times, [item[1][index] for item in history])
                for axis in plots:
                    axis.set_xlim(left, max(right, left + 1))
                    axis.relim()
                    axis.autoscale_view(scalex=False)
                rate = (collector.count - 1) / (collector.clock.elapsed_ms / 1000) if collector.clock.elapsed_ms else 0
                plots[0].set_title(f"{source} | {collector.count} samples | average {rate:.1f} Hz")
                fig.canvas.draw_idle()
            plt.pause(0.1)
    finally:
        collector.stop.set()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM7")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 records until stopped.")
    parser.add_argument("--window", type=float, default=10, help="Displayed history in seconds.")
    parser.add_argument("--output", type=Path, help="New CSV path; existing files are never overwritten.")
    parser.add_argument("--no-plot", action="store_true", help="Record without opening a graph.")
    parser.add_argument("--simulate", action="store_true", help="Use synthetic data, without accessing the serial port.")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or not math.isfinite(args.window) or args.window <= 0 or args.baud <= 0:
        parser.error("duration must be finite and >= 0; window and baud must be positive.")
    prefix = "simulated" if args.simulate else "imu"
    filename = f"{prefix}_{datetime.now(SHANGHAI):%Y%m%d_%H%M%S_%f}.csv"
    output = args.output or Path(__file__).resolve().parent / "recordings" / filename
    print(f"Source: {'SIMULATED DATA' if args.simulate else args.port} | CSV: {output}", flush=True)
    print("Close the plot or press Ctrl+C to stop and save.", flush=True)
    collector = Collector(args, output)
    worker = threading.Thread(target=collector.run, name="imu-collector", daemon=True)
    worker.start()
    try:
        if args.no_plot:
            while not collector.finished.wait(0.1):
                pass
        else:
            show_plot(collector)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Plot error: {error}. Try --no-plot to record without a graph.", file=sys.stderr)
        collector.stop.set()
        worker.join(timeout=3)
        return 1
    finally:
        collector.stop.set()
        worker.join(timeout=3)
    if worker.is_alive():
        print("Collector has not stopped; CSV finalization is unverified.", file=sys.stderr)
        return 1
    print(f"Saved {collector.count} samples to {output}")
    print(f"Skipped malformed lines: {collector.bad_lines}; missing sequence IDs: {collector.clock.missing_sequences}")
    if collector.error:
        print(f"Error: {collector.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
