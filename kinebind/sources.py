"""CSV replay and a single USB reader; no GUI or mouse actions."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any

from capture_imu import serial_lines
from imu_protocol import WIRE_COLUMNS, ProtocolError, Sample, SampleClock, parse_line
from .data import Motion


def read_csv(path: Path) -> Motion:
    if path.stat().st_size > 50_000_000:
        raise ValueError('CSV 超过 50 MB，请选择较短记录。')
    times, values = [], []
    ranges: tuple[int, int] | None = None
    clock = SampleClock()
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(WIRE_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('CSV 需要设备时间戳、序号和六轴列。')
        for row in reader:
            current_ranges = (int(row.get('accel_range_g', 4)), int(row.get('gyro_range_dps', 500)))
            if ranges is not None and current_ranges != ranges:
                raise ValueError('CSV 中途改变了传感器量程。')
            ranges = current_ranges
            sample = parse_line(','.join(str(row[k]) for k in WIRE_COLUMNS))
            if sample is None:
                raise ValueError('CSV 含非样本记录。')
            elapsed = clock.advance(sample)
            if clock.missing_sequences:
                raise ValueError('CSV 含序号缺口，不能当作连续示范。')
            times.append(elapsed)
            values.append(sample.values)
            if len(times) > 200000:
                raise ValueError('CSV 记录过长。')
    return Motion(times, values, str(path.resolve()), *(ranges or (4, 500)))


@dataclass(frozen=True)
class Point:
    t: float
    values: tuple[float, ...]
    received: float
    sample: Sample | None = None


class SerialFeed:
    def __init__(self, port: str, baud: int = 115200):
        self.port, self.baud = port, baud
        self.cancel = threading.Event()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=3000)
        self.accel_range, self.gyro_range = 4, 500
        self.thread = threading.Thread(target=self._run, name='kinebind-usb', daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.cancel.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)

    def _put(self, kind: str, value: Any) -> None:
        try:
            self.messages.put_nowait((kind, value))
        except queue.Full:
            self.cancel.set()
            try: self.messages.get_nowait()
            except queue.Empty: pass
            self.messages.put_nowait(('error', '读取队列已满，请重新连接设备。'))

    def _run(self) -> None:
        source = serial_lines(self.port, self.baud, self.cancel)
        clock = SampleClock()
        last_valid = time.monotonic()
        first = True
        bad = 0
        try:
            for line in source:
                if self.cancel.is_set(): break
                now = time.monotonic()
                if now - last_valid > 10:
                    raise ValueError('10 秒没有有效六轴数据，请检查端口和采集固件。')
                if line is None: continue
                if line.startswith('#'):
                    a = re.search(r'accel_range_g=(\d+)', line)
                    g = re.search(r'gyro_range_dps=(\d+)', line)
                    if a: self.accel_range = int(a.group(1))
                    if g: self.gyro_range = int(g.group(1))
                try:
                    sample = parse_line(line)
                except ProtocolError as error:
                    if 'HighLevelExample' in str(error): raise
                    bad += 1
                    if bad >= 5: raise ValueError('连续五条无效数据，请检查固件。') from error
                    continue
                if sample is None: continue
                bad = 0
                elapsed = clock.advance(sample)
                if clock.missing_sequences:
                    raise ValueError('串口序号出现缺口，录制和控制已停止，请重新连接。')
                if first:
                    if self.accel_range not in (2, 4, 8, 16) or self.gyro_range not in (125, 250, 500, 1000, 2000):
                        raise ValueError('设备传感器量程无效，请检查固件。')
                    self._put('connected', (self.accel_range, self.gyro_range))
                    first = False
                last_valid = now
                self._put('point', Point(elapsed, sample.values, now, sample))
        except Exception as error:
            if not self.cancel.is_set():
                self._put('error', str(error))
        finally:
            source.close()
            self._put('closed', None)
