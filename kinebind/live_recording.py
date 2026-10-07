"""Stream complete realtime test inputs to a new CSV per recognition session."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, TextIO
from uuid import uuid4

from capture_imu import CSV_COLUMNS, SHANGHAI
from imu_protocol import Sample
from .data import Profile


class LiveTestRecorder:
    def __init__(self, folder: Path, source: str = 'USB realtime test'):
        self.folder, self.source = Path(folder), source
        self.path: Path | None = None
        self.count = 0
        self._stream: TextIO | None = None
        self._writer: Any = None
        self._first_t: float | None = None
        self._metadata: tuple[Any, ...] = ()
        self._last_flush = 0.
        self._lock = threading.RLock()

    def start(self, profile: Profile) -> Path:
        with self._lock:
            self.close()
            model = profile.model
            if model is None:
                raise ValueError('实时测试保存需要已学习的手势。')
            self.folder.mkdir(parents=True, exist_ok=True)
            prefix = 'simulated_realtime_test' if 'SIMULATED' in self.source.upper() else 'realtime_test'
            path = self.folder / f'{prefix}_{datetime.now(SHANGHAI):%Y%m%d_%H%M%S_%f}_{uuid4().hex[:6]}.csv'
            stream = path.open('x', encoding='utf-8-sig', newline='')
            try:
                writer = csv.writer(stream)
                writer.writerow((*CSV_COLUMNS, 'accel_range_g', 'gyro_range_dps',
                                 'gesture_id', 'gesture_name', 'source'))
                stream.flush()
            except Exception:
                stream.close()
                raise
            self.path, self._stream, self._writer = path, stream, writer
            self.count, self._first_t = 0, None
            self._metadata = (model.accel_range, model.gyro_range, profile.id, profile.name, self.source)
            self._last_flush = time.monotonic()
            return path

    def write(self, t: float, sample: Sample | None) -> None:
        with self._lock:
            if self._stream is None:
                return
            if sample is None:
                raise ValueError('实时测试数据缺少原始设备时间戳和序号。')
            if self._first_t is None:
                self._first_t = t
            self._writer.writerow((datetime.now(SHANGHAI).isoformat(timespec='milliseconds'),
                                   f'{t - self._first_t:.6f}', sample.timestamp_ms,
                                   sample.sequence, *sample.values, *self._metadata))
            self.count += 1
            now = time.monotonic()
            if now - self._last_flush >= 1.:
                self._stream.flush()
                self._last_flush = now

    def close(self) -> tuple[Path, int] | None:
        with self._lock:
            stream = self._stream
            if stream is None:
                return None
            self._stream, self._writer = None, None
            stream.close()
            assert self.path is not None
            return self.path, self.count
