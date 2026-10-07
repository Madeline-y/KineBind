"""Validated raw motions, reviewed clips and recording state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .matching import Model

FloatArray = NDArray[np.float64]


class Motion:
    def __init__(self, times: ArrayLike, values: ArrayLike, source: str = '',
                 accel_range: int = 4, gyro_range: int = 500):
        self.times: FloatArray = np.asarray(times, dtype=float).copy()
        self.values: FloatArray = np.asarray(values, dtype=float).copy()
        self.source = source
        self.accel_range, self.gyro_range = accel_range, gyro_range
        if self.times.ndim != 1 or len(self.times) < 2 or len(self.times) > 200000:
            raise ValueError('至少需要两条记录，记录数量不能超过 200000。')
        if self.values.shape != (len(self.times), 6):
            raise ValueError('每条记录必须包含六轴数据。')
        if not np.isfinite(self.times).all() or not np.isfinite(self.values).all():
            raise ValueError('记录含无效数值。')
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('设备时间必须严格递增。')
        if accel_range not in (2, 4, 8, 16) or gyro_range not in (125, 250, 500, 1000, 2000):
            raise ValueError('未知传感器量程。')
        self.times -= self.times[0]

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def section(self, start: float, end: float) -> Motion:
        if not np.isfinite([start, end]).all() or start < 0 or end > self.duration + 1e-6 or end <= start:
            raise ValueError('请选择记录范围内的有效起止时间。')
        selected = (self.times >= start - 1e-6) & (self.times <= end + 1e-6)
        return Motion(self.times[selected], self.values[selected], self.source,
                      self.accel_range, self.gyro_range)

    def to_dict(self) -> dict[str, Any]:
        return dict(times=self.times.tolist(), values=self.values.tolist(), source=self.source,
                    accel_range=self.accel_range, gyro_range=self.gyro_range)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Motion:
        if not isinstance(data, dict):
            raise ValueError('运动记录不是有效对象。')
        return cls(data['times'], data['values'], str(data.get('source', '')),
                   int(data.get('accel_range', 4)), int(data.get('gyro_range', 500)))


@dataclass
class Clip:
    raw: Motion
    start: float
    end: float

    def __post_init__(self) -> None:
        self.raw.section(self.start, self.end)

    @property
    def motion(self) -> Motion:
        return self.raw.section(self.start, self.end)

    def validate(self) -> None:
        m = self.motion
        if len(m.times) < 10 or m.duration < .15 or m.duration > 20:
            raise ValueError('示范需要至少 10 条记录，动作长度应在 0.15～20 秒内。')
        if np.max(np.diff(m.times)) > .08:
            raise ValueError('示范有明显采样缺口，请重录。')
        # Seeed library converts signed raw endpoints using range-dependent sensitivity.
        accel_edge = 32767 * .061 * (m.accel_range // 2) / 1000
        gyro_edge = 32767 * 4.375 * (m.gyro_range / 125) / 1000
        if np.any(np.abs(m.values[:, :3]) >= accel_edge * .998) or np.any(np.abs(m.values[:, 3:]) >= gyro_edge * .998):
            raise ValueError('动作接近传感器量程极限，请轻一些重录或调整固件量程。')

    def to_dict(self) -> dict[str, Any]:
        return dict(raw=self.raw.to_dict(), start=self.start, end=self.end)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Clip:
        if not isinstance(data, dict):
            raise ValueError('样本记录不是有效对象。')
        clip = cls(Motion.from_dict(data['raw']), float(data['start']), float(data['end']))
        clip.validate()
        return clip


@dataclass
class Profile:
    id: str
    name: str
    samples: list[Clip] = field(default_factory=list)
    background: Motion | None = None
    model: Model | None = None
    pixels: int = 100
    calibration: dict[str, Motion] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, name: str) -> Profile:
        p = cls(str(uuid4()), name.strip())
        p.validate()
        return p

    def validate(self) -> None:
        if str(UUID(self.id)) != self.id:
            raise ValueError('无效手势 ID。')
        if not self.name.strip() or len(self.name) > 80:
            raise ValueError('手势名称需要 1～80 个字符。')
        if isinstance(self.pixels, bool) or not isinstance(self.pixels, int) or not 1 <= self.pixels <= 2000:
            raise ValueError('固定移动距离需要 1～2000 像素。')
        if len(self.samples) > 5:
            raise ValueError('每个手势保存五个示范。')
        for clip in self.samples:
            clip.validate()
        if not isinstance(self.calibration, dict) or any(k not in ('static', 'shake', 'pick_place') or not isinstance(v, Motion) for k, v in self.calibration.items()):
            raise ValueError('无效分类校准记录。')
        if not isinstance(self.report, dict):
            raise ValueError('无效学习报告。')
        axes = self.report.get('axes')
        if axes is not None and (not isinstance(axes, list) or any(type(i) is not int or not 0 <= i < 6 for i in axes)):
            raise ValueError('学习报告包含无效轴。')
        if 'reason' in self.report and not isinstance(self.report['reason'], str):
            raise ValueError('学习报告说明无效。')
        if 'status' in self.report and not isinstance(self.report['status'], str):
            raise ValueError('学习报告状态无效。')
        if 'category' in self.report and not isinstance(self.report['category'], str):
            raise ValueError('学习报告类别无效。')
        if 'sample' in self.report and (type(self.report['sample']) is not int or not 1 <= self.report['sample'] <= 5):
            raise ValueError('学习报告示范编号无效。')
        for key in ('threshold', 'distance'):
            if key in self.report and (not isinstance(self.report[key], (int, float)) or not np.isfinite(self.report[key]) or self.report[key] < 0):
                raise ValueError('学习报告距离无效。')
        interval = self.report.get('interval')
        if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
                                    not all(isinstance(t, (int, float)) and np.isfinite(t) for t in interval) or
                                    interval[0] < 0 or interval[1] <= interval[0]):
            raise ValueError('学习报告时间范围无效。')
        if 'categories' in self.report and (not isinstance(self.report['categories'], dict) or any(
                k not in ('static', 'shake', 'pick_place') or not isinstance(v, dict) or
                not isinstance(v.get('distance'), (int, float)) or not np.isfinite(v['distance'])
                for k, v in self.report['categories'].items())):
            raise ValueError('学习报告分类摘要无效。')
        if self.model is not None:
            new = getattr(self.model, 'version', 1) == 2
            records = list(self.calibration.values()) if new else ([self.background] if self.background else [])
            if len(self.samples) != 5 or (new and set(self.calibration) != {'static', 'shake', 'pick_place'}) or (not new and (self.background is None or self.background.duration < 25)):
                raise ValueError('模型缺少五个示范或完整校准记录。')
            ranges = {(c.raw.accel_range, c.raw.gyro_range) for c in self.samples}
            ranges.update((m.accel_range, m.gyro_range) for m in records)
            if ranges != {(self.model.accel_range, self.model.gyro_range)}:
                raise ValueError('模型与原始记录量程不一致。')
            if new:
                from .calibration import training_digest, recording_quality
                for key, motion in self.calibration.items():
                    recording_quality(key, motion, self.calibration['static'])
                if self.model.to_dict()['training_hash'] != training_digest(self):
                    raise ValueError('模型与当前训练数据不一致，请重新学习。')

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(version=2 if self.calibration or self.report or getattr(self.model, 'version', 1) == 2 else 1, id=self.id, name=self.name, pixels=self.pixels,
                    samples=[c.to_dict() for c in self.samples],
                    background=self.background.to_dict() if self.background else None,
                    model=self.model.to_dict() if self.model else None,
                    calibration={k: v.to_dict() for k, v in self.calibration.items()}, report=self.report)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        if not isinstance(data, dict):
            raise ValueError('手势记录不是有效对象。')
        if data.get('version') not in (1, 2):
            raise ValueError('不兼容的手势记录版本。')
        if not isinstance(data.get('calibration', {}), dict) or not isinstance(data.get('report', {}), dict):
            raise ValueError('分类校准或学习报告格式无效。')
        model: Model | None = None
        if data['model']:
            from .matching import Model
            model = Model.from_dict(data['model'])
        profile = cls(str(data['id']), str(data['name']),
                      [Clip.from_dict(c) for c in data['samples']],
                      Motion.from_dict(data['background']) if data['background'] else None,
                      model, data['pixels'],
                      {k: Motion.from_dict(v) for k, v in data.get('calibration', {}).items()}, data.get('report', {}))
        profile.validate()
        if profile.model and (len(profile.samples) != 5 or (getattr(profile.model, 'version', 1) == 1 and profile.background is None)):
            raise ValueError('模型缺少原始示范或校准记录。')
        return profile


class Recorder:
    def __init__(self) -> None:
        self.state = 'idle'
        self.kind = 'sample'
        self.deadline = 0.0
        self.times: list[float] = []
        self.values: list[list[float]] = []
        self.accel_range = 4
        self.gyro_range = 500

    @property
    def count(self) -> int:
        return len(self.times)

    def begin(self, now: float, kind: str = 'sample') -> None:
        if self.state in ('countdown', 'recording'):
            raise ValueError('请先结束当前录制。')
        if kind not in ('sample', 'background', 'static', 'shake', 'pick_place'):
            raise ValueError('未知录制类型。')
        self.times, self.values = [], []
        self.kind, self.state, self.deadline = kind, 'countdown', now + 3

    def feed(self, t: float, values: ArrayLike, now: float) -> None:
        if self.state == 'countdown' and now >= self.deadline:
            self.state = 'recording'
        if self.state != 'recording':
            return
        row = np.asarray(values, dtype=float)
        if row.shape != (6,) or not np.isfinite(row).all() or not np.isfinite(t):
            self.cancel()
            raise ValueError('采样数值无效，录制已结束。')
        if self.times and (t <= self.times[-1] or t - self.times[-1] > .08):
            self.cancel()
            raise ValueError('采样时间异常或有缺口，请重录。')
        if self.times and t - self.times[0] > 60:
            self.cancel()
            raise ValueError('录制超过 60 秒，请重新录制一个动作。')
        self.times.append(float(t))
        self.values.append(row.tolist())

    def finish(self) -> Motion:
        if self.state != 'recording':
            raise ValueError('录制尚未开始。')
        result = Motion(self.times, self.values, 'USB recording', self.accel_range, self.gyro_range)
        self.state = 'review'
        return result

    def cancel(self) -> None:
        self.state = 'idle'
        self.times, self.values = [], []
