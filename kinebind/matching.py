"""Six-axis constrained template matching and complete-cycle streaming events."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from scipy.spatial.distance import cdist

from .data import FloatArray, Motion, Profile

POINTS = 48
RADIUS = 9
SCALE = np.array([1., 1., 1., 100., 100., 100.])


def features(motion: Motion, bias: FloatArray) -> FloatArray:
    values = motion.values.copy()
    values[:, 3:] -= bias
    # Causal three-sample mean, also used when evaluating a streaming window.
    padded = np.vstack([values[:1], values[:1], values])
    summed = np.vstack([np.zeros((1, 6)), np.cumsum(padded, axis=0)])
    smooth = (summed[3:] - summed[:-3]) / 3
    grid = np.linspace(0, motion.duration, POINTS)
    return np.column_stack([np.interp(grid, motion.times, smooth[:, i]) for i in range(6)]) / SCALE


def distance(a: FloatArray, b: FloatArray) -> float:
    costs = cdist(a, b, metric='sqeuclidean')
    prev = np.full(POINTS + 1, np.inf)
    prev[0] = 0
    for i in range(1, POINTS + 1):
        current = np.full(POINTS + 1, np.inf)
        for j in range(max(1, i - RADIUS), min(POINTS, i + RADIUS) + 1):
            current[j] = costs[i - 1, j - 1] + min(prev[j], current[j - 1], prev[j - 1])
        prev = current
    return math.sqrt(float(prev[-1]) / (POINTS * 6))


@dataclass(frozen=True)
class Match:
    accepted: bool
    score: float
    reason: str
    template: int = -1


@dataclass
class Model:
    templates: list[FloatArray]
    durations: list[float]
    threshold: float
    bias: FloatArray
    endpoint_limit: float
    finish_speed: float
    onset_speed: float
    background_min: float
    accel_range: int = 4
    gyro_range: int = 500

    @property
    def min_duration(self) -> float:
        return max(.15, min(self.durations) * .55)

    @property
    def max_duration(self) -> float:
        return max(self.durations) * 2.1

    def match(self, motion: Motion) -> Match:
        if not self.min_duration <= motion.duration <= self.max_duration:
            return Match(False, 1e6, '动作时长超出模板范围')
        query = features(motion, self.bias)
        scores = [distance(query, template) for template in self.templates]
        order = np.argsort(scores)
        for index in order:
            template = self.templates[int(index)]
            edge = max(float(np.sqrt(np.mean((query[0] - template[0]) ** 2))),
                       float(np.sqrt(np.mean((query[-1] - template[-1]) ** 2))))
            speed = float(np.linalg.norm(motion.values[-1, 3:] - self.bias))
            if scores[index] <= self.threshold and edge <= self.endpoint_limit and speed <= self.finish_speed:
                return Match(True, float(scores[index]), '完整动作匹配', int(index))
        return Match(False, float(min(scores)), '未满足完整动作及接受阈值')

    def to_dict(self) -> dict[str, Any]:
        return dict(version=1, points=POINTS, radius=RADIUS, processing='causal-mean3-fixed-scale-v1',
                    templates=[x.tolist() for x in self.templates], durations=self.durations,
                    threshold=self.threshold, bias=self.bias.tolist(), endpoint_limit=self.endpoint_limit,
                    finish_speed=self.finish_speed, onset_speed=self.onset_speed, background_min=self.background_min,
                    accel_range=self.accel_range, gyro_range=self.gyro_range)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Model:
        if not isinstance(data, dict):
            raise ValueError('模型记录不是有效对象。')
        if data.get('version') == 2:
            from .robust import RobustModel
            return RobustModel.from_dict(data)
        if data.get('version') != 1 or data.get('points') != POINTS or data.get('radius') != RADIUS or data.get('processing') != 'causal-mean3-fixed-scale-v1':
            raise ValueError('不兼容的模型处理版本。')
        templates = [np.asarray(v, dtype=float) for v in data['templates']]
        durations = [float(v) for v in data['durations']]
        bias = np.asarray(data['bias'], dtype=float)
        numbers = [float(data[k]) for k in ('threshold', 'endpoint_limit', 'finish_speed', 'onset_speed', 'background_min')]
        if len(templates) != 5 or any(v.shape != (POINTS, 6) or not np.isfinite(v).all() for v in templates):
            raise ValueError('模型必须包含五个有效六轴模板。')
        if len(durations) != 5 or not all(.15 <= v <= 20 for v in durations):
            raise ValueError('无效模板时长。')
        if bias.shape != (3,) or not np.isfinite(bias).all() or not all(np.isfinite(v) and v > 0 for v in numbers):
            raise ValueError('无效模型参数。')
        accel, gyro = data.get('accel_range', 4), data.get('gyro_range', 500)
        if accel not in (2, 4, 8, 16) or gyro not in (125, 250, 500, 1000, 2000):
            raise ValueError('模型量程无效。')
        return cls(templates, durations, numbers[0], bias, numbers[1], numbers[2], numbers[3], numbers[4], accel, gyro)


def learn(profile: Profile, progress: Callable[[str], None] | None = None, use_new: bool = False,
          single_accel: bool = False) -> Model:
    if single_accel or use_new or profile.calibration or profile.background is None:
        from .robust import learn_robust
        return learn_robust(profile, progress, single_accel=single_accel)
    profile.validate()
    if len(profile.samples) != 5:
        raise ValueError('请先确认 5 个完整示范。')
    bg = profile.background
    if bg is None or bg.duration < 25 or np.max(np.diff(bg.times)) > .08:
        raise ValueError('请录制约 30 秒连续的非目标动作（至少 25 秒）。')
    motions = [clip.motion for clip in profile.samples]
    if len({(m.accel_range, m.gyro_range) for m in motions + [bg]}) != 1:
        raise ValueError('传感器量程不同，请使用同一固件重新录入。')
    bias = np.median(bg.values[:, 3:], axis=0)
    templates = [features(m, bias) for m in motions]
    durations = [m.duration for m in motions]
    pairs = np.array([[distance(a, b) if i != j else 1e6 for j, b in enumerate(templates)]
                      for i, a in enumerate(templates)])
    positive = float(np.max(np.min(pairs, axis=1)))
    threshold = max(.065, positive * 1.6 + .025)
    edges = [float(np.sqrt(np.mean((a[k] - b[k]) ** 2)))
             for a in templates for b in templates for k in (0, -1)]
    endpoint = max(.055, max(edges) * 1.4 + .025)
    peak = min(float(np.max(np.linalg.norm(m.values[:, 3:] - bias, axis=1))) for m in motions)
    finish_speed = max(3., min(max(float(np.linalg.norm(m.values[-1, 3:] - bias)) for m in motions) * 1.2 + 2., peak * .06 + 2.))
    onset = max(6., min(peak * .12, 20.))
    provisional = Model(templates, durations, threshold, bias, endpoint, finish_speed, onset, 1.,
                        motions[0].accel_range, motions[0].gyro_range)
    negative: list[float] = []
    sizes = sorted(set([min(durations), float(np.median(durations)), max(durations)]))
    for size in sizes:
        starts = np.linspace(0, bg.duration - size, min(60, max(2, int(bg.duration / .4))))
        for start in starts:
            window = bg.section(float(start), min(bg.duration, float(start) + size))
            query = features(window, bias)
            negative.append(min(distance(query, t) for t in templates))
        if progress:
            progress(f'校准非目标窗口：已比较 {len(negative)} 个')
    background_min = min(negative)
    if background_min <= threshold / .75:
        raise ValueError('示范与非目标动作区分不佳，请重录；不要在非目标录制中重复目标组合。')
    provisional.background_min = background_min
    if not all(provisional.match(m).accepted for m in motions):
        raise ValueError('示范的结束边界或握姿不一致，请检查完整回收和边界后重录。')
    return provisional


@dataclass(frozen=True)
class Event:
    session: int
    gesture_id: str
    event_id: str
    start: float
    end: float
    score: float


class Recognizer:
    def __new__(cls, model: Model, gesture_id: str, session: int):
        if cls is Recognizer and getattr(model, 'version', 1) == 2:
            from .robust import RobustRecognizer
            return object.__new__(RobustRecognizer)
        return object.__new__(cls)

    def __init__(self, model: Model, gesture_id: str, session: int):
        self.model, self.gesture_id, self.session = model, gesture_id, session
        self.points: deque[tuple[float, FloatArray]] = deque()
        self.starts: deque[float] = deque()
        self.active = False
        self.last_t: float | None = None
        self.last_check = -1e6
        self.consumed_end = -1e6
        self.last_match = Match(False, 1e6, '等待动作')

    def feed(self, t: float, values: Any) -> Event | None:
        row = np.asarray(values, dtype=float)
        if row.shape != (6,) or not np.isfinite(row).all() or not np.isfinite(t):
            raise ValueError('六轴输入无效。')
        if self.last_t is not None and (t <= self.last_t or t - self.last_t > .08):
            raise ValueError('识别输入时间异常或有采样缺口。')
        self.last_t = t
        self.points.append((t, row.copy()))
        while self.points and t - self.points[0][0] > self.model.max_duration + .5:
            self.points.popleft()
        while self.starts and t - self.starts[0] > self.model.max_duration:
            self.starts.popleft()
        recent = list(self.points)[-3:]
        smooth = np.mean([v for _, v in recent], axis=0)
        speed = float(np.linalg.norm(smooth[3:] - self.model.bias))
        change = 0.
        if len(self.points) >= 2:
            change = float(np.linalg.norm(row[:3] - self.points[-2][1][:3]))
        active = speed >= self.model.onset_speed or change > .045
        if active and not self.active:
            # Reviewed clips can contain a small amount of leading quiet time.
            for padding in (.06, .14, .22):
                start = max(self.points[0][0], t - padding)
                if start > self.consumed_end and start not in self.starts:
                    self.starts.append(start)
            while len(self.starts) > 24:
                self.starts.popleft()
        self.active = active
        # Completion requires a template-like endpoint, not a fixed quiet-time wait.
        if t - self.last_check < .035 or float(np.linalg.norm(row[3:] - self.model.bias)) > self.model.finish_speed:
            return None
        self.last_check = t
        points = list(self.points)
        for start in list(self.starts):
            if t - start < self.model.min_duration:
                continue
            selected = [(s, v) for s, v in points if s >= start]
            if len(selected) < 10:
                continue
            motion = Motion([s for s, _ in selected], [v for _, v in selected])
            match = self.model.match(motion)
            self.last_match = match
            if match.accepted:
                self.consumed_end = t
                self.starts.clear()
                return Event(self.session, self.gesture_id, str(uuid4()), selected[0][0], t, match.score)
        return None
