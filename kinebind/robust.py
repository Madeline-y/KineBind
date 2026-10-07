"""Version two: selected-axis motion evidence plus constrained template matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from .calibration import AXES, CATEGORIES, LABELS, SCALE, recording_quality, training_digest
from .data import FloatArray, Motion, Profile
from .matching import Event, Match, Model, POINTS, RADIUS, Recognizer, distance, features


class LearningError(ValueError):
    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.report = dict(status='failed', reason=reason, **details)


def full_features(motion: Motion, bias: FloatArray) -> FloatArray:
    x = features(motion, bias)
    # A different gravity baseline alone must not separate gesture and stillness.
    x[:, :3] -= x[0, :3]
    return x


def axis_distance(a: FloatArray, b: FloatArray) -> float:
    # Legacy distance uses six in its denominator; keep v1 unchanged.
    return distance(a, b) * float(np.sqrt(6 / a.shape[1]))


def rates(motion: Motion, bias: FloatArray) -> FloatArray:
    v = motion.values.copy()
    padded = np.vstack([v[:1], v[:1], v])
    cumulative = np.vstack([np.zeros((1, 6)), np.cumsum(padded, axis=0)])
    smooth = (cumulative[3:] - cumulative[:-3]) / 3
    result = (smooth - np.r_[np.zeros(3), bias]) / SCALE
    dt = np.r_[np.diff(motion.times)[0], np.diff(motion.times)]
    result[:, :3] = np.vstack([np.zeros((1, 3)), np.diff(smooth[:, :3], axis=0)]) * (.1 / dt[:, None])
    return result


def magnitude(values: FloatArray) -> FloatArray:
    return np.sqrt(np.mean(values ** 2, axis=1))


@dataclass
class RobustModel(Model):
    axes: tuple[int, ...] = tuple(range(6))
    min_peak: float = .02
    noise: FloatArray = field(default_factory=lambda: np.full(6, .002))
    category_results: dict[str, Any] = field(default_factory=dict)
    training_hash: str = ''
    version: int = 2

    @property
    def min_duration(self) -> float:
        return max(.15, min(self.durations) * .6)

    @property
    def max_duration(self) -> float:
        return max(self.durations) * 1.8

    def query(self, motion: Motion) -> FloatArray:
        return full_features(motion, self.bias)[:, self.axes]

    def match(self, motion: Motion) -> Match:
        if not self.min_duration <= motion.duration <= self.max_duration:
            return Match(False, 1e6, '动作时长超出示范范围')
        query = self.query(motion)
        peak = float(np.max(magnitude(query - query[0])))
        if peak < self.min_peak:
            return Match(False, 1e6, '有效运动不足，拒绝静止或轻微噪声')
        final = float(magnitude(rates(motion, self.bias)[-1:, self.axes])[0])
        if final > self.finish_speed:
            return Match(False, 1e6, '动作尚未完整结束')
        best = 1e6
        for i, template in enumerate(self.templates):
            edge = max(float(np.sqrt(np.mean((query[0] - template[0]) ** 2))),
                       float(np.sqrt(np.mean((query[-1] - template[-1]) ** 2))))
            if edge > self.endpoint_limit:
                continue
            score = axis_distance(query, template)
            best = min(best, score)
            if score <= self.threshold:
                return Match(True, score, '有效运动与完整波形匹配', i)
        return Match(False, best, '完整波形、边界或接受距离不符')

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(version=2, processing='causal-mean3-centered-axes-v2', axes=list(self.axes),
                      min_peak=self.min_peak, noise=self.noise.tolist(),
                      categories=self.category_results, training_hash=self.training_hash)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustModel:
        if data.get('version') != 2 or data.get('processing') != 'causal-mean3-centered-axes-v2' or data.get('points') != POINTS or data.get('radius') != RADIUS:
            raise ValueError('不兼容的新模型处理版本。')
        axes = tuple(data['axes'])
        if not axes or len(axes) > 6 or len(set(axes)) != len(axes) or any(type(i) is not int or not 0 <= i < 6 for i in axes):
            raise ValueError('无效有效轴集合。')
        templates = [np.asarray(t, dtype=float) for t in data['templates']]
        durations = [float(v) for v in data['durations']]
        bias, noise = np.asarray(data['bias'], dtype=float), np.asarray(data['noise'], dtype=float)
        numbers = [float(data[k]) for k in ('threshold', 'endpoint_limit', 'finish_speed', 'onset_speed', 'background_min', 'min_peak')]
        if len(templates) != 5 or any(t.shape != (POINTS, len(axes)) or not np.isfinite(t).all() for t in templates):
            raise ValueError('新模型缺少五个有效模板。')
        if len(durations) != 5 or not all(.15 <= d <= 20 for d in durations):
            raise ValueError('无效模型时长。')
        if bias.shape != (3,) or noise.shape != (6,) or not np.isfinite(bias).all() or not np.isfinite(noise).all() or np.any(noise < 0) or not all(np.isfinite(n) and n > 0 for n in numbers):
            raise ValueError('无效新模型参数。')
        accel, gyro = data['accel_range'], data['gyro_range']
        if accel not in (2, 4, 8, 16) or gyro not in (125, 250, 500, 1000, 2000):
            raise ValueError('无效新模型量程。')
        categories = data['categories']
        if not isinstance(categories, dict) or set(categories) != set(CATEGORIES):
            raise ValueError('新模型缺少分类校准摘要。')
        for item in categories.values():
            if not isinstance(item, dict) or not np.isfinite(item['distance']) or item['distance'] <= 0 or item.get('events') != 0:
                raise ValueError('无效分类校准摘要。')
        digest = data['training_hash']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('无效训练数据指纹。')
        return cls(templates, durations, numbers[0], bias, numbers[1], numbers[2], numbers[3], numbers[4],
                   accel, gyro, axes, numbers[5], noise, categories, digest)


class RobustRecognizer(Recognizer):
    model: RobustModel

    def feed(self, t: float, values: Any) -> Event | None:
        row = np.asarray(values, dtype=float)
        if row.shape != (6,) or not np.isfinite(row).all() or not np.isfinite(t):
            raise ValueError('六轴输入无效。')
        if self.last_t is not None and (t <= self.last_t or t - self.last_t > .08):
            raise ValueError('识别输入时间异常或有采样缺口。')
        previous_t = self.last_t
        self.last_t = t
        self.points.append((t, row.copy()))
        while self.points and t - self.points[0][0] > self.model.max_duration + .5:
            self.points.popleft()
        while self.starts and t - self.starts[0] > self.model.max_duration:
            self.starts.popleft()
        recent = list(self.points)[-4:]
        smooth = np.mean([v for _, v in recent[-3:]], axis=0)
        signal = (smooth - np.r_[np.zeros(3), self.model.bias]) / SCALE
        signal[:3] = 0.
        if len(recent) >= 2 and previous_t is not None:
            before = np.mean([v for _, v in recent[:-1][-3:]], axis=0)
            signal[:3] = (smooth[:3] - before[:3]) * .1 / (t - previous_t)
        activity = float(np.sqrt(np.mean(signal[list(self.model.axes)] ** 2)))
        active = activity >= self.model.onset_speed
        if active and not self.active and t > self.consumed_end:
            for padding in (.06, .14, .22):
                start = max(self.points[0][0], t - padding)
                if start > self.consumed_end and not any(abs(start - s) < .025 for s in self.starts):
                    self.starts.append(start)
            # Preserve earliest starts through internal pauses; bound computation.
            while len(self.starts) > 9:
                del self.starts[3]
        self.active = active
        if t - self.last_check < .04 or activity > self.model.finish_speed:
            return None
        self.last_check = t
        for start in list(self.starts):
            if t - start < self.model.min_duration:
                continue
            selected = [(s, v) for s, v in self.points if s >= start]
            if len(selected) < 10:
                continue
            motion = Motion([s for s, _ in selected], [v for _, v in selected],
                            accel_range=self.model.accel_range, gyro_range=self.model.gyro_range)
            self.last_match = match = self.model.match(motion)
            if match.accepted:
                self.consumed_end = t
                self.starts.clear()
                return Event(self.session, self.gesture_id, str(uuid4()), selected[0][0], t, match.score)
        return None


def learn_robust(profile: Profile, progress: Callable[[str], None] | None = None,
                 single_accel: bool = False) -> RobustModel:
    profile.validate()
    if len(profile.samples) != 5:
        raise LearningError('请先确认五个独立完整示范。')
    for category in CATEGORIES:
        if category not in profile.calibration:
            raise LearningError(f'缺少{LABELS[category]}校准，请选择该类别补录约 10 秒。', category=category)
        try:
            recording_quality(category, profile.calibration[category], profile.calibration.get('static'))
        except ValueError as error:
            raise LearningError(str(error), category=category) from error
    motions = [c.motion for c in profile.samples]
    static = profile.calibration['static']
    if len({(m.accel_range, m.gyro_range) for m in [*motions, *profile.calibration.values()]}) != 1:
        raise LearningError('示范与分类校准量程不同，请使用同一固件重新采集。')
    bias = np.median(static.values[:, 3:], axis=0)
    noise = np.maximum(.001, np.percentile(np.abs(rates(static, bias)), 99, axis=0))
    full = [full_features(m, bias) for m in motions]
    durations = [m.duration for m in motions]
    strengths = np.min([np.std(f, axis=0) for f in full], axis=0)
    informative = tuple(int(i) for i in np.flatnonzero(strengths > np.maximum(.006, noise * 4)))
    if single_accel:
        informative = tuple(i for i in informative if i < 3)
    if not informative:
        raise LearningError('五个示范没有一致的有效加速度运动，请检查边界或重新示范。' if single_accel else
                            '五个示范没有一致的有效运动，请检查边界或重新示范。')
    # Features are reused across candidate subsets; labels and timestamps remain separate.
    windows: dict[str, list[tuple[FloatArray, list[float], Motion]]] = {}
    for category in CATEGORIES:
        bg = profile.calibration[category]
        windows[category] = []
        for size in sorted(set([min(durations), float(np.median(durations)), max(durations)])):
            size = min(size, bg.duration)
            for start in np.linspace(0., bg.duration - size, 12):
                window = bg.section(float(start), min(bg.duration, float(start) + size))
                windows[category].append((full_features(window, bias), [float(start), float(start) + window.duration], window))
    all_rates = [rates(m, bias) for m in motions]
    best_failure: dict[str, Any] = {}
    for count in range(1, 2 if single_accel else len(informative) + 1):
        viable: list[tuple[float, RobustModel]] = []
        for axes in combinations(informative, count):
            if progress:
                progress('正在校准有效轴：' + '、'.join(AXES[i] for i in axes))
            templates = [f[:, axes] for f in full]
            pair = np.array([[axis_distance(a, b) if i != j else np.inf
                              for j, b in enumerate(templates)] for i, a in enumerate(templates)])
            positive = float(np.max(np.min(pair, axis=1)))
            # Keep the established fixed-scale alignment floor. It is still
            # bounded by every negative category, never raised to force a pass.
            threshold = max(.065, positive * 1.35 + .015)
            peaks = [float(np.max(magnitude(a - a[0]))) for a in templates]
            activity_peaks = [float(np.max(magnitude(r[:, axes]))) for r in all_rates]
            quiet = float(np.sqrt(np.mean(noise[list(axes)] ** 2)))
            onset = max(.012, quiet * 6, min(activity_peaks) * .1)
            finish = max(.01, quiet * 5,
                         min(max(float(magnitude(r[-1:, axes])[0]) for r in all_rates) * 1.5 + .005,
                             min(activity_peaks) * .12 + .005))
            endpoint = max(.055, max(float(np.sqrt(np.mean((a[k] - b[k]) ** 2)))
                                    for a in templates for b in templates for k in (0, -1)) * 1.25 + .02)
            model = RobustModel(templates, durations, threshold, bias, endpoint, finish, onset, 1.,
                                motions[0].accel_range, motions[0].gyro_range, axes,
                                max(.015, min(peaks) * .35), noise, {}, training_digest(profile))
            failed = False
            summaries: dict[str, Any] = {}
            for category in CATEGORIES:
                scored = []
                for q, interval, window in windows[category]:
                    raw_distance = min(axis_distance(q[:, axes], a) for a in templates)
                    if single_accel:
                        checked = model.match(window)
                        eligible_window = checked.score < 1e6
                        scored.append((checked.score if eligible_window else raw_distance,
                                       interval, eligible_window))
                    else:
                        scored.append((raw_distance, interval, True))
                raw_closest = min(scored, key=lambda item: item[0])
                # Apply exactly the same duration, motion, completion and
                # endpoint gates as recognition before checking DTW margin.
                eligible = [item for item in scored if item[2]] if single_accel else scored
                closest = min(eligible, key=lambda item: item[0]) if eligible else raw_closest
                summaries[category] = dict(distance=closest[0], interval=closest[1], events=0)
                if single_accel:
                    summaries[category].update(evidence_rejected_windows=len(scored) - len(eligible),
                                               eligible_distance=closest[0] if eligible else None)
                if eligible and closest[0] <= threshold / .72:
                    best_failure = dict(category=category, interval=closest[1], axes=list(axes),
                                        distance=closest[0], threshold=threshold)
                    failed = True
                    break
            if failed:
                continue
            for i, motion in enumerate(motions, 1):
                if not model.match(motion).accepted:
                    best_failure = dict(sample=i, axes=list(axes), threshold=threshold)
                    failed = True
                    break
            if failed:
                continue
            # Calibrate against actual candidate/event generation, not windows alone.
            for category, bg in profile.calibration.items():
                engine = RobustRecognizer(model, profile.id, 0)
                for sample_number, (t, row) in enumerate(zip(bg.times, bg.values)):
                    if progress and sample_number % 100 == 0:
                        progress(f'流式校准 {LABELS[category]}：{t:.1f} 秒')
                    event = engine.feed(float(t), row)
                    if event:
                        best_failure = dict(category=category, interval=[event.start, event.end], axes=list(axes))
                        failed = True
                        break
                if progress:
                    progress(f'流式校准：{LABELS[category]}，' + ('检测到混淆' if failed else '零事件'))
                if failed:
                    break
            if not failed:
                model.category_results = summaries
                model.background_min = min(s['distance'] for s in summaries.values())
                margin = min((s['eligible_distance'] for s in summaries.values()
                              if s.get('eligible_distance') is not None), default=model.background_min)
                viable.append((margin / threshold if single_accel else model.background_min / threshold, model))
        if viable:
            model = max(viable, key=lambda item: item[0])[1]
            weakest = min(model.category_results, key=lambda k: model.category_results[k]['distance'])
            profile.report = dict(status='learned', reason='学习完成；三类流式校准均为零事件。',
                                  axes=list(model.axes), threshold=model.threshold,
                                  categories=model.category_results, category=weakest,
                                  version=2, data_fingerprint=model.training_hash,
                                  learning_mode='single_accel' if single_accel else 'auto')
            return model
    if 'sample' in best_failure:
        raise LearningError(f'第 {best_failure["sample"]} 个示范的结束或边界不一致，请检查完整动作。', **best_failure)
    failure_category = best_failure.get('category')
    reason = f'目标与{LABELS[failure_category]}区分不足，请检查相关片段、示范边界或补录该类。' if failure_category else '示范重复性与非目标区分不足，请检查示范和三类校准。'
    raise LearningError(reason, **best_failure)
