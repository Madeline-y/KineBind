"""Frozen-model acceptance through the application boundary and FakeMouse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .actions import FakeMouse
from .calibration import CATEGORIES, LABELS
from .controller import GestureController
from .data import Motion, Profile
from .matching import Event
from .storage import ProfileStore


def motion_digest(motion: Motion) -> str:
    payload = dict(times=np.round(motion.times, 6).tolist(),
                   values=np.round(motion.values, 6).tolist(),
                   ranges=[motion.accel_range, motion.gyro_range])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def contains_training(test: Motion, training: Motion) -> bool:
    """Detect an entire reused raw clip, including clips embedded in a longer CSV."""
    n = len(training.times)
    if len(test.times) < n:
        return False
    # Anchor on the most distinctive changing row, not a common quiet endpoint.
    variation = np.abs(np.diff(training.values, axis=0)).sum(axis=1)
    anchor = int(np.argmax(variation)) + 1
    rows = np.flatnonzero(np.all(np.isclose(test.values, training.values[anchor], atol=1e-6, rtol=0), axis=1))
    for index in rows:
        start = int(index) - anchor
        if start < 0 or start + n > len(test.times):
            continue
        if np.allclose(test.values[start:start+n], training.values, atol=1e-6, rtol=0) and np.allclose(
                test.times[start:start+n] - test.times[start], training.times, atol=1e-5, rtol=0):
            return True
    return False


def score_events(events: list[Event], intervals: list[tuple[float, float]]) -> dict[str, Any]:
    """Count per annotated action, with an explicit four-sample finish tolerance."""
    assigned: list[list[Event]] = [[] for _ in intervals]
    unmatched = []
    for event in events:
        choices = [i for i, (start, end) in enumerate(intervals)
                   if start - .25 <= event.start <= end and
                   event.end <= (intervals[i+1][0] if i+1 < len(intervals) else end + 1.)]
        if not choices:
            unmatched.append(event)
            continue
        assigned[min(choices, key=lambda i: abs(event.start - intervals[i][0]))].append(event)
    details = []
    for i, ((start, end), hits) in enumerate(zip(intervals, assigned), 1):
        early = any(e.end < end - .08 for e in hits)
        correct = len(hits) == 1 and not early
        details.append(dict(action=i, interval=[start, end], correct=correct,
                            missed=not hits, early=early, duplicate=len(hits) > 1,
                            events=[dict(start=e.start, end=e.end, score=e.score,
                                         completion_delay_ms=max(0., e.end - end) * 1000) for e in hits]))
    return dict(correct=sum(d['correct'] for d in details), missed=sum(d['missed'] for d in details),
                early=sum(d['early'] for d in details), duplicate=sum(d['duplicate'] for d in details),
                unmatched=len(unmatched), actions=details, annotation_tolerance_s=.08)


def evaluate(profile: Profile, target: Motion | None, intervals: list[tuple[float, float]],
             negatives: dict[str, Motion], independent: bool = False) -> dict[str, Any]:
    base: dict[str, Any] = dict(status='incomplete', real_verified=False, mouse='simulated',
                                profile_id=profile.id, profile_name=profile.name)
    if profile.model is None or getattr(profile.model, 'version', 1) != 2:
        return dict(base, reason='需要已完成三类校准的新规则模型。')
    base['model_sha256'] = hashlib.sha256(json.dumps(profile.model.to_dict(), sort_keys=True).encode()).hexdigest()
    if target is None or len(intervals) != 10 or set(negatives) != set(CATEGORIES):
        return dict(base, reason='需要十次人工标注目标动作及静止、普通晃动、拿放各 30 秒新记录。')
    if not independent:
        return dict(base, reason='请确认这些记录未用于学习、选轴或调参，再设置 independent=true。')
    previous_end = -1.
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or end > target.duration + 1e-6 or start <= previous_end:
            raise ValueError('目标区间需要按时间排序、不重叠且位于记录范围内。')
        previous_end = end
    for category, motion in negatives.items():
        if motion.duration < 29.98:
            return dict(base, reason=f'{LABELS[category]}独立测试不足 30 秒。', category=category)
    training = [c.raw for c in profile.samples] + list(profile.calibration.values())
    if profile.background is not None:
        training.append(profile.background)
    captures = {'target': target, **negatives}
    for label, capture in captures.items():
        if any(contains_training(capture, old) for old in training):
            return dict(base, reason='验收记录包含已用于学习的原始片段，请另采独立记录。', category=label)
        if any(contains_training(capture, c.motion) for c in profile.samples):
            return dict(base, reason='验收记录包含已确认的训练示范，请另采独立记录。', category=label)
    base['data'] = {k: dict(source=m.source, sha256=motion_digest(m), duration=m.duration) for k, m in captures.items()}
    simulated = any('SIMULATED' in m.source.upper() for m in [*training, *captures.values()])
    results: dict[str, list[Event]] = {}
    requests: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as folder:
        mouse = FakeMouse()
        app = GestureController(ProfileStore(Path(folder)), mouse, threaded=False)
        app.select(profile)
        app.set_ranges(profile.model.accel_range, profile.model.gyro_range)
        app.set_connected(True)
        for label, capture in captures.items():
            if (capture.accel_range, capture.gyro_range) != (profile.model.accel_range, profile.model.gyro_range):
                raise ValueError('验收记录量程与模型不同。')
            app.start_recognition()
            count_before = len(mouse.requests)
            found = []
            for t, row in zip(capture.times, capture.values):
                app.on_sample(float(t), row, float(t))
                # Drain per sample: long tests must not overflow the UI queue.
                while not app.results.empty():
                    kind, value = app.results.get_nowait()
                    if kind == 'event':
                        found.append(value[0])
            app.stop()
            results[label], requests[label] = found, len(mouse.requests) - count_before
    positive = score_events(results['target'], intervals)
    background = {k: dict(events=len(results[k]), execution_requests=requests[k],
                          event_intervals=[[e.start, e.end] for e in results[k]]) for k in CATEGORIES}
    passed = (positive['correct'] >= 9 and positive['early'] == 0 and positive['duplicate'] == 0 and
              positive['unmatched'] == 0 and all(v['events'] == 0 and v['execution_requests'] == 0 for v in background.values()))
    status = ('simulated_pass' if passed else 'simulated_fail') if simulated else ('passed' if passed else 'failed')
    return dict(base, status=status, real_verified=passed and not simulated,
                target=positive, non_target=background, target_execution_requests=requests['target'],
                reason='合成数据只验证验收工具，不能代表真实指标。' if simulated else '独立数据验收完成；实际指针操作仍需用户核对。')
