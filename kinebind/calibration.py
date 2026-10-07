"""Classified capture quality and advisory (never auto-saved) gesture bounds."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .data import Motion, Profile

CATEGORIES = ('static', 'shake', 'pick_place')
LABELS = {'static': '静止', 'shake': '普通晃动', 'pick_place': '拿放'}
INSTRUCTIONS = {
    'static': '保持示范的起始握姿，安静持握约 10 秒。',
    'shake': '做日常小幅晃动约 10 秒，不做完整目标动作。',
    'pick_place': '正常拿起、放下设备约 10 秒，不做完整目标动作。',
}
AXES = ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
SCALE = np.array([1., 1., 1., 100., 100., 100.])


def training_digest(profile: Profile) -> str:
    body = dict(samples=[c.to_dict() for c in profile.samples],
                calibration={k: v.to_dict() for k, v in sorted(profile.calibration.items())},
                background=profile.background.to_dict() if profile.background else None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recording_quality(category: str, motion: Motion, static: Motion | None = None) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError('请选择静止、普通晃动或拿放类别。')
    label = LABELS[category]
    if motion.duration < 9.5:
        raise ValueError(f'{label}记录不足，请采集约 10 秒（至少 9.5 秒）。')
    if np.max(np.diff(motion.times)) > .08:
        raise ValueError(f'{label}记录有采样缺口，请重录这一类。')
    edges = np.array([32767 * .061 * (motion.accel_range // 2) / 1000] * 3 +
                     [32767 * 4.375 * (motion.gyro_range / 125) / 1000] * 3)
    if np.any(np.abs(motion.values) >= edges * .998):
        raise ValueError(f'{label}记录接近量程极限，请轻一些重录。')
    values = motion.values
    center = np.median(values, axis=0)
    spread = np.percentile(np.abs(values - center) / SCALE, 95, axis=0)
    if category == 'static':
        if np.max(spread[:3]) > .025 or np.max(spread[3:]) > .03:
            raise ValueError('静止记录不够稳定，请保持起始握姿重录。')
    else:
        floor = np.full(6, .01)
        if static is not None:
            base = static.values - np.median(static.values, axis=0)
            floor = np.maximum(floor, np.percentile(np.abs(base) / SCALE, 95, axis=0) * 6)
        if not np.any(spread > floor):
            raise ValueError(f'{label}记录几乎静止，请补录真实的{label}活动。')
    return dict(category=category, label=label, duration=motion.duration,
                spread=spread.tolist(), status='valid')


def suggest_bounds(motion: Motion) -> dict[str, Any]:
    if np.max(np.diff(motion.times)) > .08:
        return dict(start=0., end=motion.duration, candidates=[], status='ambiguous',
                    reason='记录有采样缺口，请检查或重录，范围尚未确认。')
    v = motion.values
    gyro_bias = np.median(v[:min(5, len(v)), 3:], axis=0)
    gyro = np.linalg.norm((v[:, 3:] - gyro_bias) / 100., axis=1)
    delta = np.vstack([np.zeros((1, 3)), np.diff(v[:, :3], axis=0)])
    dt = np.r_[np.diff(motion.times)[0], np.diff(motion.times)]
    accel = np.linalg.norm(delta, axis=1) * .1 / dt
    activity = np.maximum(gyro, accel)
    threshold = max(.015, float(np.max(activity)) * .06)
    moving = np.flatnonzero(activity > threshold)
    if len(moving) < 3:
        return dict(start=0., end=motion.duration, candidates=[], status='ambiguous',
                    reason='未发现明确运动，请人工检查或重新示范。')
    cuts = np.flatnonzero(np.diff(motion.times[moving]) > .8) + 1
    groups = np.split(moving, cuts)
    candidates = [dict(start=max(0., float(motion.times[g[0]]) - .14),
                       end=min(motion.duration, float(motion.times[g[-1]]) + .2)) for g in groups]
    start, end = candidates[0]['start'], candidates[-1]['end']
    ambiguous = len(groups) > 1 or moving[0] < 2 or moving[-1] >= len(v) - 2
    reason = ('发现多段运动或边界可能截断，请核对完整动作；建议未自动保存。' if ambiguous else
              '已建议完整运动范围，请检查内部停顿及结束阶段后确认。')
    return dict(start=start, end=end, candidates=candidates,
                status='ambiguous' if ambiguous else 'suggested', reason=reason)
