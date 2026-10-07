"""Explicit synthetic fixtures for withdrawn GUI checks, never personal templates."""
from collections.abc import Sequence
import numpy as np
from .data import Motion


def gesture(duration: float = 1., amplitude: float = 1.) -> Motion:
    t = np.arange(0, duration + .01, .02)
    u = t / duration
    pulse = np.sin(2 * np.pi * u) * np.sin(np.pi * u) ** 2
    x = np.zeros((len(t), 6))
    x[:, 2] = 1 + amplitude * .2 * np.sin(np.pi * u) ** 2
    x[:, 0] = amplitude * .6 * pulse
    x[:, 4] = amplitude * 180 * pulse
    return Motion(t, x, 'SIMULATED gesture')


def background(duration: float = 30.) -> Motion:
    t = np.arange(0, duration + .01, .02)
    x = np.zeros((len(t), 6))
    x[:, 2] = 1
    x[:, 0] = .008 * np.sin(t)
    x[:, 3] = .6 * np.cos(t)
    return Motion(t, x, 'SIMULATED non-target')


def join(motions: Sequence[Motion]) -> Motion:
    times, values, offset = [], [], 0.
    for motion in motions:
        times.extend((motion.times + offset).tolist())
        values.extend(motion.values.tolist())
        offset += motion.duration + .02
    return Motion(times, values, 'SIMULATED combined recording')


def calibration_records() -> dict[str, Motion]:
    still, shake, handling = background(10.), background(10.), background(10.)
    shake.values[:, 3] += 12 * np.sin(shake.times * 2.3)
    handling.values[:, 2] += .15 * np.sin(handling.times * .8)
    handling.values[:, 5] += 12 * np.sin(handling.times * .7)
    return {'static': still, 'shake': shake, 'pick_place': handling}
