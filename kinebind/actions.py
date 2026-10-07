"""Explicitly enabled mouse actions with locked session invalidation."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import threading
from typing import Protocol

from .matching import Event


class Mouse(Protocol):
    def move_left(self, pixels: int) -> tuple[int, int]: ...


class FakeMouse:
    def __init__(self, x: int = 500, y: int = 300, fail: bool = False):
        self.position = (x, y)
        self.requests: list[int] = []
        self.fail = fail

    def move_left(self, pixels: int) -> tuple[int, int]:
        self.requests.append(pixels)
        if self.fail:
            raise OSError('模拟鼠标执行失败')
        self.position = (max(0, self.position[0] - pixels), self.position[1])
        return self.position


class WindowsMouse:
    def move_left(self, pixels: int) -> tuple[int, int]:
        if os.name != 'nt':
            raise OSError('实际鼠标控制需要 Windows。')
        api = ctypes.WinDLL('user32', use_last_error=True)
        api.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        api.GetCursorPos.restype = wintypes.BOOL
        api.GetClipCursor.argtypes = [ctypes.POINTER(wintypes.RECT)]
        api.GetClipCursor.restype = wintypes.BOOL
        api.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        api.SetCursorPos.restype = wintypes.BOOL
        point, rect = wintypes.POINT(), wintypes.RECT()
        if not api.GetCursorPos(ctypes.byref(point)) or not api.GetClipCursor(ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        x = max(rect.left, min(rect.right - 1, point.x - pixels))
        y = point.y
        if not api.SetCursorPos(x, y):
            raise ctypes.WinError(ctypes.get_last_error())
        return x, y


class ExecutionGate:
    def __init__(self, mouse: Mouse):
        self.mouse = mouse
        self._lock = threading.Lock()
        self._enabled = False
        self._session = 0
        self.gesture_id = ''
        self.pixels = 100
        self.seen: set[str] = set()
        self.end = -math.inf

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def start(self, gesture_id: str, pixels: int = 100, execute: bool = True) -> int:
        if not gesture_id or isinstance(pixels, bool) or not isinstance(pixels, int) or not 1 <= pixels <= 2000:
            raise ValueError('无效动作绑定。')
        with self._lock:
            self._session += 1
            self._enabled, self.gesture_id, self.pixels = execute, gesture_id, pixels
            self.seen.clear()
            self.end = -math.inf
            return self._session

    def stop(self) -> None:
        # Holding this same lock through execution makes stop linearizable.
        with self._lock:
            self._enabled = False
            self._session += 1
            self.seen.clear()

    def execute(self, event: Event) -> bool:
        with self._lock:
            if not self._enabled or event.session != self._session or event.gesture_id != self.gesture_id:
                return False
            if event.event_id in self.seen or event.start <= self.end:
                return False
            if not all(math.isfinite(v) for v in (event.start, event.end, event.score)) or event.end <= event.start:
                return False
            self.seen.add(event.event_id)
            self.end = event.end
            if len(self.seen) > 2000:
                self.seen.clear()  # The monotonic consumed interval still guards older events.
            try:
                self.mouse.move_left(self.pixels)
            except Exception:
                self._enabled = False
                self._session += 1
                raise
            return True
