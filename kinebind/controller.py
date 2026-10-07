"""Application boundary shared by UI, deterministic replay and behavior tests."""
from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ExecutionGate, Mouse
from .data import Clip, Motion, Profile, Recorder
from .matching import Event, Recognizer, learn
from .storage import ProfileStore
from .calibration import CATEGORIES, LABELS, recording_quality, suggest_bounds, training_digest
from .live_recording import LiveTestRecorder
from imu_protocol import Sample


class GestureController:
    def __init__(self, store: ProfileStore, mouse: Mouse, threaded: bool = True,
                 test_folder: Path | None = None, test_source: str = 'USB realtime test'):
        self.store = store
        self.gate = ExecutionGate(mouse)
        self.threaded = threaded
        self.profile: Profile | None = None
        self.recorder = Recorder()
        self.review: Motion | None = None
        self.suggestion: dict[str, Any] = {}
        self.connected = False
        self.revision = 0
        self.recognizer: Recognizer | None = None
        self.results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1000)
        self._cancel = threading.Event()
        self._inputs: queue.Queue[tuple[float, float, Any]] = queue.Queue(maxsize=300)
        self._session = 0
        self._active = False
        self._replay = False
        self._learning_generation = 0
        self._learning_lock = threading.RLock()
        self.test_recorder = LiveTestRecorder(test_folder, test_source) if test_folder is not None else None

    @property
    def enabled(self) -> bool:
        return self.gate.enabled

    @property
    def active(self) -> bool:
        return self._active

    def _profile(self) -> Profile:
        if self.profile is None:
            raise ValueError('请先新建或加载一个手势。')
        return self.profile

    def notify(self, kind: str, value: Any) -> None:
        try:
            self.results.put_nowait((kind, value))
        except queue.Full:
            try: self.results.get_nowait()
            except queue.Empty: pass
            self.results.put_nowait((kind, value))

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        if not connected:
            self.stop()
            self.recorder.cancel()

    def set_ranges(self, accel_range: int, gyro_range: int) -> None:
        if accel_range not in (2, 4, 8, 16) or gyro_range not in (125, 250, 500, 1000, 2000):
            raise ValueError('设备传感器量程无效。')
        self.recorder.accel_range, self.recorder.gyro_range = accel_range, gyro_range

    def new_profile(self, name: str) -> Profile:
        profile = Profile.new(name)
        existing, _ = self.store.list_profiles()
        if any(p.name.casefold() == profile.name.casefold() for p in existing):
            raise ValueError('该名称已存在，请加载原有手势或使用新名称。')
        self.select(profile)
        self.store.save(profile)
        return profile

    def select(self, profile: Profile) -> None:
        profile.validate()
        self.stop()
        self.recorder.cancel()
        self.profile = copy.deepcopy(profile)
        self.review = None
        self.suggestion = {}
        self.revision += 1

    def begin_recording(self, now: float, kind: str = 'sample') -> None:
        self._profile()
        if not self.connected:
            raise ValueError('请先连接设备并等待有效数据。')
        self.stop()
        self.review = None
        self.suggestion = {}
        self.recorder.begin(now, kind)

    def on_sample(self, t: float, values: Any, now: float, sample: Sample | None = None) -> None:
        self.recorder.feed(t, values, now)
        if self.recorder.kind in ('background', *CATEGORIES) and self.recorder.state == 'recording' and self.recorder.count >= 2:
            if self.recorder.times[-1] - self.recorder.times[0] >= (30 if self.recorder.kind == 'background' else 10):
                self.finish_recording()
        if self.recognizer is None or not self._active:
            return
        if self.test_recorder is not None and not self._replay:
            try:
                self.test_recorder.write(t, sample)
            except Exception:
                self.stop()
                raise
        if self.threaded:
            try:
                self._inputs.put_nowait((time.monotonic(), t, np.asarray(values).copy()))
            except queue.Full:
                self.stop()
                self.notify('error', '识别队列已满，控制已停止。')
        else:
            event = self.recognizer.feed(t, values)
            if event:
                self.execute(event)

    def finish_recording(self) -> Motion:
        motion = self.recorder.finish()
        if self.recorder.kind in CATEGORIES:
            self.set_calibration(self.recorder.kind, motion)
            self.review = None
        elif self.recorder.kind == 'background':
            p = self._profile()
            p.background, p.model = motion, None
            p.report = dict(status='needs_learning', reason='旧非目标记录已更新，请重新学习。')
            self.revision += 1
            self._save_draft()
            self.review = None
            self.notify('background', motion)
        else:
            self.review = motion
            self.suggestion = suggest_bounds(motion)
            self.notify('review', motion)
        return motion

    def import_review(self, motion: Motion) -> None:
        self._profile()
        self.stop()
        self.recorder.cancel()
        self.review = motion
        self.suggestion = suggest_bounds(motion)
        self.notify('review', motion)

    def suggest_review(self) -> None:
        if self.review is None or self.recorder.state in ('countdown', 'recording'):
            raise ValueError('请先结束录制或选择一个示范。')
        self.suggestion = suggest_bounds(self.review)
        self.notify('review', self.review)

    def set_calibration(self, category: str, motion: Motion) -> None:
        self.stop()
        self.recorder.cancel()
        p = self._profile()
        try:
            quality = recording_quality(category, motion, p.calibration.get('static'))
        except ValueError as error:
            p.report = dict(status='failed', category=category, reason=str(error),
                            data_fingerprint=training_digest(p))
            self._save_draft()
            self.notify('report', p.report)
            raise
        p.calibration[category] = motion
        p.model = None
        p.report = dict(status='needs_learning', reason=f'{LABELS[category]}已更新，请重新学习。')
        p.report['data_fingerprint'] = training_digest(p)
        self.revision += 1
        self._save_draft()
        self.recorder.state = 'review'
        self.notify('calibration', (category, motion, quality))

    def set_background(self, motion: Motion) -> None:
        self._profile()
        self.stop()
        self.recorder.cancel()
        p = self._profile()
        p.background, p.model = motion, None
        p.report = dict(status='needs_learning', reason='旧非目标记录已更新，请重新学习。')
        self.revision += 1
        self._save_draft()
        self.notify('background', motion)

    def save_review(self, start: float, end: float, slot: int | None = None) -> None:
        if self.review is None:
            raise ValueError('请先结束录制或导入记录。')
        p = self._profile()
        clip = Clip(self.review, start, end)
        clip.validate()
        if slot is None:
            if len(p.samples) >= 5:
                raise ValueError('已有五个示范，请先选择要替换的样本。')
            if any(np.array_equal(clip.motion.values, old.motion.values) for old in p.samples):
                raise ValueError('这段示范已经保存，请选择下一次独立动作。')
            p.samples.append(clip)
        elif 0 <= slot < len(p.samples):
            p.samples[slot] = clip
        else:
            raise ValueError('请选择已有样本进行替换。')
        p.model = None
        p.report = dict(status='needs_learning', reason='示范已修改，请重新学习。')
        p.report['data_fingerprint'] = training_digest(p)
        self.revision += 1
        self._save_draft()
        self.notify('saved', f'已确认 {len(p.samples)}/5 个示范')

    def _save_draft(self) -> None:
        p = self._profile()
        # Keep a previous usable model on disk until an edited draft learns successfully.
        try:
            saved = self.store.load(p.id)
        except FileNotFoundError:
            saved = None
        if saved is None or saved.model is None:
            self.store.save(p)
        self.store.save_draft(p)

    def learn_current(self, use_new: bool = False, single_accel: bool = False) -> None:
        self.stop()
        revision = self.revision
        generation = self._learning_generation
        snapshot = copy.deepcopy(self._profile())
        def progress(message: str) -> None:
            if generation != self._learning_generation or revision != self.revision:
                raise ValueError('学习已停止或数据已改变，本次结果未保存。')
            self.notify('progress', message)
        try:
            if single_accel:
                snapshot.model = learn(snapshot, progress, use_new=use_new, single_accel=True)
            else:
                snapshot.model = learn(snapshot, progress, use_new=use_new)
        except ValueError as error:
            if generation == self._learning_generation and revision == self.revision and self._profile().id == snapshot.id:
                current = self._profile()
                current.report = getattr(error, 'report', dict(status='failed', reason=str(error)))
                current.report['data_fingerprint'] = training_digest(current)
                self._save_draft()
                self.notify('report', current.report)
            raise
        # A concurrent stop returns either before this commit, cancelling it,
        # or after the complete commit; no late save after stop has returned.
        with self._learning_lock:
            if generation != self._learning_generation or revision != self.revision or self._profile().id != snapshot.id:
                raise ValueError('学习已停止或手势已改变，本次结果未保存。')
            self.store.save(snapshot)
            self.profile = snapshot
            self.store.save_draft(snapshot)
            self.notify('report', snapshot.report)
            self.notify('learned', f'学习完成；接受距离 ≤ {snapshot.model.threshold:.3f}')

    def change_binding(self, pixels: int) -> None:
        self.stop()
        p = copy.deepcopy(self._profile())
        try:
            saved = self.store.load(p.id)
        except FileNotFoundError:
            saved = None
        if p.model is None and saved is not None and saved.model is not None:
            raise ValueError('示范有未学习的修改，请先学习，或重新加载已保存手势后修改绑定。')
        p.pixels = pixels
        p.validate()
        self.store.save(p)
        self.profile = p
        self.revision += 1

    def start_recognition(self, replay: bool = False) -> None:
        p = self._profile()
        if p.model is None:
            raise ValueError('请先完成五个示范、非目标校准和学习。')
        if not replay and not self.connected:
            raise ValueError('请先连接设备。')
        if not replay and (p.model.accel_range, p.model.gyro_range) != (self.recorder.accel_range, self.recorder.gyro_range):
            raise ValueError('设备量程与模板不同，请使用录入时的固件或重新学习。')
        if self.recorder.state in ('countdown', 'recording'):
            raise ValueError('请先结束录制。')
        self.stop()
        self._replay = replay
        self._session = self.gate.start(p.id, p.pixels, execute=not replay)
        if self.test_recorder is not None and not replay:
            try:
                path = self.test_recorder.start(p)
            except Exception:
                self.stop()
                raise
            self.notify('test_recording', path)
        self.recognizer = Recognizer(p.model, p.id, self._session)
        self._active = True
        self._cancel = threading.Event()
        self._inputs = queue.Queue(maxsize=300)
        if self.threaded and not replay:
            threading.Thread(target=self._run_recognition, args=(self.recognizer, self._inputs, self._cancel),
                             name='kinebind-recognition', daemon=True).start()

    def replay_motion(self, motion: Motion) -> None:
        engine = self.recognizer
        cancel = self._cancel
        if engine is None or not self._replay:
            raise ValueError('请先开始回放会话。')
        if (motion.accel_range, motion.gyro_range) != (engine.model.accel_range, engine.model.gyro_range):
            raise ValueError('回放记录与模板量程不同。')
        count = 0
        last_report = -1e6
        for t, row in zip(motion.times, motion.values):
            if cancel.is_set(): return
            event = engine.feed(float(t), row)
            last_report = self._report_rejection(engine, float(t), last_report)
            if event:
                count += 1
                self.execute(event)
        if not cancel.is_set():
            self.notify('replay_done', f'回放完成：{count} 个完整动作事件（不执行鼠标）')

    def _run_recognition(self, engine: Recognizer, inputs: queue.Queue, cancel: threading.Event) -> None:
        last_report = -1e6
        try:
            while not cancel.is_set():
                try: received, t, values = inputs.get(timeout=.1)
                except queue.Empty: continue
                if time.monotonic() - received > .5:
                    raise ValueError('识别处理延时过大，控制已停止。')
                event = engine.feed(t, values)
                last_report = self._report_rejection(engine, t, last_report)
                if event and not cancel.is_set():
                    self.execute(event)
        except Exception as error:
            if not cancel.is_set() and engine.session == self._session:
                self.stop()
                self.notify('error', str(error))

    def _report_rejection(self, engine: Recognizer, t: float, previous: float) -> float:
        if engine.starts and engine.last_check == t and not engine.last_match.accepted and engine.last_match.score < 1e6 and t - previous >= .5:
            self.notify('rejected', (engine.session, t, engine.last_match))
            return t
        return previous

    def execute(self, event: Event) -> bool:
        if not self._active or event.session != self._session:
            return False
        if self._replay:
            self.notify('event', (event, '回放匹配；鼠标未执行'))
            return False
        try:
            moved = self.gate.execute(event)
        except Exception as error:
            self.stop()
            self.notify('error', f'已识别，但鼠标执行失败：{error}')
            return False
        if moved:
            self.notify('event', (event, f'鼠标左移 {self._profile().pixels} 像素'))
        return moved

    def stop(self) -> None:
        with self._learning_lock:
            self._learning_generation += 1
            self.gate.stop()
            self._active = False
            self._cancel.set()
            self.recognizer = None
            if self.test_recorder is not None:
                saved = self.test_recorder.close()
                if saved is not None:
                    self.notify('test_saved', saved)
