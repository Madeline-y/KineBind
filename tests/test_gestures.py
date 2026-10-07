from pathlib import Path
import copy
import csv
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from kinebind.data import Clip, Motion, Profile, Recorder
from kinebind.storage import ProfileStore


def gesture(duration=1.0, amplitude=1.0):
    t = np.arange(0, duration + 0.01, 0.02)
    u = t / duration
    pulse = np.sin(2 * np.pi * u) * np.sin(np.pi * u) ** 2
    x = np.zeros((len(t), 6))
    x[:, 2] = 1
    x[:, 0] = amplitude * 0.6 * pulse
    x[:, 2] += amplitude * 0.2 * np.sin(np.pi * u) ** 2
    x[:, 4] = amplitude * 180 * pulse
    return Motion(t, x, 'test gesture')


def background(duration=30.0):
    t = np.arange(0, duration + 0.01, 0.02)
    x = np.zeros((len(t), 6))
    x[:, 2] = 1
    x[:, 0] = 0.008 * np.sin(t)
    x[:, 3] = 0.6 * np.cos(t)
    return Motion(t, x, 'test non-target')


def profile_fixture():
    p = Profile.new('左挥')
    for duration, amplitude in [(0.9, .94), (1., 1.), (1.1, 1.04), (.95, .98), (1.05, 1.02)]:
        m = gesture(duration, amplitude)
        p.samples.append(Clip(m, 0, m.duration))
    p.background = background()
    return p


class EnrollmentTests(unittest.TestCase):
    def test_countdown_manual_end_and_review_keep_complete_return(self):
        recorder = Recorder()
        recorder.begin(0, 'sample')
        recorder.feed(10, (0, 0, 1, 0, 0, 0), 2.9)
        self.assertEqual(recorder.count, 0)
        m = gesture()
        for t, value in zip(m.times, m.values):
            recorder.feed(11 + t, value, 3 + t)
        recorded = recorder.finish()
        self.assertAlmostEqual(recorded.duration, m.duration)
        np.testing.assert_allclose(recorded.values, m.values)
        self.assertEqual(recorder.state, 'review')
        clip = Clip(recorded, 0, recorded.duration)
        np.testing.assert_allclose(clip.motion.values[-1], m.values[-1])

    def test_review_bounds_and_invalid_samples(self):
        m = gesture()
        with self.assertRaises(ValueError): Clip(m, .8, .4)
        with self.assertRaises(ValueError): Clip(m, -.1, .8)
        with self.assertRaises(ValueError): Motion([0, .02], [[0] * 6, [float('nan')] * 6])
        with self.assertRaises(ValueError): Motion([0, 0], [[0] * 6, [0] * 6])
        saturated = m.values.copy()
        saturated[12, 4] = 573.37
        with self.assertRaisesRegex(ValueError, '量程'): Clip(Motion(m.times, saturated), 0, m.duration).validate()

    def test_profile_storage_roundtrip_collision_and_corrupt_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            p = profile_fixture()
            store.save(p)
            loaded = store.load(p.id)
            self.assertEqual(loaded.name, '左挥')
            self.assertEqual(len(loaded.samples), 5)
            np.testing.assert_allclose(loaded.samples[0].raw.values, p.samples[0].raw.values)
            with self.assertRaisesRegex(ValueError, '名称'): store.save(Profile.new('左挥'))
            Path(folder, 'broken.json').write_text('{bad', encoding='utf-8')
            good, errors = store.list_profiles()
            self.assertEqual(len(good), 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(store.load(p.id).name, '左挥')

    def test_failed_save_preserves_previous_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            p = profile_fixture()
            store.save(p)
            before = Path(folder, p.id + '.json').read_bytes()
            p.name = ''
            with self.assertRaises(ValueError): store.save(p)
            self.assertEqual(Path(folder, p.id + '.json').read_bytes(), before)

    def test_interrupted_atomic_replace_keeps_last_usable_file(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            p = profile_fixture()
            store.save(p)
            before = Path(folder, p.id + '.json').read_bytes()
            p.pixels = 80
            with patch.object(Path, 'replace', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError): store.save(p)
            self.assertEqual(Path(folder, p.id + '.json').read_bytes(), before)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])


class LearningTests(unittest.TestCase):
    def test_five_examples_rejection_speed_and_model_roundtrip(self):
        from kinebind.matching import Model, learn
        p = profile_fixture()
        model = learn(p)
        self.assertTrue(model.match(gesture(.82, 1.03)).accepted)
        self.assertTrue(model.match(gesture(1.25, .96)).accepted)
        self.assertFalse(model.match(background(1)).accepted)
        m = gesture()
        self.assertFalse(model.match(m.section(0, .48)).accepted)
        self.assertFalse(model.match(m.section(.5, 1)).accepted)
        restored = Model.from_dict(model.to_dict())
        self.assertAlmostEqual(restored.match(m).score, model.match(m).score)

    def test_incomplete_and_conflicting_calibration_not_ready(self):
        from kinebind.matching import learn
        p = profile_fixture()
        p.samples.pop()
        with self.assertRaisesRegex(ValueError, '5'): learn(p)
        p = profile_fixture()
        p.background = background(2)
        with self.assertRaisesRegex(ValueError, '非目标'): learn(p)
        p = profile_fixture()
        m = gesture()
        times = np.arange(0, 30.02, .02)
        values = np.tile(m.values[:-1], (31, 1))[:len(times)]
        p.background = Motion(times, values)
        with self.assertRaisesRegex(ValueError, '区分'): learn(p)

    def test_learned_profile_roundtrip(self):
        from kinebind.matching import learn
        with tempfile.TemporaryDirectory() as folder:
            p = profile_fixture()
            p.model = learn(p)
            store = ProfileStore(Path(folder))
            store.save(p)
            loaded = store.load(p.id)
            self.assertIsNotNone(loaded.model)
            self.assertTrue(loaded.model.match(gesture(.85)).accepted)

    def test_demonstrations_ending_during_return_are_not_marked_ready(self):
        from kinebind.matching import learn
        p = profile_fixture()
        for clip in p.samples:
            clip.end = clip.raw.duration * .84
        with self.assertRaisesRegex(ValueError, '结束边界'): learn(p)


def join_motions(motions):
    times, values, offset = [], [], 0.
    for m in motions:
        times.extend((m.times + offset).tolist())
        values.extend(m.values.tolist())
        offset += m.duration + .02
    return Motion(times, values)


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kinebind.matching import learn
        cls.profile = profile_fixture()
        cls.model = learn(cls.profile)

    def events(self, motion):
        from kinebind.matching import Recognizer
        recognizer = Recognizer(self.model, self.profile.id, 7)
        result = []
        for t, row in zip(motion.times, motion.values):
            event = recognizer.feed(float(t), row)
            if event: result.append(event)
        return result

    def test_speed_variation_two_cycles_no_repeat_during_rest(self):
        m = join_motions([background(.5), gesture(.8), background(.4), gesture(1.3), background(1)])
        events = self.events(m)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.session == 7 and e.gesture_id == self.profile.id for e in events))
        self.assertGreaterEqual(events[0].end, .5 + .8 * .9)
        self.assertGreater(events[1].start, events[0].end)

    def test_prefix_and_return_only_rejected(self):
        m = gesture()
        for segment in [m.section(0, .48), m.section(.5, 1)]:
            series = join_motions([background(.5), segment, background(1)])
            self.assertEqual(self.events(series), [])

    def test_internal_pause_stays_in_one_complete_event(self):
        m = gesture()
        rows = np.vstack([m.values[:25], np.tile(m.values[25], (10, 1)), m.values[25:]])
        paused = Motion(np.arange(len(rows)) * .02, rows)
        series = join_motions([background(.3), paused, background(.5)])
        events = self.events(series)
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0].end, .3 + paused.duration * .9)

    def test_non_target_and_invalid_time(self):
        from kinebind.matching import Recognizer
        self.assertEqual(self.events(background(3)), [])
        r = Recognizer(self.model, self.profile.id, 1)
        r.feed(1, (0, 0, 1, 0, 0, 0))
        with self.assertRaises(ValueError): r.feed(.9, (0, 0, 1, 0, 0, 0))


class ExecutionTests(unittest.TestCase):
    def test_explicit_start_one_event_once_and_edge(self):
        from kinebind.actions import ExecutionGate, FakeMouse
        from kinebind.matching import Event
        mouse = FakeMouse(250, 300)
        gate = ExecutionGate(mouse)
        self.assertFalse(gate.execute(Event(0, 'left', 'before', 0, 1, .01)))
        session = gate.start('left', 100)
        event = Event(session, 'left', 'first', 0, 1, .01)
        self.assertTrue(gate.execute(event))
        self.assertEqual(mouse.position, (150, 300))
        self.assertFalse(gate.execute(event))
        self.assertFalse(gate.execute(Event(session, 'left', 'overlap', .5, 1.1, .01)))
        self.assertTrue(gate.execute(Event(session, 'left', 'next', 1.2, 2, .01)))
        self.assertTrue(gate.execute(Event(session, 'left', 'edge', 2.2, 3, .01)))
        self.assertEqual(mouse.position, (0, 300))
        self.assertEqual(mouse.requests, [100, 100, 100])

    def test_stop_invalidates_queued_event_and_restart(self):
        from kinebind.actions import ExecutionGate, FakeMouse
        from kinebind.matching import Event
        mouse = FakeMouse()
        gate = ExecutionGate(mouse)
        session = gate.start('left')
        queued = Event(session, 'left', 'queued', 0, 1, .01)
        gate.stop()
        self.assertFalse(gate.execute(queued))
        gate.start('left')
        self.assertFalse(gate.execute(queued))
        self.assertEqual(mouse.requests, [])

    def test_failure_is_not_retried_and_disables_execution(self):
        from kinebind.actions import ExecutionGate, FakeMouse
        from kinebind.matching import Event
        mouse = FakeMouse(fail=True)
        gate = ExecutionGate(mouse)
        session = gate.start('left')
        event = Event(session, 'left', 'failed', 0, 1, .01)
        with self.assertRaises(OSError): gate.execute(event)
        self.assertFalse(gate.enabled)
        self.assertFalse(gate.execute(event))
        self.assertEqual(len(mouse.requests), 1)


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kinebind.matching import learn
        cls.profile = profile_fixture()
        cls.profile.model = learn(cls.profile)

    def test_application_start_stop_switch_and_reload_disabled(self):
        from kinebind.actions import FakeMouse
        from kinebind.controller import GestureController
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            store.save(self.profile)
            mouse = FakeMouse()
            app = GestureController(store, mouse, threaded=False)
            app.select(store.load(self.profile.id))
            self.assertFalse(app.enabled)
            app.set_connected(True)
            series = join_motions([background(.3), gesture(.9), background(.3)])
            for t, row in zip(series.times, series.values): app.on_sample(float(t), row, float(t))
            self.assertEqual(mouse.requests, [])
            app.start_recognition()
            for t, row in zip(series.times, series.values): app.on_sample(float(t), row, float(t))
            self.assertEqual(mouse.requests, [100])
            app.stop()
            for t, row in zip(series.times, series.values): app.on_sample(float(t), row, float(t))
            self.assertEqual(mouse.requests, [100])
            app.start_recognition()
            app.new_profile('第二个手势')
            self.assertFalse(app.enabled)
            self.assertIsNone(app.profile.model)
            self.assertIsNotNone(store.load(self.profile.id).model)
            restarted = GestureController(store, FakeMouse(), threaded=False)
            restarted.select(store.load(self.profile.id))
            self.assertFalse(restarted.enabled)

    def test_enrollment_does_not_execute_and_failed_learning_keeps_saved_model(self):
        from kinebind.actions import FakeMouse
        from kinebind.controller import GestureController
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            store.save(self.profile)
            mouse = FakeMouse()
            app = GestureController(store, mouse, threaded=False)
            app.select(store.load(self.profile.id))
            app.set_connected(True)
            app.start_recognition()
            app.begin_recording(0, 'sample')
            m = gesture()
            for t, row in zip(m.times, m.values): app.on_sample(float(t), row, 3 + float(t))
            app.finish_recording()
            app.save_review(0, m.duration, slot=0)
            app.profile.background = background(1)
            with self.assertRaises(ValueError): app.learn_current()
            self.assertIsNotNone(store.load(self.profile.id).model)
            self.assertEqual(mouse.requests, [])


    def test_bindings_models_and_old_sessions_are_independent(self):
        from kinebind.actions import FakeMouse
        from kinebind.controller import GestureController
        from kinebind.matching import Event, learn
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            first = copy.deepcopy(self.profile)
            second = profile_fixture()
            second.name, second.pixels = '另一种动作', 75
            for clip in second.samples:
                clip.raw.values[:, [0, 1]] = clip.raw.values[:, [1, 0]]
                clip.raw.values[:, [3, 4]] = clip.raw.values[:, [4, 3]]
            second.model = learn(second)
            store.save(first)
            store.save(second)
            mouse = FakeMouse()
            app = GestureController(store, mouse, threaded=False)
            app.select(first)
            app.set_connected(True)
            app.start_recognition()
            old_session = app.recognizer.session
            app.select(second)
            self.assertFalse(app.execute(Event(old_session, first.id, 'old', 0, 1, .01)))
            app.change_binding(60)
            app.start_recognition()
            app.execute(Event(app.recognizer.session, second.id, 'new', 0, 1, .01))
            self.assertEqual(mouse.requests, [60])
            self.assertEqual(store.load(first.id).pixels, 100)
            self.assertEqual(store.load(second.id).pixels, 60)
            self.assertFalse(second.model.match(gesture()).accepted)
            original = Path(folder, second.id + '.json').read_bytes()
            app.import_review(gesture())
            app.save_review(0, 1, slot=0)
            with self.assertRaisesRegex(ValueError, '修改'): app.change_binding(70)
            self.assertEqual(Path(folder, second.id + '.json').read_bytes(), original)

    def test_disconnect_restart_range_and_background_auto_finish(self):
        from kinebind.actions import FakeMouse
        from kinebind.controller import GestureController
        with tempfile.TemporaryDirectory() as folder:
            app = GestureController(ProfileStore(Path(folder)), FakeMouse(), threaded=False)
            app.select(self.profile)
            app.set_connected(True)
            app.start_recognition()
            app.set_connected(False)
            app.set_connected(True)
            self.assertFalse(app.enabled)
            app.set_ranges(4, 1000)
            with self.assertRaisesRegex(ValueError, '量程'): app.start_recognition()
            app.set_ranges(4, 500)
            app.begin_recording(0, 'background')
            m = background(30)
            for t, row in zip(m.times, m.values): app.on_sample(float(t), row, 3 + float(t))
            self.assertEqual(app.recorder.state, 'review')
            self.assertAlmostEqual(app.profile.background.duration, 30)
            self.assertFalse(app.enabled)

    def test_replay_never_executes_mouse_and_stale_queue_stops_live(self):
        from kinebind.actions import FakeMouse
        from kinebind.controller import GestureController
        with tempfile.TemporaryDirectory() as folder:
            mouse = FakeMouse()
            app = GestureController(ProfileStore(Path(folder)), mouse, threaded=False)
            app.select(self.profile)
            app.start_recognition(replay=True)
            app.replay_motion(join_motions([background(.3), gesture(.85), background(.3)]))
            results = list(app.results.queue)
            self.assertEqual(sum(kind == 'event' for kind, _ in results), 1)
            self.assertEqual(mouse.requests, [])
            app.set_connected(True)
            app.start_recognition()
            app._inputs.put((time.monotonic() - 1, 0, [0, 0, 1, 0, 0, 0]))
            app._run_recognition(app.recognizer, app._inputs, app._cancel)
            self.assertFalse(app.enabled)
            self.assertEqual(mouse.requests, [])


class SourceTests(unittest.TestCase):
    def test_real_format_csv_and_gap_rejection(self):
        from imu_protocol import WIRE_COLUMNS
        from kinebind.sources import read_csv
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'samples.csv')
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(WIRE_COLUMNS)
                writer.writerow([1000, 0, 0, 0, 1, 0, 0, 0])
                writer.writerow([1020, 1, 0, 0, 1, 0, 0, 0])
            self.assertAlmostEqual(read_csv(path).duration, .02)
            path.write_text(path.read_text().replace('1020,1,', '1020,3,'))
            with self.assertRaisesRegex(ValueError, '序号'): read_csv(path)
            path.write_text('elapsed_s,ax\n0,1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '六轴'): read_csv(path)

    def test_serial_protocol_reset_bad_metadata_and_legacy_are_visible(self):
        from kinebind.sources import SerialFeed
        cases = [
            (['# accel_range_g=4 gyro_range_dps=500', '1000,0,0,0,1,0,0,0', '1020,1,0,0,1,0,0,0'], False),
            (['1000,0,0,0,1,0,0,0', '20,0,0,0,1,0,0,0'], True),
            (['# accel_range_g=99 gyro_range_dps=500', '1000,0,0,0,1,0,0,0'], True),
            (['Accelerometer:'], True),
        ]
        for lines, fails in cases:
            def source(*args):
                yield from lines
            with self.subTest(lines=lines), patch('kinebind.sources.serial_lines', side_effect=source):
                feed = SerialFeed('FAKE')
                feed._run()
                results = list(feed.messages.queue)
                self.assertEqual(any(k == 'error' for k, _ in results), fails)
                self.assertEqual(results[-1][0], 'closed')
                if not fails:
                    self.assertEqual(sum(k == 'connected' for k, _ in results), 1)
                    self.assertEqual(sum(k == 'point' for k, _ in results), 2)

    def test_corrupt_schema_is_isolated_from_usable_profile(self):
        from kinebind.matching import learn
        with tempfile.TemporaryDirectory() as folder:
            store = ProfileStore(Path(folder))
            good = profile_fixture()
            good.model = learn(good)
            store.save(good)
            for value in (None, [], {'version': 300}):
                broken = Profile.new('broken')
                Path(folder, broken.id + '.json').write_text(json.dumps(value), encoding='utf-8')
            profiles, errors = store.list_profiles()
            self.assertEqual(len(profiles), 1)
            self.assertEqual(len(errors), 3)
            data = good.to_dict()
            data['model']['gyro_range'] = 1000
            with self.assertRaisesRegex(ValueError, '量程'): Profile.from_dict(data)


class StopRaceTests(unittest.TestCase):
    def test_stop_waits_for_inflight_action_then_rejects_every_queued_event(self):
        from kinebind.actions import ExecutionGate
        from kinebind.matching import Event
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        class BlockingMouse:
            calls = 0
            def move_left(self, pixels):
                self.calls += 1
                entered.set()
                if not release.wait(2): raise TimeoutError('test action was not released')
                return 0, 0
        mouse = BlockingMouse()
        gate = ExecutionGate(mouse)
        session = gate.start('left')
        worker = threading.Thread(target=gate.execute, args=(Event(session, 'left', 'one', 0, 1, .01),))
        worker.start()
        self.assertTrue(entered.wait(2))
        def stop():
            gate.stop()
            stopped.set()
        stopper = threading.Thread(target=stop)
        stopper.start()
        self.assertFalse(stopped.wait(.02))
        release.set()
        worker.join(2)
        stopper.join(2)
        self.assertTrue(stopped.is_set())
        self.assertFalse(gate.execute(Event(session, 'left', 'queued', 1.1, 2, .01)))
        self.assertEqual(mouse.calls, 1)


if __name__ == '__main__':
    unittest.main()
