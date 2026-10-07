"""Application behavior tests; synthetic data never represents real acceptance."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kinebind.actions import FakeMouse
from kinebind.controller import GestureController
from kinebind.data import Clip, Motion, Profile
from kinebind.storage import ProfileStore


def still(duration=10.):
    t = np.arange(0, duration + .01, .02)
    x = np.zeros((len(t), 6))
    x[:, 2] = 1.
    x[:, 0] = .001 * np.sin(t * 2)
    x[:, 3] = .1 * np.cos(t)
    return Motion(t, x, 'SIMULATED still')


def target(axis=4, duration=1., amplitude=1.):
    t = np.arange(0, duration + .01, .02)
    u = t / duration
    pulse = np.sin(2 * np.pi * u) * np.sin(np.pi * u) ** 2
    x = np.zeros((len(t), 6))
    x[:, 2] = 1.
    x[:, axis] += (180 if axis >= 3 else .6) * amplitude * pulse
    return Motion(t, x, 'SIMULATED target')


def join(motions):
    t, x, offset = [], [], 0.
    for m in motions:
        t.extend((m.times + offset).tolist())
        x.extend(m.values.tolist())
        offset += m.duration + .02
    return Motion(t, x, 'SIMULATED joined')


def negatives():
    a, b = still(), still()
    a.values[:, 3] += 12 * np.sin(a.times * 2.3)
    b.values[:, 2] += .15 * np.sin(b.times * .8)
    b.values[:, 5] += 12 * np.sin(b.times * .7)
    return {'static': still(), 'shake': a, 'pick_place': b}


def examples(axis=4):
    p = Profile.new('个人动作')
    for duration, amplitude in [(1., 1.), (.9, .95), (1.1, 1.04), (.96, .98), (1.04, 1.02)]:
        m = target(axis, duration, amplitude)
        p.samples.append(Clip(m, 0., m.duration))
    return p


class EnrollmentBehavior(unittest.TestCase):
    def test_categories_replace_independently_and_survive_reload(self):
        with tempfile.TemporaryDirectory() as d:
            store = ProfileStore(Path(d))
            app = GestureController(store, FakeMouse(), threaded=False)
            app.new_profile('分类录入')
            for key, m in negatives().items():
                app.set_calibration(key, m)
            app.set_calibration('static', still(11.))
            p = store.load(app.profile.id)
            self.assertEqual(set(p.calibration), {'static', 'shake', 'pick_place'})
            self.assertAlmostEqual(p.calibration['static'].duration, 11.)
            self.assertAlmostEqual(p.calibration['shake'].duration, 10.)
            self.assertIsNone(p.model)

    def test_motion_category_rejects_still_with_specific_report(self):
        with tempfile.TemporaryDirectory() as d:
            app = GestureController(ProfileStore(Path(d)), FakeMouse(), threaded=False)
            app.new_profile('质量检查')
            app.set_calibration('static', still())
            with self.assertRaisesRegex(ValueError, '普通晃动'):
                app.set_calibration('shake', still())
            self.assertEqual(app.profile.report['category'], 'shake')
            self.assertNotIn('shake', app.profile.calibration)

    def test_category_recording_auto_finishes_at_ten_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            mouse = FakeMouse()
            app = GestureController(ProfileStore(Path(d)), mouse, threaded=False)
            app.new_profile('定时录制')
            app.set_connected(True)
            app.begin_recording(0., 'static')
            m = still(10.2)
            for t, row in zip(m.times, m.values):
                app.on_sample(float(t), row, float(t) + 3.)
            self.assertEqual(app.recorder.state, 'review')
            self.assertIn('static', app.profile.calibration)
            self.assertEqual(mouse.requests, [])

    def test_suggested_range_requires_confirmation_and_keeps_internal_pause(self):
        with tempfile.TemporaryDirectory() as d:
            app = GestureController(ProfileStore(Path(d)), FakeMouse(), threaded=False)
            app.new_profile('范围建议')
            g = target()
            rows = np.vstack([g.values[:25], np.tile(g.values[25], (12, 1)), g.values[25:]])
            paused = Motion(np.arange(len(rows)) * .02, rows, 'SIMULATED paused')
            recording = join([still(.8), paused, still(1.2)])
            app.import_review(recording)
            self.assertEqual(app.profile.samples, [])
            bounds = app.suggestion
            self.assertLess(bounds['start'], 1.)
            self.assertGreater(bounds['end'], .8 + paused.duration * .9)
            self.assertLess(bounds['end'], recording.duration - .5)
            app.save_review(bounds['start'], bounds['end'])
            p = app.store.load(app.profile.id)
            self.assertAlmostEqual(p.samples[0].raw.duration, recording.duration)
            self.assertEqual(len(p.samples), 1)


class RecognitionBehavior(unittest.TestCase):
    def test_single_acceleration_learning_reload_and_execution(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d, axis=0)
            app.learn_current(use_new=True, single_accel=True)
            loaded = app.store.load(app.profile.id)
            self.assertEqual(loaded.model.axes, (0,))
            self.assertEqual(loaded.report['learning_mode'], 'single_accel')
            app.select(loaded)
            app.set_connected(True)
            for motion in loaded.calibration.values():
                app.start_recognition()
                for t, row in zip(motion.times, motion.values):
                    app.on_sample(float(t), row, float(t))
                app.stop()
            self.assertEqual(app.gate.mouse.requests, [])
            app.start_recognition()
            series = join([still(.4), target(0, .85), still(.4), target(0, 1.2), still(.6)])
            for t, row in zip(series.times, series.values):
                app.on_sample(float(t), row, float(t))
            self.assertEqual(app.gate.mouse.requests, [100, 100])
            app.stop()

    def test_single_acceleration_never_falls_back_to_gyro(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d, axis=4)
            with self.assertRaisesRegex(ValueError, '加速度'):
                app.learn_current(use_new=True, single_accel=True)
            self.assertIsNone(app.profile.model)

    def test_single_acceleration_rejects_a_real_streaming_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d, axis=0)
            app.set_calibration('shake', join([still(2.), target(0), still(7.)]))
            with self.assertRaises(ValueError):
                app.learn_current(use_new=True, single_accel=True)
            self.assertEqual(app.profile.report['category'], 'shake')
            self.assertIsNone(app.profile.model)

    def app_with_examples(self, folder, axis=4):
        app = GestureController(ProfileStore(Path(folder)), FakeMouse(), threaded=False)
        app.select(examples(axis))
        for key, motion in negatives().items():
            app.set_calibration(key, motion)
        return app

    def test_new_learning_reloads_and_executes_two_complete_cycles_once(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.learn_current(use_new=True)
            loaded = app.store.load(app.profile.id)
            self.assertEqual(loaded.model.to_dict()['version'], 2)
            self.assertEqual(loaded.report['status'], 'learned')
            app.select(loaded)
            app.set_connected(True)
            app.start_recognition()
            series = join([still(.4), target(4, .85), still(.4), target(4, 1.2), still(.6)])
            for t, row in zip(series.times, series.values):
                app.on_sample(float(t), row, float(t))
            self.assertEqual(app.gate.mouse.requests, [100, 100])
            app.stop()
            for t, row in zip(series.times, series.values):
                app.on_sample(float(t), row, float(t))
            self.assertEqual(app.gate.mouse.requests, [100, 100])

    def test_translation_and_another_gyro_axis_use_same_learning_flow(self):
        for axis in (0, 5):
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as d:
                app = self.app_with_examples(d, axis)
                app.learn_current(use_new=True)
                self.assertTrue(app.profile.model.match(target(axis, .95)).accepted)
                self.assertFalse(app.profile.model.match(still(1.)).accepted)
                other = target(4 if axis != 4 else 5)
                self.assertFalse(app.profile.model.match(other).accepted)

    def test_backgrounds_and_half_gestures_produce_no_execution(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.learn_current(use_new=True)
            app.set_connected(True)
            g = target()
            for motion in [*negatives().values(), g.section(0, .48), g.section(.5, 1.)]:
                app.start_recognition()
                series = join([still(.3), motion, still(.6)])
                for t, row in zip(series.times, series.values):
                    app.on_sample(float(t), row, float(t))
                app.stop()
            self.assertEqual(app.gate.mouse.requests, [])

    def test_incomplete_calibration_reports_category_and_preserves_old_model(self):
        from kinebind.matching import learn
        from kinebind.synthetic import background
        with tempfile.TemporaryDirectory() as d:
            p = examples()
            p.background = background()
            p.model = learn(p)
            store = ProfileStore(Path(d))
            store.save(p)
            original = (Path(d) / (p.id + '.json')).read_bytes()
            app = GestureController(store, FakeMouse(), threaded=False)
            app.select(p)
            app.set_calibration('static', still())
            with self.assertRaisesRegex(ValueError, '普通晃动'):
                app.learn_current(use_new=True)
            self.assertEqual((Path(d) / (p.id + '.json')).read_bytes(), original)
            self.assertEqual(store.load_draft(p.id).report['category'], 'shake')

    def test_same_pose_and_different_pose_static_are_both_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.learn_current(use_new=True)
            for pose in [(0, 0, 1), (1, 0, 0)]:
                m = still(1.)
                m.values[:, :3] = pose
                self.assertFalse(app.profile.model.match(m).accepted)

    def test_internal_pause_does_not_end_or_duplicate_the_gesture(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.learn_current(use_new=True)
            g = target()
            rows = np.vstack([g.values[:25], np.tile(g.values[25], (10, 1)), g.values[25:]])
            paused = Motion(np.arange(len(rows)) * .02, rows)
            app.set_connected(True)
            app.start_recognition()
            series = join([still(.3), paused, still(.6)])
            for t, row in zip(series.times, series.values):
                app.on_sample(float(t), row, float(t))
            events = []
            while not app.results.empty():
                kind, value = app.results.get_nowait()
                if kind == 'event':
                    events.append(value[0])
            self.assertEqual(app.gate.mouse.requests, [100])
            self.assertGreater(events[0].end, .3 + paused.duration * .9)

    def test_stop_during_learning_cannot_save_late_results(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            before = (Path(d) / (app.profile.id + '.json')).read_bytes()
            def cancelled(profile, progress, use_new=False):
                progress('第一步')
                app.stop()
                progress('停止后的结果')
            with patch('kinebind.controller.learn', side_effect=cancelled):
                with self.assertRaisesRegex(ValueError, '停止'):
                    app.learn_current(use_new=True)
            self.assertEqual((Path(d) / (app.profile.id + '.json')).read_bytes(), before)
            self.assertIsNone(app.profile.model)

    def test_conflicting_background_reports_the_class_and_time_range(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.set_calibration('shake', join([target() for _ in range(10)]))
            with self.assertRaisesRegex(ValueError, '普通晃动'):
                app.learn_current(use_new=True)
            report = app.store.load_draft(app.profile.id).report
            self.assertEqual(report['category'], 'shake')
            self.assertEqual(len(report['interval']), 2)
            self.assertIsNone(app.profile.model)

    def test_malformed_record_is_isolated_and_changed_training_cannot_load(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.app_with_examples(d)
            app.learn_current(use_new=True)
            p = app.profile.to_dict()
            p['calibration'] = []
            Path(d, 'broken.json').write_text(json.dumps(p), encoding='utf-8')
            good, errors = app.store.list_profiles()
            self.assertEqual(len(good), 1)
            self.assertEqual(len(errors), 1)
            p = app.profile.to_dict()
            p['samples'][0]['raw']['values'][10][4] += .1
            with self.assertRaisesRegex(ValueError, '训练数据'):
                Profile.from_dict(p)

    def test_multiple_axes_can_distinguish_two_single_axis_impostors(self):
        with tempfile.TemporaryDirectory() as d:
            p = examples()
            for c in p.samples:
                c.raw.values[:, 3] = c.raw.values[:, 4] * .5
                c.raw.values[:, 4] *= .5
            a, b = target(3), target(4)
            a.values[:, 3] *= .5
            b.values[:, 4] *= .5
            app = GestureController(ProfileStore(Path(d)), FakeMouse(), threaded=False)
            app.select(p)
            for k, m in {'static': still(), 'shake': join([a] * 10), 'pick_place': join([b] * 10)}.items():
                app.set_calibration(k, m)
            app.learn_current(use_new=True)
            self.assertTrue(app.profile.model.match(p.samples[0].motion).accepted)
            self.assertFalse(app.profile.model.match(a).accepted)
            self.assertFalse(app.profile.model.match(b).accepted)
            loaded = app.store.load(p.id)
            self.assertTrue(loaded.model.match(p.samples[1].motion).accepted)

    def test_single_direction_motion_is_not_forced_to_have_return_phase(self):
        with tempfile.TemporaryDirectory() as d:
            p = examples()
            p.name = '单方向转动'
            for c in p.samples:
                c.raw.values[:, 4] = 180 * np.sin(np.pi * c.raw.times / c.raw.duration) ** 2
            app = GestureController(ProfileStore(Path(d)), FakeMouse(), threaded=False)
            app.select(p)
            for k, m in negatives().items():
                app.set_calibration(k, m)
            app.learn_current(use_new=True)
            self.assertTrue(app.profile.model.match(p.samples[0].motion).accepted)


class AcceptanceBehavior(unittest.TestCase):
    def test_nine_correct_plus_one_missing_is_not_total_event_count(self):
        from kinebind.acceptance import score_events
        from kinebind.matching import Event
        intervals = [(i * 2., i * 2. + 1.) for i in range(10)]
        events = [Event(1, 'g', str(i), start, end, .01) for i, (start, end) in enumerate(intervals[:9])]
        result = score_events(events, intervals)
        self.assertEqual(result['correct'], 9)
        self.assertEqual(result['missed'], 1)
        events.append(Event(1, 'g', 'duplicate', 0., 1.02, .01))
        result = score_events(events, intervals)
        self.assertEqual(result['correct'], 8)
        self.assertEqual(result['duplicate'], 1)

    def test_early_completion_fails_even_with_ten_events(self):
        from kinebind.acceptance import score_events
        from kinebind.matching import Event
        intervals = [(i * 2., i * 2. + 1.) for i in range(10)]
        events = [Event(1, 'g', str(i), start, end - .3, .01) for i, (start, end) in enumerate(intervals)]
        result = score_events(events, intervals)
        self.assertEqual(result['correct'], 0)
        self.assertEqual(result['early'], 10)

    def test_reused_training_clip_in_longer_capture_is_detected(self):
        from kinebind.acceptance import contains_training
        g = target()
        combined = join([still(.3), g, still(.6)])
        self.assertTrue(contains_training(combined, g))
        self.assertFalse(contains_training(combined, target(4, 1.1, 1.03)))

    def test_missing_data_cannot_be_reported_as_verified(self):
        from kinebind.acceptance import evaluate
        with tempfile.TemporaryDirectory() as d:
            app = RecognitionBehavior().app_with_examples(d)
            app.learn_current(use_new=True)
            report = evaluate(app.profile, None, [], {}, True)
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(report['real_verified'])

    def test_full_simulated_evaluation_never_claims_real_acceptance(self):
        from kinebind.acceptance import evaluate
        with tempfile.TemporaryDirectory() as d:
            app = RecognitionBehavior().app_with_examples(d)
            app.learn_current(use_new=True)
            motions, intervals, offset = [still(.5)], [], .52
            for i in range(10):
                g = target(4, .82 + i * .04, 1.01)
                intervals.append((offset, offset + g.duration))
                motions.extend([g, still(.3)])
                offset += g.duration + .02 + .3 + .02
            bg = {key: still(30.) for key in ('static', 'shake', 'pick_place')}
            for m in bg.values():
                m.values[:, 0] += .0003 * np.sin(m.times * .13 + .4)
            bg['shake'].values[:, 3] += 13 * np.sin(bg['shake'].times * 1.71)
            bg['pick_place'].values[:, 2] += .14 * np.sin(bg['pick_place'].times * .83)
            report = evaluate(app.profile, join(motions), intervals, bg, True)
            self.assertEqual(report['status'], 'simulated_pass', report)
            self.assertFalse(report['real_verified'])
            self.assertGreaterEqual(report['target']['correct'], 9)
            self.assertTrue(all(v['events'] == 0 for v in report['non_target'].values()))


if __name__ == '__main__':
    unittest.main()
