from pathlib import Path
import csv
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from imu_protocol import Sample
from kinebind.actions import FakeMouse
from kinebind.controller import GestureController
from kinebind.data import Clip, Profile
from kinebind.matching import learn
from kinebind.sources import SerialFeed, read_csv
from kinebind.storage import ProfileStore
from kinebind.synthetic import gesture, calibration_records
from kinebind.live_recording import LiveTestRecorder


class LiveRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = Profile.new('实时测试')
        for duration in (1., .9, 1.1, .95, 1.05):
            motion = gesture(duration)
            cls.profile.samples.append(Clip(motion, 0., motion.duration))
        cls.profile.calibration = calibration_records()
        cls.profile.model = learn(cls.profile, use_new=True, single_accel=True)

    def app(self, folder):
        app = GestureController(ProfileStore(Path(folder) / 'profiles'), FakeMouse(),
                                threaded=False, test_folder=Path(folder) / 'recordings')
        app.select(self.profile)
        app.set_connected(True)
        return app

    def feed(self, app, count, first_sequence=1234, first_timestamp=567800):
        for i in range(count):
            values = (0., 0., 1., 0., 0., 0.)
            sample = Sample(first_timestamp + i * 20, first_sequence + i, values)
            app.on_sample(100. + i * .02, values, 100. + i * .02, sample=sample)

    def test_complete_session_is_saved_beyond_plot_buffer_and_can_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            app.start_recognition()
            self.feed(app, 801)
            app.stop()
            path, = (Path(folder) / 'recordings').glob('*.csv')
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 801)
            self.assertEqual(int(rows[0]['timestamp_ms']), 567800)
            self.assertEqual(int(rows[-1]['sequence']), 2034)
            self.assertAlmostEqual(float(rows[-1]['elapsed_s']), 16.)
            self.assertEqual(rows[0]['gesture_id'], self.profile.id)
            motion = read_csv(path)
            self.assertAlmostEqual(motion.duration, 16.)
            np.testing.assert_array_equal(motion.values, np.tile([0., 0., 1., 0., 0., 0.], (801, 1)))

    def test_restart_disconnect_and_replay_keep_sessions_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            app.start_recognition()
            self.feed(app, 10)
            app.set_connected(False)
            first, = (Path(folder) / 'recordings').glob('*.csv')
            first_bytes = first.read_bytes()
            self.feed(app, 5)
            self.assertEqual(first.read_bytes(), first_bytes)
            app.set_connected(True)
            app.start_recognition()
            self.feed(app, 12)
            app.stop()
            self.assertEqual(len(list((Path(folder) / 'recordings').glob('*.csv'))), 2)
            app.start_recognition(replay=True)
            app.replay_motion(gesture())
            app.stop()
            self.assertEqual(len(list((Path(folder) / 'recordings').glob('*.csv'))), 2)

    def test_save_failure_stops_control_before_sample_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            app.start_recognition()
            with patch.object(app.test_recorder, 'write', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    self.feed(app, 1)
            self.assertFalse(app.active)
            self.assertFalse(app.enabled)
            self.assertEqual(app.gate.mouse.requests, [])

    def test_serial_points_preserve_device_metadata(self):
        feed = SerialFeed('TEST')
        lines = (line for line in ['456000,123,0,0,1,0,0,0', '456020,124,0,0,1,0,0,0'])
        with patch('kinebind.sources.serial_lines', return_value=lines):
            feed._run()
        points = []
        while not feed.messages.empty():
            kind, value = feed.messages.get_nowait()
            if kind == 'point':
                points.append(value)
        self.assertEqual(points[0].sample.timestamp_ms, 456000)
        self.assertEqual(points[1].sample.sequence, 124)

    def test_nondefault_ranges_and_device_counter_wrap_survive_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = copy.deepcopy(self.profile)
            profile.model.accel_range, profile.model.gyro_range = 8, 1000
            recorder = LiveTestRecorder(Path(folder), 'SIMULATED range check')
            path = recorder.start(profile)
            recorder.write(500., Sample(4294967290, 4294967295, (0., 0., 1., 0., 0., 0.)))
            recorder.write(500.02, Sample(14, 0, (0., 0., 1., 0., 0., 0.)))
            recorder.close()
            motion = read_csv(path)
            self.assertEqual((motion.accel_range, motion.gyro_range), (8, 1000))
            self.assertAlmostEqual(motion.duration, .02)


if __name__ == '__main__':
    unittest.main()
