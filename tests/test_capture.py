import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_imu import CSV_COLUMNS, Collector
from imu_protocol import LegacyFirmwareError, LineBuffer, ProtocolError, SampleClock, UINT32_MAX, WIRE_COLUMNS, parse_line


def wire(timestamp=100, sequence=0, values="0.1,-0.2,1.0,10,-20,30"):
    return f"{timestamp},{sequence},{values}"


def args(**overrides):
    values = dict(port="COM7", baud=115200, duration=0, window=10, no_plot=True, simulate=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class ProtocolTests(unittest.TestCase):
    def test_complete_six_axis_sample_and_metadata(self):
        sample = parse_line(wire())
        self.assertEqual(sample.timestamp_ms, 100)
        self.assertEqual(sample.sequence, 0)
        self.assertEqual(sample.values, (0.1, -0.2, 1, 10, -20, 30))
        for metadata in ("", "# KineBind IMU v1", ",".join(WIRE_COLUMNS)):
            self.assertIsNone(parse_line(metadata))

    def test_reject_bad_values_and_field_count(self):
        for line in ("100,0,1,2", wire(values="nan,0,1,2,3,4"), wire(values="inf,0,1,2,3,4"),
                     wire(timestamp=-1), wire(sequence="1.2"), wire(timestamp=2**32),
                     wire(values="nope,0,1,2,3,4")):
            with self.subTest(line=line), self.assertRaises(ProtocolError):
                parse_line(line)

    def test_legacy_firmware_is_actionable(self):
        with self.assertRaisesRegex(LegacyFirmwareError, "Upload firmware"):
            parse_line("Accelerometer:")
        with self.assertRaisesRegex(RuntimeError, "IMU initialization failed"):
            parse_line("# ERROR: IMU initialization failed")

    def test_serial_fragments_do_not_become_partial_samples(self):
        buffer = LineBuffer()
        self.assertEqual(buffer.feed(b"100,0,0.1,-"), [])
        lines = buffer.feed(b"0.2,1,10,-20,30\r\n120,1,0,0,1,0,0,0\n140,")
        self.assertEqual(len(lines), 2)
        self.assertEqual(parse_line(lines[0]).values[1], -0.2)
        self.assertEqual(parse_line(lines[1]).timestamp_ms, 120)
        self.assertEqual(buffer.feed(b"2,0,0,1,0,0,0\n"), ["140,2,0,0,1,0,0,0"])

    def test_serial_invalid_encoding_and_long_lines(self):
        with self.assertRaises(ProtocolError):
            LineBuffer().feed(b"\xff\n")
        with self.assertRaises(ProtocolError):
            LineBuffer().feed(b"x" * 513)

    def test_clock_wrap_gap_and_device_reset(self):
        clock = SampleClock()
        self.assertEqual(clock.advance(parse_line(wire(UINT32_MAX - 9, UINT32_MAX))), 0)
        self.assertAlmostEqual(clock.advance(parse_line(wire(10, 0))), 0.020)
        self.assertAlmostEqual(clock.advance(parse_line(wire(50, 2))), 0.060)
        self.assertEqual(clock.missing_sequences, 1)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            clock.advance(parse_line(wire(0, 0)))

    def test_clock_rejects_duplicate_and_backwards_timestamp(self):
        for second in (wire(120, 0), wire(80, 1)):
            clock = SampleClock()
            clock.advance(parse_line(wire()))
            with self.assertRaises(RuntimeError):
                clock.advance(parse_line(second))


class RecordingTests(unittest.TestCase):
    def test_csv_preserves_only_complete_valid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "record.csv"
            collector = Collector(args(), target)
            records = (line for line in ["# metadata", ",".join(WIRE_COLUMNS), "broken",
                                        wire(100, 4), wire(120, 5), wire(160, 7)])
            with patch("capture_imu.serial_lines", return_value=records):
                collector.run()
            self.assertIsNone(collector.error)
            self.assertTrue(collector.finished.is_set())
            self.assertEqual(collector.count, 3)
            self.assertEqual(collector.bad_lines, 1)
            self.assertEqual(collector.clock.missing_sequences, 1)
            with target.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["elapsed_s"], "0.060")
            self.assertEqual(rows[0]["ax_g"], "0.1")
            self.assertTrue(rows[0]["received_at"].endswith("+08:00"))

    def test_does_not_overwrite_existing_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.csv"
            target.write_text("keep this", encoding="utf-8")
            collector = Collector(args(), target)
            collector.run()
            self.assertIsNotNone(collector.error)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep this")

    def test_source_closed_and_csv_flushed_after_disconnect(self):
        source_closed = []

        def disconnecting_source():
            try:
                yield wire()
                raise OSError("USB disconnected")
            finally:
                source_closed.append(True)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "disconnect.csv"
            collector = Collector(args(), target)
            with patch("capture_imu.serial_lines", return_value=disconnecting_source()):
                collector.run()
            self.assertIn("USB disconnected", collector.error)
            self.assertTrue(source_closed)
            with target.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_legacy_output_stops_instead_of_recording_garbage(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = Collector(args(), Path(directory) / "legacy.csv")
            with patch("capture_imu.serial_lines", return_value=(line for line in ["Accelerometer:"])):
                collector.run()
            self.assertEqual(collector.count, 0)
            self.assertIn("HighLevelExample", collector.error)

    def test_no_valid_data_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = Collector(args(), Path(directory) / "timeout.csv")
            with patch("capture_imu.serial_lines", return_value=(line for line in [None])), \
                 patch("capture_imu.time.monotonic", side_effect=[0, 11]):
                collector.run()
            self.assertIn("10 seconds", collector.error)


if __name__ == "__main__":
    unittest.main()
