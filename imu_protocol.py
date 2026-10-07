"""Validated serial records shared by the collector and offline checks."""

from dataclasses import dataclass
import math

AXES = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")
WIRE_COLUMNS = ("timestamp_ms", "sequence", *AXES)
UINT32_MAX = 2**32 - 1


class ProtocolError(ValueError):
    pass


class LegacyFirmwareError(ProtocolError):
    pass


@dataclass(frozen=True)
class Sample:
    timestamp_ms: int
    sequence: int
    values: tuple[float, ...]


def parse_line(line: str) -> Sample | None:
    """Skip metadata, reject malformed/legacy records, accept eight CSV fields."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("# ERROR:"):
        raise RuntimeError(line[2:])
    if line.startswith("#") or line == ",".join(WIRE_COLUMNS):
        return None
    if line in ("Accelerometer:", "Gyroscope:", "Thermometer:", "Device OK!", "Device error"):
        raise LegacyFirmwareError(
            "The board is still running HighLevelExample. Upload firmware/imu_stream/imu_stream.ino first."
        )
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 8:
        raise ProtocolError("Expected timestamp, sequence and six sensor values.")
    try:
        if not all(field.isascii() and field.isdigit() for field in fields[:2]):
            raise ValueError("Timestamp and sequence must be unsigned integers.")
        timestamp, sequence = map(int, fields[:2])
        if timestamp > UINT32_MAX or sequence > UINT32_MAX:
            raise ValueError("Timestamp/sequence exceeds uint32.")
        values = tuple(float(field) for field in fields[2:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Sensor values must be finite.")
    except ValueError as error:
        raise ProtocolError(str(error)) from error
    return Sample(timestamp, sequence, values)


class LineBuffer:
    """Frame serial chunks without treating timeout fragments as complete rows."""

    def __init__(self):
        self.pending = bytearray()

    def feed(self, chunk: bytes) -> list[str]:
        self.pending.extend(chunk)
        lines = []
        while b"\n" in self.pending:
            line, _, remainder = self.pending.partition(b"\n")
            self.pending = bytearray(remainder)
            if len(line) > 512:
                raise ProtocolError("Serial line exceeds 512 bytes.")
            try:
                lines.append(line.decode("ascii"))
            except UnicodeDecodeError as error:
                raise ProtocolError("Serial data is not ASCII; check the firmware and baud rate.") from error
        if len(self.pending) > 512:
            raise ProtocolError("No newline found within 512 bytes.")
        return lines


class SampleClock:
    """Unwrap device milliseconds; stop on reset instead of mixing sessions."""

    def __init__(self):
        self.previous: Sample | None = None
        self.elapsed_ms = 0
        self.missing_sequences = 0

    def advance(self, sample: Sample) -> float:
        if self.previous is not None:
            delta_sequence = (sample.sequence - self.previous.sequence) & UINT32_MAX
            if delta_sequence == 0 or delta_sequence > 2**31:
                raise RuntimeError("Duplicate sequence or device reset detected; start a new recording.")
            delta_ms = (sample.timestamp_ms - self.previous.timestamp_ms) & UINT32_MAX
            if delta_ms > 2**31:
                raise RuntimeError("Device timestamp moved backwards; start a new recording.")
            self.elapsed_ms += delta_ms
            self.missing_sequences += delta_sequence - 1
        self.previous = sample
        return self.elapsed_ms / 1000.0
