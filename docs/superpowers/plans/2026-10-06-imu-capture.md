# IMU Capture Implementation Plan

> **For agentic workers:** Execute inline in this session, as requested by the user; no delegation is required.

**Goal:** Record and plot six-axis data from COM7 using the kinebind environment.

**Architecture:** A 50 Hz firmware emits timestamped CSV. A serial worker validates records and saves all accepted samples; the main thread draws a rolling plot. Simulation and offline tests exercise the same parser and writer.

**Tech Stack:** Python 3.11, pyserial, matplotlib, Arduino Seeed mbed, Seeed LSM6DS3.

**Spec:** ../specs/2026-10-06-imu-capture-design.md

## Global Constraints

- Default COM7/115200; target output 50 Hz; sensor refresh 104 Hz.
- ±4 g and ±500 deg/s; raw readings, no calibration or filtering.
- Preserve existing files; UTC+08:00 output names and reception timestamps.
- Stop on reset, malformed firmware, or 10 seconds without valid samples.
- No automatic firmware upload; distinguish software checks from hardware validation.

## Task 1: Protocol and collector

Files: imu_protocol.py, capture_imu.py, tests/test_capture.py, environment.yml, .gitignore, README.md.

Interfaces: parse_line(str) -> Sample | None; LineBuffer.feed(bytes) -> list[str];
SampleClock.advance(Sample) -> float; Collector.run() -> None; main() -> int.

- [x] Check parser rejection of invalid columns/NaN and old HighLevelExample; test fragmented serial lines, uint32 wrap and reset.
- [x] Implement protocol validation, serial worker, exclusive CSV creation, two plots and simulated source.
- [x] Verify CSV header, complete saved samples, protected existing files and informative serial errors.
- [x] Run `python -m unittest discover -s tests -v` and `python capture_imu.py --simulate --no-plot --duration 2`.

## Task 2: Firmware and delivery

Files: firmware/imu_stream/imu_stream.ino, README.md.

Interface: timestamp_ms,sequence,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps, one record per line.

- [x] Configure sensor ranges/refresh; turn user LEDs off; emit CSV at scheduled 20 ms intervals with no catch-up bursts.
- [x] Compile for Seeeduino:mbed:xiaonRF52840Sense using the installed Arduino CLI.
- [x] Check simulated graphical startup and clean close; report actual test results and hardware validation limit.
- [x] Provide the firmware path and COM7 launch command to the user.

Results so far: 12 automated tests pass; both console and plot simulation saved 100 samples in 2 seconds
at 50 Hz with no rejected rows or sequence gaps. Native TkAgg event-loop smoke test also passed.
The plot layout was rendered and visually inspected. Real hardware output remains unverified until
the new firmware is uploaded by the user. Firmware compilation passed: 97,848 bytes of program
storage (12%) and 46,000 bytes of global RAM (19%). No upload was performed.
