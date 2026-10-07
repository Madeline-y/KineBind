# Gesture Control Implementation Plan

> Execute inline: the user approved five tickets and requested implementation. Keep the acquisition tool and firmware intact.

**Goal:** Teach a complete gesture, calibrate rejection and explicitly enable one mouse movement per recognition.

**Architecture:** Tkinter calls an application controller; recording, matching and persistence are domain services. Serial input and mouse output are replaceable boundaries. Replay and real-time use the same recognizer.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, Tkinter, pySerial, Windows ctypes.

**Spec:** ../specs/2026-10-06-gesture-control-design.md

## Global Constraints

- Complete left swing plus return is one gesture. No execution for a half gesture or twice for overlapping windows.
- Five reviewed demonstrations; manual end; 3-second countdown; approximately 30 seconds of non-target recording.
- Similar grip and initial orientation; natural speed variation.
- No additional 0.5-second end wait. Unknown motions rejected.
- Only explicit start enables execution; stop invalidates queued events and previous sessions.
- Mouse moves left 100 pixels, y unchanged, clamped at desktop bounds.
- Use installed dependencies, replay and fake mouse for automatic checks. Real hardware acceptance is separate.
- This directory has no Git branch. Preserve the checkout without initializing a repository solely for a commit.

## Task 01: Record and review

**Files:** kinebind/data.py, kinebind/storage.py, kinebind/controller.py, gesture_app.py, tests/test_gestures.py.

**Interfaces:** Motion(times, values, source), Clip(raw, start, end), Recorder.begin(now, kind), feed(t, values, now), finish(); Profile.new(name); ProfileStore.save(profile), load(id), list_profiles().

- [ ] Test countdown, full return retained, review bounds, invalid clip, save/load and failed save preserving prior data.
- [ ] Implement data services and the basic recording window; reuse existing serial parser and source.
- [ ] Run the gesture test file and mypy.

```python
recorder.begin(now=0, kind="sample")
recorder.feed(t=0, values=stationary, now=2)
assert recorder.count == 0
recorder.feed(t=1, values=stationary, now=3)
assert recorder.count == 1
```

## Task 02: Learn and calibrate

**Files:** kinebind/matching.py; controller, storage, window and tests.

**Interfaces:** learn(profile) -> Model; Model.match(motion) -> Match; Model stores preprocessing, templates, duration bounds and rejection threshold.

- [ ] Test five samples, insufficient background, positive/negative separation and model round-trip.
- [ ] Implement causal smoothing, consistent scaling, constrained multidimensional DTW and conservative calibration.
- [ ] Expose progress, background collection and threaded learning results; retain last saved model on failure.
- [ ] Run tests, mypy and matching/calibration benchmarks.

```python
model = learn(profile_with_five_examples_and_background)
assert model.match(held_out_complete_gesture).accepted
assert not model.match(non_target_motion).accepted
```

## Task 03: Replay complete gestures

**Files:** matching, kinebind/sources.py, controller, window and tests.

**Interfaces:** Recognizer.feed(t, values) -> Event | None; replay feeds the same engine as serial input.

- [ ] Test full cycle, prefix, return-only, pause, different speed, duplicates, repeated gestures and invalid time.
- [ ] Implement bounded windows, endpoint/completion checks, consumption and event logs.
- [ ] Add replay selection; replay never automatically enables actual mouse output.
- [ ] Run tests, mypy and real-recording diagnostics without inventing ground-truth boundaries.

```python
assert len(replay_stream(model, held_out_series)) == expected_cycles
assert len(replay_stream(model, prefix_only)) == 0
```

## Task 04: Live mouse control

**Files:** kinebind/actions.py, controller, sources, window and tests.

**Interfaces:** ExecutionGate.start() -> session ID; stop(); execute(event) -> bool; Mouse.move_left(pixels). Worker feeds recognizer; fake mouse records requests.

- [ ] Test startup disabled, one event once, x-100/y unchanged, edge clamp, old queued events, restart, failures and disconnect.
- [ ] Implement locked session invalidation, worker event delivery and explicit Windows execution.
- [ ] Connect actual serial source and start/stop UI.
- [ ] Run tests, mypy and withdrawn GUI smoke with fake mouse; provide real-hardware instructions.

```python
gate.stop()
assert not gate.execute(event_from_previous_session)
assert fake_mouse.requests == []
```

## Task 05: Switch and restore

**Files:** storage, controller, window, tests, README, .gitignore.

**Interfaces:** list_profiles() returns valid profiles and errors; controller.select(profile), new_profile(name), change_binding(pixels) invalidate old sessions.

- [ ] Test independent profiles, duplicate-name protection, selection invalidation, corruption isolation and restart disabled.
- [ ] Finish profile selection, binding configuration, persisted drafts and reload UX.
- [ ] Run full tests at the end, final mypy, GUI smoke and inline code review; repair defects and rerun affected checks.
- [ ] Update local ticket state with verification evidence and outstanding real-hardware acceptance.

```python
controller.select(other_profile)
assert not controller.enabled
assert not controller.execute(event_from_previous_profile)
```
