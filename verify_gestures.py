"""Verify frozen gesture models with independent annotated CSV recordings; FakeMouse only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kinebind.acceptance import evaluate
from kinebind.data import Profile
from kinebind.sources import read_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = dict(status='incomplete', real_verified=False, mouse='simulated')
    try:
        body = args.manifest.read_bytes()
        manifest = json.loads(body.decode('utf-8-sig'))
        if not isinstance(manifest, dict):
            raise ValueError('验收清单需要 JSON 对象。')
        if manifest.get('version') != 1:
            raise ValueError('不支持的验收清单版本。')
        root = args.manifest.resolve().parent
        profile_file = (root / manifest['profile']).resolve()
        profile_bytes = profile_file.read_bytes()
        profile = Profile.from_dict(json.loads(profile_bytes.decode('utf-8-sig')))
        target_spec = manifest.get('target')
        if target_spec is not None and not isinstance(target_spec, dict):
            raise ValueError('target 需要 csv 与 intervals 对象。')
        target = read_csv(root / target_spec['csv']) if target_spec else None
        raw_intervals = target_spec.get('intervals', []) if target_spec else []
        if not isinstance(raw_intervals, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in raw_intervals):
            raise ValueError('intervals 中每个区间需要 [起点秒数, 终点秒数]。')
        intervals = [(float(pair[0]), float(pair[1])) for pair in raw_intervals]
        negatives = {k: read_csv(root / path) for k, path in manifest.get('non_target', {}).items() if path}
        report = evaluate(profile, target, intervals, negatives, manifest.get('independent') is True)
        report['manifest_sha256'] = hashlib.sha256(body).hexdigest()
        report['profile_file_sha256'] = hashlib.sha256(profile_bytes).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(status='incomplete', real_verified=False, mouse='simulated', reason=str(error))
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        try:
            with args.output.open('x', encoding='utf-8') as file:
                file.write(text + '\n')
        except OSError as error:
            print(f'报告未保存（已有文件不会覆盖）：{error}')
            return 2
    return 0 if report['status'] == 'passed' else (1 if report['status'] in ('failed', 'simulated_fail') else 2)


if __name__ == '__main__':
    raise SystemExit(main())
