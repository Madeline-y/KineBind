"""Atomic, versioned per-gesture persistence with corrupt-record isolation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from uuid import UUID

from .data import Profile


class ProfileStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def _path(self, profile_id: str) -> Path:
        if str(UUID(profile_id)) != profile_id:
            raise ValueError('无效手势 ID。')
        return self.folder / (profile_id + '.json')

    def save(self, profile: Profile) -> None:
        data = profile.to_dict()  # Validate before touching a previously saved version.
        existing, _ = self.list_profiles()
        if any(p.id != profile.id and p.name.casefold() == profile.name.casefold() for p in existing):
            raise ValueError('该手势名称已存在，请换一个名称或加载原有手势。')
        self.folder.mkdir(parents=True, exist_ok=True)
        destination = self._path(profile.id)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=self.folder,
                                             prefix='saving-', suffix='.tmp', delete=False) as f:
                temp_path = Path(f.name)
                json.dump(data, f, ensure_ascii=False, allow_nan=False)
                f.flush()
                if f.tell() > 30_000_000:
                    raise ValueError('手势记录超过 30 MB，请导入较短的原始记录。')
                os.fsync(f.fileno())
            temp_path.replace(destination)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def load(self, profile_id: str) -> Profile:
        path = self._path(profile_id)
        if path.stat().st_size > 30_000_000:
            raise ValueError('手势记录过大，无法加载。')
        with path.open(encoding='utf-8-sig') as f:
            p = Profile.from_dict(json.load(f))
        if p.id != profile_id:
            raise ValueError('手势 ID 与保存记录不一致。')
        return p

    def list_profiles(self) -> tuple[list[Profile], list[str]]:
        profiles, errors = [], []
        for path in sorted(self.folder.glob('*.json')):
            try:
                profiles.append(self.load(path.stem))
            except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
                errors.append(f'{path.name}: {error}')
        return profiles, errors

    def save_draft(self, profile: Profile) -> None:
        ProfileStore(self.folder / '.drafts').save(profile)

    def load_draft(self, profile_id: str) -> Profile:
        return ProfileStore(self.folder / '.drafts').load(profile_id)
