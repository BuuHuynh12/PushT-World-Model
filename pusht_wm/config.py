"""Đọc config YAML + ghi đè từ dòng lệnh dạng `--set train.steps=1000 policy.crop=84`."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    return apply_overrides(yaml.safe_load(Path(path).read_text(encoding="utf-8")), overrides)


def apply_overrides(cfg: dict, overrides: list[str] | None = None) -> dict:
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)  # tự suy kiểu: số, bool, list...
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(copy.deepcopy(cfg), allow_unicode=True, sort_keys=False),
                          encoding="utf-8")
