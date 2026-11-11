from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmrouter import config as cfg


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_include_merges_then_file_wins(tmp_path: Path) -> None:
    _write(tmp_path / "base" / "embedder.yaml", "embedder: {name: hashing, params: {dim: 8}}\n")
    main = _write(
        tmp_path / "exp" / "main.yaml",
        "include: [../base/embedder.yaml]\nname: x\nembedder: {params: {dim: 16}}\n",
    )
    config = cfg.load_config(main)
    assert config == {"name": "x", "embedder": {"name": "hashing", "params": {"dim": 16}}}


def test_overrides_are_parsed_as_yaml_and_create_keys(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.yaml", "router: {params: {gamma: 10}}\n")
    config = cfg.load_config(main, ["router.params.gamma=5", "router.params.top_k=null", "a.b=x"])
    assert config["router"]["params"] == {"gamma": 5, "top_k": None}
    assert config["a"]["b"] == "x"


def test_bad_override_and_circular_include(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.yaml", "name: x\n")
    with pytest.raises(ValueError, match="key.path=value"):
        cfg.load_config(main, ["novalue"])
    _write(tmp_path / "a.yaml", "include: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "include: [a.yaml]\n")
    with pytest.raises(ValueError, match="circular include"):
        cfg.load_config(tmp_path / "a.yaml")


def test_get_with_and_without_default() -> None:
    config = {"a": {"b": 1}}
    assert cfg.get(config, "a.b") == 1
    assert cfg.get(config, "a.c", None) is None
    with pytest.raises(KeyError, match="a.c"):
        cfg.get(config, "a.c")


def test_env_placeholders_expand_only_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = _write(tmp_path / "main.yaml", 'api: {key: "${TEST_ROUTER_KEY}"}\n')
    config = cfg.load_config(main)
    cfg.save_resolved(config, tmp_path / "out" / "config.resolved.json")
    saved = json.loads((tmp_path / "out" / "config.resolved.json").read_text(encoding="utf-8"))
    assert saved["api"]["key"] == "${TEST_ROUTER_KEY}"

    monkeypatch.setenv("TEST_ROUTER_KEY", "secret")
    assert cfg.expand_env(config["api"]["key"]) == "secret"
    monkeypatch.delenv("TEST_ROUTER_KEY")
    with pytest.raises(KeyError, match="TEST_ROUTER_KEY"):
        cfg.expand_env(config["api"]["key"])
