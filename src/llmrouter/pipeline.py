"""Experiment steps behind the CLI: toy data, info, fit, calibrate, evaluate, route, report.

Everything here is router-agnostic: it only talks to routers through the ``Router``
interface, and every path, model name and hyper-parameter comes from the config.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import llmrouter.embeddings  # noqa: F401 - registers embedders
import llmrouter.routers  # noqa: F401 - registers routers
from llmrouter import config as cfg
from llmrouter.data import tiers as tiers_mod
from llmrouter.data.contamination import find_contaminated
from llmrouter.data.download import download_files
from llmrouter.data.filters import filter_battles
from llmrouter.data.io import load_battles, load_benchmark
from llmrouter.data.prepare import convert_all
from llmrouter.data.split import split_calibration
from llmrouter.data.toy import make_toy_battles, make_toy_benchmark
from llmrouter.embeddings import CachingEmbedder, Embedder
from llmrouter.evaluation import runner
from llmrouter.evaluation.calibrate import calibrate as calibrate_thresholds
from llmrouter.llm.cost import ModelPrice, routed_cost_usd
from llmrouter.registry import EMBEDDERS, ROUTERS
from llmrouter.routers.base import Router

log = logging.getLogger(__name__)

Config = dict[str, Any]


@dataclass
class PreparedData:
    fit: pd.DataFrame
    calibration: pd.DataFrame
    ratings: dict[str, float]
    tiers: dict[str, int]
    strong_tier: int
    weak_tier: int


# ----------------------------------------------------------------------- builders
def experiment_dir(config: Config) -> Path:
    return Path(cfg.get(config, "output_dir", "experiments")) / str(config["name"])


def model_pair(config: Config) -> tuple[str, str]:
    return str(cfg.get(config, "models.strong.name")), str(cfg.get(config, "models.weak.name"))


def build_embedder(config: Config) -> Embedder:
    """The configured embedder, wrapped so every call goes through the on-disk cache."""
    inner = EMBEDDERS.create(
        cfg.get(config, "embedder.name"), **(cfg.get(config, "embedder.params", None) or {})
    )
    return CachingEmbedder(
        inner,
        cfg.get(config, "embedder.cache_dir", None),
        cfg.get(config, "embedder.dtype", "float32"),
    )


def build_router(config: Config) -> Router:
    strong, weak = model_pair(config)
    return ROUTERS.create(
        cfg.get(config, "router.name"),
        strong_model=strong,
        weak_model=weak,
        seed=int(cfg.get(config, "seed", 0)),
        **(cfg.get(config, "router.params", None) or {}),
    )


def prepare_data(config: Config) -> PreparedData:
    """Load, filter, rate and tier the battles, then split off the calibration set."""
    battles = load_battles(cfg.get(config, "data.sources"))
    battles = filter_battles(battles, int(cfg.get(config, "data.min_prompt_chars", 16)))
    ratings = tiers_mod.compute_ratings(battles)
    tiers = tiers_mod.segment_tiers(ratings, int(cfg.get(config, "data.n_tiers", 10)))
    aliases = cfg.get(config, "data.model_aliases", None) or {}
    strong, weak = model_pair(config)
    strong_tier, weak_tier = tiers_mod.check_pair(tiers, strong, weak, aliases)
    battles = tiers_mod.assign_tiers(battles, tiers)
    fit, calibration = split_calibration(
        battles,
        float(cfg.get(config, "data.calibration_fraction", 0.1)),
        int(cfg.get(config, "seed", 0)),
    )
    return PreparedData(fit, calibration, ratings, tiers, strong_tier, weak_tier)


def _load_router(config: Config) -> tuple[Router, Embedder]:
    embedder = build_embedder(config)
    router_dir = experiment_dir(config) / "router"
    if not router_dir.exists():
        raise FileNotFoundError(f"{router_dir} not found; run 'fit' first")
    return Router.load(router_dir, embedder), embedder


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------- steps
def toy_data(config: Config) -> dict[str, Any]:
    """Write synthetic battles and benchmark to the paths named in the config."""
    strong, weak = model_pair(config)
    seed = int(cfg.get(config, "data.toy.seed", cfg.get(config, "seed", 0)))
    battles = make_toy_battles(
        n_battles=int(cfg.get(config, "data.toy.n_battles", 2000)),
        seed=seed,
        strong_model=strong,
        weak_model=weak,
    )
    benchmark = make_toy_benchmark(
        n_questions=int(cfg.get(config, "data.toy.n_benchmark", 300)), seed=seed + 1
    )
    battles_path = Path(cfg.get(config, "data.toy.battles_path"))
    benchmark_path = Path(cfg.get(config, "eval.benchmark_path"))
    for path, frame in ((battles_path, battles), (benchmark_path, benchmark)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return {
        "battles_path": str(battles_path),
        "n_battles": len(battles),
        "benchmark_path": str(benchmark_path),
        "n_benchmark": len(benchmark),
    }


def prepare_sources(config: Config) -> dict[str, Any]:
    """Download the raw files listed in ``data.prepare`` and write the processed tables."""
    prepare = cfg.get(config, "data.prepare", None)
    if not prepare:
        raise ValueError("config has no data.prepare block")
    raw_dir = prepare.get("raw_dir", "data/raw")
    downloads = download_files(prepare.get("downloads") or [], raw_dir)
    return {"downloads": downloads, "written": convert_all(prepare, raw_dir)}


def info(config: Config) -> dict[str, Any]:
    """Describe the data an experiment would fit on."""
    data = prepare_data(config)
    battles = pd.concat([data.fit, data.calibration], ignore_index=True)
    strong, weak = model_pair(config)
    tier_pairs = (
        battles.assign(
            low=np.minimum(battles["tier_a"], battles["tier_b"]),
            high=np.maximum(battles["tier_a"], battles["tier_b"]),
        )
        .groupby(["low", "high"])
        .size()
    )
    return {
        "experiment": config["name"],
        "router": cfg.get(config, "router.name"),
        "embedder": build_embedder(config).name,
        "n_battles": len(battles),
        "n_fit": len(data.fit),
        "n_calibration": len(data.calibration),
        "by_source": battles["source"].value_counts().to_dict(),
        "by_winner": battles["winner"].value_counts().to_dict(),
        "n_models": len(data.tiers),
        "tiers": {
            str(tier): sorted(m for m, t in data.tiers.items() if t == tier)
            for tier in sorted(set(data.tiers.values()))
        },
        "strong": {"model": strong, "tier": data.strong_tier},
        "weak": {"model": weak, "tier": data.weak_tier},
        "battles_by_tier_pair": {f"{low}-{high}": int(n) for (low, high), n in tier_pairs.items()},
    }


def fit(config: Config) -> dict[str, Any]:
    """Fit the configured router and save it under ``experiments/<name>/router``."""
    out_dir = experiment_dir(config)
    cfg.save_resolved(config, out_dir / "config.resolved.json")
    data = prepare_data(config)
    router = build_router(config)
    embedder: Embedder | None = None
    embeddings: NDArray[np.floating] | None = None
    if router.uses_embeddings:
        embedder = build_embedder(config)
        embeddings = embedder.embed(data.fit["prompt"].tolist())
    router.fit(data.fit, embeddings, embedder)
    router.save(out_dir / "router")
    return {
        "experiment": config["name"],
        "router": router.registry_name,
        "n_fit": len(data.fit),
        "router_dir": str(out_dir / "router"),
    }


def calibrate(config: Config, strong_rates: Sequence[float]) -> dict[str, Any]:
    """Find alpha for each target strong-call rate on the held-out calibration prompts."""
    router, _ = _load_router(config)
    data = prepare_data(config)
    prompts = sorted(data.calibration["prompt"].unique())
    if not prompts:
        raise ValueError("calibration set is empty; raise data.calibration_fraction")
    thresholds = calibrate_thresholds(router.strong_win_rate(prompts), strong_rates)
    payload = {"n_calibration_prompts": len(prompts), "thresholds": thresholds}
    _write_json(experiment_dir(config) / "thresholds.json", payload)
    return payload


def _cost_block(config: Config, summary: dict[str, Any]) -> dict[str, Any] | None:
    models = cfg.get(config, "models", {})
    tokens = models.get("avg_tokens")
    if not tokens or "input_usd_per_mtok" not in models.get("strong", {}):
        return None
    strong = ModelPrice.from_config(models["strong"])
    weak = ModelPrice.from_config(models["weak"])
    tokens_in, tokens_out = float(tokens["input"]), float(tokens["output"])
    all_strong = routed_cost_usd(1.0, strong, weak, tokens_in, tokens_out)
    block: dict[str, Any] = {"usd_per_query_all_strong": all_strong}
    for key in ("cpt_50", "cpt_80"):
        cost = routed_cost_usd(summary[key], strong, weak, tokens_in, tokens_out)
        block[f"usd_per_query_at_{key}"] = cost
        block[f"saving_ratio_at_{key}"] = all_strong / cost
    return block


def evaluate(config: Config) -> dict[str, Any]:
    """Score the fitted router on the benchmark after removing contaminated questions."""
    router, embedder = _load_router(config)
    benchmark = load_benchmark(cfg.get(config, "eval.benchmark_path"))
    name = str(cfg.get(config, "eval.benchmark_name"))

    train_prompts = sorted(prepare_data(config).fit["prompt"].unique())
    threshold = float(cfg.get(config, "eval.contamination_threshold", 0.95))
    contaminated = find_contaminated(
        embedder.embed(benchmark["prompt"].tolist()),
        embedder.embed(train_prompts),
        threshold,
    )
    n_removed = int(contaminated.sum())
    log.info("contamination (cosine >= %.2f): removed %d questions", threshold, n_removed)
    benchmark = benchmark.loc[~contaminated].reset_index(drop=True)

    seed = int(cfg.get(config, "seed", 0))
    scores = router.strong_win_rate(benchmark["prompt"].tolist())
    summary = {
        "experiment": config["name"],
        "router": router.registry_name,
        "benchmark": name,
        "seed": seed,
        "n_contaminated_removed": n_removed,
        **runner.summarise(scores, benchmark, int(cfg.get(config, "eval.random_seeds", 20)), seed),
    }
    cost = _cost_block(config, summary)
    if cost is not None:
        summary["cost"] = cost
    runner.write_results(experiment_dir(config) / "eval" / name, summary, scores, benchmark)
    return summary


def route(
    config: Config,
    prompts: Sequence[str],
    strong_rate: float | None = None,
    alpha: float | None = None,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """Route prompts with a calibrated threshold (``strong_rate``) or an explicit ``alpha``."""
    router, _ = _load_router(config)
    if alpha is None:
        thresholds_path = experiment_dir(config) / "thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(f"{thresholds_path} not found; run 'calibrate' first")
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
        key = f"{0.5 if strong_rate is None else strong_rate:g}"
        if key not in thresholds:
            raise KeyError(f"no calibrated threshold for rate {key}; have {sorted(thresholds)}")
        alpha = float(thresholds[key]["alpha"])
    details = router.explain(prompts) if explain else [{} for _ in prompts]
    decisions = router.route(prompts, alpha)
    strong, weak = model_pair(config)
    return [
        {
            "prompt": prompt,
            "model": strong if use_strong else weak,
            "alpha": alpha,
            **detail,
        }
        for prompt, use_strong, detail in zip(prompts, decisions, details, strict=True)
    ]


def report(config: Config) -> list[dict[str, Any]]:
    """Collect the headline numbers of every benchmark evaluated for this experiment."""
    rows = []
    for path in sorted((experiment_dir(config) / "eval").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "benchmark": summary["benchmark"],
                "router": summary["router"],
                "n_questions": summary["n_questions"],
                "apgr": summary["apgr"],
                "cpt_50": summary["cpt_50"],
                "cpt_80": summary["cpt_80"],
                "random_apgr": summary["random"]["apgr_mean"],
                "random_apgr_ci95": summary["random"]["apgr_ci95"],
                "random_cpt_50": summary["random"]["cpt_50_mean"],
                "random_cpt_80": summary["random"]["cpt_80_mean"],
            }
        )
    return rows
