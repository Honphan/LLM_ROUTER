"""Download raw public files into ``data/raw`` and record what was fetched.

Raw files are immutable: an existing file is never downloaded again or overwritten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_FILE = "MANIFEST.json"
_USER_AGENT = "python-urllib"
_TIMEOUT_SECONDS = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_downloads(specs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Flatten download specs into ``(url, relative path)`` pairs.

    A spec is either ``{url, path}`` or ``{base_url, dir, files: [...]}``; the second form
    fetches ``base_url/<file>`` into ``dir/<file>``.
    """
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        if "url" in spec:
            pairs.append((str(spec["url"]), str(spec["path"])))
        else:
            base = str(spec["base_url"]).rstrip("/")
            directory = str(spec["dir"]).strip("/")
            pairs.extend((f"{base}/{name}", f"{directory}/{name}") for name in spec["files"])
    return pairs


def download_files(specs: Sequence[Mapping[str, Any]], raw_dir: str | Path) -> dict[str, Any]:
    """Fetch every file that is not on disk yet; return counts and update the manifest."""
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_FILE
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    downloaded = skipped = 0
    for url, relative in expand_downloads(specs):
        target = root / relative
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        log.info("downloading %s", url)
        with (
            urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,  # noqa: S310
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        partial.replace(target)
        manifest[relative] = {
            "url": url,
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        }
        downloaded += 1

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info("downloads: %d fetched, %d already present", downloaded, skipped)
    return {"downloaded": downloaded, "already_present": skipped, "raw_dir": str(root)}
