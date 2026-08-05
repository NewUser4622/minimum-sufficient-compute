"""
msc_lib.py -- Minimum Sufficient Compute: full Kaggle/HuggingFace pipeline.

Companion to:
    msc_core.py   -- the MSC oracle and every analysis statistic (numpy/scipy only)
    msc_torch.py  -- reference exit heads, ordinal head, loss, LTT calibration

This module is the operational layer: everything needed to run ~1,200 T4-hours
of experiments across six Kaggle accounts without colliding, losing work, or
producing a number that cannot be traced back to a config.

Design principle, inherited from E2AM and unchanged:
    HuggingFace is the ONLY permanent store. The Kaggle disk is scratch.
    /kaggle/temp  (~1 TB, session-local) holds datasets and intermediates.
    /kaggle/working (20 GB, persistent-ish) holds artifacts awaiting push.
    Once HF confirms a run's artifacts, the local copy is deleted.

Sections
--------
    1.  utils                -- atomic IO, seeding, hashing, env capture
    2.  hf_uploader          -- batched commits, token-bucket rate limiter, 429 handling
    3.  hf_run_sync          -- per-run wrapper + dual-repo router
    4.  registry             -- multi-account claim protocol, run ledger
    5.  lifecycle            -- SIGTERM / atexit / KeyboardInterrupt flush, session watchdog
    6.  data                 -- CIFAR-100 from the Kaggle mirror, in-memory tensors
    7.  zoo                  -- 13 architectures, all exposing forward_features()
    8.  budgets              -- FLOPs per compute configuration, per axis
    9.  exits                -- exit heads, multi-exit wrapper, ordinal sufficiency head
    10. energy               -- NVML power sampling at >=10 Hz
    11. dynamics             -- EL2N, forgetting events, prediction depth
    12. config               -- run registry: architecture x dataset x phase x seed
    13. train                -- resumable backbone training with full RNG capture
    14. oracle               -- depth / resolution / precision sweeps -> per-sample Parquet
    15. method               -- MSC-KD, baselines, matched-FLOPs evaluation
    16. analysis             -- thin wrappers over msc_core + aggregation
    17. selftest

Run `python msc_lib.py --selftest` for the offline checks (no GPU required).
"""
from __future__ import annotations

import atexit
import base64
import csv
import hashlib
import io
import json
import math
import os
import platform
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Torch is imported lazily-but-eagerly: the analysis notebooks run CPU-only and
# should not pay for it, but every training path needs it. A missing torch is a
# hard error only when a training entry point is actually called.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    _TORCH_OK = True
except Exception as _e:                                    # pragma: no cover
    torch = None; nn = None; F = None
    DataLoader = object; Dataset = object
    _TORCH_OK = False
    _TORCH_ERR = str(_e)

try:
    import pandas as pd
except Exception:                                          # pragma: no cover
    pd = None

try:
    import yaml
except Exception:                                          # pragma: no cover
    yaml = None

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# Platform constants
# --------------------------------------------------------------------------
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
# /kaggle/temp is ~1 TB and session-local. Datasets and any large intermediate
# tensor goes here. /kaggle/working is 20 GB and is artifact space -- putting a
# dataset there is how a session dies at hour six.
SCRATCH_ROOT = Path("/kaggle/temp") if ON_KAGGLE else Path(
    os.environ.get("MSC_SCRATCH", Path.cwd() / "scratch"))

# One repo per dataset. A second dataset gets `msc-tinyimagenet`, etc.
HF_REPO = "Shanmuk4622/msc-cifar100"
# Retained so older notebooks and the audit tool can still name the previous
# two-repo layout.
HF_MODEL_REPO = "Shanmuk4622/msc-kd"
HF_DATA_REPO = "Shanmuk4622/msc-kd-data"

# The Kaggle mirror the team uses. Direct in-datacentre download; far faster
# than reaching out to cs.toronto.edu from a Kaggle worker.
KAGGLE_CIFAR100_SLUG = "shanmuk4622/dataset-cifar100-python"

TAU_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

# Compute-configuration grids. Frozen here so budgets/{arch}.json is
# deterministic across accounts and sessions.
DEPTH_FRACTIONS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
RESOLUTIONS: Tuple[int, ...] = (16, 20, 24, 28, 32)
PRECISIONS: Tuple[str, ...] = ("int4", "int6", "int8", "fp16", "fp32")
PRECISION_BITS: Dict[str, int] = {"int4": 4, "int6": 6, "int8": 8, "fp16": 16, "fp32": 32}


# =============================================================================
# 1. utils
# =============================================================================
def _no_grad():
    """`torch.no_grad()` where torch exists, a no-op decorator where it does not.

    The analysis notebooks run CPU-only and legitimately have no torch. A bare
    module-level `@torch.no_grad()` would make this whole module unimportable
    there, which would be an absurd reason to be unable to compute a Spearman
    correlation.
    """
    if _TORCH_OK:
        return torch.no_grad()

    def _identity(fn):
        return fn
    return _identity


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path, text: str) -> None:
    """Write via a temp file and rename.

    Never write in place. A session killed mid-write leaves a truncated file,
    and for ckpt_last.pt that means the run is gone. os.replace is atomic on
    POSIX, which Kaggle is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str, sort_keys=False))


def atomic_write_yaml(path, obj) -> None:
    if yaml is None:
        atomic_write_json(Path(path).with_suffix(".json"), obj)
        return
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=True, default_flow_style=False))


def atomic_save_torch(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_of_obj(obj) -> str:
    """Stable hash of a config dict. Sorted keys, so key order never matters."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_array(a: np.ndarray) -> str:
    """Fingerprint of the canonical sample order.

    Every per-sample table stores this over its label vector. At analysis time
    two tables that disagree are refusing to be correlated, loudly, instead of
    silently producing a meaningless transfer coefficient. Index misalignment
    between models is the single most likely way to fabricate a result here.
    """
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every stream that affects the run.

    `deterministic` trades ~10% throughput for bit-reproducibility. The spec
    says enable it where it does not cost more than that, and record the choice
    in the config either way.
    """
    random.seed(seed)
    np.random.seed(seed)
    if not _TORCH_OK:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def capture_rng_state() -> Dict[str, Any]:
    """All four RNG streams.

    Omitting this is the subtlest way to destroy this project. Without it a
    resumed run sees a different augmentation and shuffling sequence than an
    uninterrupted one, so "same architecture, same data, different seed" stops
    meaning what Q1 needs it to mean -- and Q1's seed ceiling is the
    denominator of every transfer number in the paper.
    """
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if _TORCH_OK:
        st["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng_state(st: Optional[Dict[str, Any]]) -> bool:
    if not st:
        return False
    ok = True
    try:
        random.setstate(st["python"])
    except Exception:
        ok = False
    try:
        np.random.set_state(st["numpy"])
    except Exception:
        ok = False
    if _TORCH_OK:
        try:
            torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
        except Exception:
            ok = False
        if torch.cuda.is_available() and "cuda" in st:
            try:
                torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                              for s in st["cuda"]])
            except Exception:
                ok = False
    return ok


def shell(cmd: List[str], timeout: float = 20.0) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def free_mb(path) -> int:
    try:
        return shutil.disk_usage(str(path)).free // (1024 * 1024)
    except Exception:
        return -1


def dir_size_mb(path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) // (1024 * 1024)
    except Exception:
        return 0


def environment_report() -> Dict[str, Any]:
    """Everything needed to explain a number six months from now.

    T4 sessions vary (driver versions, whether you got a T4 or a P100 on a
    fallback). Record which you got.
    """
    rep: Dict[str, Any] = {
        "captured_utc": now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "on_kaggle": ON_KAGGLE,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "cpu_count": os.cpu_count(),
        "msc_lib_version": __version__,
    }
    if _TORCH_OK:
        rep.update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn": (torch.backends.cudnn.version()
                      if torch.backends.cudnn.is_available() else None),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())]
                         if torch.cuda.is_available() else [],
            "gpu_total_mem_mb": [
                torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
                for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else [],
        })
    rc, out, _ = shell(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if rc == 0:
        rep["nvidia_driver"] = out.strip().splitlines()[0] if out.strip() else None
    rc, out, _ = shell([sys.executable, "-m", "pip", "freeze"], timeout=90)
    rep["pip_freeze"] = out.splitlines() if rc == 0 else []
    rep["free_mb_working"] = free_mb(WORK_ROOT)
    rep["free_mb_scratch"] = free_mb(SCRATCH_ROOT if SCRATCH_ROOT.exists() else WORK_ROOT)
    return rep


class Tee:
    """Mirror stdout to a file so the console log is an artifact like any other.

    Kaggle truncates long outputs in the rendered notebook; the pushed log is
    the copy that survives.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        try:
            self._f.write(s)
        except Exception:
            pass

    def flush(self):
        self._stdout.flush()
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def log(msg: str, tag: str = "MSC") -> None:
    print(f"[{tag}] {msg}", flush=True)


# =============================================================================
# 2. hf_uploader -- batched commits, token bucket, 429 handling, dedup
# =============================================================================
@dataclass
class _PendingFile:
    local_path: str
    repo_path: str
    is_heavy: bool
    fingerprint: str
    enqueued_at: float


class _SharedRateLimiter:
    """One commit budget per HuggingFace TOKEN, shared by every uploader.

    HF's write limit is per USER, not per repository. A limiter that lives on
    the uploader therefore multiplies the budget by the number of repos: two
    uploaders each capped at 20/hour let one account emit 40/hour, and six
    accounts 240/hour against a real ceiling near 128. The cap silently stopped
    meaning anything.

    So the bucket is keyed by token and shared process-wide. Adding repos no
    longer inflates the budget.
    """

    _buckets: Dict[str, "_SharedRateLimiter"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._times: List[float] = []
        self._lock = threading.Lock()

    @classmethod
    def for_token(cls, token: Optional[str], limit: int) -> "_SharedRateLimiter":
        key = hashlib.sha256((token or "anon").encode()).hexdigest()[:16]
        with cls._registry_lock:
            b = cls._buckets.get(key)
            if b is None:
                b = cls(limit)
                cls._buckets[key] = b
            else:
                b.limit = min(b.limit, int(limit))    # most conservative wins
            return b

    def count_last_hour(self) -> int:
        now = time.time()
        with self._lock:
            self._times = [t for t in self._times if now - t < 3600]
            return len(self._times)

    def record(self) -> None:
        with self._lock:
            self._times.append(time.time())

    def wait_for_slot(self, stop: threading.Event, label: str = "") -> None:
        while not stop.is_set():
            now = time.time()
            with self._lock:
                self._times = [t for t in self._times if now - t < 3600]
                if len(self._times) < self.limit:
                    return
                oldest = self._times[0]
            wait = max(1.0, 3600 - (now - oldest) + 2.0)
            print(f"[HF:{label}] shared rate-limit guard: {self.limit} commits used "
                  f"this hour (budget is per HF token, across all repos) -- "
                  f"sleeping {wait:.0f}s")
            if stop.wait(wait):
                return


class BackgroundUploader:
    """One worker thread, one buffer, one commit per cycle.

    The single most important property is that every file enqueued inside a
    push window collapses into ONE HuggingFace commit. Pushing six files as six
    commits consumes six times the rate-limit quota for exactly no benefit, and
    HF's write limit (~128 commits/hour/user) is shared across all six team
    accounts if they use one token -- or across all repos if they do not.

    Flush triggers:
        - BATCH_INTERVAL_SEC elapsed (default 1800 = the 30-minute policy)
        - buffer exceeds BATCH_MAX_FILES or BATCH_MAX_BYTES
        - flush() called explicitly (stage completion, interrupt, exit)

    Rate limiting is a token bucket over a rolling hour. When the cap is
    reached the worker SLEEPS until the oldest commit ages out rather than
    failing -- a failed push that kills training is worse than a slow one.
    """

    MAX_BACKOFF_SEC = 300.0
    MAX_ATTEMPTS = 8
    BATCH_INTERVAL_SEC = 1800.0                  # 30 min, per engineering spec 5
    BATCH_MAX_FILES = 400
    BATCH_MAX_BYTES = 3 * 1024 * 1024 * 1024     # 3 GB
    # HF's cap is ~128/hr. Six accounts share the org quota, so 20 each leaves
    # headroom (6 x 20 = 120) even when everyone is running flat out.
    COMMITS_PER_HOUR_LIMIT = 20

    def __init__(self, repo_id: str, token: str, repo_type: str = "dataset",
                 batch_interval_sec: Optional[float] = None,
                 batch_max_files: Optional[int] = None,
                 batch_max_bytes: Optional[int] = None,
                 commits_per_hour_limit: Optional[int] = None,
                 private: bool = True,
                 label: str = ""):
        self.repo_id = repo_id
        self.token = token
        self.repo_type = repo_type
        self.private = private
        self.label = label or repo_id.split("/")[-1]
        if batch_interval_sec is not None:
            self.BATCH_INTERVAL_SEC = float(batch_interval_sec)
        if batch_max_files is not None:
            self.BATCH_MAX_FILES = int(batch_max_files)
        if batch_max_bytes is not None:
            self.BATCH_MAX_BYTES = int(batch_max_bytes)
        if commits_per_hour_limit is not None:
            self.COMMITS_PER_HOUR_LIMIT = int(commits_per_hour_limit)

        self._buffer: Dict[str, _PendingFile] = {}
        self._buf_lock = threading.Lock()
        self._fingerprints: Set[str] = set()
        self._fp_lock = threading.Lock()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        # Commit budget is shared across every uploader using this token.
        self._limiter = _SharedRateLimiter.for_token(token, self.COMMITS_PER_HOUR_LIMIT)
        self._thread: Optional[threading.Thread] = None
        self._in_commit = False
        self._api = None
        self._stats = {"queued": 0, "uploaded": 0, "skipped_dedup": 0,
                       "commits_made": 0, "retries": 0, "rate_limit_waits": 0,
                       "failed_permanent": 0, "bytes_uploaded": 0}
        self._stats_lock = threading.Lock()

    # ------------------------------ lifecycle ------------------------------
    def start(self) -> bool:
        try:
            from huggingface_hub import HfApi, create_repo
            create_repo(repo_id=self.repo_id, token=self.token, exist_ok=True,
                        repo_type=self.repo_type, private=self.private)
            self._api = HfApi(token=self.token)
        except Exception as e:
            print(f"[HF:{self.label}] init failed: {e}")
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"hf-uploader-{self.label}")
        self._thread.start()
        print(f"[HF:{self.label}] uploader started -> {self.repo_id} "
              f"({self.repo_type}, batch {self.BATCH_INTERVAL_SEC/60:.0f} min, "
              f"max {self.COMMITS_PER_HOUR_LIMIT} commits/hr)")
        return True

    def stop(self, drain: bool = True, timeout: float = 900.0) -> None:
        if self._thread is None:
            return
        if drain:
            self.flush(timeout=timeout)
        self._stop.set()
        self._wakeup.set()
        self._thread.join(timeout=30)
        self._thread = None

    # ------------------------------ public api -----------------------------
    def enqueue(self, local_path, repo_path: str, *, is_heavy: bool = False) -> bool:
        """Buffer a file for the next batched commit. False if deduplicated."""
        local_path = Path(local_path)
        if not local_path.exists():
            return False
        fp = self._fingerprint(local_path, repo_path)
        with self._fp_lock:
            if fp in self._fingerprints:
                with self._stats_lock:
                    self._stats["skipped_dedup"] += 1
                return False
        repo_path = repo_path.replace("\\", "/").lstrip("/")
        with self._buf_lock:
            # A newer version of the same repo_path supersedes the pending one.
            # Rolling checkpoints hit this every cycle.
            self._buffer[repo_path] = _PendingFile(
                local_path=str(local_path), repo_path=repo_path,
                is_heavy=is_heavy, fingerprint=fp, enqueued_at=time.time())
            n = len(self._buffer)
            nbytes = sum(self._safe_size(p.local_path) for p in self._buffer.values())
        with self._stats_lock:
            self._stats["queued"] += 1
        if n >= self.BATCH_MAX_FILES or nbytes >= self.BATCH_MAX_BYTES:
            self._wakeup.set()
        return True

    def enqueue_dir(self, local_dir, repo_prefix: str, *,
                    patterns: Sequence[str] = ("*",), recursive: bool = True,
                    heavy_suffixes: Sequence[str] = (".pt", ".pth", ".safetensors",
                                                     ".parquet")) -> int:
        local_dir = Path(local_dir)
        if not local_dir.exists():
            return 0
        n = 0
        globber = local_dir.rglob if recursive else local_dir.glob
        seen: Set[Path] = set()
        for pat in patterns:
            for f in globber(pat):
                if not f.is_file() or f in seen:
                    continue
                seen.add(f)
                rel = f.relative_to(local_dir).as_posix()
                heavy = f.suffix in heavy_suffixes
                n += int(self.enqueue(f, f"{repo_prefix.rstrip('/')}/{rel}", is_heavy=heavy))
        return n

    def flush(self, timeout: float = 900.0) -> bool:
        """Force a commit now and block until the buffer is empty."""
        self._wakeup.set()
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._buf_lock:
                empty = not self._buffer
            if empty and not self._in_commit:
                return True
            time.sleep(0.5)
        return False

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            with self._buf_lock:
                pending = len(self._buffer)
            return dict(self._stats, pending_in_buffer=pending,
                        commits_in_last_hour=self._commits_in_last_hour(),
                        repo=self.repo_id)

    def list_repo_files(self) -> Set[str]:
        try:
            return set(self._api.list_repo_files(repo_id=self.repo_id,
                                                 repo_type=self.repo_type))
        except Exception as e:
            print(f"[HF:{self.label}] list_repo_files: {e}")
            return set()

    def download(self, local_dir, allow_patterns: Optional[Sequence[str]] = None,
                 quiet: bool = False) -> bool:
        """Scoped snapshot. ALWAYS pass allow_patterns on a 20 GB disk.

        An unscoped snapshot of the model repo late in the project is several
        hundred GB and will kill the session instantly.
        """
        try:
            from huggingface_hub import snapshot_download
            ensure_dir(local_dir)
            snapshot_download(repo_id=self.repo_id, repo_type=self.repo_type,
                              local_dir=str(local_dir), token=self.token,
                              allow_patterns=list(allow_patterns) if allow_patterns else None)
            return True
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "repository not found" in msg:
                if not quiet:
                    print(f"[HF:{self.label}] no prior snapshot (fresh repo)")
                return False
            if not quiet:
                print(f"[HF:{self.label}] snapshot warning: {e}")
            return False

    def download_file(self, repo_path: str, local_dir) -> Optional[Path]:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type,
                                filename=repo_path, token=self.token,
                                local_dir=str(ensure_dir(local_dir)))
            return Path(p)
        except Exception:
            return None

    def delete_prefix(self, prefix: str) -> int:
        """Remove every file under a repo prefix in one commit.

        Used by broken-stub demotion: a run marked complete but truncated by a
        crash must be erased from HF too, or the next session resurrects it.
        """
        try:
            from huggingface_hub import CommitOperationDelete
            files = [f for f in self.list_repo_files() if f.startswith(prefix)]
            if not files:
                return 0
            self._api.create_commit(
                repo_id=self.repo_id, repo_type=self.repo_type,
                operations=[CommitOperationDelete(path_in_repo=f) for f in files],
                commit_message=f"msc: wipe {prefix} ({len(files)} files)")
            self._limiter.record()
            return len(files)
        except Exception as e:
            print(f"[HF:{self.label}] delete_prefix({prefix}): {e}")
            return 0

    # ------------------------------ internals ------------------------------
    @staticmethod
    def _fingerprint(local_path: Path, repo_path: str) -> str:
        try:
            st = local_path.stat()
            return f"{repo_path}|{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            return f"{repo_path}|?|{time.time()}"

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return Path(path).stat().st_size
        except Exception:
            return 0

    def _commits_in_last_hour(self) -> int:
        return self._limiter.count_last_hour()

    def _wait_for_rate_limit(self) -> None:
        before = self._limiter.count_last_hour()
        self._limiter.wait_for_slot(self._stop, self.label)
        if before >= self._limiter.limit:
            with self._stats_lock:
                self._stats["rate_limit_waits"] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait(timeout=self.BATCH_INTERVAL_SEC)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            with self._buf_lock:
                if not self._buffer:
                    continue
                batch = list(self._buffer.values())
                self._buffer.clear()
            self._wait_for_rate_limit()
            self._in_commit = True
            try:
                if not self._commit_batch(batch):
                    # Requeue for the next cycle, but never clobber a newer
                    # version of the same path that arrived while we were trying.
                    with self._buf_lock:
                        for pf in batch:
                            self._buffer.setdefault(pf.repo_path, pf)
            finally:
                self._in_commit = False
        # Final drain on stop.
        with self._buf_lock:
            final = list(self._buffer.values())
            self._buffer.clear()
        if final:
            self._wait_for_rate_limit()
            self._commit_batch(final)

    def _commit_batch(self, batch: List[_PendingFile]) -> bool:
        if not batch:
            return True
        try:
            from huggingface_hub import CommitOperationAdd
        except Exception as e:
            print(f"[HF:{self.label}] huggingface_hub import failed: {e}")
            return False

        ops, total_bytes = [], 0
        for pf in batch:
            if not Path(pf.local_path).exists():
                continue
            ops.append(CommitOperationAdd(path_in_repo=pf.repo_path,
                                          path_or_fileobj=pf.local_path))
            total_bytes += self._safe_size(pf.local_path)
        if not ops:
            return True

        backoff = 2.0
        last_err: Optional[str] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if self._stop.is_set():
                return False
            try:
                self._api.create_commit(
                    repo_id=self.repo_id, repo_type=self.repo_type, operations=ops,
                    commit_message=(f"msc: batch {len(ops)} files "
                                    f"({total_bytes // 1024} KB) @ {now_iso()}"))
                with self._fp_lock:
                    for pf in batch:
                        self._fingerprints.add(pf.fingerprint)
                self._limiter.record()
                with self._stats_lock:
                    self._stats["uploaded"] += len(ops)
                    self._stats["commits_made"] += 1
                    self._stats["bytes_uploaded"] += total_bytes
                print(f"[HF:{self.label}] committed {len(ops)} files "
                      f"({total_bytes/1e6:.1f} MB)")
                return True
            except Exception as e:
                last_err = str(e)
                low = last_err.lower()
                with self._stats_lock:
                    self._stats["retries"] += 1
                # Auth problems will never fix themselves. Stop immediately
                # rather than burning eight attempts.
                if any(s in low for s in ("401", "403", "unauthorized",
                                          "forbidden", "permission")):
                    print(f"[HF:{self.label}] AUTH FAILURE -- check HF_TOKEN write scope "
                          f"and access to {self.repo_id}")
                    break
                if "429" in low or "rate limit" in low or "too many requests" in low:
                    wait = self._parse_retry_after(last_err)
                    print(f"[HF:{self.label}] 429 rate limit, sleeping {wait:.0f}s "
                          f"(attempt {attempt}/{self.MAX_ATTEMPTS})")
                    if self._stop.wait(wait):
                        return False
                    continue
                sleep_for = min(backoff, self.MAX_BACKOFF_SEC)
                print(f"[HF:{self.label}] commit attempt {attempt} failed: "
                      f"{last_err[:160]} -> retry in {sleep_for:.0f}s")
                if self._stop.wait(sleep_for):
                    return False
                backoff = min(backoff * 2.0, self.MAX_BACKOFF_SEC)

        with self._stats_lock:
            self._stats["failed_permanent"] += len(ops)
        print(f"[HF:{self.label}] BATCH FAILED after {self.MAX_ATTEMPTS} attempts "
              f"({len(ops)} files): {last_err}")
        return False

    @staticmethod
    def _parse_retry_after(err: str) -> float:
        """HF's 429 body carries a human-readable hint. Obey it.

        Sleeping the exact advertised interval beats blind exponential backoff:
        it neither wastes a window nor hammers the endpoint early.
        """
        m = re.search(r"[Rr]etry[- ]?[Aa]fter[:= ]+(\d+)", err)
        if m:
            return float(m.group(1)) + 2.0
        m = re.search(r"retry after (\d+)\s*second", err, re.I)
        if m:
            return float(m.group(1)) + 2.0
        m = re.search(r"in about (\d+)\s*hour", err, re.I)
        if m:
            return min(3600.0, float(m.group(1)) * 3600.0)
        m = re.search(r"in about (\d+)\s*minute", err, re.I)
        if m:
            return float(m.group(1)) * 60.0 + 5.0
        return 120.0


def get_hf_token(secret_name: str = "HF_TOKEN") -> Optional[str]:
    """Kaggle Secrets first, environment variable second."""
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(secret_name)
        if tok:
            return tok
    except Exception:
        pass
    tok = os.environ.get(secret_name)
    if not tok:
        print(f"[HF] no token: add '{secret_name}' to Kaggle Secrets "
              f"(Add-ons -> Secrets) or export it as an env var")
    return tok


# =============================================================================
# 3. hf_run_sync -- dual-repo router
# =============================================================================
class MSCHub:
    """ONE repository. See 06_DATA_SCHEMA.md 1.

    Everything a run produces lives under `runs/{run_id}/` -- checkpoints,
    metrics, telemetry, per-sample tables. Two reasons this replaced the
    earlier two-repo split:

      * HuggingFace's write limit is per USER, not per repo. Two uploaders each
        capped at 20 commits/hour let one account emit 40, and six accounts 240
        against a real ceiling near 128. One repo means one commit per cycle and
        the cap means what it says. (The shared limiter now enforces this
        regardless, but halving the commit count is free.)
      * A run's artifacts belong together. Reading a run's history should not
        require knowing which of two repos to look in.

    A DATASET repo rather than a model repo, because HuggingFace renders CSV and
    Parquet previews for datasets -- every metrics table becomes browsable in
    the web UI without downloading anything. For a project whose contribution is
    partly the artifact, that is worth more than the model-repo badge.

    `.models` and `.data` both point at the same uploader, so older call sites
    keep working.
    """

    def __init__(self, token: Optional[str] = None,
                 repo: str = HF_REPO, enable: bool = True,
                 repo_type: str = "dataset", **uploader_kwargs):
        self.token = token if token is not None else get_hf_token()
        self.repo_id = repo
        self.hub: Optional[BackgroundUploader] = None
        self.enabled = False
        if not enable or not self.token:
            print("[HF] disabled (no token or explicitly off) -- "
                  "runs will be LOCAL ONLY and lost when the session ends")
            self.models = self.data = None
            return
        u = BackgroundUploader(repo, self.token, repo_type=repo_type,
                               label="hub", **uploader_kwargs)
        if u.start():
            self.hub = self.models = self.data = u
            self.enabled = True
        else:
            print(f"[HF] {repo} failed to initialise -- disabling")
            self.models = self.data = None
            try:
                u.stop(drain=False)
            except Exception:
                pass

    def flush(self, timeout: float = 900.0) -> bool:
        return self.hub.flush(timeout=timeout) if self.enabled else True

    def stop(self, drain: bool = True) -> None:
        if self.enabled:
            try:
                self.hub.stop(drain=drain)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        return {"enabled": False} if not self.enabled else {"hub": self.hub.stats()}

    def print_stats(self) -> None:
        if not self.enabled:
            print("[HF] disabled")
            return
        v = self.hub.stats()
        print(f"[HF] {self.repo_id}  uploaded={v['uploaded']:5d} "
              f"commits={v['commits_made']:4d} dedup={v['skipped_dedup']:5d} "
              f"retries={v['retries']:3d} ratewaits={v['rate_limit_waits']:2d} "
              f"pending={v['pending_in_buffer']:4d} "
              f"lasthour={v['commits_in_last_hour']:3d}/{self.hub._limiter.limit} "
              f"MB={v['bytes_uploaded']/1e6:.0f}")


# Everything a run produces, under one folder. See 06_DATA_SCHEMA.md 2.
RUN_SUBDIRS = ("metrics", "telemetry", "per_sample", "checkpoints", "env")


def run_layout(root, run_id: str) -> Dict[str, Path]:
    """Canonical paths for one run. Local tree mirrors the repo tree exactly,
    so a push is a relative-path calculation and never a guess.
    """
    base = Path(root) / "runs" / run_id
    d = {"base": base}
    for s in RUN_SUBDIRS:
        d[s] = base / s
    return d


class RunSync:
    """Per-run artifact router for the single-repo layout.

        {scratch}/runs/{run_id}/...   ->   runs/{run_id}/...

    Push tiers exist because the files have very different sizes and
    freshness requirements:

      light   config, STATUS, summary, metrics/*.csv -- small, pushed every
              30-minute cycle so the record on HF is never far behind
      heavy   checkpoints -- large but essential for resume
      bulk    telemetry/* and per_sample/* -- energy_samples.csv reaches several
              MB, and re-uploading it every half hour would churn LFS storage
              for data nobody reads until the run ends. Pushed at 10-epoch
              milestones and at completion.
    """

    def __init__(self, hub: MSCHub, run_id: str, run_dir, data_dir=None):
        self.hub = hub
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        # data_dir is the repo-root staging area (registry, analysis, tables).
        self.data_dir = Path(data_dir) if data_dir is not None \
            else self.run_dir.parent.parent
        self.enabled = hub.enabled
        self._last_push_ts = 0.0

    @property
    def prefix(self) -> str:
        return f"runs/{self.run_id}"

    def _dir(self, sub: Optional[str] = None) -> int:
        if not self.enabled:
            return 0
        local = self.run_dir / sub if sub else self.run_dir
        repo = f"{self.prefix}/{sub}" if sub else self.prefix
        return self.hub.hub.enqueue_dir(local, repo)

    # ------------------------------ tiers ----------------------------------
    def push_light(self) -> int:
        """Config, status, summary and every metrics table. Cheap, every cycle."""
        if not self.enabled:
            return 0
        n = 0
        for pat in ("*.yaml", "*.json", "*.txt", "*.md"):
            n += self.hub.hub.enqueue_dir(self.run_dir, self.prefix,
                                          patterns=(pat,), recursive=False)
        n += self._dir("metrics")
        n += self._dir("env")
        return n

    def push_checkpoints(self) -> int:
        return self._dir("checkpoints")

    def push_bulk(self) -> int:
        """Raw telemetry and per-sample tables. Milestones only."""
        return self._dir("telemetry") + self._dir("per_sample")

    def push_registry(self) -> int:
        if not self.enabled:
            return 0
        n = self.push_root("registry/events")
        n += self.push_root(f"registry/claims/{self.run_id}.json")
        return n

    def push_root(self, rel: str) -> int:
        """Push a file or directory at the repo root (registry, analysis, tables)."""
        if not self.enabled:
            return 0
        p = self.data_dir / rel
        if p.is_dir():
            return self.hub.hub.enqueue_dir(p, rel)
        return int(self.hub.hub.enqueue(p, rel)) if p.exists() else 0

    def push_all(self, heavy: bool = True, bulk: bool = True) -> int:
        n = self.push_light()
        if heavy:
            n += self.push_checkpoints()
        if bulk:
            n += self.push_bulk()
        n += self.push_registry()
        self._last_push_ts = time.time()
        return n

    # Back-compat aliases for call sites written against the two-repo layout.
    def push_models(self, heavy: bool = True) -> int:
        return self.push_light() + (self.push_checkpoints() if heavy else 0)

    def push_logs(self) -> int:
        return self._dir("telemetry")

    def push_per_sample(self) -> int:
        return self._dir("per_sample")

    def push_data_path(self, rel: str) -> int:
        return self.push_root(rel)

    def due_for_timer_push(self, interval_sec: float = 1800.0) -> bool:
        return (time.time() - self._last_push_ts) >= interval_sec

    def flush(self, timeout: float = 900.0) -> bool:
        return self.hub.flush(timeout=timeout) if self.enabled else True

    def verify_present(self, required: Sequence[str]) -> Set[str]:
        """Which required repo paths are NOT on HF.

        Confirm-then-delete depends on this. Never wipe a local run on the
        strength of a flush() that merely did not time out.
        """
        if not self.enabled:
            return set(required)
        have = self.hub.hub.list_repo_files()
        return {r for r in required if r not in have}


# =============================================================================
# 4. registry -- optimistic claim protocol for six accounts
# =============================================================================
CLAIM_STALE_SEC = 2 * 3600


class RunRegistry:
    """HF Hub is the only shared filesystem, and it has no locking primitive.

    So: optimistic claims. Pull the ledger, refuse anything with a live claim,
    take over anything whose heartbeat has gone stale for two hours (that
    session died), and heartbeat your own claim on every push cycle.

    With six people this is sufficient. The failure mode it does not prevent --
    two accounts claiming the same run within the same few seconds -- is
    caught downstream because both write the same deterministic run_id and the
    later one's checkpoint simply wins.
    """

    def __init__(self, hub: MSCHub, data_dir, account: str = "unknown",
                 worker_id: int = 0):
        self.hub = hub
        self.data_dir = Path(data_dir)
        self.account = account
        self.worker_id = int(worker_id)
        self.session_id = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local") + "-" + \
            hashlib.sha256(f"{platform.node()}{time.time()}".encode()).hexdigest()[:10]

        # ---------------------------------------------------------------
        # The ledger is SHARDED PER WORKER. This is not an optimisation.
        #
        # HuggingFace has no append operation -- you upload a whole file. So if
        # every worker appends to one shared `runs.jsonl` and pushes it, the
        # last push wins and every other worker's lines are silently destroyed.
        # Worker 0 records "s1 running", worker 1 pushes its own copy a few
        # minutes later, and worker 0's line is gone. Nothing errors. The ledger
        # just quietly forgets what happened.
        #
        # That is a lost-update race, and it is expensive here: `plan_work`
        # reads completion state FROM the ledger, so a lost "completed" entry
        # means a finished 3-hour run looks unfinished and gets trained again.
        #
        # Fix: each (account, worker, session) owns its own event file that no
        # other writer ever touches, and reads merge every shard. This is the
        # same collision-safe pattern the NB05 generator pipeline used -- unique
        # filename per writer, reconcile on read.
        # ---------------------------------------------------------------
        self.events_dir = self.data_dir / "registry" / "events"
        ensure_dir(self.events_dir)
        self.shard_name = f"{account}_w{self.worker_id}_{self.session_id}.jsonl"
        self.shard_path = self.events_dir / self.shard_name
        self.shard_repo_path = f"registry/events/{self.shard_name}"
        # Legacy single-file ledger, still read so nothing written before this
        # change is lost. Never written to again.
        self.ledger_path = self.data_dir / "registry" / "runs.jsonl"
        ensure_dir(self.data_dir / "registry" / "claims")

    # ------------------------------ ledger ---------------------------------
    def pull(self) -> None:
        if not self.hub.enabled:
            return
        self.hub.hub.download(self.data_dir, allow_patterns=["registry/**"], quiet=True)

    def _shard_files(self) -> List[Path]:
        files = sorted(self.events_dir.glob("*.jsonl")) if self.events_dir.exists() else []
        if self.ledger_path.exists():
            files.append(self.ledger_path)           # legacy, read-only
        return files

    def entries(self) -> List[Dict[str, Any]]:
        """Every event from every worker's shard, oldest first.

        Ordered by `updated_at` rather than by file, because two workers'
        shards interleave in time and `latest()` must resolve to the genuinely
        most recent state, not to whichever filename sorts last.
        """
        out: List[Dict[str, Any]] = []
        for p in self._shard_files():
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        def _key(e):
            ts = e.get("ts")
            if isinstance(ts, (int, float)):
                return (0, float(ts), "")
            # Legacy entries carry no float clock; fall back to the string
            # timestamp and sort them before anything with a real one.
            return (0, -1.0, str(e.get("updated_at") or e.get("created_at") or ""))
        out.sort(key=_key)
        return out

    def latest(self) -> Dict[str, Dict[str, Any]]:
        """Event log collapsed to the most recent state per run_id.

        `completed` is sticky: once any worker reports a run finished, a later
        stale `running` heartbeat from a different shard must not resurrect it.
        Without this, a worker whose push landed out of order could cause a
        finished run to be trained a second time.
        """
        st: Dict[str, Dict[str, Any]] = {}
        for e in self.entries():
            rid = e.get("run_id")
            if not rid:
                continue
            prev = st.get(rid)
            if prev is not None and prev.get("state") == "completed" \
                    and e.get("state") != "completed":
                continue
            st[rid] = e
        return st

    def append(self, run_id: str, state: str, **fields) -> None:
        """Record an event in THIS worker's shard. Never touches another's."""
        # `ts` is a float epoch seconds alongside the human-readable timestamp.
        # now_iso() has one-second granularity, and two events landing in the
        # same second would otherwise sort ambiguously ACROSS shards -- which is
        # precisely where ordering has to be trustworthy, because that is how
        # `latest()` decides a run's current state.
        rec = {"run_id": run_id, "state": state, "account": self.account,
               "worker_id": self.worker_id, "session_id": self.session_id,
               "updated_at": now_iso(), "ts": time.time(), **fields}
        with open(self.shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self.hub.enabled:
            self.hub.hub.enqueue(self.shard_path, self.shard_repo_path)

    # ------------------------------ claims ---------------------------------
    @staticmethod
    def _age_sec(ts: Optional[str]) -> float:
        if not ts:
            return 1e18
        try:
            t = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            return max(0.0, time.time() - (t - time.timezone))
        except Exception:
            return 1e18

    def can_claim(self, run_id: str, force: bool = False) -> Tuple[bool, str]:
        """May this worker start (or continue) this run?

        The staleness window exists to stop worker A stealing a run that worker
        B is actively training. It must NOT stop worker A resuming its OWN
        interrupted run -- which is the single most common thing that happens in
        this pipeline. A session pauses at the 8.5-hour limit, you open a fresh
        one two minutes later, and the ledger still says "running, updated 2
        minutes ago". Treating that as a live claim by someone else would make
        the run unresumable for two hours, which defeats the entire resumability
        contract.

        So ownership is checked before freshness:

            same account   -> always allowed. It is your run. A previous session
                              of yours died, or you are deliberately taking over.
            other account  -> the original rule: blocked while the heartbeat is
                              fresh, stealable once it goes stale.
        """
        if force:
            return True, "forced"
        st = self.latest().get(run_id)
        if st is None:
            return True, "unclaimed"
        state = st.get("state")
        if state == "completed":
            return False, "already completed"
        if state in ("running", "paused"):
            owner = st.get("account")
            age = self._age_sec(st.get("updated_at"))
            if owner == self.account:
                same_session = st.get("session_id") == self.session_id
                if same_session:
                    return True, f"continuing this session's own run (state={state})"
                if age < CLAIM_STALE_SEC:
                    # Almost always: your previous Kaggle session died and this
                    # is the new one. Flagged rather than blocked, because the
                    # alternative -- two live sessions on one account with the
                    # same WORKER_ID -- is user error and much rarer.
                    log(f"{run_id} was left '{state}' by an earlier session of "
                        f"{owner} {age/60:.0f} min ago -- resuming it. If you "
                        f"genuinely have two live sessions on this account, give "
                        f"them different WORKER_IDs.", "CLAIM")
                return True, (f"resuming own run from a previous session "
                              f"({age/60:.0f} min ago, state={state})")
            if age < CLAIM_STALE_SEC:
                return False, (f"held by {owner} "
                               f"({age/60:.0f} min ago, state={state})")
            return True, (f"stale claim from {owner} "
                          f"({age/3600:.1f} h) -- taking over")
        return True, f"previous state {state}"

    def claim(self, run_id: str, **fields) -> None:
        cp = self.data_dir / "registry" / "claims" / f"{run_id}.json"
        atomic_write_json(cp, {"run_id": run_id, "account": self.account,
                               "session_id": self.session_id,
                               "started_at": now_iso(),
                               "hostname": platform.node(), **fields})
        if self.hub.enabled:
            self.hub.hub.enqueue(cp, f"registry/claims/{run_id}.json")
        self.append(run_id, "running", **fields)

    def heartbeat(self, run_id: str, run_dir, **fields) -> None:
        """STATUS.json is the heartbeat. Staleness detection depends on it."""
        sp = Path(run_dir) / "STATUS.json"
        atomic_write_json(sp, {"run_id": run_id, "account": self.account,
                               "session_id": self.session_id,
                               "hostname": platform.node(),
                               "updated_at": now_iso(), **fields})
        if self.hub.enabled:
            self.hub.hub.enqueue(sp, f"runs/{run_id}/STATUS.json")

    def finish(self, run_id: str, **metrics) -> None:
        self.append(run_id, "completed", **metrics)

    def pause(self, run_id: str, **fields) -> None:
        self.append(run_id, "paused", **fields)

    def fail(self, run_id: str, error: str) -> None:
        self.append(run_id, "failed", error=error[:500])

    def summary(self) -> "Any":
        rows = [{"run_id": k, **{kk: vv for kk, vv in v.items() if kk != "run_id"}}
                for k, v in sorted(self.latest().items())]
        if pd is None:
            return rows
        return pd.DataFrame(rows)


# =============================================================================
# 4b. worker sharding -- N Kaggle accounts, zero coordination
# =============================================================================
# Ported from the NB05 generator pipeline, where it cut a multi-day job to a
# fraction of the wall-clock across parallel accounts.
#
# The idea, in one line: DECIDE OWNERSHIP BY ARITHMETIC, NOT BY NEGOTIATION.
#
#     owner(run_id) = sha256(run_id) % NUM_WORKERS
#
# Every worker computes the same function over the same universe of work and
# keeps only the slice that hashes to its own WORKER_ID. This gives three
# properties for free, none of which requires the workers to talk to each other:
#
#   no overlap  two workers can never pick the same run, because a hash has
#               exactly one value
#   no gaps     every run hashes to SOME worker, so nothing is orphaned
#   restart-proof  ownership depends only on the id, not on start time, not on
#               how far anyone else has got, not on who crashed
#
# Compare with the claim protocol in RunRegistry, which needs a shared ledger, a
# heartbeat, and a staleness window. That is still here and still useful -- but
# as a SAFETY NET for taking over dead workers, not as the primary mechanism.
# Sharding is what makes six accounts safe by default; claims are what let you
# recover when one of them dies.
#
# The one thing that must stay fixed is NUM_WORKERS. Changing it re-shuffles
# every assignment. That is not a correctness problem -- global progress is read
# from HF, so already-finished runs are skipped by everyone -- but it does mean
# a worker's slice changes shape mid-project. `WorkerPlan.describe()` prints the
# assignment so you can see it.

def hash_owner(key: str, num_workers: int) -> int:
    """Deterministic worker assignment. Same answer on every machine, forever."""
    if num_workers <= 1:
        return 0
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) % int(num_workers)


# --------------------------------------------------------------------------
# Balancing: hash sharding is uniform only IN EXPECTATION
# --------------------------------------------------------------------------
# Pure hashing is the right tool when the universe is huge and open-ended --
# 10,000 images, ids arriving over time, workers joining late. That is the NB05
# situation and hashing is perfect there.
#
# The MSC atlas is the opposite situation: a small, fixed, known-in-advance
# universe (45 runs) whose members differ enormously in cost. Hashing 45 items
# into 6 buckets gives splits like [11, 7, 4, 10, 3, 10] -- a 3.7x imbalance.
# At ~3 h per run that is one account working 33 hours while another finishes in
# 9 and sits idle. The wall-clock of the whole phase is set by the SLOWEST
# worker, so that imbalance is a direct, pure loss.
#
# Worse, the cost spread is not uniform either: a resnet20 for 240 epochs is
# maybe 1 GPU-hour; a vit_tiny for 300 epochs is closer to 6. Balancing the
# COUNT of runs still leaves the wall-clock unbalanced.
#
# So we offer three modes and default to the one that balances TIME:
#
#   "hash"      NB05 behaviour. Stateless, open-universe, unbalanced.
#   "balanced"  Deterministic round-robin over the sorted universe. Counts
#               differ by at most 1.
#   "cost"      Longest-processing-time-first bin packing on estimated GPU
#               cost. Balances hours, not items. DEFAULT.
#
# All three are deterministic: every worker computes the same assignment from
# the same inputs with no communication. "cost" and "balanced" additionally
# require every worker to see the same universe list, which they do because it
# is generated from the same config code.

# Relative GPU cost per epoch, normalised so resnet20 = 1.0.
#
# CALIBRATED against real Phase 0 timings on a Kaggle T4 (2026-08-02):
#   resnet32x4  240 epochs in 10,389 s  ->  43.3 s/epoch
#   wrn_40_2    240 epochs in  6,758 s  ->  28.2 s/epoch
#
# Those two fix both the scale and the ratio. The first-guess table predicted
# 1.73 h for the resnet32x4 run that actually took 2.89 h -- a 40% underestimate,
# which matters when the whole point of these numbers is telling you how long a
# phase will take before you commit to it.
#
# The rest remain estimates. `estimate_costs_from_history` replaces any entry
# with a measured median as soon as that architecture has finished a run, so the
# table self-corrects as the atlas progresses.
MEASURED_ARCHS = frozenset({"resnet32x4", "wrn_40_2"})

ARCH_COST_HINT: Dict[str, float] = {
    "resnet20": 1.0, "resnet56": 2.4, "resnet110": 4.6,
    "resnet8x4": 1.6, "resnet32x4": 5.2,          # measured
    "wrn_40_2": 3.38, "wrn_16_2": 1.3, "wrn_40_1": 1.7,   # wrn_40_2 measured
    "vgg13": 3.4, "vgg8": 1.8,
    "mobilenetv2": 3.0, "shufflenetv2": 2.2,
    "convnext_femto": 6.0, "vit_tiny": 7.5, "mixer_nano": 4.0,
}

# Seconds of T4 wall-clock per cost-unit-epoch. Derived from the anchor above:
#   10,389 s / (240 epochs x 5.2 units) = 8.32
SECONDS_PER_COST_UNIT = 8.32


def estimate_run_hours(run_id: str, epochs_hint: Optional[int] = None,
                       costs: Optional[Dict[str, float]] = None) -> float:
    """Estimated wall-clock hours for one run on a single T4."""
    return (estimate_run_cost(run_id, epochs_hint, costs)
            * SECONDS_PER_COST_UNIT / 3600.0)


def estimate_phase(run_ids: Sequence[str], num_workers: int = 1,
                   costs: Optional[Dict[str, float]] = None,
                   session_limit_h: float = 8.5) -> Dict[str, Any]:
    """Total GPU-hours, wall-clock at N workers, and sessions needed.

    Wall-clock is NOT total/N: work is assigned in whole runs, so the phase ends
    when the busiest worker does. This uses the same cost-balanced packing the
    scheduler uses, so the number matches what will actually happen.
    """
    costs = costs or ARCH_COST_HINT
    per_run = {r: estimate_run_hours(r, costs=costs) for r in run_ids}
    total = float(sum(per_run.values()))
    owner = assign_workers(list(run_ids), max(1, num_workers), mode="cost",
                           costs=costs)
    loads = [sum(per_run[r] for r, w in owner.items() if w == i)
             for i in range(max(1, num_workers))]
    wall = max(loads) if loads else 0.0
    n_measured = sum(1 for r in run_ids
                     if str(r).split("-")[1] in MEASURED_ARCHS)
    return {
        "n_runs": len(run_ids), "total_gpu_hours": total,
        "wall_clock_hours": wall, "per_worker_hours": loads,
        "sessions_needed": int(math.ceil(wall / session_limit_h)) if wall else 0,
        "per_run_hours": per_run, "num_workers": max(1, num_workers),
        "frac_measured": (n_measured / len(run_ids)) if run_ids else 0.0,
    }


def estimate_run_cost(run_id: str, epochs_hint: Optional[int] = None,
                      costs: Optional[Dict[str, float]] = None) -> float:
    """Relative cost of a run, in arbitrary units proportional to GPU-time.

    Parsed from the run_id so this works with nothing but a list of names --
    the scheduler must not need checkpoints or configs to plan.
    """
    costs = costs or ARCH_COST_HINT
    parts = str(run_id).split("-")
    arch = parts[1] if len(parts) > 1 else ""
    per_epoch = costs.get(arch, float(np.median(list(costs.values()))))
    ep = epochs_hint if epochs_hint else (300 if arch in TRANSFORMER_LIKE else 240)
    return float(per_epoch) * float(ep)


def estimate_costs_from_history(data_dir) -> Dict[str, float]:
    """Replace the hints with measured seconds-per-epoch, once we have them.

    After the first few runs finish, real timings exist in history.csv and are
    strictly better than any hint. This makes the scheduler self-correcting:
    the more of the atlas you have run, the better it balances the rest.
    """
    out: Dict[str, List[float]] = {}
    logs = Path(data_dir) / "runs"
    if pd is None or not logs.exists():
        return {}
    for d in logs.iterdir():
        h = d / "metrics" / "epochs.csv"
        if not (d.is_dir() and h.exists()):
            continue
        try:
            df = pd.read_csv(h)
            if df.empty or "epoch_time_sec" not in df:
                continue
            arch = (df["arch"].iloc[0] if "arch" in df.columns
                    else d.name.split("-")[1])
            out.setdefault(str(arch), []).append(float(df["epoch_time_sec"].median()))
        except Exception:
            continue
    if not out:
        return {}
    med = {a: float(np.median(v)) for a, v in out.items()}
    base = med.get("resnet20") or min(med.values())
    return {a: v / max(1e-9, base) for a, v in med.items()}


def assign_workers(run_ids: Sequence[str], num_workers: int,
                   mode: str = "cost",
                   costs: Optional[Dict[str, float]] = None,
                   epochs_hint: Optional[Dict[str, int]] = None
                   ) -> Dict[str, int]:
    """run_id -> worker_id, deterministically, for the whole universe.

    Every worker calls this with identical arguments and reads off its own
    slice. No communication, no locking, no negotiation.

    `costs` MUST be a stable table -- in practice, always leave it None so
    ARCH_COST_HINT is used. Passing measured timings here makes the assignment
    depend on how much of the project has finished, which means two sessions of
    the same worker can disagree about what it owns. Use estimate_phase() if you
    want time predictions refined by measurements; that is a display concern and
    has no effect on ownership.
    """
    ids = sorted(run_ids)                       # canonical order on every machine
    n = max(1, int(num_workers))
    if n == 1:
        return {r: 0 for r in ids}

    if mode == "hash":
        return {r: hash_owner(r, n) for r in ids}

    if mode == "balanced":
        return {r: i % n for i, r in enumerate(ids)}

    if mode == "cost":
        # Longest-processing-time-first: sort by descending cost and repeatedly
        # give the next job to whichever worker currently has the least work.
        # A classic greedy scheduler with a (4/3 - 1/3n) worst-case bound -- and
        # in practice, on this kind of input, near-perfect.
        eh = epochs_hint or {}
        jobs = sorted(ids, key=lambda r: (-estimate_run_cost(r, eh.get(r), costs), r))
        load = [0.0] * n
        owner: Dict[str, int] = {}
        for r in jobs:
            w = int(np.argmin(load))
            owner[r] = w
            load[w] += estimate_run_cost(r, eh.get(r), costs)
        return owner

    raise ValueError(f"unknown shard mode '{mode}' (use hash / balanced / cost)")


@dataclass
class WorkerPlan:
    """What THIS worker should do, given the whole universe of work.

    universe -> mine (hash-owned slice) -> todo (mine, minus what is already
    finished anywhere). `done` is read from HuggingFace and is GLOBAL: if
    another account already finished one of my runs, I skip it.
    """
    worker_id: int
    num_workers: int
    universe: List[str]
    mine: List[str]
    done: Set[str]
    todo: List[str]
    stolen: List[str] = field(default_factory=list)
    in_progress_elsewhere: List[str] = field(default_factory=list)
    mode: str = "cost"
    stage: str = "train"
    est_cost: float = 0.0

    @property
    def work(self) -> List[str]:
        """Everything to attempt this session: my slice first, then any stolen."""
        return list(self.todo) + list(self.stolen)

    def describe(self, title: str = "work plan") -> None:
        print(f"\n{'='*74}")
        print(f"  {title}   worker {self.worker_id} of {self.num_workers}"
              f"   (stage: {self.stage}, split: {self.mode})")
        print(f"{'='*74}")
        print(f"  universe (all runs in this phase) : {len(self.universe)}")
        print(f"  my slice                          : {len(self.mine)}"
              f"   (~{self.est_cost * SECONDS_PER_COST_UNIT / 3600.0:.1f} GPU-h estimated)")
        print(f"  already finished (GLOBAL, from HF): {len(self.done)}"
              f"   <- for the '{self.stage}' stage")
        print(f"  MY REMAINING WORK                 : {len(self.todo)}")
        if self.in_progress_elsewhere:
            print(f"  live on another worker (skipped)  : {len(self.in_progress_elsewhere)}")
        if self.stolen:
            print(f"  stale, taken over from a dead run : {len(self.stolen)}")
        print(f"{'-'*74}")
        for r in self.work:
            tag = "STOLEN" if r in self.stolen else "mine"
            print(f"    [{tag:6s}] {r}")
        if not self.work:
            print("    (nothing to do -- either finished, or owned by other workers)")
        print(f"{'='*74}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {"worker_id": self.worker_id, "num_workers": self.num_workers,
                "n_universe": len(self.universe), "n_mine": len(self.mine),
                "n_done_global": len(self.done), "n_todo": len(self.todo),
                "n_stolen": len(self.stolen), "mine": self.mine, "todo": self.todo,
                "stolen": self.stolen, "planned_utc": now_iso()}


def plan_work(run_ids: Sequence[str], registry: "RunRegistry",
              worker_id: int = 0, num_workers: int = 1,
              steal_stale: bool = True, mode: str = "cost",
              costs: Optional[Dict[str, float]] = None,
              done_states: Sequence[str] = ("completed",),
              done_fn: Optional[Callable[[str], bool]] = None,
              stage: str = "train") -> WorkerPlan:
    """Build this worker's plan. Call it right before the training loop.

    `steal_stale=True` means: after my own slice is exhausted, also pick up runs
    owned by OTHER workers whose claim has gone stale (>2 h without a
    heartbeat). That is how a dead account's share gets finished without anyone
    intervening. It is deliberately second in priority -- you always do your own
    work first, so two live workers never fight over the same run.

    Stealing is also what rescues an unlucky split: if the estimated costs were
    wrong and one worker finishes early, it starts absorbing stalled work
    instead of idling.
    """
    assert 0 <= worker_id < num_workers, \
        f"WORKER_ID must be in 0..{num_workers-1}, got {worker_id}"
    registry.pull()
    latest = registry.latest()

    universe = list(run_ids)
    owner = assign_workers(universe, num_workers, mode=mode, costs=costs)
    mine = [r for r in universe if owner.get(r) == worker_id]

    # WHAT COUNTS AS DONE DEPENDS ON THE STAGE.
    #
    # A run passes through several stages -- train, then measure, then method --
    # but the ledger carries one state per run. Asking "is state == completed?"
    # from the measurement notebook therefore returns True because TRAINING
    # completed, and the measurement stage plans zero work and exits in seconds
    # looking like a success. That is exactly what happened on the first real
    # Phase 0 run.
    #
    # So the caller supplies a predicate for its own stage. The training stage
    # uses ledger state; the measurement stage asks whether the per-sample
    # tables actually exist, which is both stage-correct and robust to a lost
    # ledger event -- the same "trust the artifacts, not the status file"
    # principle used when repairing progress on resume.
    if done_fn is not None:
        done = {r for r in universe if done_fn(r)}
    else:
        done = {r for r in universe
                if latest.get(r, {}).get("state") in done_states}
    todo = [r for r in mine if r not in done]

    stolen, live_elsewhere = [], []
    if steal_stale and num_workers > 1:
        for r in universe:
            if r in done or owner.get(r) == worker_id:
                continue
            st = latest.get(r)
            if st is None:
                continue                       # never started; leave it to its owner
            if st.get("state") in ("running", "paused"):
                if registry._age_sec(st.get("updated_at")) >= CLAIM_STALE_SEC:
                    stolen.append(r)
                else:
                    live_elsewhere.append(r)

    p = WorkerPlan(worker_id=worker_id, num_workers=num_workers,
                   universe=universe, mine=mine, done=done, todo=todo,
                   stolen=stolen, in_progress_elsewhere=live_elsewhere)
    p.stage = stage
    p.mode = mode
    p.est_cost = sum(estimate_run_cost(r, costs=costs) for r in mine)
    return p


def shard_report(run_ids: Sequence[str], num_workers: int, mode: str = "cost",
                 costs: Optional[Dict[str, float]] = None) -> "Any":
    """How the universe splits, and -- more importantly -- how balanced it is.

    Print this BEFORE starting a long phase. The wall-clock of the phase is set
    by the slowest worker, so a 3x imbalance is a 3x-longer phase, and it is
    much cheaper to notice now than on day four.
    """
    owner = assign_workers(run_ids, num_workers, mode=mode, costs=costs)
    rows = [{"run_id": r, "owner": owner[r],
             "est_cost": estimate_run_cost(r, costs=costs),
             "arch": str(r).split("-")[1] if "-" in str(r) else "?"}
            for r in sorted(run_ids)]
    if pd is None:
        return rows
    df = pd.DataFrame(rows)
    df["est_hours"] = df.est_cost * SECONDS_PER_COST_UNIT / 3600.0
    g = (df.groupby("owner")
           .agg(n_runs=("run_id", "count"), est_hours=("est_hours", "sum"),
                archs=("arch", lambda s: ", ".join(sorted(set(s)))))
           .reset_index().sort_values("owner"))
    g["est_hours"] = g.est_hours.round(1)
    lo, hi = g.est_hours.min(), g.est_hours.max()
    print(f"\n  shard mode = '{mode}'   workers = {num_workers}")
    print(f"  estimated wall-clock: {hi:.1f} h (slowest worker sets the phase)")
    print(f"  imbalance: {hi/max(1e-9, lo):.2f}x between fastest and slowest")
    if hi / max(1e-9, lo) > 1.5:
        print("  ^ consider mode='cost', or a different worker count")
    print(f"  total GPU-hours across all workers: {g.est_hours.sum():.1f} h\n")
    return g


# =============================================================================
# 5. lifecycle -- interrupt / SIGTERM / atexit / session watchdog
# =============================================================================
class LifecycleGuard:
    """Guarantees a final push on every way a Kaggle session can end.

    Four exits are handled:
        KeyboardInterrupt  -- you pressed stop
        SIGTERM            -- Kaggle is about to kill the session; it sends this
                              first, and those seconds are enough for one commit
        atexit             -- normal or exceptional interpreter shutdown
        watchdog           -- elapsed > session_limit_h, push and mark paused
                              BEFORE the platform intervenes

    E2AM caught only KeyboardInterrupt. On Kaggle the common death is SIGTERM at
    the 9-12 hour boundary, which that misses entirely -- and losing the last
    30 minutes of a 3-hour run is exactly the outcome the push policy exists to
    prevent.
    """

    def __init__(self, on_flush: Callable[[str], None],
                 session_limit_h: float = 8.5, verbose: bool = True):
        self.on_flush = on_flush
        self.session_limit_sec = session_limit_h * 3600.0
        self.started = time.time()
        self.verbose = verbose
        self._fired = threading.Event()
        self._prev_sigterm = None
        self._prev_sigint = None
        self._installed = False

    def install(self) -> "LifecycleGuard":
        if self._installed:
            return self
        try:
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            pass
        atexit.register(self._handle_atexit)
        self._installed = True
        if self.verbose:
            log(f"lifecycle guard armed (SIGTERM + atexit, "
                f"session limit {self.session_limit_sec/3600:.1f} h)", "LIFE")
        return self

    def _fire(self, reason: str) -> None:
        if self._fired.is_set():
            return
        self._fired.set()
        try:
            print(f"\n[LIFE] {reason} -- flushing everything to HuggingFace now")
            self.on_flush(reason)
        except Exception:
            traceback.print_exc()

    def _handle_signal(self, signum, frame):
        self._fire(f"SIGTERM ({signum})")
        if callable(self._prev_sigterm):
            try:
                self._prev_sigterm(signum, frame)
            except Exception:
                pass
        raise KeyboardInterrupt(f"SIGTERM received at {now_iso()}")

    def _handle_atexit(self):
        self._fire("interpreter exit")

    @property
    def elapsed_h(self) -> float:
        return (time.time() - self.started) / 3600.0

    def session_expiring(self) -> bool:
        return (time.time() - self.started) >= self.session_limit_sec

    def rearm(self) -> None:
        """Allow the guard to fire again after a handled interruption."""
        self._fired.clear()


# =============================================================================
# 6. data -- CIFAR-100 from the Kaggle mirror
# =============================================================================
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _has_cifar100(root: Path) -> bool:
    p = Path(root) / "cifar-100-python"
    return p.is_dir() and (p / "train").exists() and (p / "test").exists()


def locate_cifar100(prefer_scratch: bool = True, verbose: bool = True) -> Path:
    """Find or fetch CIFAR-100, preferring sources in this order:

        1. any attached Kaggle input dataset          (instant, no download)
        2. a previous extraction under scratch        (instant)
        3. the team's Kaggle mirror via the CLI       (in-datacentre, fast)
        4. torchvision auto-download                  (last resort, slow)

    Extraction target is /kaggle/temp, never /kaggle/working: the 20 GB working
    disk is artifact space, and a CIFAR-100 tarball plus its extraction is a
    meaningful bite out of it for no reason.
    """
    def _say(m):
        if verbose:
            log(m, "DATA")

    # 1. attached Kaggle datasets
    inp = Path("/kaggle/input")
    if inp.exists():
        candidates = [inp / "dataset-cifar100-python", inp / "cifar100",
                      inp / "cifar-100", inp / "cifar100-python"]
        candidates += [p for p in inp.iterdir() if p.is_dir()]
        for base in candidates:
            if _has_cifar100(base):
                _say(f"found attached Kaggle dataset at {base}")
                return Path(base)
            # Mirrors sometimes nest one level deeper.
            if base.is_dir():
                for sub in base.iterdir():
                    if sub.is_dir() and _has_cifar100(sub):
                        _say(f"found attached Kaggle dataset at {sub}")
                        return sub

    data_root = ensure_dir((SCRATCH_ROOT if prefer_scratch else WORK_ROOT) / "data")

    # 2. previous extraction
    if _has_cifar100(data_root):
        _say(f"reusing extraction at {data_root}")
        return data_root

    # 3. Kaggle CLI against the team's mirror
    _say(f"not found locally -- downloading {KAGGLE_CIFAR100_SLUG} via Kaggle CLI")
    try:
        rc, _, _ = shell(["kaggle", "--version"], timeout=30)
        if rc != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle",
                            "--break-system-packages"], check=False, timeout=180)
        for slug in (KAGGLE_CIFAR100_SLUG, "melikechan/cifar100", "fedesoriano/cifar100"):
            try:
                _say(f"  kaggle datasets download -d {slug}")
                r = subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                                    "-p", str(data_root), "--unzip"],
                                   capture_output=True, text=True, timeout=900)
                if r.returncode != 0:
                    _say(f"  {slug}: {r.stderr.strip()[:180]}")
                    continue
                if _has_cifar100(data_root):
                    _say(f"  extracted to {data_root}")
                    return data_root
                # Extracted one level deep -- promote it so torchvision finds it.
                for sub in data_root.rglob("cifar-100-python"):
                    if (sub / "train").exists():
                        target = data_root / "cifar-100-python"
                        if sub.resolve() != target.resolve():
                            shutil.move(str(sub), str(target))
                        if _has_cifar100(data_root):
                            _say(f"  promoted nested extraction to {data_root}")
                            return data_root
            except Exception as e:
                _say(f"  {slug} failed: {e}")
    except Exception as e:
        _say(f"kaggle CLI unavailable: {e}")

    # 4. torchvision
    _say("falling back to torchvision auto-download")
    from torchvision.datasets import CIFAR100 as _TVC100
    _TVC100(root=str(data_root), train=True, download=True)
    _TVC100(root=str(data_root), train=False, download=True)
    if not _has_cifar100(data_root):
        raise RuntimeError(
            "Could not obtain CIFAR-100 from any source. Attach "
            f"https://www.kaggle.com/datasets/{KAGGLE_CIFAR100_SLUG} to the notebook.")
    _say(f"downloaded to {data_root}")
    return data_root


class CIFARTensor(Dataset):
    """Whole dataset resident in a uint8 tensor; augmentation on the fly.

    50k x 32 x 32 x 3 is ~150 MB as uint8, so num_workers=0 with in-memory
    indexing beats a worker pool -- no IPC, no pickling, no worker startup on
    every epoch. That matters here because the oracle sweep re-reads the test
    set fifteen times per model (5 depth x 5 resolution x 5 precision configs).

    IMPORTANT: the test set is never shuffled and never augmented, so
    `sample_idx` is the canonical order that every per-sample table is aligned
    to. Do not add a shuffle to the eval loader.
    """

    def __init__(self, data_root, dataset: str = "cifar100", train: bool = True,
                 augment: bool = True):
        import pickle
        dataset = dataset.lower()
        folder = "cifar-100-python" if dataset == "cifar100" else "cifar-10-batches-py"
        root = Path(data_root) / folder
        self.dataset = dataset
        self.train = train
        self.augment = augment and train

        if dataset == "cifar100":
            fn = root / ("train" if train else "test")
            with open(fn, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            data = d["data"]
            labels = np.asarray(d["fine_labels"], dtype=np.int64)
            meta = root / "meta"
            with open(meta, "rb") as f:
                m = pickle.load(f, encoding="latin1")
            self.classes = list(m["fine_label_names"])
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        else:
            files = ([f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"])
            chunks, labs = [], []
            for fn in files:
                with open(root / fn, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
                chunks.append(d["data"])
                labs.extend(d["labels"])
            data = np.concatenate(chunks, axis=0)
            labels = np.asarray(labs, dtype=np.int64)
            with open(root / "batches.meta", "rb") as f:
                m = pickle.load(f, encoding="latin1")
            self.classes = list(m["label_names"])
            mean, std = CIFAR10_MEAN, CIFAR10_STD

        images = data.reshape(-1, 3, 32, 32)
        self.images = torch.from_numpy(np.ascontiguousarray(images))          # uint8 CHW
        self.labels = torch.from_numpy(labels)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        # Fingerprint the label order once. Every per-sample table carries it,
        # and the analysis refuses to correlate tables whose fingerprints differ.
        self.order_hash = sha256_of_array(labels)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def _normalize(self, img_u8: "torch.Tensor") -> "torch.Tensor":
        x = img_u8.float().div_(255.0)
        return (x - self.mean) / self.std

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.augment:
            # Standard CIFAR recipe: 4px reflect pad + random crop, hflip.
            img = F.pad(img.unsqueeze(0).float(), (4, 4, 4, 4), mode="reflect").squeeze(0)
            i = int(torch.randint(0, 9, (1,)).item())
            j = int(torch.randint(0, 9, (1,)).item())
            img = img[:, i:i + 32, j:j + 32]
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            x = img.div(255.0)
            x = (x - self.mean) / self.std
        else:
            x = self._normalize(img.clone())
        # sample_idx travels with the batch so the oracle can write rows back
        # in canonical order regardless of loader ordering.
        return x, int(self.labels[idx]), int(idx)


def build_loaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any, List[str], str]:
    """train / val(test) / train-holdout loaders.

    The train-holdout is a fixed 5,000-sample slice of the training set,
    evaluated with augmentation off. It costs one extra inference sweep and
    answers a free question: does MSC structure look different on data the
    model has already seen?
    """
    data_root = cfg["data_root"]
    ds = str(cfg.get("dataset_name", "cifar100"))
    bs = int(cfg.get("batch_size", 64))
    eval_bs = int(cfg.get("eval_batch_size", 512))

    train_set = CIFARTensor(data_root, ds, train=True, augment=True)
    test_set = CIFARTensor(data_root, ds, train=False, augment=False)
    train_clean = CIFARTensor(data_root, ds, train=True, augment=False)

    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 1)))

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False,
                              generator=g)
    # Never shuffle eval loaders. sample_idx alignment depends on it.
    val_loader = DataLoader(test_set, batch_size=eval_bs, shuffle=False,
                            num_workers=0, pin_memory=True)

    n_hold = int(cfg.get("train_holdout_n", 5000))
    rng = np.random.default_rng(12345)                 # fixed across ALL runs
    hold_idx = np.sort(rng.choice(len(train_clean), size=min(n_hold, len(train_clean)),
                                  replace=False))
    holdout = torch.utils.data.Subset(train_clean, hold_idx.tolist())
    holdout_loader = DataLoader(holdout, batch_size=eval_bs, shuffle=False,
                                num_workers=0, pin_memory=True)

    return (train_loader, val_loader, holdout_loader,
            train_set.classes, test_set.order_hash)


# =============================================================================
# 7. zoo -- 13 architectures behind one staged interface
# =============================================================================
# Every backbone in this project must answer three questions identically,
# regardless of whether it is a ResNet or an MLP-Mixer:
#
#   forward(x)              -> logits at full compute
#   forward_features(x)     -> list of K intermediate feature tensors
#   forward_prefix(x, k)    -> features after only the first k stages
#
# forward_prefix is what makes the depth axis honest. An early exit that still
# runs the whole backbone and merely reads a mid-layer activation costs full
# compute; the FLOPs saving it claims would be fictional. Exiting at stage k
# must actually stop at stage k.
#
# Feature tensors are (B, C, H, W) for convolutional families and (B, N, C) for
# ViT / Mixer. ExitHead dispatches on rank, so nothing downstream cares.

if _TORCH_OK:

    class StagedBackbone(nn.Module):
        """Stem + ordered blocks partitioned into K stages + classifier.

        The partition is by *fraction of blocks*, matching
        01_PHASE0_GO_NOGO.md 3: exits at {0.2, 0.4, 0.6, 0.8, 1.0} of depth.
        Partitioning by block count rather than by parameter count is the right
        choice because the depth axis is about how far the computation got, and
        because it makes the exit points comparable across architectures with
        very different width profiles.
        """

        is_token_model = False
        # Can this architecture run at an input resolution other than 32x32?
        # Convolutional backbones can. Token models with a learned positional
        # embedding can only if that embedding is interpolated, and MLP-Mixer
        # cannot at all -- see MixerBackbone.
        supports_native_resolution = True

        def __init__(self, stem: nn.Module, blocks: Sequence[nn.Module],
                     classifier: nn.Module, feature_dim_fn: Callable[[int], int],
                     depth_fractions: Sequence[float] = DEPTH_FRACTIONS,
                     final_norm: Optional[nn.Module] = None):
            super().__init__()
            self.stem = stem
            self.blocks = nn.ModuleList(blocks)
            self.classifier = classifier
            self.final_norm = final_norm
            n = len(self.blocks)

            # Cut points are the *inclusive* last block index of each stage.
            #
            # K is ADAPTIVE, not fixed at 5. A network with fewer blocks than
            # requested exits cannot have five distinct depth budgets --
            # resnet8x4 has only 3 blocks, so asking for exits at
            # {0.2,0.4,0.6,0.8,1.0} produces cuts (1,2,3,3,3) and hence
            # rho = [0.295, 0.648, 1.0, 1.0, 1.0].
            #
            # Those duplicate 1.0 entries are not a cosmetic problem. The MSC
            # oracle requires strictly ascending costs (msc_core.compute_msc
            # raises on non-ascending rho), because "the smallest sufficient
            # budget" is ill-defined when two budgets cost the same. Silently
            # emitting duplicates would have crashed the oracle three hours into
            # Phase 1b, or -- worse -- produced an MSC that depends on which of
            # several identical budgets argmax happened to return.
            #
            # So we take as many distinct cuts as the depth allows and record
            # the fractions we actually achieved. Cross-architecture comparison
            # is unaffected: MSC is a cost FRACTION in (0,1], not an exit index,
            # so architectures may legitimately carry different K.
            cuts, prev = [], 0
            for fr in depth_fractions:
                c = min(n, max(prev + 1, int(round(fr * n))))
                if c > prev:
                    cuts.append(c)
                    prev = c
                if prev >= n:
                    break
            if not cuts or cuts[-1] != n:
                cuts.append(n)
            seen, uniq = set(), []
            for c in cuts:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)

            self.stage_cuts = tuple(uniq)
            self.requested_depth_fractions = tuple(depth_fractions)
            self.depth_fractions = tuple(c / n for c in uniq)
            self.feature_dims = tuple(feature_dim_fn(c - 1) for c in self.stage_cuts)
            if len(uniq) < len(depth_fractions):
                log(f"{type(self).__name__} has only {n} blocks -- using "
                    f"K={len(uniq)} depth exits at "
                    f"{[round(f,2) for f in self.depth_fractions]} instead of "
                    f"{list(depth_fractions)}", "ZOO")

        def _run_to(self, x, upto_block: int):
            x = self.stem(x)
            for i in range(upto_block):
                x = self.blocks[i](x)
            return x

        def forward_prefix(self, x, k: int):
            """Features after stage k only. Stops early -- really."""
            k = max(0, min(k, len(self.stage_cuts) - 1))
            return self._run_to(x, self.stage_cuts[k])

        def forward_features(self, x) -> List["torch.Tensor"]:
            feats, h, prev = [], self.stem(x), 0
            for c in self.stage_cuts:
                for i in range(prev, c):
                    h = self.blocks[i](h)
                prev = c
                feats.append(h)
            return feats

        def pooled(self, feat):
            if feat.dim() == 4:
                return F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat.mean(dim=1)            # (B, N, C) -> (B, C)

        def forward(self, x):
            h = self._run_to(x, len(self.blocks))
            if self.final_norm is not None:
                h = self.final_norm(h)
            return self.classifier(self.pooled(h))

    # ---------------------------------------------------------------- ResNet
    class _BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.short = nn.Sequential()
            if stride != 1 or cin != cout:
                self.short = nn.Sequential(
                    nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            return F.relu(out + self.short(x), inplace=True)

    def build_resnet_cifar(depth: int, width_mult: int = 1,
                           num_classes: int = 100) -> StagedBackbone:
        """CIFAR ResNet as used by CRD / DKD / mdistiller.

        depth in {8, 20, 32, 56, 110}; width_mult=4 gives the x4 variants.
        These exact configurations are what the published benchmark numbers in
        02_ENGINEERING_SPEC.md 7 refer to, so reproducing them is how we know
        the recipe is right before generating any MSC table.
        """
        assert (depth - 2) % 6 == 0, f"CIFAR ResNet depth must be 6n+2, got {depth}"
        n = (depth - 2) // 6
        widths = [16 * width_mult, 32 * width_mult, 64 * width_mult]
        stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        blocks, dims, cin = [], [], 16
        for gi, w in enumerate(widths):
            for bi in range(n):
                stride = 2 if (gi > 0 and bi == 0) else 1
                blocks.append(_BasicBlock(cin, w, stride))
                cin = w
                dims.append(w)
        return StagedBackbone(stem, blocks, nn.Linear(cin, num_classes),
                              lambda i: dims[i])

    # ----------------------------------------------------------- WideResNet
    class _WideBlock(nn.Module):
        """Pre-activation wide block (Zagoruyko & Komodakis)."""

        def __init__(self, cin, cout, stride, drop=0.0):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.drop = drop
            self.equal = (cin == cout and stride == 1)
            self.short = None if self.equal else nn.Conv2d(cin, cout, 1, stride, bias=False)

        def forward(self, x):
            o = F.relu(self.bn1(x), inplace=True)
            s = x if self.equal else self.short(o)
            o = self.conv1(o)
            o = F.relu(self.bn2(o), inplace=True)
            if self.drop > 0:
                o = F.dropout(o, self.drop, self.training)
            return self.conv2(o) + s

    def build_wrn(depth: int, widen: int, num_classes: int = 100) -> StagedBackbone:
        assert (depth - 4) % 6 == 0, f"WRN depth must be 6n+4, got {depth}"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False))
        blocks, dims, cin = [], [], 16
        for gi in range(3):
            for bi in range(n):
                stride = 2 if (gi > 0 and bi == 0) else 1
                blocks.append(_WideBlock(cin, widths[gi + 1], stride))
                cin = widths[gi + 1]
                dims.append(cin)
        final_norm = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        return StagedBackbone(stem, blocks, nn.Linear(cin, num_classes),
                              lambda i: dims[i], final_norm=final_norm)

    # ------------------------------------------------------------------ VGG
    _VGG_CFG = {
        13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
        8:  [64, "M", 128, "M", 256, "M", 512, "M", 512],
        11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    }

    def build_vgg(depth: int, num_classes: int = 100) -> StagedBackbone:
        """CIFAR VGG with batch norm, no residuals.

        Present specifically because H3 predicts across-CNN-family transfer
        sits between within-family and CNN->ViT. A CNN without skip connections
        is the intermediate point that makes that ordering testable.
        """
        cfg = _VGG_CFG[depth]
        blocks, dims, cin = [], [], 3
        for v in cfg:
            if v == "M":
                blocks.append(nn.MaxPool2d(2, 2))
                dims.append(cin)
            else:
                blocks.append(nn.Sequential(nn.Conv2d(cin, v, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(v), nn.ReLU(inplace=True)))
                cin = v
                dims.append(cin)
        return StagedBackbone(nn.Identity(), blocks, nn.Linear(cin, num_classes),
                              lambda i: dims[i])

    # ---------------------------------------------------------- MobileNetV2
    class _InvertedResidual(nn.Module):
        def __init__(self, cin, cout, stride, expand):
            super().__init__()
            hidden = cin * expand
            self.use_res = (stride == 1 and cin == cout)
            layers = []
            if expand != 1:
                layers += [nn.Conv2d(cin, hidden, 1, bias=False),
                           nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
            layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                       nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.conv(x) if self.use_res else self.conv(x)

    def build_mobilenetv2(num_classes: int = 100, width: float = 1.0) -> StagedBackbone:
        # CIFAR adaptation: stem stride 1 and the first two stages kept at 32px,
        # otherwise a 32x32 input is down to 1x1 before the network has done
        # anything.
        cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
               (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        c0 = int(32 * width)
        stem = nn.Sequential(nn.Conv2d(3, c0, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(c0), nn.ReLU6(inplace=True))
        blocks, dims, cin = [], [], c0
        for t, c, n, s in cfg:
            cout = int(c * width)
            for i in range(n):
                blocks.append(_InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
                dims.append(cin)
        last = int(1280 * max(1.0, width))
        blocks.append(nn.Sequential(nn.Conv2d(cin, last, 1, bias=False),
                                    nn.BatchNorm2d(last), nn.ReLU6(inplace=True)))
        dims.append(last)
        return StagedBackbone(stem, blocks, nn.Linear(last, num_classes),
                              lambda i: dims[i])

    # --------------------------------------------------------- ShuffleNetV2
    def _channel_shuffle(x, groups: int):
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    class _ShuffleUnit(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.stride = stride
            branch = cout // 2
            if stride > 1:
                self.b1 = nn.Sequential(
                    nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                    nn.BatchNorm2d(cin),
                    nn.Conv2d(cin, branch, 1, bias=False),
                    nn.BatchNorm2d(branch), nn.ReLU(inplace=True))
                b2in = cin
            else:
                self.b1 = None
                b2in = cin // 2
            self.b2 = nn.Sequential(
                nn.Conv2d(b2in, branch, 1, bias=False),
                nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, stride, 1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False),
                nn.BatchNorm2d(branch), nn.ReLU(inplace=True))

        def forward(self, x):
            if self.stride > 1:
                out = torch.cat([self.b1(x), self.b2(x)], 1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat([x1, self.b2(x2)], 1)
            return _channel_shuffle(out, 2)

    def build_shufflenetv2(num_classes: int = 100, width: str = "1.0x") -> StagedBackbone:
        chans = {"0.5x": [48, 96, 192, 1024], "1.0x": [116, 232, 464, 1024],
                 "1.5x": [176, 352, 704, 1024]}[width]
        stem = nn.Sequential(nn.Conv2d(3, 24, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        blocks, dims, cin = [], [], 24
        for stage, (cout, reps) in enumerate(zip(chans[:3], [4, 8, 4])):
            for i in range(reps):
                stride = 2 if (i == 0 and stage > 0) else (2 if i == 0 else 1)
                blocks.append(_ShuffleUnit(cin, cout, stride if i == 0 else 1))
                cin = cout
                dims.append(cin)
        blocks.append(nn.Sequential(nn.Conv2d(cin, chans[3], 1, bias=False),
                                    nn.BatchNorm2d(chans[3]), nn.ReLU(inplace=True)))
        dims.append(chans[3])
        return StagedBackbone(stem, blocks, nn.Linear(chans[3], num_classes),
                              lambda i: dims[i])

    # ------------------------------------------------------------- ConvNeXt
    class _LayerNorm2d(nn.Module):
        def __init__(self, c, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(c))
            self.bias = nn.Parameter(torch.zeros(c))
            self.eps = eps

        def forward(self, x):
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

    class _ConvNeXtBlock(nn.Module):
        def __init__(self, dim, drop_path=0.0, ls_init=1e-6):
            super().__init__()
            self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.norm = _LayerNorm2d(dim)
            self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
            self.pw2 = nn.Conv2d(4 * dim, dim, 1)
            self.gamma = nn.Parameter(ls_init * torch.ones(dim)) if ls_init > 0 else None
            self.drop_path = drop_path

        def forward(self, x):
            r = x
            x = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
            if self.gamma is not None:
                x = x * self.gamma[:, None, None]
            if self.drop_path > 0.0 and self.training:
                keep = 1.0 - self.drop_path
                mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
                x = x * mask / keep
            return r + x

    def build_convnext_femto(num_classes: int = 100,
                             dims: Sequence[int] = (48, 96, 192, 384),
                             depths: Sequence[int] = (2, 2, 6, 2),
                             drop_path: float = 0.1) -> StagedBackbone:
        """ConvNeXt-Femto adapted to 32x32.

        Patchify stem is 2x2 stride 2 rather than 4x4 stride 4 -- the ImageNet
        stem would take a 32px input straight to 8px and leave the network
        almost nothing to work with.
        """
        stem = nn.Sequential(nn.Conv2d(3, dims[0], 2, 2), _LayerNorm2d(dims[0]))
        blocks, bdims = [], []
        total = sum(depths)
        dp = [drop_path * i / max(1, total - 1) for i in range(total)]
        k = 0
        for si, (d, n) in enumerate(zip(dims, depths)):
            if si > 0:
                blocks.append(nn.Sequential(_LayerNorm2d(dims[si - 1]),
                                            nn.Conv2d(dims[si - 1], d, 2, 2)))
                bdims.append(d)
            for _ in range(n):
                blocks.append(_ConvNeXtBlock(d, dp[k]))
                bdims.append(d)
                k += 1
        return StagedBackbone(stem, blocks, nn.Linear(dims[-1], num_classes),
                              lambda i: bdims[i], final_norm=_LayerNorm2d(dims[-1]))

    # ------------------------------------------------------- ViT / DeiT-Tiny
    class _PatchEmbed(nn.Module):
        """Patchify + CLS token + positional embedding, resolution-agnostic.

        The positional embedding is learned for a fixed grid -- 8x8 = 64 patches
        at 32px with patch 4, plus one CLS token, so 65 entries. Feed a 16px
        image and you get 4x4 = 16 patches plus CLS = 17 tokens, and adding a
        65-entry embedding to a 17-token tensor is a shape error.

        That matters here because the resolution axis is one of the three
        compute dials we measure, so a ViT that cannot run below 32px cannot be
        measured on that axis at all.

        The fix is the standard one from ViT/DeiT fine-tuning: keep the CLS
        entry, reshape the patch entries back to their square grid, and
        bicubically resample to the grid the current input needs. This is what
        every ViT implementation does when transferring between resolutions, so
        it is not an invention -- and it means the resolution axis measures
        genuine token-count reduction, which is where a transformer's compute
        saving actually comes from.
        """

        def __init__(self, img=32, patch=4, cin=3, dim=192):
            super().__init__()
            self.proj = nn.Conv2d(cin, dim, patch, patch)
            self.patch = patch
            self.n_patches = (img // patch) ** 2
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def _pos_for(self, n_tokens: int):
            if n_tokens == self.pos.shape[1]:
                return self.pos
            cls_pos, grid_pos = self.pos[:, :1], self.pos[:, 1:]
            s_old = int(round(grid_pos.shape[1] ** 0.5))
            s_new = int(round((n_tokens - 1) ** 0.5))
            if s_new < 1 or s_new * s_new != n_tokens - 1:
                raise ValueError(
                    f"cannot interpolate positional embedding to {n_tokens} tokens "
                    f"-- the patch grid is not square")
            g = grid_pos.reshape(1, s_old, s_old, -1).permute(0, 3, 1, 2)
            g = F.interpolate(g.float(), size=(s_new, s_new), mode="bicubic",
                              align_corners=False).to(grid_pos.dtype)
            g = g.permute(0, 2, 3, 1).reshape(1, s_new * s_new, -1)
            return torch.cat([cls_pos, g], dim=1)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)        # (B, N, C)
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            return x + self._pos_for(x.size(1))

    class _TransformerBlock(nn.Module):
        def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0):
            super().__init__()
            self.n1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.n2 = nn.LayerNorm(dim)
            h = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
            self.drop_path = drop_path

        def _dp(self, x):
            if self.drop_path <= 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
            return x * mask / keep

        def forward(self, x):
            h = self.n1(x)
            x = x + self._dp(self.attn(h, h, h, need_weights=False)[0])
            return x + self._dp(self.mlp(self.n2(x)))

    class TokenBackbone(StagedBackbone):
        """Token models pool by taking the CLS token, not a spatial mean."""

        is_token_model = True

        def pooled(self, feat):
            return feat[:, 0]                     # CLS

    def build_vit_tiny(num_classes: int = 100, dim: int = 192, depth: int = 12,
                       heads: int = 3, patch: int = 4,
                       drop_path: float = 0.1) -> TokenBackbone:
        """DeiT-Tiny geometry, CIFAR patchification (4px -> 64 tokens).

        This entry and the Mixer below are what make Q3 interesting. H3 predicts
        CNN->ViT transfer T < 0.6 precisely because the inductive bias differs;
        drop them and the transfer study covers only CNNs and H3 becomes
        untestable. Do not remove them for convenience.
        """
        stem = _PatchEmbed(32, patch, 3, dim)
        dp = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        blocks = [_TransformerBlock(dim, heads, 4.0, dp[i]) for i in range(depth)]
        return TokenBackbone(stem, blocks, nn.Linear(dim, num_classes),
                             lambda i: dim, final_norm=nn.LayerNorm(dim))

    # --------------------------------------------------------- MLP-Mixer
    class _MixerBlock(nn.Module):
        def __init__(self, dim, n_tokens, token_mlp=0.5, chan_mlp=4.0, drop_path=0.0):
            super().__init__()
            th, ch = int(dim * token_mlp), int(dim * chan_mlp)
            self.n1 = nn.LayerNorm(dim)
            self.token_mlp = nn.Sequential(nn.Linear(n_tokens, th), nn.GELU(),
                                           nn.Linear(th, n_tokens))
            self.n2 = nn.LayerNorm(dim)
            self.chan_mlp = nn.Sequential(nn.Linear(dim, ch), nn.GELU(),
                                          nn.Linear(ch, dim))
            self.drop_path = drop_path

        def _dp(self, x):
            if self.drop_path <= 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
            return x * mask / keep

        def forward(self, x):
            x = x + self._dp(self.token_mlp(self.n1(x).transpose(1, 2)).transpose(1, 2))
            return x + self._dp(self.chan_mlp(self.n2(x)))

    class MixerBackbone(StagedBackbone):
        """MLP-Mixer. Fixed token count, by construction.

        The token-mixing block is `Linear(n_tokens -> hidden)` -- the weight
        matrix's input dimension IS the number of patches. Feed a 16px image
        (16 tokens instead of 64) and you get
        "mat1 and mat2 shapes cannot be multiplied (192x16 and 64x96)".

        Unlike the ViT case there is no principled fix. A ViT's positional
        embedding is a lookup that can be resampled; a Mixer's token-mixing
        weights are a learned linear map whose domain is the token grid. You
        cannot run a trained Mixer at a different token count, full stop. That
        is a real property of the architecture, not a limitation of our code.

        So for this architecture the resolution axis is measured with the
        downsample-upsample proxy only: the image is degraded to r px and
        restored to 32, so information content drops while the token count is
        unchanged. 01_PHASE0_GO_NOGO.md 3 anticipates exactly this and says to
        use native resolution "if the architecture tolerates it". This one does
        not, and we record that rather than quietly dropping the model or
        quietly reporting a different quantity under the same name.
        """

        is_token_model = True
        supports_native_resolution = False

        def pooled(self, feat):
            return feat.mean(dim=1)

    class _MixerStem(nn.Module):
        def __init__(self, img=32, patch=4, dim=192):
            super().__init__()
            self.proj = nn.Conv2d(3, dim, patch, patch)
            self.n_tokens = (img // patch) ** 2

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    def build_mixer_nano(num_classes: int = 100, dim: int = 192, depth: int = 8,
                         patch: int = 4, drop_path: float = 0.1) -> MixerBackbone:
        """MLP-Mixer-Nano: the weakest spatial prior in the zoo.

        This is the extreme point of H3. If compute requirements transfer even
        to a model with essentially no convolutional inductive bias, the
        "property of the input" reading is strongly supported; if they collapse
        here specifically, that localises the effect.
        """
        stem = _MixerStem(32, patch, dim)
        n_tok = (32 // patch) ** 2
        dp = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        blocks = [_MixerBlock(dim, n_tok, drop_path=dp[i]) for i in range(depth)]
        return MixerBackbone(stem, blocks, nn.Linear(dim, num_classes),
                             lambda i: dim, final_norm=nn.LayerNorm(dim))


# --------------------------------------------------------------------------
# Zoo registry
# --------------------------------------------------------------------------
# family is the Q3 grouping variable: within-family transfer is expected to
# exceed across-family, which exceeds CNN->token. Keep it accurate.
ZOO: Dict[str, Dict[str, Any]] = {
    "resnet20":     dict(family="resnet", builder=("resnet", dict(depth=20, width_mult=1))),
    "resnet56":     dict(family="resnet", builder=("resnet", dict(depth=56, width_mult=1))),
    "resnet110":    dict(family="resnet", builder=("resnet", dict(depth=110, width_mult=1))),
    "resnet8x4":    dict(family="resnet", builder=("resnet", dict(depth=8, width_mult=4))),
    "resnet32x4":   dict(family="resnet", builder=("resnet", dict(depth=32, width_mult=4))),
    "wrn_40_2":     dict(family="wrn",    builder=("wrn", dict(depth=40, widen=2))),
    "wrn_16_2":     dict(family="wrn",    builder=("wrn", dict(depth=16, widen=2))),
    "wrn_40_1":     dict(family="wrn",    builder=("wrn", dict(depth=40, widen=1))),
    "vgg13":        dict(family="vgg",    builder=("vgg", dict(depth=13))),
    "vgg8":         dict(family="vgg",    builder=("vgg", dict(depth=8))),
    "mobilenetv2":  dict(family="mobile", builder=("mobilenetv2", dict(width=1.0))),
    "shufflenetv2": dict(family="mobile", builder=("shufflenetv2", dict(width="1.0x"))),
    "convnext_femto": dict(family="convnext", builder=("convnext_femto", dict())),
    "vit_tiny":     dict(family="vit",    builder=("vit_tiny", dict())),
    "mixer_nano":   dict(family="mixer",  builder=("mixer_nano", dict())),
}

# Architectures that need the DeiT-style recipe (AdamW, long warmup, strong
# augmentation, label smoothing). SGD flatlines these on CIFAR from scratch --
# the same failure E2AM documented for ConvNeXtV2 under SGD.
TRANSFORMER_LIKE = {"vit_tiny", "mixer_nano", "convnext_femto"}


def build_model(arch: str, num_classes: int = 100, **overrides):
    if not _TORCH_OK:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERR}")
    if arch not in ZOO:
        raise KeyError(f"unknown architecture '{arch}'. Known: {sorted(ZOO)}")
    kind, kwargs = ZOO[arch]["builder"]
    kwargs = dict(kwargs)
    kwargs.update(overrides)
    fn = {
        "resnet": build_resnet_cifar, "wrn": build_wrn, "vgg": build_vgg,
        "mobilenetv2": build_mobilenetv2, "shufflenetv2": build_shufflenetv2,
        "convnext_femto": build_convnext_femto, "vit_tiny": build_vit_tiny,
        "mixer_nano": build_mixer_nano,
    }[kind]
    return fn(num_classes=num_classes, **kwargs)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model) -> float:
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    b += sum(x.numel() * x.element_size() for x in model.buffers())
    return b / (1024 ** 2)


# =============================================================================
# 8. budgets -- FLOPs per compute configuration
# =============================================================================
# rho(c) = FLOPs(f, c) / FLOPs(f, c_full) is the load-bearing methodological
# choice of the whole project (protocol 2.1). It is what puts a ResNet and a
# ViT on a common dimensionless scale and makes "did MSC transfer?" a
# well-posed question. Two consequences that are easy to get wrong:
#
#   1. The SAME profiler and the SAME accounting convention must be used for
#      every architecture and every axis. A budget table built with fvcore for
#      one model and thop for another silently corrupts every transfer number.
#      So: one profiler is chosen, its name and version are recorded in
#      budgets/{arch}.json, and a second is used only as a cross-check.
#
#   2. The depth axis must cost the PREFIX, not the whole network. That is why
#      StagedBackbone.forward_prefix exists and why we profile a wrapper that
#      truncates rather than reading a mid-layer activation from a full pass.

_PROFILER_CACHE: Dict[str, Any] = {}


def _get_profiler() -> Tuple[str, Optional[Callable], str]:
    """Pick one profiler and stick with it. fvcore > ptflops > thop > analytic."""
    if "chosen" in _PROFILER_CACHE:
        return _PROFILER_CACHE["chosen"]
    chosen = ("analytic", None, "builtin")
    try:
        import fvcore
        from fvcore.nn import FlopCountAnalysis

        def _f(model, shape):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fca = FlopCountAnalysis(model, torch.zeros(*shape))
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                # fvcore counts MACs; x2 for FLOPs, consistently everywhere.
                return int(fca.total()) * 2
        chosen = ("fvcore", _f, getattr(fvcore, "__version__", "unknown"))
    except Exception:
        try:
            import thop

            def _f(model, shape):
                macs, _ = thop.profile(model, inputs=(torch.zeros(*shape),), verbose=False)
                return int(macs) * 2
            chosen = ("thop", _f, getattr(thop, "__version__", "unknown"))
        except Exception:
            pass
    _PROFILER_CACHE["chosen"] = chosen
    return chosen


def _analytic_flops(model, shape) -> int:
    """Hook-based fallback: conv + linear only, which dominate these models."""
    total = [0]
    hooks = []

    def conv_hook(m, i, o):
        total[0] += 2 * int(o.numel()) * (m.in_channels // m.groups) * \
            int(np.prod(m.kernel_size))

    def lin_hook(m, i, o):
        total[0] += 2 * int(o.numel()) * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*shape))
    model.train(was)
    for h in hooks:
        h.remove()
    return int(total[0])


def measure_flops(model, input_shape=(1, 3, 32, 32)) -> int:
    name, fn, _ = _get_profiler()
    model = model.eval()
    try:
        if fn is not None:
            return int(fn(model, input_shape))
    except Exception as e:
        log(f"profiler {name} failed ({str(e)[:80]}); using analytic fallback", "FLOP")
    return _analytic_flops(model, input_shape)


if _TORCH_OK:

    class _PrefixWrapper(nn.Module):
        """Backbone truncated at stage k, plus its exit head. Profiled as one unit."""

        def __init__(self, backbone, k: int, head: Optional[nn.Module] = None):
            super().__init__()
            self.backbone = backbone
            self.k = k
            self.head = head

        def forward(self, x):
            f = self.backbone.forward_prefix(x, self.k)
            if self.head is None:
                return f
            return self.head(f)


def build_budget_table(arch: str, num_classes: int = 100,
                       resolutions: Sequence[int] = RESOLUTIONS,
                       depth_fractions: Sequence[float] = DEPTH_FRACTIONS,
                       precisions: Sequence[str] = PRECISIONS,
                       model=None) -> Dict[str, Any]:
    """FLOPs for every configuration on every axis, plus normalised rho.

    Measured once per architecture, written to budgets/{arch}.json, and never
    recomputed -- a budget table that drifts between sessions makes MSC values
    from different sessions incomparable.
    """
    model = model if model is not None else build_model(arch, num_classes)
    model = model.eval().cpu()
    prof_name, _, prof_ver = _get_profiler()

    full = measure_flops(model, (1, 3, 32, 32))

    # --- depth: prefix cost + a linear exit head -------------------------
    # K comes from the MODEL, not the global constant: a shallow backbone
    # legitimately carries fewer distinct depth budgets (see StagedBackbone).
    feat_dims = list(model.feature_dims)
    achieved_fractions = list(getattr(model, "depth_fractions", depth_fractions))
    depth_flops = []
    for k in range(len(feat_dims)):
        head = ExitHead(feat_dims[k], num_classes,
                        token_model=getattr(model, "is_token_model", False)).eval()
        depth_flops.append(measure_flops(_PrefixWrapper(model, k, head), (1, 3, 32, 32)))
    depth_rho = [f / depth_flops[-1] for f in depth_flops]
    if not all(depth_rho[i] < depth_rho[i + 1] for i in range(len(depth_rho) - 1)):
        # The oracle needs strictly ascending costs; equal budgets make "the
        # smallest sufficient one" ill-defined. Fail here, where it is one line
        # of output, rather than mid-sweep in Phase 1b.
        raise ValueError(
            f"{arch}: depth costs are not strictly ascending: "
            f"{[round(r, 4) for r in depth_rho]}. The stage partition is wrong.")

    # --- resolution -------------------------------------------------------
    # Two honest cost models, per 01_PHASE0_GO_NOGO.md 3:
    #   native  the network really runs at r x r. Cleaner, but requires the
    #           architecture to tolerate a different input size.
    #   proxy   the image is degraded to r and restored to 32. Works for every
    #           architecture; cost is the same table but labelled idealised.
    #
    # We measure native where possible and always measure proxy, so the
    # resolution axis is defined uniformly across the whole zoo -- which is what
    # makes a cross-architecture comparison on this axis legitimate at all.
    native_ok = bool(getattr(model, "supports_native_resolution", True))
    res_flops, native_err = [], None
    if native_ok:
        try:
            res_flops = [measure_flops(model, (1, 3, r, r)) for r in resolutions]
        except Exception as e:
            native_ok, native_err = False, f"{type(e).__name__}: {str(e)[:160]}"
            log(f"{arch} cannot run at non-32px input ({native_err}); "
                f"resolution axis will use the proxy only", "FLOP")
    if not res_flops:
        # Analytic stand-in: cost scales with pixel count for a convolutional
        # network and with token count for a patch model -- both quadratic in r.
        res_flops = [int(full * (r / 32.0) ** 2) for r in resolutions]
    res_rho = [f / res_flops[-1] for f in res_flops]

    # --- precision: analytic bit-operation accounting ---------------------
    # There is no INT4 kernel to time on a T4, so this axis is priced, not
    # measured. Reported as an analytic cost model and never as measured
    # latency -- see the limitations section of the paper.
    prec_rho = [PRECISION_BITS[p] / 32.0 for p in precisions]
    prec_flops = [int(full * r) for r in prec_rho]

    table = {
        "arch": arch,
        "num_classes": int(num_classes),
        "full_flops": int(full),
        "profiler": {"name": prof_name, "version": prof_ver,
                     "convention": "FLOPs = 2 x MACs",
                     "measured_utc": now_iso()},
        "params": count_parameters(model),
        "axes": {
            "depth": {
                "configs": [f"d{i+1}" for i in range(len(depth_flops))],
                "K": len(depth_flops),
                "fractions": [float(f) for f in achieved_fractions],
                "requested_fractions": list(depth_fractions),
                "stage_cuts": list(model.stage_cuts),
                "n_blocks": len(model.blocks),
                "feature_dims": feat_dims,
                "flops": [int(f) for f in depth_flops],
                "rho": [float(r) for r in depth_rho],
                "note": ("prefix backbone + linear exit head; forward_prefix stops "
                         "early. K is adaptive: a backbone with fewer blocks than "
                         "requested exits carries fewer distinct depth budgets."),
            },
            "resolution": {
                "configs": [f"r{r}" for r in resolutions],
                "values": list(resolutions),
                "flops": [int(f) for f in res_flops],
                "rho": [float(r) for r in res_rho],
                "native_supported": bool(native_ok),
                "native_error": native_err,
                "note": ("cost measured at NATIVE input size where the "
                         "architecture tolerates it; otherwise an analytic "
                         "quadratic-in-r model. The proxy sweep "
                         "(downsample-then-upsample to 32px) shares this cost "
                         "table and is labelled idealised."),
            },
            "precision": {
                "configs": list(precisions),
                "bits": [PRECISION_BITS[p] for p in precisions],
                "flops": [int(f) for f in prec_flops],
                "rho": [float(r) for r in prec_rho],
                "note": ("analytic bit-operation model rho = bits/32. INT4/INT6 "
                         "are simulated by fake quantisation; no T4 kernel exists "
                         "to time. Never reported as measured latency."),
            },
        },
    }
    return table


def load_or_build_budgets(arch: str, data_dir, num_classes: int = 100,
                          hub: Optional[MSCHub] = None, force: bool = False,
                          model=None) -> Dict[str, Any]:
    p = Path(data_dir) / "budgets" / f"{arch}.json"
    if p.exists() and not force:
        t = read_json(p)
        if t and t.get("full_flops"):
            return t
    log(f"measuring FLOPs budget for {arch}", "FLOP")
    t = build_budget_table(arch, num_classes, model=model)
    atomic_write_json(p, t)
    if hub is not None and hub.enabled:
        hub.hub.enqueue(p, f"budgets/{arch}.json")
    return t


# =============================================================================
# 9. exits -- exit heads, multi-exit wrapper, ordinal sufficiency head
# =============================================================================
if _TORCH_OK:

    class ExitHead(nn.Module):
        """Pool -> normalise -> project. Deliberately minimal.

        A heavier head would do its own representation learning, which
        confounds the measurement: we want to read what the backbone has
        computed by this depth, not what a capable head can recover from it.

        Rank dispatch is what lets the same head class attach to a ResNet
        (B,C,H,W) and a ViT (B,N,C) without the caller knowing which it has.
        """

        def __init__(self, in_dim: int, num_classes: int, token_model: bool = False):
            super().__init__()
            self.token_model = token_model
            self.norm = nn.BatchNorm1d(in_dim)
            self.fc = nn.Linear(in_dim, num_classes)

        def forward(self, feat):
            if feat.dim() == 4:
                x = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            elif feat.dim() == 3:
                # CLS token if the model has one, else mean over tokens.
                x = feat[:, 0] if self.token_model else feat.mean(dim=1)
            else:
                x = feat.flatten(1)
            return self.fc(self.norm(x))

    class MultiExitModel(nn.Module):
        """Frozen backbone + K exit heads.

        Freezing is not an optimisation, it is the definition. If the backbone
        adapts while the heads train, each exit reads a *different* network and
        the "same model under reduced compute" interpretation -- which the
        entire MSC construct rests on -- collapses. train() is overridden so a
        stray model.train() cannot silently un-freeze BatchNorm statistics.
        """

        def __init__(self, backbone, num_classes: int, freeze: bool = True):
            super().__init__()
            self.backbone = backbone
            self.token_model = getattr(backbone, "is_token_model", False)
            self.heads = nn.ModuleList([
                ExitHead(d, num_classes, self.token_model)
                for d in backbone.feature_dims])
            self.frozen = freeze
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
                self.backbone.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            if self.frozen:
                self.backbone.eval()
            return self

        def forward(self, x) -> List["torch.Tensor"]:
            if self.frozen:
                with torch.no_grad():
                    feats = self.backbone.forward_features(x)
            else:
                feats = self.backbone.forward_features(x)
            return [h(f) for h, f in zip(self.heads, feats)]

        def forward_at(self, x, k: int):
            """Single exit, prefix only -- the deployment path."""
            f = self.backbone.forward_prefix(x, k)
            return self.heads[k](f)

    class OrdinalSufficiencyHead(nn.Module):
        """Monotone sufficiency curve, by construction.

            theta_1 = t_1,  theta_{k+1} = theta_k + softplus(delta_k)
            s_k(x)  = sigmoid(theta_k - u(x))

        Since theta is increasing, s_k is non-decreasing in k automatically.
        This replaces the auxiliary monotonicity penalty from the earlier CEB-KD
        plan. An architectural constraint beats a soft penalty on three counts:
        it cannot be violated, it adds no hyperparameter, and it cannot trade
        off against the other loss terms during optimisation.

        Placed on the EARLIEST exit's features so the routing decision is
        available cheaply and early -- a router that needs deep features to
        decide not to compute deep features is useless.
        """

        def __init__(self, in_dim: int, n_budgets: int, hidden: int = 128,
                     token_model: bool = False):
            super().__init__()
            self.n_budgets = n_budgets
            self.token_model = token_model
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True), nn.Linear(hidden, 1))
            self.theta_0 = nn.Parameter(torch.zeros(1))
            self.deltas = nn.Parameter(torch.zeros(n_budgets - 1))

        def _pool(self, feat):
            if feat.dim() == 4:
                return F.adaptive_avg_pool2d(feat, 1).flatten(1)
            if feat.dim() == 3:
                return feat[:, 0] if self.token_model else feat.mean(dim=1)
            return feat.flatten(1)

        def thresholds(self):
            steps = F.softplus(self.deltas) + 1e-4
            return torch.cat([self.theta_0, self.theta_0 + torch.cumsum(steps, 0)])

        def logits(self, feat):
            """The pre-sigmoid score `theta_k - u(x)`, shape (B, K).

            Exposed because the loss must not be given probabilities. D-21:
            `F.binary_cross_entropy` refuses to run under AMP autocast, and the
            fix is not to disable autocast but to use the logit form, which is
            both autocast-safe and numerically stable. Monotonicity is
            unaffected -- `thresholds()` is increasing and sigmoid is monotone,
            so s_k is non-decreasing in k whether or not you apply the sigmoid.
            """
            u = self.mlp(self._pool(feat))                       # (B, 1)
            return self.thresholds().unsqueeze(0) - u

        def forward(self, feat):
            return torch.sigmoid(self.logits(feat))

        @torch.no_grad()
        def route(self, feat, gamma: float):
            s = self.forward(feat)
            hit = s >= gamma
            return torch.where(hit.any(dim=1), hit.float().argmax(dim=1),
                               torch.full((s.size(0),), self.n_budgets - 1,
                                          device=s.device, dtype=torch.long))


# =============================================================================
# 10. energy -- NVML power sampling
# =============================================================================
class GPUEnergyMonitor:
    """Direct power sampling on EVERY visible GPU, trapezoidal integration.

    pynvml at >=10 Hz where available, nvidia-smi at ~1 Hz as fallback. The
    protocol (7.1) makes theoretical FLOPs the PRIMARY efficiency metric and
    energy strictly secondary -- FLOP-based proxies underestimate real energy by
    2-6x due to memory traffic and kernel-launch overhead, which is exactly why
    we sample directly and exactly why energy is reported as measurement
    methodology rather than as a contribution (7.3).
    """

    def __init__(self, sample_hz: float = 10.0, device_index: Optional[int] = None):
        self.interval = 1.0 / max(1.0, sample_hz)
        self.sample_hz = sample_hz
        self._samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handles: List[Tuple[int, Any]] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            idx = ([device_index] if device_index is not None
                   else list(range(pynvml.nvmlDeviceGetCount())))
            self._handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idx]
        except Exception:
            self._nvml = None
            self._fallback_index = device_index if device_index is not None else 0

    def _read(self) -> List[Dict[str, Any]]:
        base = {"unix_ts": time.time(), "datetime_utc": now_iso(),
                "monotonic_sec": time.monotonic()}
        if self._nvml is not None and self._handles:
            out = []
            for i, h in self._handles:
                try:
                    out.append(dict(base, gpu_index=i,
                                    power_w=self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0))
                except Exception:
                    pass
            return out
        rc, o, _ = shell(["nvidia-smi", "--query-gpu=index,power.draw",
                          "--format=csv,noheader,nounits"], timeout=5)
        if rc != 0 or not o.strip():
            return []
        out = []
        for line in o.strip().splitlines():
            try:
                i, w = line.split(",")
                out.append(dict(base, gpu_index=int(i), power_w=float(w)))
            except Exception:
                continue
        return out

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._samples.extend(self._read())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nvml")
        self._thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        return list(self._samples)

    @staticmethod
    def integrate_j(samples: List[Dict[str, Any]], fallback_sec: float = 0.0,
                    fallback_w: float = 70.0) -> float:
        """Total joules across all GPUs, integrating each device separately."""
        if not samples:
            return fallback_sec * fallback_w
        by_gpu: Dict[int, List[Dict[str, Any]]] = {}
        for s_ in samples:
            by_gpu.setdefault(int(s_.get("gpu_index", 0)), []).append(s_)
        total = 0.0
        for rows in by_gpu.values():
            if len(rows) < 2:
                continue
            t = np.asarray([r["monotonic_sec"] for r in rows], dtype=float)
            w = np.asarray([r["power_w"] for r in rows], dtype=float)
            o = np.argsort(t)
            total += float(np.trapezoid(w[o], t[o])) if hasattr(np, "trapezoid") \
                else float(np.trapz(w[o], t[o]))
        return total if total > 0 else fallback_sec * fallback_w

    @staticmethod
    def power_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        w = [s_["power_w"] for s_ in samples if "power_w" in s_]
        if not w:
            return {"power_mean_w": NA, "power_max_w": NA, "power_min_w": NA}
        return {"power_mean_w": float(np.mean(w)), "power_max_w": float(np.max(w)),
                "power_min_w": float(np.min(w))}


def energy_to_kwh(j: float) -> float:
    return j / 3.6e6


def energy_to_co2_kg(j: float, intensity_kg_per_kwh: float = 0.475) -> float:
    return energy_to_kwh(j) * intensity_kg_per_kwh


# =============================================================================
# 11. dynamics -- the three difficulty scores that cannot be computed post hoc
# =============================================================================
class TrainingDynamics:
    """Per-sample instrumentation of the TRAINING set, recorded during training.

    Q4 is the question that decides whether MSC is a new object or a rebranded
    one, so it is treated as the primary threat rather than a footnote. Four of
    its seven difficulty scores (msp, margin, entropy, ce_loss) are trivially
    computable from a final checkpoint. Three are not:

      EL2N            ||softmax(f(x)) - onehot(y)||_2, captured at a fixed early
                      epoch. The DURING-TRAINING variant specifically -- the
                      GraNd-at-init variant failed reproduction (arXiv
                      2303.14753) and the protocol excludes it by name.
      forgetting      count of 1->0 transitions in per-sample training
                      correctness across epochs (Toneva et al., ICLR 2019).
                      Needs every epoch; cannot be reconstructed later.
      prediction depth computed post hoc from exit-head features, but only
                      because we keep the exit heads.

    Cost is one extra forward-free bookkeeping array per epoch: we reuse the
    logits the training loop has already computed. Re-running the 110-hour
    atlas because one of these was forgotten is not a recoverable mistake, so
    the instrumentation is unconditional.
    """

    def __init__(self, n_train: int, el2n_epoch: int = 10):
        self.n = int(n_train)
        self.el2n_epoch = int(el2n_epoch)
        self.correct_prev = np.zeros(self.n, dtype=np.int8)
        self.ever_correct = np.zeros(self.n, dtype=bool)
        self.forget_events = np.zeros(self.n, dtype=np.int32)
        self.el2n = np.full(self.n, np.nan, dtype=np.float32)
        self._epoch_correct = np.zeros(self.n, dtype=np.int8)
        self._epoch_seen = np.zeros(self.n, dtype=bool)
        self.epochs_recorded = 0

    def observe_batch(self, idx, logits, labels, epoch: int) -> None:
        """Called once per training batch with what the loop already has."""
        with torch.no_grad():
            i = idx.detach().cpu().numpy().astype(np.int64)
            pred = logits.detach().argmax(dim=1)
            corr = (pred == labels).detach().cpu().numpy().astype(np.int8)
            self._epoch_correct[i] = corr
            self._epoch_seen[i] = True
            if epoch == self.el2n_epoch:
                p = F.softmax(logits.detach().float(), dim=1)
                oh = F.one_hot(labels, num_classes=p.size(1)).float()
                self.el2n[i] = (p - oh).norm(dim=1).cpu().numpy().astype(np.float32)

    def end_epoch(self) -> None:
        seen = self._epoch_seen
        if seen.any():
            # A forgetting event is a 1 -> 0 transition on a sample that was
            # previously learned. Samples never yet learned cannot be forgotten.
            forgot = seen & (self.correct_prev == 1) & (self._epoch_correct == 0)
            self.forget_events[forgot] += 1
            self.correct_prev[seen] = self._epoch_correct[seen]
            self.ever_correct[seen] |= self._epoch_correct[seen].astype(bool)
        self._epoch_correct[:] = 0
        self._epoch_seen[:] = False
        self.epochs_recorded += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "el2n_epoch": self.el2n_epoch,
                "correct_prev": self.correct_prev, "ever_correct": self.ever_correct,
                "forget_events": self.forget_events, "el2n": self.el2n,
                "epochs_recorded": self.epochs_recorded}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        if not st or int(st.get("n", -1)) != self.n:
            return
        self.correct_prev = np.asarray(st["correct_prev"])
        self.ever_correct = np.asarray(st["ever_correct"])
        self.forget_events = np.asarray(st["forget_events"])
        self.el2n = np.asarray(st["el2n"])
        self.epochs_recorded = int(st.get("epochs_recorded", 0))

    def to_frame(self):
        return pd.DataFrame({
            "sample_idx": np.arange(self.n),
            "forget_events": self.forget_events,
            "ever_correct": self.ever_correct,
            "el2n": self.el2n,
            # Toneva's "unforgettable" set: learned and never lost. A useful
            # sanity check -- it should be a large, easy majority.
            "unforgettable": (self.ever_correct & (self.forget_events == 0)),
        })


@_no_grad()
def prediction_depth(multi_exit, loader, device, k_neighbors: int = 30,
                     max_support: int = 5000) -> np.ndarray:
    """Baldock, Maennel & Neyshabur (NeurIPS 2021), adapted to our exits.

    For each sample, the earliest layer at which a k-NN probe on that layer's
    representation already predicts the network's final answer, and keeps
    predicting it at every deeper layer. The suffix requirement mirrors the
    stable-sufficiency closure in 2.2 for exactly the same reason: without it,
    an accidental early agreement is recorded as a genuine one.

    Returned as a fraction in [0,1] so it is comparable across architectures
    with different exit counts.
    """
    multi_exit.eval()
    feats_all: List[List[np.ndarray]] = []
    finals: List[np.ndarray] = []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        fs = multi_exit.backbone.forward_features(x)
        pooled = []
        for f in fs:
            if f.dim() == 4:
                pooled.append(F.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu().numpy())
            elif f.dim() == 3:
                pooled.append((f[:, 0] if multi_exit.token_model
                               else f.mean(1)).float().cpu().numpy())
            else:
                pooled.append(f.flatten(1).float().cpu().numpy())
        feats_all.append(pooled)
        finals.append(multi_exit.backbone(x).argmax(1).cpu().numpy())

    n_layers = len(feats_all[0])
    layers = [np.concatenate([b[l] for b in feats_all], axis=0) for l in range(n_layers)]
    final = np.concatenate(finals, axis=0)
    n = final.shape[0]

    rng = np.random.default_rng(0)
    sup = rng.choice(n, size=min(max_support, n), replace=False)

    agree = np.zeros((n, n_layers), dtype=bool)
    for l, X in enumerate(layers):
        Xs = X[sup]
        Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
        Xq = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        ys = final[sup]
        # Chunked cosine kNN vote; full pairwise on 10k x 5k would be fine but
        # the chunking keeps peak memory flat for larger test sets.
        preds = np.empty(n, dtype=final.dtype)
        step = 1024
        for s in range(0, n, step):
            sim = Xq[s:s + step] @ Xs.T
            nb = np.argpartition(-sim, kth=min(k_neighbors, sim.shape[1] - 1),
                                 axis=1)[:, :k_neighbors]
            votes = ys[nb]
            preds[s:s + step] = [np.bincount(v).argmax() for v in votes]
        agree[:, l] = (preds == final)

    # Suffix closure: earliest layer from which agreement never breaks.
    suffix = np.ones_like(agree)
    suffix[:, -1] = agree[:, -1]
    for j in range(n_layers - 2, -1, -1):
        suffix[:, j] = agree[:, j] & suffix[:, j + 1]
    any_ok = suffix.any(axis=1)
    depth = np.where(any_ok, suffix.argmax(axis=1), n_layers - 1)
    return (depth + 1).astype(np.float32) / float(n_layers)


# =============================================================================
# 12. config -- run identity and recipes
# =============================================================================
def make_run_id(phase: str, arch: str, dataset: str, method: str, seed: int) -> str:
    """`{phase}-{arch}-{dataset}-{method}-s{seed}`

    Deterministic and collision-free by construction. Never auto-generate a
    UUID: six weeks from now you will need to find a specific run by reading
    its name, and a UUID makes that impossible.
    """
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.]+", "", str(s))
    return f"{safe(phase)}-{safe(arch)}-{safe(dataset)}-{safe(method)}-s{int(seed)}"


def parse_run_id(run_id: str) -> Dict[str, Any]:
    """Recover a run's identity from its id, which is authoritative by design.

        {phase}-{arch}-{dataset}-{method}-s{seed}

    Use this rather than reading `arch`/`seed` out of ledger events. Not every
    event carries every field -- `repair_ledger`, for instance, reconstructs a
    completion from history.csv and knows the run_id but not the architecture.
    Trusting the ledger for metadata therefore yields None where the id has the
    answer sitting in plain text. That is what broke NB08 (defect D-13).

    The run_id format exists precisely so that identity never needs a lookup.
    """
    parts = str(run_id).split("-")
    out: Dict[str, Any] = {"run_id": run_id, "phase": None, "arch": None,
                           "dataset": None, "method": None, "seed": None}
    if len(parts) < 5:
        return out
    out["phase"] = parts[0]
    out["arch"] = parts[1]
    out["dataset"] = parts[2]
    out["method"] = "-".join(parts[3:-1])
    tail = parts[-1]
    if tail.startswith("s") and tail[1:].isdigit():
        out["seed"] = int(tail[1:])
    out["family"] = ZOO.get(out["arch"], {}).get("family")
    return out


def run_meta(run_id: str, ledger_entry: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """Identity from the run_id, enriched with whatever the ledger happens to
    carry. The id always wins for the fields it defines."""
    meta = dict(ledger_entry or {})
    meta.update({k: v for k, v in parse_run_id(run_id).items() if v is not None})
    return meta


def base_config(arch: str, dataset: str = "cifar100", seed: int = 1,
                phase: str = "p1", method: str = "base", **overrides) -> Dict[str, Any]:
    """Standard CRD/DKD recipe for CNNs, DeiT-style recipe for token models.

    The CNN recipe (240 epochs, SGD 0.05, x0.1 at 150/180/210, bs 64, wd 5e-4)
    is chosen so that the resulting accuracies are directly comparable to the
    published benchmark table in 02_ENGINEERING_SPEC.md 7. That comparison is
    the acceptance test for the whole atlas: MSC computed from an undertrained
    model is meaningless, and an undertrained model is otherwise very hard to
    notice.
    """
    n_classes = {"cifar100": 100, "cifar10": 10, "tinyimagenet": 200}[dataset]
    transformer = arch in TRANSFORMER_LIKE

    cfg: Dict[str, Any] = {
        "run_id": make_run_id(phase, arch, dataset, method, seed),
        "phase": phase, "arch": arch, "dataset_name": dataset, "method": method,
        "seed": int(seed), "num_classes": n_classes,
        "family": ZOO.get(arch, {}).get("family", "unknown"),

        "num_epochs": 240 if not transformer else 300,
        "batch_size": 64 if not transformer else 128,
        "eval_batch_size": 512,
        "optimizer": "sgd" if not transformer else "adamw",
        "learning_rate": 0.05 if not transformer else 1e-3,
        "weight_decay": 5e-4 if not transformer else 0.05,
        "momentum": 0.9,
        "nesterov": True,
        "scheduler": "multistep" if not transformer else "cosine",
        "lr_milestones": [150, 180, 210],
        "lr_gamma": 0.1,
        "warmup_epochs": 0 if not transformer else 20,
        "label_smoothing": 0.0 if not transformer else 0.1,
        "grad_clip_norm": 0.0 if not transformer else 1.0,
        "amp_enabled": True,
        "gradient_accumulation_steps": 1,
        "deterministic": False,

        # Q4 instrumentation
        "el2n_epoch": 10,
        "train_holdout_n": 5000,

        # exit heads: backbone frozen, per 01_PHASE0_GO_NOGO.md 3
        "exit_epochs": 20,
        "exit_lr": 0.01,

        # infrastructure
        "milestone_push_every_epochs": 10,
        "timer_push_sec": 1800,
        "session_limit_h": 8.5,
        "cleanup_local_after_complete": True,
        "energy_sample_hz": 10.0,
        "carbon_intensity_kg_per_kwh": 0.475,
        "force_rerun": False,
        "msc_lib_version": __version__,
    }
    cfg.update(overrides)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


# Fields that legitimately vary between sessions and must NOT participate in
# the resume hash. Everything else is frozen at run start.
_HASH_EXCLUDE = {"config_hash", "output_root", "data_root", "force_rerun",
                 "cleanup_local_after_complete", "milestone_push_every_epochs",
                 "timer_push_sec", "session_limit_h", "energy_sample_hz",
                 "sysmon_hz", "eval_batch_size", "msc_lib_version",
                 "worker_id", "run_id", "_debug_interrupt_after_epoch"}


def config_hash(cfg: Dict[str, Any]) -> str:
    return sha256_of_obj({k: v for k, v in sorted(cfg.items())
                          if k not in _HASH_EXCLUDE})


def phase0_configs(dataset: str = "cifar100") -> List[Dict[str, Any]]:
    """The four runs of 01_PHASE0_GO_NOGO.md 2.

    resnet32x4 and wrn-40-2, two seeds each. Two seeds per architecture is not
    a convenience -- it is what produces the noise ceiling, which is the
    denominator of every transfer claim in the project.
    """
    out = []
    for arch in ("resnet32x4", "wrn_40_2"):
        for seed in (1, 2):
            out.append(base_config(arch, dataset, seed, phase="p0", method="base"))
    return out


def phase1_configs(dataset: str = "cifar100", seeds: Sequence[int] = (1, 2, 3),
                   archs: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    archs = list(archs) if archs else list(ZOO.keys())
    return [base_config(a, dataset, s, phase="p1", method="base")
            for a in archs for s in seeds]


# Published CIFAR-100 top-1 for the standard recipe (DKD paper / mdistiller).
# If a trained model lands more than ~1 point below its reference, the recipe
# is wrong and every MSC table derived from it is worthless. Checked, loudly,
# at the end of every backbone run.
REFERENCE_ACC = {
    "resnet56": 72.34, "resnet110": 74.31, "resnet32x4": 79.42,
    "resnet20": 69.06, "resnet8x4": 72.50,
    "wrn_40_2": 75.61, "wrn_16_2": 73.26, "wrn_40_1": 71.98,
    "vgg13": 74.64, "vgg8": 70.36,
    "mobilenetv2": 64.60, "shufflenetv2": 70.50,
}


# =============================================================================
# 13. train -- resumable backbone training
# =============================================================================
# Every column recorded per epoch. The instruction was "save every single
# detail -- we only train once", and that is the right instinct: an atlas run
# costs ~3 T4-hours and re-running it to recover a metric nobody thought to
# record is unrecoverable time.
#
# Grouped by what question each column lets you answer later:
#
#   learning     did it learn?              losses, accuracies, f1/precision/recall
#   optimisation was the optimiser healthy? LR per group, grad norms pre/post
#                                           clip, weight norm, update ratio,
#                                           AMP scale, clip-hit fraction
#   speed        where did the time go?     step-time p50/p90/p99, dataload vs
#                                           compute split, throughput
#   hardware     was the GPU the problem?   VRAM allocated/reserved/peak, GPU
#                                           util, temperature, SM clock, CPU, RAM
#   energy       what did it cost?          per-epoch and cumulative J, kWh, CO2
#   provenance   which run was this?        run_id, worker, session, host, epoch
# Loss terms whose columns always exist but are only populated when the term
# is actually part of the objective. 00_RESEARCH_PROTOCOL.md 1 deletes
# feature / attention / Pareto and drops counterfactual, so the current
# objective is CE + alpha*KD + beta*MSC -- three terms, two weights. Writing a
# number into a column for a loss the model never computed would be worse than
# writing NA, so these stay NA unless the matching cfg flag turns them on.
OPTIONAL_LOSS_TERMS = ("feature", "attention", "energy_boundary",
                       "counterfactual", "pareto")

# Number of GPUs given their own columns. Dual T4 is the platform; anything
# beyond is still captured per device in telemetry/system_samples.csv.
N_GPU_COLUMNS = 2

NA = "NA"          # what a column holds when the quantity does not exist


def _gpu_fields(n: int = N_GPU_COLUMNS) -> List[str]:
    """Per-device columns. The spec asks for GPU utilisation 'each GPU
    separate', and it matters: training uses one T4 while the second idles, so
    an aggregate would hide the fact that half the allocation does nothing.
    """
    out: List[str] = []
    for i in range(n):
        out += [f"gpu{i}_util_mean_pct", f"gpu{i}_util_max_pct",
                f"gpu{i}_mem_used_mb", f"gpu{i}_mem_total_mb",
                f"gpu{i}_mem_util_pct",
                f"gpu{i}_temp_mean_c", f"gpu{i}_temp_max_c",
                f"gpu{i}_power_mean_w", f"gpu{i}_power_max_w",
                f"gpu{i}_sm_clock_mhz", f"gpu{i}_mem_clock_mhz",
                f"gpu{i}_energy_j", f"gpu{i}_throttle_reasons"]
    return out


# Every column recorded per epoch. The instruction was "save every single
# detail -- we only train once", and that is the right instinct: an atlas run
# costs ~3 T4-hours and re-running it to recover a metric nobody thought to
# record is unrecoverable time.
#
# Full column-by-column mapping to requirement 15.1 is in 06_DATA_SCHEMA.md 6.
HISTORY_FIELDS = (
    # ---- identity & provenance ----
    ["run_id", "epoch", "global_step", "timestamp_utc", "unix_ts",
     "account", "worker_id", "session_id", "hostname",
     "arch", "family", "dataset", "seed", "phase", "method", "config_hash"]

    # ---- learning ----
    + ["train_loss", "val_loss", "train_accuracy", "val_accuracy",
       "train_accuracy_top5", "val_accuracy_top5",
       "f1_macro", "f1_micro", "f1_weighted",
       "precision_macro", "precision_micro", "precision_weighted",
       "recall_macro", "recall_micro", "recall_weighted",
       "balanced_accuracy", "cohen_kappa", "matthews_corrcoef",
       "train_loss_min", "train_loss_max", "train_loss_std", "train_loss_median",
       "best_val_accuracy_so_far", "epochs_since_best", "is_best"]

    # ---- calibration (beyond spec: Q5's mechanism claim is about calibration,
    #      so measuring it per epoch turns an assertion into evidence) ----
    + ["val_ece", "val_mce", "val_nll", "val_brier",
       "val_confidence_mean", "val_entropy_mean"]

    # ---- loss components ----
    + ["loss_total", "loss_ce", "loss_kd", "loss_msc", "loss_l1",
       "alpha", "beta", "temperature"]
    + [f"loss_{t}" for t in OPTIONAL_LOSS_TERMS]

    # ---- optimisation health ----
    + ["learning_rate", "lr_min_group", "lr_max_group", "lr_groups_json",
       "momentum", "weight_decay",
       "grad_norm_mean", "grad_norm_max", "grad_norm_min",
       "grad_norm_p50", "grad_norm_p95", "grad_norm_p99", "grad_norm_std",
       "grad_clip_value", "grad_clip_hit_frac",
       "weight_norm", "update_norm", "update_to_weight_ratio",
       "amp_scale", "amp_scale_decreases",
       "n_batches", "n_optimizer_steps", "n_skipped_steps", "nan_or_inf_batches"]

    # ---- time ----
    + ["epoch_time_sec", "train_time_sec", "val_time_sec", "cumulative_time_sec",
       "dataload_time_sec", "compute_time_sec", "backward_time_sec",
       "optimizer_time_sec", "dataload_frac",
       "step_time_mean_ms", "step_time_p50_ms", "step_time_p90_ms",
       "step_time_p99_ms", "step_time_max_ms",
       "throughput_train_img_s", "throughput_val_img_s",
       "samples_seen", "cumulative_samples_seen", "eta_sec"]

    # ---- GPU, per device ----
    + _gpu_fields()
    + ["vram_allocated_mb", "vram_reserved_mb", "peak_vram_mb", "vram_total_mb",
       "n_gpus_visible"]

    # ---- host ----
    + ["cpu_percent", "cpu_count", "ram_used_mb", "ram_total_mb", "ram_percent",
       "proc_rss_mb", "disk_free_scratch_mb", "disk_free_working_mb"]

    # ---- energy & carbon ----
    + ["epoch_energy_j", "epoch_energy_wh", "epoch_energy_kwh",
       "cumulative_energy_j", "cumulative_energy_wh", "cumulative_energy_kwh",
       "epoch_co2_g", "epoch_co2_kg", "cumulative_co2_g", "cumulative_co2_kg",
       "carbon_intensity_g_per_kwh",
       "power_mean_w", "power_max_w", "power_min_w",
       "energy_per_sample_mj", "energy_samples_n", "energy_sample_hz"]

    # ---- config echo, so the CSV is self-describing ----
    + ["batch_size", "effective_batch_size", "gradient_accumulation_steps",
       "amp_enabled", "num_epochs", "optimizer", "scheduler", "image_size",
       "num_classes", "label_smoothing", "deterministic", "msc_lib_version"]
)


class EpochTelemetry:
    """Accumulates everything measurable during one epoch.

    Deliberately cheap: the expensive quantities (gradient norm, weight norm)
    are computed once per optimizer step rather than per batch, and the
    step-time trace is a list of floats. Total overhead is well under 1% of
    epoch time, which is the right trade for never having to re-run a 3-hour job
    because a number was not recorded.
    """

    def __init__(self):
        self.step_times: List[float] = []
        self.dataload_times: List[float] = []
        self.compute_times: List[float] = []
        self.backward_times: List[float] = []
        self.optimizer_times: List[float] = []
        self.grad_norms: List[float] = []
        self.losses: List[float] = []
        self.lrs: List[float] = []
        self.clip_hits = 0
        self.opt_steps = 0
        self.skipped_steps = 0
        self.n_batches = 0
        self.bad_batches = 0
        self.samples = 0
        self.amp_decreases = 0

    def add_batch(self, loss: float, step_t: float, load_t: float, comp_t: float,
                  backward_t: float = 0.0, opt_t: float = 0.0,
                  lr: Optional[float] = None):
        self.n_batches += 1
        self.step_times.append(step_t)
        self.dataload_times.append(load_t)
        self.compute_times.append(comp_t)
        self.backward_times.append(backward_t)
        self.optimizer_times.append(opt_t)
        if lr is not None:
            self.lrs.append(float(lr))
        if loss != loss or loss in (float("inf"), float("-inf")):
            # NaN/Inf losses are silent killers under AMP -- the run keeps going
            # and quietly learns nothing. Counting them makes it visible.
            self.bad_batches += 1
        else:
            self.losses.append(loss)

    def add_step(self, grad_norm: Optional[float], clipped: bool,
                 skipped: bool = False):
        self.opt_steps += 1
        if skipped:
            self.skipped_steps += 1
        if grad_norm is not None and np.isfinite(grad_norm):
            self.grad_norms.append(float(grad_norm))
        if clipped:
            self.clip_hits += 1

    @staticmethod
    def _p(a: List[float], q: float, scale: float = 1.0):
        return float(np.percentile(a, q) * scale) if a else NA

    @staticmethod
    def _f(a: List[float], fn, scale: float = 1.0):
        return float(fn(a) * scale) if a else NA

    def summary(self) -> Dict[str, Any]:
        L, S, G = self.losses, self.step_times, self.grad_norms
        tot_step = float(np.sum(S)) if S else 0.0
        return {
            "n_batches": self.n_batches,
            "n_optimizer_steps": self.opt_steps,
            "n_skipped_steps": self.skipped_steps,
            "nan_or_inf_batches": self.bad_batches,
            "train_loss_min": self._f(L, np.min),
            "train_loss_max": self._f(L, np.max),
            "train_loss_std": self._f(L, np.std),
            "train_loss_median": self._f(L, np.median),
            "grad_norm_mean": self._f(G, np.mean),
            "grad_norm_max": self._f(G, np.max),
            "grad_norm_min": self._f(G, np.min),
            "grad_norm_std": self._f(G, np.std),
            "grad_norm_p50": self._p(G, 50),
            "grad_norm_p95": self._p(G, 95),
            "grad_norm_p99": self._p(G, 99),
            "grad_clip_hit_frac": (self.clip_hits / self.opt_steps)
                                  if self.opt_steps else 0.0,
            "step_time_mean_ms": self._f(S, np.mean, 1e3),
            "step_time_p50_ms": self._p(S, 50, 1e3),
            "step_time_p90_ms": self._p(S, 90, 1e3),
            "step_time_p99_ms": self._p(S, 99, 1e3),
            "step_time_max_ms": self._f(S, np.max, 1e3),
            "dataload_time_sec": float(np.sum(self.dataload_times)),
            "compute_time_sec": float(np.sum(self.compute_times)),
            "backward_time_sec": float(np.sum(self.backward_times)),
            "optimizer_time_sec": float(np.sum(self.optimizer_times)),
            "dataload_frac": (float(np.sum(self.dataload_times)) / tot_step)
                             if tot_step > 0 else NA,
        }

    def step_trace(self, max_points: int = 2000) -> Dict[str, List[float]]:
        """Downsampled per-step trace. Enough to plot a within-epoch slowdown,
        small enough that 240 epochs of it is still a few MB.
        """
        n = len(self.step_times)
        idx = (np.linspace(0, n - 1, min(max_points, n)).astype(int)
               if n else np.array([], dtype=int))
        def pick(seq):
            return [float(seq[i]) for i in idx if i < len(seq)]
        return {"step": idx.tolist(),
                "step_time_ms": [self.step_times[i] * 1e3 for i in idx],
                "loss": pick(self.losses), "lr": pick(self.lrs),
                "grad_norm": pick(self.grad_norms)}


@_no_grad()
def optimisation_health(model, prev_flat: Optional["torch.Tensor"] = None):
    """Weight norm, update norm, and the update-to-weight ratio.

    The update ratio (||dw|| / ||w||) is the single most useful number for
    spotting a broken learning rate without waiting for the loss curve to say
    so. Healthy training sits around 1e-3; 1e-1 means the LR is far too high,
    1e-6 means nothing is moving.
    """
    flat = torch.cat([p.detach().float().reshape(-1) for p in model.parameters()
                      if p.requires_grad])
    wn = float(flat.norm())
    un = ratio = NA
    if prev_flat is not None and prev_flat.numel() == flat.numel():
        un = float((flat - prev_flat).norm())
        ratio = un / max(1e-12, wn)
    return wn, un, ratio, flat


class SystemMonitor:
    """Background sampler for GPU utilisation, temperature, clocks, CPU and RAM.

    Samples EVERY visible GPU, not just device 0. The requirement says GPU
    utilisation "each GPU separate", and it is genuinely informative here: a
    dual-T4 Kaggle session trains on one card while the other sits idle, so an
    aggregate would report ~50% utilisation and hide the fact that half the
    allocation does nothing.

    Together with the power sampler this is what lets you answer, months later,
    "was that epoch slow because the GPU throttled, or because the dataloader
    starved it?" -- when the session is long gone and re-measuring is not an
    option.
    """

    def __init__(self, sample_hz: float = 1.0):
        self.interval = 1.0 / max(0.1, sample_hz)
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handles: List[Any] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self._nvml = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:
            self._psutil = self._proc = None

    @property
    def n_gpus(self) -> int:
        return len(self._handles)

    def _host(self) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        if self._psutil is None:
            return rec
        try:
            rec["cpu_percent"] = float(self._psutil.cpu_percent(interval=None))
            vm = self._psutil.virtual_memory()
            rec["ram_used_mb"] = float(vm.used / 1024 ** 2)
            rec["ram_total_mb"] = float(vm.total / 1024 ** 2)
            rec["ram_percent"] = float(vm.percent)
            rec["proc_rss_mb"] = float(self._proc.memory_info().rss / 1024 ** 2)
        except Exception:
            pass
        return rec

    def _sample(self) -> List[Dict[str, Any]]:
        base = {"unix_ts": time.time(), "datetime_utc": now_iso(),
                "monotonic_sec": time.monotonic(), **self._host()}
        if self._nvml is None or not self._handles:
            return [dict(base, gpu_index=-1)]
        out = []
        for i, h in enumerate(self._handles):
            rec = dict(base, gpu_index=i)
            nv = self._nvml
            for key, fn in (
                ("util_pct", lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu),
                ("mem_util_pct", lambda: nv.nvmlDeviceGetUtilizationRates(h).memory),
                ("temp_c", lambda: nv.nvmlDeviceGetTemperature(
                    h, nv.NVML_TEMPERATURE_GPU)),
                ("sm_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                ("mem_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                ("power_w", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000.0),
            ):
                try:
                    rec[key] = float(fn())
                except Exception:
                    pass
            try:
                mi = nv.nvmlDeviceGetMemoryInfo(h)
                rec["mem_used_mb"] = float(mi.used / 1024 ** 2)
                rec["mem_total_mb"] = float(mi.total / 1024 ** 2)
            except Exception:
                pass
            try:
                # Non-zero means the card is clocking down -- thermal, power cap,
                # or a hardware slowdown. Without it, a slow epoch is a mystery.
                rec["throttle_reasons"] = int(
                    nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
            except Exception:
                pass
            out.append(rec)
        return out

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.extend(self._sample())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sysmon")
        self._thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        return list(self.samples)

    @staticmethod
    def aggregate(samples: List[Dict[str, Any]],
                  n_gpu_cols: int = N_GPU_COLUMNS) -> Dict[str, Any]:
        """Collapse the sample stream into one row's worth of columns."""
        def agg(rows, key, fn):
            v = [r[key] for r in rows if key in r and r[key] == r[key]]
            return float(fn(v)) if v else NA

        out: Dict[str, Any] = {}
        for k, fn in (("cpu_percent", np.mean), ("ram_used_mb", np.mean),
                      ("ram_total_mb", np.max), ("ram_percent", np.mean),
                      ("proc_rss_mb", np.max)):
            out[k] = agg(samples, k, fn)

        by_gpu: Dict[int, List[Dict[str, Any]]] = {}
        for r in samples:
            by_gpu.setdefault(int(r.get("gpu_index", -1)), []).append(r)
        out["n_gpus_visible"] = len([g for g in by_gpu if g >= 0])

        for i in range(n_gpu_cols):
            rows = by_gpu.get(i, [])
            out[f"gpu{i}_util_mean_pct"] = agg(rows, "util_pct", np.mean)
            out[f"gpu{i}_util_max_pct"] = agg(rows, "util_pct", np.max)
            out[f"gpu{i}_mem_used_mb"] = agg(rows, "mem_used_mb", np.max)
            out[f"gpu{i}_mem_total_mb"] = agg(rows, "mem_total_mb", np.max)
            out[f"gpu{i}_mem_util_pct"] = agg(rows, "mem_util_pct", np.mean)
            out[f"gpu{i}_temp_mean_c"] = agg(rows, "temp_c", np.mean)
            out[f"gpu{i}_temp_max_c"] = agg(rows, "temp_c", np.max)
            out[f"gpu{i}_power_mean_w"] = agg(rows, "power_w", np.mean)
            out[f"gpu{i}_power_max_w"] = agg(rows, "power_w", np.max)
            out[f"gpu{i}_sm_clock_mhz"] = agg(rows, "sm_clock_mhz", np.mean)
            out[f"gpu{i}_mem_clock_mhz"] = agg(rows, "mem_clock_mhz", np.mean)
            out[f"gpu{i}_throttle_reasons"] = agg(rows, "throttle_reasons", np.max)
            # Integrate this card's own power draw over the epoch.
            t = [r["monotonic_sec"] for r in rows if "power_w" in r]
            w = [r["power_w"] for r in rows if "power_w" in r]
            if len(t) >= 2:
                o = np.argsort(t)
                tt, ww = np.asarray(t)[o], np.asarray(w)[o]
                area = np.trapezoid(ww, tt) if hasattr(np, "trapezoid") \
                    else np.trapz(ww, tt)
                out[f"gpu{i}_energy_j"] = float(area)
            else:
                out[f"gpu{i}_energy_j"] = NA
        return out


SYSTEM_SAMPLE_COLUMNS = [
    "unix_ts", "datetime_utc", "monotonic_sec", "epoch", "stage", "gpu_index",
    "util_pct", "mem_util_pct", "mem_used_mb", "mem_total_mb", "temp_c",
    "sm_clock_mhz", "mem_clock_mhz", "power_w", "throttle_reasons",
    "cpu_percent", "ram_used_mb", "ram_total_mb", "ram_percent", "proc_rss_mb",
]

ENERGY_SAMPLE_COLUMNS = [
    "unix_ts", "datetime_utc", "monotonic_sec", "epoch", "stage",
    "gpu_index", "power_w",
]


def build_optimizer(model, cfg):
    name = str(cfg.get("optimizer", "sgd")).lower()
    lr, wd = float(cfg["learning_rate"]), float(cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              momentum=float(cfg.get("momentum", 0.9)),
                              weight_decay=wd, nesterov=bool(cfg.get("nesterov", True)))
    elif name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"unknown optimizer {name}")

    sched_name = str(cfg.get("scheduler", "none")).lower()
    n_ep = int(cfg["num_epochs"])
    warm = int(cfg.get("warmup_epochs", 0))
    if sched_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep - warm))
    elif sched_name == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(m) for m in cfg.get("lr_milestones", [])],
            gamma=float(cfg.get("lr_gamma", 0.1)))
    else:
        sched = None
    return opt, sched


def calibration_metrics(probs: np.ndarray, labels: np.ndarray,
                        n_bins: int = 15) -> Dict[str, Any]:
    """ECE, MCE, NLL, Brier and the reliability-diagram bins.

    Q5's mechanism claim is that small students are MISCALIBRATED, so their own
    confidence is a poor gate for routing. Recording calibration every epoch
    costs one pass over probabilities we already have, and turns that claim
    from an assertion into something measured -- including the case where the
    method wins but the stated mechanism is wrong, which we would have to
    report.
    """
    n, C = probs.shape
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        k = int(m.sum())
        if k == 0:
            bins.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "confidence": NA, "accuracy": NA, "gap": NA})
            continue
        acc_b, conf_b = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc_b - conf_b)
        ece += (k / n) * gap
        mce = max(mce, gap)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": k,
                     "confidence": conf_b, "accuracy": acc_b,
                     "gap": float(acc_b - conf_b)})

    p_true = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    ent = float((-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)).mean())

    return {"ece": float(ece), "mce": float(mce), "nll": nll, "brier": brier,
            "confidence_mean": float(conf.mean()), "entropy_mean": ent,
            "overconfidence_gap": float(conf.mean() - correct.mean()),
            "bins": bins}


@_no_grad()
def evaluate(model, loader, device, amp: bool = True, criterion=None,
             collect_probs: bool = False, n_bins: int = 15) -> Dict[str, Any]:
    """Full evaluation pass: losses, accuracies, macro/micro/weighted P-R-F1,
    agreement statistics, and calibration.

    Everything is computed from ONE pass. The probability matrix is 10,000 x 100
    floats (~4 MB), which is cheap enough to keep and is what the confusion
    matrix, per-class table and reliability diagram are all derived from.
    """
    model.eval()
    crit = criterion or nn.CrossEntropyLoss()
    loss_sum = correct = correct5 = total = 0
    preds, targets, prob_chunks = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        pr = logits.argmax(1)
        correct += int((pr == y).sum().item())
        k = min(5, logits.size(1))
        if k > 1:
            _, t5 = logits.topk(k, dim=1)
            correct5 += int((t5 == y.unsqueeze(1)).any(1).sum().item())
        total += int(y.size(0))
        preds.extend(pr.cpu().tolist())
        targets.extend(y.cpu().tolist())
        prob_chunks.append(F.softmax(logits.float(), dim=1).cpu().numpy())

    probs = np.concatenate(prob_chunks) if prob_chunks else np.zeros((0, 1))
    y_true = np.asarray(targets)
    y_pred = np.asarray(preds)

    out: Dict[str, Any] = {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "accuracy_top5": correct5 / max(1, total),
        "preds": preds, "targets": targets, "n": total,
    }
    try:
        from sklearn.metrics import (precision_recall_fscore_support,
                                     balanced_accuracy_score, cohen_kappa_score,
                                     matthews_corrcoef)
        for avg in ("macro", "micro", "weighted"):
            pr_, rc_, f1_, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg, zero_division=0)
            out[f"precision_{avg}"] = float(pr_)
            out[f"recall_{avg}"] = float(rc_)
            out[f"f1_{avg}"] = float(f1_)
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception as e:
        for avg in ("macro", "micro", "weighted"):
            out[f"precision_{avg}"] = out[f"recall_{avg}"] = out[f"f1_{avg}"] = NA
        out["balanced_accuracy"] = out["cohen_kappa"] = out["matthews_corrcoef"] = NA
        out["metrics_error"] = str(e)[:120]
    # Legacy aliases used elsewhere in this module.
    out["precision"] = out.get("precision_macro", NA)
    out["recall"] = out.get("recall_macro", NA)
    out["f1"] = out.get("f1_macro", NA)

    if probs.size:
        out["calibration"] = calibration_metrics(probs, y_true, n_bins=n_bins)
    if collect_probs:
        out["probs"] = probs
    return out


FINAL_FIELDS = (
    ["run_id", "arch", "family", "dataset", "seed", "phase", "method",
     "config_hash", "sample_order_hash", "baseline_run_id",
     "num_epochs_planned", "num_epochs_run", "started_utc", "completed_utc",
     "account", "worker_id", "msc_lib_version", "torch_version", "cuda_version",
     "driver_version", "gpu_names", "n_gpus"]
    + ["top1_accuracy", "top5_accuracy", "val_loss",
       "f1_macro", "f1_micro", "f1_weighted",
       "precision_macro", "precision_micro", "precision_weighted",
       "recall_macro", "recall_micro", "recall_weighted",
       "balanced_accuracy", "cohen_kappa", "matthews_corrcoef",
       "worst_class_f1", "best_class_f1", "n_classes_below_50pct_f1"]
    + ["ece", "mce", "nll", "brier", "confidence_mean", "overconfidence_gap"]
    + ["params_total", "params_trainable", "params_nonzero", "sparsity_pct",
       "model_size_mb", "model_size_mb_fp16", "model_size_mb_int8",
       "flops", "macs", "flops_per_param",
       "n_layers", "n_conv_layers", "n_linear_layers"]
    + ["latency_bs1_mean_ms", "latency_bs1_median_ms", "latency_bs1_p90_ms",
       "latency_bs1_p99_ms", "latency_bs1_std_ms",
       "latency_bs32_median_ms", "latency_bs128_median_ms",
       "throughput_bs1_img_s", "throughput_bs32_img_s", "throughput_bs128_img_s",
       "warmup_batches_discarded", "n_repeats"]
    + ["train_energy_j", "train_energy_kwh", "train_co2_kg", "total_gpu_hours",
       "inference_energy_j_per_image", "inference_power_mean_w",
       "inference_co2_g_per_1k_images", "energy_per_accuracy_point"]
    + ["energy_reduction_pct", "accuracy_change_pts", "compression_ratio",
       "speedup_vs_baseline", "flops_reduction_pct"]
    + ["exit_accuracies_json", "msc_mean_depth_tau0.1", "msc_std_depth_tau0.1",
       "frac_irreducible_tau0.1", "reference_accuracy",
       "accuracy_gap_vs_reference", "recipe_ok"]
)


@_no_grad()
def benchmark_inference(model, device, batch_sizes: Sequence[int] = (1, 32, 128),
                        n_repeats: int = 5, n_iters: int = 30,
                        warmup: int = 10, image_size: int = 32,
                        measure_energy: bool = True) -> Dict[str, Any]:
    """Latency, throughput and inference energy.

    Methodology, because these numbers are easy to get wrong:
      * warm-up iterations are DISCARDED -- the first passes pay for cudnn
        autotuning and allocator warm-up and are not representative
      * `torch.cuda.synchronize()` around every timed region, or you time the
        kernel *launch* rather than the work
      * `n_repeats` independent measurements, median reported -- a single
        timing on a shared cloud GPU is noise

    Batch-1 latency is the number that matters for this project. Per-sample
    adaptive routing gives no wall-clock gain under batched inference unless
    the batch is split by route (protocol 7.2), so the deployment claim is
    scoped to the batch-1 / edge / streaming regime and measured there.
    """
    model.eval()
    out: Dict[str, Any] = {"warmup_batches_discarded": warmup,
                           "n_repeats": n_repeats}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, image_size, image_size, device=device)
        try:
            for _ in range(warmup):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            mon = GPUEnergyMonitor(sample_hz=20.0) if (
                measure_energy and bs == 1 and device.type == "cuda") else None
            if mon is not None:
                mon.start()

            per_iter = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                per_iter.append((time.perf_counter() - t0) / n_iters)

            samples = mon.stop() if mon is not None else []
            a = np.asarray(per_iter) * 1e3           # ms per forward pass
            out[f"latency_bs{bs}_median_ms"] = float(np.median(a))
            out[f"throughput_bs{bs}_img_s"] = float(bs / (np.median(a) / 1e3))
            if bs == 1:
                out.update({
                    "latency_bs1_mean_ms": float(a.mean()),
                    "latency_bs1_p90_ms": float(np.percentile(a, 90)),
                    "latency_bs1_p99_ms": float(np.percentile(a, 99)),
                    "latency_bs1_std_ms": float(a.std()),
                })
                if samples:
                    total_s = float(np.sum(per_iter) * n_iters)
                    j = GPUEnergyMonitor.integrate_j(samples, total_s)
                    n_img = n_repeats * n_iters * bs
                    out["inference_energy_j_per_image"] = j / max(1, n_img)
                    out.update({k.replace("power_", "inference_power_"): v
                                for k, v in GPUEnergyMonitor.power_stats(samples).items()
                                if k == "power_mean_w"})
        except RuntimeError as e:
            # Out of memory at a large batch is expected on a T4 for some models
            # and is not a failure of the run.
            out[f"latency_bs{bs}_median_ms"] = NA
            out[f"throughput_bs{bs}_img_s"] = NA
            out[f"bs{bs}_error"] = f"{type(e).__name__}: {str(e)[:80]}"
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def model_statistics(model, flops: Optional[int] = None) -> Dict[str, Any]:
    """Parameter counts, sparsity, size in three precisions, layer census."""
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    nonzero = int(sum(int((p != 0).sum()) for p in model.parameters()))
    bytes_p = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_b = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (bytes_p + bytes_b) / 1024 ** 2
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return {
        "params_total": total, "params_trainable": trainable,
        "params_nonzero": nonzero,
        "sparsity_pct": 100.0 * (1.0 - nonzero / max(1, total)),
        "model_size_mb": size_mb,
        "model_size_mb_fp16": size_mb / 2.0,
        "model_size_mb_int8": size_mb / 4.0,
        "flops": int(flops) if flops else NA,
        "macs": int(flops // 2) if flops else NA,
        "flops_per_param": (float(flops) / max(1, total)) if flops else NA,
        "n_layers": sum(1 for _ in model.modules()),
        "n_conv_layers": n_conv, "n_linear_layers": n_lin,
    }


def final_evaluation(cfg: Dict[str, Any], model, val_loader, device, classes,
                     run_dir, budgets: Optional[Dict[str, Any]] = None,
                     train_summary: Optional[Dict[str, Any]] = None,
                     baseline: Optional[Dict[str, Any]] = None,
                     amp: bool = True, hub: Optional[MSCHub] = None,
                     ) -> Dict[str, Any]:
    """Everything in requirement 15.2, in one pass over the trained model.

    Writes metrics/final.csv, final.json, confusion_matrix.csv, per_class.csv,
    calibration.csv and inference_bench.csv into the run folder.

    `baseline` supplies the reference for the comparative metrics (energy
    reduction, accuracy change, compression, speedup). Without one, those read
    against the model's own full-precision self and are 0/0/1.0 -- which is
    correct, not missing. `baseline_run_id` records what each was measured
    against, because a compression ratio with no stated reference is
    uninterpretable.
    """
    L = run_layout(Path(run_dir).parent.parent, cfg["run_id"])
    met = ensure_dir(L["metrics"])

    ev = evaluate(model, val_loader, device, amp=amp, collect_probs=True)
    y_true, y_pred = np.asarray(ev["targets"]), np.asarray(ev["preds"])
    cal = ev.get("calibration", {}) or {}

    cm = confusion_matrix_frame(y_true, y_pred, classes)
    pc = per_class_frame(y_true, y_pred, classes)
    if pd is not None:
        cm.to_csv(met / "confusion_matrix.csv")
        pc.to_csv(met / "per_class.csv", index=False)
        if cal.get("bins"):
            pd.DataFrame(cal["bins"]).to_csv(met / "calibration.csv", index=False)

    bench = benchmark_inference(model, device,
                                image_size=int(cfg.get("image_size", 32)))
    if pd is not None:
        pd.DataFrame([bench]).to_csv(met / "inference_bench.csv", index=False)

    flops = (budgets or {}).get("full_flops")
    stats = model_statistics(model, flops)

    ts = train_summary or {}
    train_j = float(ts.get("total_energy_j") or 0.0)
    acc = float(ev["accuracy"])
    carbon = float(cfg.get("carbon_intensity_kg_per_kwh", 0.475))
    inf_j = bench.get("inference_energy_j_per_image")

    row: Dict[str, Any] = {
        "run_id": cfg["run_id"], "arch": cfg["arch"],
        "family": cfg.get("family", NA), "dataset": cfg["dataset_name"],
        "seed": int(cfg["seed"]), "phase": cfg.get("phase", NA),
        "method": cfg.get("method", NA), "config_hash": cfg["config_hash"],
        "sample_order_hash": cfg.get("sample_order_hash", NA),
        "baseline_run_id": (baseline or {}).get("run_id", "self"),
        "num_epochs_planned": int(cfg.get("num_epochs", 0)),
        "num_epochs_run": ts.get("num_epochs_run", NA),
        "started_utc": ts.get("started_utc", NA), "completed_utc": now_iso(),
        "account": cfg.get("account", NA), "worker_id": cfg.get("worker_id", 0),
        "msc_lib_version": __version__,
        "torch_version": torch.__version__ if _TORCH_OK else NA,
        "cuda_version": torch.version.cuda if _TORCH_OK else NA,
        "driver_version": environment_report().get("nvidia_driver", NA),
        "gpu_names": ";".join(
            torch.cuda.get_device_properties(i).name
            for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else NA,
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,

        "top1_accuracy": acc, "top5_accuracy": float(ev["accuracy_top5"]),
        "val_loss": float(ev["loss"]),
        **{k: ev.get(k, NA) for k in
           ("f1_macro", "f1_micro", "f1_weighted", "precision_macro",
            "precision_micro", "precision_weighted", "recall_macro",
            "recall_micro", "recall_weighted", "balanced_accuracy",
            "cohen_kappa", "matthews_corrcoef")},

        "ece": cal.get("ece", NA), "mce": cal.get("mce", NA),
        "nll": cal.get("nll", NA), "brier": cal.get("brier", NA),
        "confidence_mean": cal.get("confidence_mean", NA),
        "overconfidence_gap": cal.get("overconfidence_gap", NA),

        **stats, **bench,

        "train_energy_j": train_j or NA,
        "train_energy_kwh": energy_to_kwh(train_j) if train_j else NA,
        "train_co2_kg": energy_to_co2_kg(train_j, carbon) if train_j else NA,
        "total_gpu_hours": (float(ts["total_time_sec"]) / 3600.0
                            if ts.get("total_time_sec") else NA),
        "inference_energy_j_per_image": inf_j if inf_j is not None else NA,
        "inference_co2_g_per_1k_images": (
            energy_to_co2_kg(inf_j * 1000.0, carbon) * 1000.0
            if inf_j is not None else NA),
        "energy_per_accuracy_point": (energy_to_kwh(train_j) / max(1e-9, acc * 100)
                                      if train_j else NA),
        "reference_accuracy": REFERENCE_ACC.get(cfg["arch"], NA),
    }

    # Comparative metrics. Meaningful only against a stated reference.
    if baseline:
        b_acc = float(baseline.get("top1_accuracy", acc))
        b_size = float(baseline.get("model_size_mb", stats["model_size_mb"]))
        b_lat = baseline.get("latency_bs1_median_ms")
        b_flops = baseline.get("flops")
        b_energy = baseline.get("train_energy_j")
        row["accuracy_change_pts"] = (acc - b_acc) * 100.0
        row["compression_ratio"] = b_size / max(1e-9, stats["model_size_mb"])
        row["speedup_vs_baseline"] = (
            float(b_lat) / max(1e-9, bench.get("latency_bs1_median_ms", np.nan))
            if b_lat and bench.get("latency_bs1_median_ms") not in (None, NA) else NA)
        row["flops_reduction_pct"] = (
            100.0 * (1.0 - float(flops) / float(b_flops))
            if flops and b_flops else NA)
        row["energy_reduction_pct"] = (
            100.0 * (1.0 - train_j / float(b_energy))
            if train_j and b_energy else NA)
    else:
        # The model IS its own reference at full compute.
        row.update({"accuracy_change_pts": 0.0, "compression_ratio": 1.0,
                    "speedup_vs_baseline": 1.0, "flops_reduction_pct": 0.0,
                    "energy_reduction_pct": 0.0})

    ref = REFERENCE_ACC.get(cfg["arch"])
    if ref is not None and int(cfg.get("num_epochs", 0)) >= 100:
        row["accuracy_gap_vs_reference"] = ref - acc * 100.0
        row["recipe_ok"] = bool((ref - acc * 100.0) <= 1.0)

    if pd is not None and len(pc):
        row["worst_class_f1"] = float(pc.f1.min())
        row["best_class_f1"] = float(pc.f1.max())
        row["n_classes_below_50pct_f1"] = int((pc.f1 < 0.5).sum())

    for c in FINAL_FIELDS:
        row.setdefault(c, NA)

    atomic_write_json(met / "final.json", row)
    if pd is not None:
        pd.DataFrame([{k: row.get(k, NA) for k in FINAL_FIELDS}]).to_csv(
            met / "final.csv", index=False)
    log(f"final evaluation written: top1={acc:.4f} "
        f"top5={ev['accuracy_top5']:.4f} ece={cal.get('ece', float('nan')):.4f} "
        f"bs1={bench.get('latency_bs1_median_ms', float('nan')):.2f} ms", "EVAL")
    return row


def confusion_matrix_frame(y_true, y_pred, classes: Sequence[str]):
    """Full confusion matrix as a labelled DataFrame (true x predicted)."""
    C = len(classes)
    m = np.zeros((C, C), dtype=np.int64)
    for t, p_ in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p_)] += 1
    if pd is None:
        return m
    return pd.DataFrame(m, index=[f"true_{c}" for c in classes],
                        columns=[f"pred_{c}" for c in classes])


def per_class_frame(y_true, y_pred, classes: Sequence[str]):
    """Precision / recall / F1 / support / accuracy for every class.

    Worth having on CIFAR-100 specifically: 100 classes at ~600 test images
    each means a headline accuracy hides a lot, and per-class support is what
    tells you whether a low F1 is a hard class or a rare one.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
        pr, rc, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    except Exception:
        return pd.DataFrame() if pd is not None else []
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = [float((y_pred[y_true == i] == i).mean()) if int((y_true == i).sum()) else 0.0
           for i in range(len(classes))]
    rows = [{"class_index": i, "class_name": classes[i], "precision": float(pr[i]),
             "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i]),
             "accuracy": acc[i]} for i in range(len(classes))]
    return pd.DataFrame(rows) if pd is not None else rows


def save_checkpoint(path, cfg, model, optimizer, scheduler, scaler, epoch: int,
                    best_metric: float, dynamics: Optional[TrainingDynamics],
                    wall_seconds: float, energy_joules: float) -> None:
    """The full resumability contract of 02_ENGINEERING_SPEC.md 3.

    Every field here prevents a specific silent corruption:
      scaler   -- omit it and AMP loss scale resets, so the first post-resume
                  steps behave differently from an uninterrupted run
      rng      -- omit it and augmentation/shuffling diverge, which makes the
                  seeds meaningless and destroys Q1
      config_hash -- omit it and you resume under an edited config, forever
      energy/wall -- omit them and cumulative totals restart at zero mid-run
    """
    atomic_save_torch(path, {
        "run_id": cfg["run_id"],
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "best_metric": float(best_metric),
        "config_hash": cfg["config_hash"],
        "wall_seconds": float(wall_seconds),
        "energy_joules": float(energy_joules),
        "dynamics": dynamics.state_dict() if dynamics is not None else None,
        "msc_lib_version": __version__,
        "saved_utc": now_iso(),
    })


def msckd_dry_run(cfg: Dict[str, Any], teacher, device, amp: bool,
                  alpha: float, beta: float, temperature: float
                  ) -> Tuple[bool, str]:
    """Exercise the whole MSC-KD step on two synthetic images, before any
    expensive work. Returns (ok, reason).

    **O-19**, opened after D-21 and D-22 each cost an hour of GPU time to
    surface. `train_msc_kd` loads a teacher, trains exit heads and sweeps 50,000
    images before the first student batch, and writes its first history row only
    at the *end* of that epoch. Both defects were trivial and both hid behind
    that hour.

    This runs the same objects the real loop uses -- `MSCStudent` under
    `autocast`, `MSCLoss`, `backward`, and one `msckd_history_row` through
    `append_history_row` -- on a 2-image batch and a temp file. Under a second,
    no dataset, no teacher sweep.
    """
    if not _TORCH_OK:
        return True, "torch unavailable; dry run skipped"
    import tempfile as _tf
    try:
        n_cls = int(cfg["num_classes"])
        student = MSCStudent(build_model(cfg["arch"], n_cls), n_cls, 5).to(device)
        x = torch.randn(2, 3, int(cfg.get("image_size", 32)),
                        int(cfg.get("image_size", 32)), device=device)
        y = torch.zeros(2, dtype=torch.long, device=device)
        tgt = torch.zeros(2, 5, device=device)
        tgt[:, 3:] = 1.0
        opt = torch.optim.SGD(student.parameters(), lr=1e-4)
        lossfn = MSCLoss(alpha=alpha, beta=beta, temperature=temperature)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits, suff, _ = student(x, suff_logits=True)
            loss, parts = lossfn(s_logits[-1], t_logits, y, suff, tgt)
        loss.backward()
        opt.step()
        if not bool(torch.isfinite(loss).item()):
            return False, f"loss is not finite ({float(loss)})"

        # The history write is the OTHER thing that only fails after an epoch.
        with _tf.TemporaryDirectory() as td:
            row = msckd_history_row(
                run_id=cfg["run_id"], cfg=cfg, epoch=0,
                agg={k: float(parts.get(k, 0.0)) for k in
                     ("loss", "ce", "kd", "msc")},
                nb=1,
                val={"loss": 0.0, "accuracy_top5": 0.0, "f1": 0.0,
                     "precision": 0.0, "recall": 0.0},
                acc=0.0, best_before=0.0, lr=1e-4, amp=amp, dt=1.0,
                cum_time=1.0, cum_energy=0.0, n_train_images=2,
                alpha=alpha, beta=beta, temperature=temperature)
            append_history_row(Path(td) / "epochs.csv", row, strict=True)
        del student, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True, "ok"
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def exit_heads_path(work, run_id: str) -> Path:
    """THE canonical location of a run's trained exit heads.

    **D-23.** No such function existed, so the writer and every reader
    hard-coded a path of their own -- and they disagreed. `run_oracle` writes to
    the run root; `train_msc_kd` looked in `checkpoints/`. The teacher's heads
    were therefore never found, and **every MSC-KD run retrained them from
    scratch**: ~20 epochs of GPU time per run, nine times over, for a file
    already sitting on HuggingFace.

    D-16 recorded this split as *"cosmetic ... Contamination: none. Nothing
    reads the path by convention."* That was wrong. Three call sites read it by
    convention, and one of them was in the hot path of the entire method.
    """
    return run_layout(work, run_id)["base"] / "exit_heads.pt"


def find_exit_heads(work, run_id: str) -> Optional[Path]:
    """Canonical path, or the legacy `checkpoints/` one if that is what exists.

    Reads tolerate both locations so runs written before D-23 still work;
    writes only ever use `exit_heads_path`. Returns None if neither exists.
    """
    L = run_layout(work, run_id)
    for p in (L["base"] / "exit_heads.pt", L["checkpoints"] / "exit_heads.pt"):
        if p.exists():
            return p
    return None


_HISTORY_SET = frozenset(HISTORY_FIELDS)
_HISTORY_WARNED: Set[str] = set()


def msckd_history_row(run_id: str, cfg: Dict[str, Any], epoch: int,
                      agg: Dict[str, float], nb: int, val: Dict[str, Any],
                      acc: float, best_before: float, lr: float, amp: bool,
                      dt: float, cum_time: float, cum_energy: float,
                      n_train_images: int, alpha: float, beta: float,
                      temperature: float) -> Dict[str, Any]:
    """One MSC-KD epoch, as a `HISTORY_FIELDS`-valid row.

    Extracted from the training loop so the self-test can validate its key set
    **offline, with no GPU** (D-22). Previously the only way to discover that
    this row used `f1_score` where the schema says `f1_macro` was to finish an
    epoch of real training on a real teacher -- about an hour in.

    It also now records the **three-term loss decomposition**, which the old row
    computed every epoch and threw away. For a method notebook that is the most
    important curve in the file: the whole argument is about how L_CE, L_KD and
    L_MSC trade off, and none of it was being written down.
    """
    per = lambda k: agg[k] / max(1, nb)
    return {
        # identity -- the atlas rows carry these, so these must too or the
        # combined table cannot be grouped by architecture or method.
        "run_id": run_id, "epoch": int(epoch), "timestamp_utc": now_iso(),
        "unix_ts": time.time(),
        "arch": cfg.get("arch", NA), "family": cfg.get("family", NA),
        "dataset": cfg.get("dataset", NA), "seed": cfg.get("seed", NA),
        "phase": cfg.get("phase", NA), "method": cfg.get("method", NA),
        "config_hash": cfg.get("config_hash", NA),

        # learning
        "train_loss": per("loss"), "val_loss": float(val["loss"]),
        "train_accuracy": float("nan"), "val_accuracy": float(acc),
        "val_accuracy_top5": float(val["accuracy_top5"]),
        "f1_macro": float(val["f1"]),
        "precision_macro": float(val["precision"]),
        "recall_macro": float(val["recall"]),
        "best_val_accuracy_so_far": float(max(best_before, acc)),
        "is_best": bool(acc > best_before),

        # the three-term decomposition -- the point of the whole notebook
        "loss_total": per("loss"), "loss_ce": per("ce"),
        "loss_kd": per("kd"), "loss_msc": per("msc"),
        "alpha": float(alpha), "beta": float(beta),
        "temperature": float(temperature),

        # optimisation
        "learning_rate": float(lr),
        "batch_size": int(cfg["batch_size"]),
        "effective_batch_size": int(cfg["batch_size"]),
        "amp_enabled": bool(amp), "n_batches": int(nb),

        # time
        "epoch_time_sec": float(dt), "cumulative_time_sec": float(cum_time),
        "throughput_train_img_s": n_train_images / max(1e-9, dt),
        "samples_seen": int(nb) * int(cfg["batch_size"]),

        # energy (MSC-KD does not run the power sampler; recorded as zero
        # rather than omitted so the column stays type-stable across phases)
        "epoch_energy_j": 0.0, "cumulative_energy_j": float(cum_energy),
        "epoch_co2_kg": 0.0, "cumulative_co2_kg": 0.0, "peak_vram_mb": 0.0,
    }


def append_history_row(path, row: Dict[str, Any], strict: bool = True) -> None:
    """Append one epoch to a run's `metrics/epochs.csv`, schema-checked.

    **D-22.** The two training paths disagreed about what an unknown column
    means, and both answers were wrong:

    - `train_msc_kd` used `csv.DictWriter`'s default, which **raises** -- at the
      END of the first epoch, after the work is done and unrecoverable. Five
      misspelled keys (`f1_score` for `f1_macro`, `precision` for
      `precision_macro`, `recall`, `grad_norm`, `throughput_img_s`) therefore
      killed every MSC-KD run at epoch 0, an hour into setup, nine times over.
    - `train_backbone` used `extrasaction="ignore"`, which **silently drops**
      them. That is worse in the long run: a typo becomes a column of blanks in
      a 171-column table nobody reads by eye, and the standing instruction on
      this project is that we train once and collect everything.

    So: `strict=True` fails loudly *and* names the column you probably meant.
    `strict=False` still writes -- `train_backbone` merges dynamically-built GPU
    and power dicts whose keys legitimately vary by machine -- but **logs what
    it dropped**, once per key, so silent loss becomes visible loss.
    """
    unknown = [k for k in row if k not in _HISTORY_SET]
    if unknown:
        if strict:
            hint = {}
            for u in unknown:
                stem = u.split("_")[0]
                near = [c for c in HISTORY_FIELDS if c.startswith(stem)]
                if near:
                    hint[u] = near[:3]
            raise KeyError(
                f"{len(unknown)} column(s) are not in HISTORY_FIELDS: "
                f"{sorted(unknown)}."
                + (f" Did you mean: {hint}?" if hint else "")
                + " Either use the documented name or add the column to "
                  "HISTORY_FIELDS (and to 06_DATA_SCHEMA.md).")
        fresh = [k for k in unknown if k not in _HISTORY_WARNED]
        if fresh:
            _HISTORY_WARNED.update(fresh)
            log(f"dropping {len(fresh)} column(s) absent from HISTORY_FIELDS: "
                f"{sorted(fresh)[:8]}. They will NOT be in epochs.csv.",
                "SCHEMA")
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def ensure_run_local(hub, work, run_id: str, why: str = "") -> bool:
    """Pull a run's own artifacts back from HF before concluding it never ran.

    **D-19.** `load_checkpoint` returns "start from scratch" when the file is
    merely absent. That is correct in isolation and catastrophic in context:
    Kaggle wipes the scratch disk between sessions, so on a fresh session
    *every* run looks unstarted unless something pulled it back first.

    `run_oracle` already did this for itself. Neither training entry point did,
    so both depended entirely on the notebook having called `sync_state` with
    the right scope beforehand -- an invisible coupling between a cell near the
    top of a notebook and a decision taken deep inside the library. When that
    coupling broke for NB13, nine completed MSC-KD runs restarted at epoch 0
    and nothing said a word.

    Cheap when the checkpoint is already local, which is the common case within
    a session. Returns True if a resumable checkpoint is present afterwards.
    """
    L = run_layout(work, run_id)
    ck = L["checkpoints"] / "ckpt_last.pt"
    if ck.exists():
        return True
    if hub is None or not getattr(hub, "enabled", False):
        return False
    log(f"no local checkpoint for {run_id} -- pulling from HF before deciding "
        f"whether it has already run" + (f" ({why})" if why else ""), "RESUME")
    try:
        hub.hub.download(Path(work), allow_patterns=[f"runs/{run_id}/**"],
                         quiet=True)
    except Exception as e:                                   # noqa: BLE001
        log(f"pull failed for {run_id}: {type(e).__name__}: {e}", "RESUME")
        return False
    if ck.exists():
        log(f"recovered checkpoint for {run_id} from HF", "RESUME")
        return True
    if (L["base"] / "summary.json").exists():
        log(f"{run_id} has a summary.json on HF but no ckpt_last.pt -- it "
            f"finished and its checkpoint was pruned. Nothing to resume.",
            "RESUME")
    return False


def already_finished(hub, work, run_id: str, cfg: Dict[str, Any],
                     registry=None) -> Optional[Dict[str, Any]]:
    """Has this run already finished, on the evidence of its own artifacts?

    **D-19.** `can_claim` consults the ledger and nothing else, so a lost or
    unpushed completion event is indistinguishable from "never ran" -- and the
    programmed response to "never ran" is to spend the GPU-hours again. The
    run's `summary.json` is durable evidence and lives on HF whether or not the
    ledger event survived the session.

    `run_oracle` has always had this guard (`per-sample tables already present`).
    The two *training* entry points did not, which is why a lost ledger could
    cost 30 GPU-hours rather than 30 seconds.

    Self-healing: when the artifact says finished but the ledger disagrees, the
    completion event is re-emitted so the next worker inherits the answer
    instead of rediscovering it.
    """
    if cfg.get("force_rerun"):
        return None
    ensure_run_local(hub, work, run_id, why="completion check")
    p = run_layout(work, run_id)["base"] / "summary.json"
    if not p.exists():
        return None
    prev = read_json(p, default=None)
    if not isinstance(prev, dict):
        return None
    ran = int(prev.get("num_epochs_run") or 0)
    want = int(cfg.get("num_epochs") or 0)
    if ran < want:
        return None
    log(f"{run_id} already finished: {ran}/{want} epochs, "
        f"acc={prev.get('best_accuracy')}. NOT retraining -- pass "
        f"force_rerun=True to override.", "DONE")
    if registry is not None:
        try:
            st = registry.latest().get(run_id, {}).get("state")
            if st != "completed":
                log(f"ledger said '{st}' but the artifact says finished -- "
                    f"repairing the ledger", "DONE")
                registry.finish(run_id, **{k: prev[k] for k in
                                           ("best_accuracy", "num_epochs_run",
                                            "final_accuracy")
                                           if k in prev})
        except Exception as e:                               # noqa: BLE001
            log(f"ledger repair skipped: {type(e).__name__}: {e}", "DONE")
    return {**prev, "status": "cached"}


def load_checkpoint(path, cfg, model, optimizer, scheduler, scaler,
                    dynamics: Optional[TrainingDynamics], device,
                    strict_hash: bool = True) -> Dict[str, Any]:
    """Returns {start_epoch, best_metric, wall_seconds, energy_joules, resumed}."""
    blank = {"start_epoch": 0, "best_metric": 0.0, "wall_seconds": 0.0,
             "energy_joules": 0.0, "resumed": False, "rng_restored": False}
    p = Path(path)
    if not p.exists():
        return blank
    try:
        try:
            ck = torch.load(p, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(p, map_location=device)
    except Exception as e:
        log(f"could not read {p.name}: {e} -- starting fresh", "RESUME")
        return blank

    if ck.get("config_hash") != cfg["config_hash"]:
        msg = (f"config_hash mismatch for {cfg['run_id']}: "
               f"checkpoint {str(ck.get('config_hash'))[:12]} != "
               f"config {cfg['config_hash'][:12]}")
        if strict_hash:
            # Fail loudly. A silent mismatch means you are continuing a run
            # under a config that has been edited since it started, and nobody
            # ever notices until the numbers do not reproduce.
            raise RuntimeError(
                msg + "\nThe config changed since this run started. Either restore "
                      "the original config, or set force_rerun=True to discard the "
                      "checkpoint and retrain from scratch.")
        log(msg + " -- starting fresh", "RESUME")
        return blank

    try:
        model.load_state_dict(ck["model"], strict=True)
    except Exception as e:
        log(f"state_dict mismatch: {e} -- starting fresh", "RESUME")
        return blank
    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if obj is not None and ck.get(key) is not None:
            try:
                obj.load_state_dict(ck[key])
            except Exception as e:
                log(f"{key} restore failed: {e}", "RESUME")
    rng_ok = restore_rng_state(ck.get("rng"))
    if dynamics is not None and ck.get("dynamics") is not None:
        dynamics.load_state_dict(ck["dynamics"])
    return {"start_epoch": int(ck.get("epoch", -1)) + 1,
            "best_metric": float(ck.get("best_metric", 0.0)),
            "wall_seconds": float(ck.get("wall_seconds", 0.0)),
            "energy_joules": float(ck.get("energy_joules", 0.0)),
            "resumed": True, "rng_restored": rng_ok}


def _truncate_history(path: Path, start_epoch: int) -> None:
    """Drop rows at or beyond the resume point.

    A milestone push can land after the checkpoint was written, so history.csv
    may contain epochs the checkpoint does not know about. Without truncation
    the resumed run appends duplicate epoch numbers and every downstream
    cumulative statistic is wrong.
    """
    if not path.exists() or pd is None:
        return
    try:
        h = pd.read_csv(path)
        if h.empty:
            return
        h = h[h["epoch"] < start_epoch]
        h.to_csv(path, index=False)
    except Exception as e:
        log(f"history truncate failed: {e}", "RESUME")


def train_backbone(cfg: Dict[str, Any], hub: MSCHub, registry: RunRegistry,
                   work_root=None, data_root_out=None,
                   show_progress: bool = True) -> Dict[str, Any]:
    """One backbone run, fully resumable, HF-first.

    Push policy:
        - every `timer_push_sec` (default 1800)
        - every `milestone_push_every_epochs` epochs
        - on a new best, but suppressed if fewer than 3 epochs since the last
          push (early on, every epoch is a new best, which would defeat batching)
        - on interrupt / SIGTERM / exception / session expiry: immediate,
          blocking, then stop
    """
    if not _TORCH_OK:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERR}")

    run_id = cfg["run_id"]
    work = Path(work_root or (WORK_ROOT / "msc"))
    data_out = Path(data_root_out or (work / "data"))
    L = run_layout(work, run_id)
    run_dir = ensure_dir(L["base"])
    for _s in RUN_SUBDIRS:
        ensure_dir(L[_s])
    log_dir = L["telemetry"]          # raw sample streams
    met_dir = L["metrics"]            # the tables
    ckpt_last = L["checkpoints"] / "ckpt_last.pt"
    ckpt_best = L["checkpoints"] / "ckpt_best.pt"
    history_path = met_dir / "epochs.csv"
    energy_path = log_dir / "energy_samples.csv"

    sync = RunSync(hub, run_id, run_dir, data_out)

    # --- claim -----------------------------------------------------------
    registry.pull()
    ok, why = registry.can_claim(run_id, force=bool(cfg.get("force_rerun")))
    if not ok:
        log(f"SKIP {run_id}: {why}", "CLAIM")
        return {"run_id": run_id, "status": "skipped", "reason": why}
    log(f"claiming {run_id} ({why})", "CLAIM")

    # D-19: the ledger is not the only evidence. Check the artifact before
    # spending the GPU-hours again.
    _cached = already_finished(hub, work, run_id, cfg, registry)
    if _cached is not None:
        return _cached

    if cfg.get("force_rerun") and run_dir.exists():
        log(f"force_rerun -- wiping {run_dir}", "RUN")
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)
        L = run_layout(work, run_id)
        run_dir = ensure_dir(L["base"])
        for _s in RUN_SUBDIRS:
            ensure_dir(L[_s])
        log_dir, met_dir = L["telemetry"], L["metrics"]

    # config.yaml is frozen at run start and never edited.
    atomic_write_yaml(run_dir / "config.yaml", cfg)
    atomic_write_json(L["env"] / "environment.json", environment_report())
    atomic_write_text(run_dir / "config_hash.txt", cfg["config_hash"])

    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("no CUDA -- energy logging will be empty and this will be very slow", "WARN")

    train_loader, val_loader, holdout_loader, classes, order_hash = build_loaders(cfg)
    cfg["sample_order_hash"] = order_hash
    n_train = len(train_loader.dataset)

    model = build_model(cfg["arch"], cfg["num_classes"]).to(device)
    optimizer, scheduler = build_optimizer(model, cfg)
    amp = bool(cfg.get("amp_enabled", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    dynamics = TrainingDynamics(n_train, el2n_epoch=int(cfg.get("el2n_epoch", 10)))

    # --- resume ----------------------------------------------------------
    # D-19: pull this run's own artifacts first. Without it, resume silently
    # depends on the notebook having called sync_state with checkpoints in
    # scope, and a fresh Kaggle session makes every run look unstarted.
    ensure_run_local(hub, work, run_id, why="backbone resume")
    st = load_checkpoint(ckpt_last, cfg, model, optimizer, scheduler, scaler,
                         dynamics, device, strict_hash=not cfg.get("force_rerun"))
    start_epoch = st["start_epoch"]
    best_metric = st["best_metric"]
    cumulative_time = st["wall_seconds"]
    cumulative_energy = st["energy_joules"]
    cumulative_co2 = energy_to_co2_kg(cumulative_energy,
                                      float(cfg.get("carbon_intensity_kg_per_kwh", 0.475)))
    if st["resumed"]:
        _truncate_history(history_path, start_epoch)
        log(f"{run_id} resuming at epoch {start_epoch} "
            f"(best={best_metric:.4f}, rng_restored={st['rng_restored']})", "RESUME")
        if not st["rng_restored"]:
            log("RNG state could not be restored -- augmentation order will differ "
                "from an uninterrupted run. Note this in the run record.", "WARN")
    else:
        log(f"{run_id} starting fresh", "RUN")

    num_epochs = int(cfg["num_epochs"])
    accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    warm = int(cfg.get("warmup_epochs", 0))
    base_lr = float(cfg["learning_rate"])
    milestone_every = max(1, int(cfg.get("milestone_push_every_epochs", 10)))
    timer_sec = float(cfg.get("timer_push_sec", 1800))
    carbon = float(cfg.get("carbon_intensity_kg_per_kwh", 0.475))
    clip = float(cfg.get("grad_clip_norm", 0.0))
    last_push_epoch = -10 ** 9
    cumulative_samples = 0
    cumulative_steps = 0
    epochs_since_best = 0
    loss_extra: Dict[str, Any] = {}       # optional loss terms, NA when absent
    prev_flat = None                      # for the update-to-weight ratio
    state = {"epoch": start_epoch - 1, "best": best_metric}

    registry.claim(run_id, arch=cfg["arch"], dataset=cfg["dataset_name"],
                   seed=cfg["seed"], phase=cfg["phase"], num_epochs=num_epochs,
                   config_hash=cfg["config_hash"])

    def _emergency_flush(reason: str) -> None:
        try:
            save_checkpoint(ckpt_last, cfg, model, optimizer, scheduler, scaler,
                            state["epoch"], state["best"], dynamics,
                            cumulative_time, cumulative_energy)
        except Exception:
            traceback.print_exc()
        try:
            _write_dynamics(L["per_sample"], dynamics)
        except Exception:
            pass
        registry.heartbeat(run_id, run_dir, state="paused", epoch=state["epoch"],
                           best_metric=state["best"], reason=reason)
        registry.pause(run_id, epoch=state["epoch"], best_metric=state["best"],
                       reason=reason)
        sync.push_all(heavy=True)
        sync.flush(timeout=600)
        hub.print_stats()

    guard = LifecycleGuard(_emergency_flush,
                           session_limit_h=float(cfg.get("session_limit_h", 8.5))).install()

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    try:
        for epoch in range(start_epoch, num_epochs):
            if warm > 0 and epoch < warm:
                lr = base_lr * float(epoch + 1) / float(warm)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            model.train()
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.reset_accumulated_memory_stats(device)
            mon = GPUEnergyMonitor(sample_hz=float(cfg.get("energy_sample_hz", 10.0)))
            sysmon = SystemMonitor(sample_hz=float(cfg.get("sysmon_hz", 1.0)))
            mon.start()
            sysmon.start()
            tel = EpochTelemetry()

            run_loss = correct = total = 0
            optimizer.zero_grad(set_to_none=True)
            it = train_loader
            if tqdm is not None and show_progress:
                it = tqdm(train_loader, desc=f"{run_id} ep {epoch+1}/{num_epochs}",
                          leave=False, dynamic_ncols=True, mininterval=2.0)

            _t_batch = time.time()
            for step, batch in enumerate(it):
                # Time spent waiting for data vs. time spent computing. If
                # dataload_frac is high the GPU is starving and the fix is the
                # loader, not the model -- a distinction that is impossible to
                # recover after the fact.
                _t_loaded = time.time()
                load_t = _t_loaded - _t_batch

                x, y, idx = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss / accum).backward()

                did_step, gn_val, clipped = False, None, False
                if ((step + 1) % accum == 0) or ((step + 1) == len(train_loader)):
                    if clip > 0:
                        scaler.unscale_(optimizer)
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        gn_val = float(gn)
                        clipped = gn_val > clip
                    else:
                        # Measure the gradient norm even when not clipping --
                        # it is the cheapest early warning of a diverging run,
                        # and only computed once per optimizer step.
                        scaler.unscale_(optimizer)
                        gn_val = float(torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float("inf")))
                    _scale_before = scaler.get_scale() if amp else 0.0
                    scaler.step(optimizer)
                    scaler.update()
                    if amp and scaler.get_scale() < _scale_before:
                        # AMP halved the loss scale: that step's gradients
                        # overflowed and were DISCARDED. Silent by default.
                        tel.amp_decreases += 1
                    optimizer.zero_grad(set_to_none=True)
                    did_step = True

                # Q4 instrumentation, reusing logits the loop already computed.
                dynamics.observe_batch(idx, logits, y, epoch)

                loss_v = float(loss.item())
                run_loss += loss_v * y.size(0)
                correct += int((logits.argmax(1) == y).sum().item())
                total += int(y.size(0))

                _t_end = time.time()
                tel.add_batch(loss_v, _t_end - _t_batch, load_t,
                              _t_end - _t_loaded,
                              lr=float(optimizer.param_groups[0]["lr"]))
                if did_step:
                    tel.add_step(gn_val, clipped)
                _t_batch = _t_end

            tel.samples = total
            dynamics.end_epoch()
            train_time = time.time() - t0

            _t_eval = time.time()
            val = evaluate(model, val_loader, device, amp, criterion)
            eval_time = time.time() - _t_eval

            samples = mon.stop()
            sys_samples = sysmon.stop()
            epoch_time = time.time() - t0
            epoch_energy = GPUEnergyMonitor.integrate_j(samples, epoch_time)

            # Raw sample streams are appended, not summarised away. The
            # aggregate goes in history.csv; the full trace goes here so a
            # power or throttling question can be answered later.
            if samples:
                new = not energy_path.exists()
                with open(energy_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=ENERGY_SAMPLE_COLUMNS,
                                       extrasaction="ignore")
                    if new:
                        w.writeheader()
                    for s_ in samples:
                        w.writerow({**s_, "epoch": int(epoch), "stage": "train"})
            if sys_samples:
                sp = log_dir / "system_samples.csv"
                new = not sp.exists()
                with open(sp, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=SYSTEM_SAMPLE_COLUMNS,
                                       extrasaction="ignore")
                    if new:
                        w.writeheader()
                    for s_ in sys_samples:
                        w.writerow({**s_, "epoch": int(epoch), "stage": "train"})

            # Per-step trace, downsampled. Enough to plot a within-epoch
            # slowdown; small enough that 240 epochs of it is still tiny.
            try:
                tp = log_dir / "step_traces.jsonl"
                with open(tp, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": int(epoch), **tel.step_trace()}) + "\n")
            except Exception:
                pass

            if scheduler is not None and (warm == 0 or epoch >= warm):
                scheduler.step()

            val_acc = float(val["accuracy"])
            cumulative_time += epoch_time
            cumulative_energy += epoch_energy
            epoch_co2 = energy_to_co2_kg(epoch_energy, carbon)
            cumulative_co2 += epoch_co2
            cumulative_samples += total

            wnorm, upd_norm, upd_ratio, prev_flat = optimisation_health(
                model, prev_flat)
            cumulative_steps += tel.opt_steps
            epochs_since_best = 0 if val_acc > best_metric else epochs_since_best + 1

            # ---- assemble the epoch row -----------------------------------
            # Every column in HISTORY_FIELDS gets a value. Quantities that do
            # not exist for this configuration are written NA rather than 0 or
            # omitted -- an absent loss term and a loss term that happened to be
            # zero are different facts.
            cal = val.get("calibration", {}) or {}
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            g = tel.summary()
            sysagg = SystemMonitor.aggregate(sys_samples)
            pw = GPUEnergyMonitor.power_stats(samples)

            if device.type == "cuda":
                vram_alloc = torch.cuda.memory_allocated(device) / 1024 ** 2
                vram_resv = torch.cuda.memory_reserved(device) / 1024 ** 2
                peak_vram = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                vram_total = (torch.cuda.get_device_properties(device).total_memory
                              / 1024 ** 2)
            else:
                vram_alloc = vram_resv = peak_vram = vram_total = NA

            remaining = max(0, num_epochs - (epoch + 1))
            row = {
                # identity & provenance
                "run_id": run_id, "epoch": epoch,
                "global_step": int(cumulative_steps),
                "timestamp_utc": now_iso(), "unix_ts": time.time(),
                "account": registry.account, "worker_id": cfg.get("worker_id", 0),
                "session_id": registry.session_id, "hostname": platform.node(),
                "arch": cfg["arch"], "family": cfg.get("family", NA),
                "dataset": cfg["dataset_name"], "seed": int(cfg["seed"]),
                "phase": cfg.get("phase", NA), "method": cfg.get("method", NA),
                "config_hash": cfg["config_hash"],

                # learning
                "train_loss": run_loss / max(1, total),
                "val_loss": float(val["loss"]),
                "train_accuracy": correct / max(1, total),
                "val_accuracy": val_acc,
                "train_accuracy_top5": NA,
                "val_accuracy_top5": float(val["accuracy_top5"]),
                "f1_macro": val.get("f1_macro", NA),
                "f1_micro": val.get("f1_micro", NA),
                "f1_weighted": val.get("f1_weighted", NA),
                "precision_macro": val.get("precision_macro", NA),
                "precision_micro": val.get("precision_micro", NA),
                "precision_weighted": val.get("precision_weighted", NA),
                "recall_macro": val.get("recall_macro", NA),
                "recall_micro": val.get("recall_micro", NA),
                "recall_weighted": val.get("recall_weighted", NA),
                "balanced_accuracy": val.get("balanced_accuracy", NA),
                "cohen_kappa": val.get("cohen_kappa", NA),
                "matthews_corrcoef": val.get("matthews_corrcoef", NA),
                "best_val_accuracy_so_far": float(max(best_metric, val_acc)),
                "epochs_since_best": int(epochs_since_best),
                "is_best": bool(val_acc > best_metric),

                # calibration
                "val_ece": cal.get("ece", NA), "val_mce": cal.get("mce", NA),
                "val_nll": cal.get("nll", NA), "val_brier": cal.get("brier", NA),
                "val_confidence_mean": cal.get("confidence_mean", NA),
                "val_entropy_mean": cal.get("entropy_mean", NA),

                # loss components -- CE only for a plain backbone run
                "loss_total": run_loss / max(1, total),
                "loss_ce": run_loss / max(1, total),
                "loss_kd": NA, "loss_msc": NA,
                "loss_l1": NA, "alpha": NA, "beta": NA, "temperature": NA,

                # optimisation
                "learning_rate": float(lrs[0]),
                "lr_min_group": float(min(lrs)), "lr_max_group": float(max(lrs)),
                "lr_groups_json": json.dumps([round(float(x), 8) for x in lrs]),
                "momentum": float(cfg.get("momentum", NA))
                            if cfg.get("optimizer") == "sgd" else NA,
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
                "grad_clip_value": float(clip) if clip > 0 else NA,
                "weight_norm": wnorm, "update_norm": upd_norm,
                "update_to_weight_ratio": upd_ratio,
                "amp_scale": float(scaler.get_scale()) if amp else NA,
                "amp_scale_decreases": int(tel.amp_decreases),

                # time
                "epoch_time_sec": float(epoch_time),
                "train_time_sec": float(train_time),
                "val_time_sec": float(eval_time),
                "cumulative_time_sec": float(cumulative_time),
                "throughput_train_img_s": total / max(1e-9, train_time),
                "throughput_val_img_s": (len(val_loader.dataset)
                                         / max(1e-9, eval_time)),
                "samples_seen": int(total),
                "cumulative_samples_seen": int(cumulative_samples),
                "eta_sec": float(remaining * epoch_time),

                # GPU (torch's own view; per-device columns come from sysagg)
                "vram_allocated_mb": vram_alloc, "vram_reserved_mb": vram_resv,
                "peak_vram_mb": peak_vram, "vram_total_mb": vram_total,

                # host
                "cpu_count": os.cpu_count(),
                "disk_free_scratch_mb": free_mb(SCRATCH_ROOT),
                "disk_free_working_mb": free_mb(WORK_ROOT),

                # energy & carbon
                "epoch_energy_j": float(epoch_energy),
                "epoch_energy_wh": epoch_energy / 3600.0,
                "epoch_energy_kwh": energy_to_kwh(epoch_energy),
                "cumulative_energy_j": float(cumulative_energy),
                "cumulative_energy_wh": cumulative_energy / 3600.0,
                "cumulative_energy_kwh": energy_to_kwh(cumulative_energy),
                "epoch_co2_g": epoch_co2 * 1000.0, "epoch_co2_kg": float(epoch_co2),
                "cumulative_co2_g": cumulative_co2 * 1000.0,
                "cumulative_co2_kg": float(cumulative_co2),
                "carbon_intensity_g_per_kwh": carbon * 1000.0,
                "energy_per_sample_mj": (epoch_energy / max(1, total)) * 1000.0,
                "energy_samples_n": len(samples),
                "energy_sample_hz": float(cfg.get("energy_sample_hz", 10.0)),

                # config echo
                "batch_size": int(cfg["batch_size"]),
                "effective_batch_size": int(cfg["batch_size"]) * accum,
                "gradient_accumulation_steps": int(accum),
                "amp_enabled": bool(amp), "num_epochs": int(num_epochs),
                "optimizer": cfg.get("optimizer", NA),
                "scheduler": cfg.get("scheduler", NA),
                "image_size": int(cfg.get("image_size", 32)),
                "num_classes": int(cfg["num_classes"]),
                "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
                "deterministic": bool(cfg.get("deterministic", False)),
                "msc_lib_version": __version__,

                **g, **sysagg, **pw,
            }
            # Loss terms deleted by the protocol: columns exist, values are NA
            # unless a config flag switches the term on.
            for _t in OPTIONAL_LOSS_TERMS:
                row[f"loss_{_t}"] = (float(loss_extra.get(_t))
                                     if loss_extra.get(_t) is not None else NA)
            for _c in HISTORY_FIELDS:
                row.setdefault(_c, NA)

            # strict=False: the merged GPU/system/power dicts legitimately vary
            # by machine. Anything dropped is now LOGGED rather than silently
            # lost -- see D-22.
            append_history_row(history_path, row, strict=False)

            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc
                atomic_save_torch(ckpt_best, {
                    "run_id": run_id, "model": model.state_dict(), "epoch": epoch,
                    "val_accuracy": val_acc, "config_hash": cfg["config_hash"],
                    "classes": classes, "config": cfg, "saved_utc": now_iso()})
            state["epoch"], state["best"] = epoch, best_metric

            save_checkpoint(ckpt_last, cfg, model, optimizer, scheduler, scaler,
                            epoch, best_metric, dynamics, cumulative_time,
                            cumulative_energy)

            print(f"  ep {epoch+1}/{num_epochs}  train={row['train_accuracy']:.4f}  "
                  f"val={val_acc:.4f}  top5={row['val_accuracy_top5']:.4f}  "
                  f"lr={row['learning_rate']:.5f}  E={epoch_energy:.0f}J  "
                  f"t={epoch_time:.1f}s" + ("  [BEST]" if is_best else ""))

            # --- push decision -------------------------------------------
            since = epoch - last_push_epoch
            due = (((epoch + 1) % milestone_every == 0)
                   or (is_best and since >= 3)
                   or (epoch == num_epochs - 1)
                   or sync.due_for_timer_push(timer_sec)
                   or guard.session_expiring())
            if due:
                last_push_epoch = epoch
                registry.heartbeat(run_id, run_dir, state="running", epoch=epoch,
                                   best_metric=best_metric,
                                   elapsed_h=round(guard.elapsed_h, 2))
                _write_dynamics(L["per_sample"], dynamics)
                sync.push_all(heavy=True)
                log(f"pushed at epoch {epoch+1} "
                    f"(elapsed {guard.elapsed_h:.1f} h)", "HF")

            if guard.session_expiring():
                log(f"session limit reached at {guard.elapsed_h:.1f} h -- "
                    f"pausing cleanly at epoch {epoch+1}", "LIFE")
                _emergency_flush("session limit")
                return {"run_id": run_id, "status": "paused", "epoch": epoch,
                        "best_accuracy": best_metric}

            # Debug hook, used only by resume_acceptance_test. Simulates a
            # session death at an epoch boundary by taking the REAL interrupt
            # path -- emergency flush, paused state, re-raise -- rather than
            # letting a short run finish cleanly. Those are different code
            # paths, and only one of them is the one that matters.
            # Excluded from config_hash so the resumed run matches.
            if int(cfg.get("_debug_interrupt_after_epoch", -1)) == epoch:
                raise KeyboardInterrupt(
                    f"simulated session death after epoch {epoch + 1}")

    except KeyboardInterrupt:
        log(f"{run_id} interrupted -- immediate push", "STOP")
        _emergency_flush("KeyboardInterrupt")
        raise
    except Exception as e:
        traceback.print_exc()
        registry.fail(run_id, f"{type(e).__name__}: {e}")
        _emergency_flush(f"exception: {type(e).__name__}")
        raise

    # --- completion -------------------------------------------------------
    final = evaluate(model, val_loader, device, amp, criterion)
    _write_dynamics(L["per_sample"], dynamics)
    budgets = load_or_build_budgets(cfg["arch"], data_out, cfg["num_classes"],
                                    hub=hub, model=build_model(cfg["arch"],
                                                               cfg["num_classes"]))

    summary = {
        "run_id": run_id, "arch": cfg["arch"], "family": cfg["family"],
        "dataset": cfg["dataset_name"], "seed": cfg["seed"], "phase": cfg["phase"],
        "config_hash": cfg["config_hash"], "sample_order_hash": order_hash,
        "num_epochs_planned": num_epochs, "num_epochs_run": state["epoch"] + 1,
        "best_accuracy": float(best_metric),
        "final_accuracy": float(final["accuracy"]),
        "final_accuracy_top5": float(final["accuracy_top5"]),
        "final_f1": float(final["f1"]),
        "total_time_sec": float(cumulative_time),
        "total_energy_j": float(cumulative_energy),
        "total_energy_kwh": energy_to_kwh(cumulative_energy),
        "total_co2_kg": float(cumulative_co2),
        "num_parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "full_flops": budgets["full_flops"],
        "reference_accuracy": REFERENCE_ACC.get(cfg["arch"]),
        "status": "completed", "completed_utc": now_iso(),
        "msc_lib_version": __version__,
    }

    # Recipe acceptance check. MSC computed from an undertrained model is
    # meaningless, and undertrained models are otherwise easy to miss.
    #
    # Only meaningful for a full-length run. A 4-epoch smoke test reaching 37%
    # against a 240-epoch published 69% is not a broken recipe, it is a 4-epoch
    # run -- and shouting about it in NB00 trains you to ignore the warning that
    # actually matters in NB01.
    ref = REFERENCE_ACC.get(cfg["arch"])
    full_length = num_epochs >= int(cfg.get("recipe_check_min_epochs", 100))
    if ref is not None and full_length:
        gap = ref - best_metric * 100.0
        summary["accuracy_gap_vs_reference"] = float(gap)
        summary["recipe_ok"] = bool(gap <= 1.0)
        if gap > 1.0:
            log(f"{cfg['arch']} reached {best_metric*100:.2f}% vs published "
                f"{ref:.2f}% (gap {gap:.2f} pts). Fix the recipe BEFORE generating "
                f"MSC tables from this checkpoint.", "WARN")
        else:
            log(f"{cfg['arch']} {best_metric*100:.2f}% vs published {ref:.2f}% -- OK",
                "CHECK")
    elif ref is not None:
        summary["accuracy_gap_vs_reference"] = None
        summary["recipe_ok"] = None
        summary["recipe_check_skipped"] = (
            f"short run ({num_epochs} epochs) -- the published {ref:.2f}% is for "
            f"the full recipe, so the comparison is not meaningful")

    atomic_write_json(run_dir / "summary.json", summary)
    registry.heartbeat(run_id, run_dir, state="completed", epoch=state["epoch"],
                       best_metric=best_metric)
    registry.finish(run_id, **{k: summary[k] for k in
                               ("arch", "dataset", "seed", "best_accuracy",
                                "final_accuracy", "num_epochs_run", "config_hash")})
    sync.push_all(heavy=True)
    if hub.enabled:
        log(f"flushing {run_id} (blocks until HF confirms)", "HF")
        ok = sync.flush(timeout=1800)
        missing = sync.verify_present([f"runs/{run_id}/ckpt_last.pt",
                                       f"runs/{run_id}/ckpt_best.pt",
                                       f"runs/{run_id}/config.yaml"])
        if ok and not missing and bool(cfg.get("cleanup_local_after_complete", True)):
            # Confirm-then-delete. A flush that merely did not time out is not
            # evidence the files are on HF.
            log(f"HF confirmed -- wiping local {run_dir}", "CLEAN")
            shutil.rmtree(run_dir, ignore_errors=True)
        elif missing:
            log(f"keeping local copy -- HF is missing {sorted(missing)}", "CLEAN")
    hub.print_stats()
    return summary


def _write_dynamics(log_dir, dynamics: TrainingDynamics) -> None:
    if pd is None:
        return
    p = Path(log_dir) / "train_dynamics.parquet"
    df = dynamics.to_frame()
    try:
        df.to_parquet(p, index=False)
    except Exception:
        df.to_csv(Path(log_dir) / "train_dynamics.csv", index=False)


# =============================================================================
# 14. oracle -- depth / resolution / precision sweeps -> per-sample Parquet
# =============================================================================
def train_exit_heads(cfg: Dict[str, Any], backbone, train_loader, val_loader,
                     device, hub: Optional[MSCHub] = None,
                     run_dir=None, show_progress: bool = True) -> "MultiExitModel":
    """Attach K exit heads and train them with the backbone FROZEN.

    Freezing is the definitional requirement from 01_PHASE0_GO_NOGO.md 3, not a
    speed optimisation: if the backbone adapts, each exit is reading a different
    network, and "the same model under reduced compute" -- the interpretation
    the entire MSC construct rests on -- stops being true.

    ~20 epochs at LR 0.01 with cosine decay, roughly 15 minutes per model.
    """
    me = MultiExitModel(backbone, cfg["num_classes"], freeze=True).to(device)
    params = [p for p in me.heads.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(cfg.get("exit_lr", 0.01)),
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    n_ep = int(cfg.get("exit_epochs", 20))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    crit = nn.CrossEntropyLoss()
    amp = bool(cfg.get("amp_enabled", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    for ep in range(n_ep):
        me.train()
        tot = corr = 0
        it = train_loader
        if tqdm is not None and show_progress:
            it = tqdm(train_loader, desc=f"exits ep {ep+1}/{n_ep}", leave=False,
                      dynamic_ncols=True, mininterval=2.0)
        for batch in it:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                # Every head is trained on the same forward pass; the backbone
                # is under no_grad inside MultiExitModel.forward.
                loss = sum(crit(lg, y) for lg in me(x)) / len(me.heads)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += y.size(0)
        sched.step()

    # Per-exit accuracy is a useful sanity signal: it should increase roughly
    # monotonically with depth. A shallow exit beating a deep one usually means
    # the stage partition is wrong.
    me.eval()
    accs = [0] * len(me.heads)
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            for k, lg in enumerate(me(x)):
                accs[k] += int((lg.argmax(1) == y).sum().item())
            n += y.size(0)
    accs = [a / max(1, n) for a in accs]
    log("exit accuracies: " + "  ".join(f"d{i+1}={a:.4f}" for i, a in enumerate(accs)),
        "EXIT")
    if any(accs[i] > accs[i + 1] + 0.02 for i in range(len(accs) - 1)):
        log("a shallower exit beats a deeper one by >2 points -- check the stage "
            "partition before trusting the depth axis", "WARN")

    if run_dir is not None:
        atomic_save_torch(Path(run_dir) / "exit_heads.pt",
                          {"heads": me.heads.state_dict(), "exit_accuracies": accs,
                           "config_hash": cfg["config_hash"], "saved_utc": now_iso()})
    return me


# --------------------------------------------------------------------------
# Precision axis: simulated quantisation
# --------------------------------------------------------------------------
@contextmanager
def fake_quantized(model, bits: int, per_channel: bool = True):
    """Temporarily replace weights with their quantise-dequantise round trip.

    INT8 has real PyTorch kernels; INT4 and INT6 do not, and no T4 kernel
    exists to time them. So the precision axis is *simulated*: we measure the
    accuracy effect exactly, and price the cost analytically as rho = bits/32.
    That distinction is stated wherever this axis appears -- claiming measured
    INT4 latency on a T4 would be false.

    Symmetric per-output-channel affine quantisation, which is what a
    reasonable PTQ implementation would do.
    """
    if bits >= 32:
        yield model
        return
    saved = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2:                      # leave biases and norms alone
                continue
            saved[name] = p.detach().clone()
            qmax = 2 ** (bits - 1) - 1
            if per_channel:
                flat = p.reshape(p.shape[0], -1)
                scale = flat.abs().amax(dim=1, keepdim=True) / qmax
                scale = torch.clamp(scale, min=1e-12)
                q = torch.clamp(torch.round(flat / scale), -qmax - 1, qmax)
                p.copy_((q * scale).reshape(p.shape))
            else:
                scale = torch.clamp(p.abs().max() / qmax, min=1e-12)
                q = torch.clamp(torch.round(p / scale), -qmax - 1, qmax)
                p.copy_(q * scale)
    try:
        yield model
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name])


def _resize_proxy(x, r: int):
    """Downsample to r then back to 32. Information content drops; shape does not.

    Idealised cost: the network really runs at 32px, so the FLOPs we attribute
    are those of a native-r run. Labelled as such everywhere.
    """
    if r == x.shape[-1]:
        return x
    small = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(32, 32), mode="bilinear", align_corners=False)


@_no_grad()
def sweep_all_axes(cfg: Dict[str, Any], multi_exit, loader, device,
                   resolutions: Sequence[int] = RESOLUTIONS,
                   precisions: Sequence[str] = PRECISIONS,
                   amp: bool = True, show_progress: bool = True) -> Dict[str, np.ndarray]:
    """Run every configuration on every sample and return the full grid.

    There is no early-exit shortcut here. The stable-sufficiency definition
    quantifies over ALL larger budgets, so the oracle must observe all of them
    -- stopping at the first agreement would record exactly the accidental
    early agreement that 2.2 exists to reject.

    Returns arrays keyed by axis, each (N, K): preds, top1p, top2p.
    """
    multi_exit.eval()
    backbone = multi_exit.backbone
    n_depth = len(multi_exit.heads)

    def _collect(fn, k: int, tag: str):
        P = np.zeros((0, k), dtype=np.int16)
        T1 = np.zeros((0, k), dtype=np.float32)
        T2 = np.zeros((0, k), dtype=np.float32)
        idxs = np.zeros((0,), dtype=np.int64)
        labs = np.zeros((0,), dtype=np.int64)
        chunks_p, chunks_1, chunks_2, chunks_i, chunks_l = [], [], [], [], []
        it = loader
        try:
            from tqdm.auto import tqdm
            if show_progress:
                it = tqdm(loader, desc=f"sweep {tag}", leave=False,
                          dynamic_ncols=True, mininterval=2.0)
        except Exception:
            pass
        for batch in it:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]
            idx = batch[2] if len(batch) > 2 else torch.arange(y.numel())
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(amp and device.type == "cuda")):
                logits_list = fn(x)
            probs = torch.stack([F.softmax(l.float(), dim=1) for l in logits_list], dim=1)
            top2 = probs.topk(2, dim=2)
            chunks_p.append(top2.indices[:, :, 0].cpu().numpy().astype(np.int16))
            chunks_1.append(top2.values[:, :, 0].cpu().numpy().astype(np.float32))
            chunks_2.append(top2.values[:, :, 1].cpu().numpy().astype(np.float32))
            chunks_i.append(np.asarray(idx).astype(np.int64))
            chunks_l.append(np.asarray(y).astype(np.int64))
        P = np.concatenate(chunks_p); T1 = np.concatenate(chunks_1)
        T2 = np.concatenate(chunks_2); idxs = np.concatenate(chunks_i)
        labs = np.concatenate(chunks_l)
        # Restore canonical order regardless of how the loader emitted batches.
        order = np.argsort(idxs, kind="stable")
        return P[order], T1[order], T2[order], idxs[order], labs[order]

    out: Dict[str, Any] = {}

    # --- depth ------------------------------------------------------------
    pd_, t1, t2, idxs, labs = _collect(lambda x: multi_exit(x), n_depth, "depth")
    out["depth"] = {"preds": pd_, "top1p": t1, "top2p": t2}
    out["sample_idx"] = idxs
    out["labels"] = labs

    # --- resolution, native -----------------------------------------------
    # The network genuinely runs at r x r. Adaptive pooling before the
    # classifier means the shape works; this is option (a) from
    # 01_PHASE0_GO_NOGO.md 3, the cleaner one -- where the architecture allows.
    # MLP-Mixer's token-mixing weights are sized to the token count and cannot,
    # so it gets the proxy only and the table records that.
    if bool(getattr(backbone, "supports_native_resolution", True)):
        def native_fn(x):
            outs = []
            for r in resolutions:
                xr = x if r == 32 else F.interpolate(x, size=(r, r), mode="bilinear",
                                                     align_corners=False)
                outs.append(backbone(xr))
            return outs
        try:
            p, a, b, _, _ = _collect(native_fn, len(resolutions), "res-native")
            out["res_native"] = {"preds": p, "top1p": a, "top2p": b}
        except Exception as e:
            log(f"native-resolution sweep failed ({type(e).__name__}: "
                f"{str(e)[:120]}); proxy only for this model", "ORACLE")
    else:
        log("architecture cannot run at non-32px input -- resolution axis "
            "measured with the proxy only", "ORACLE")

    # --- resolution, proxy -------------------------------------------------
    # Option (b): downsample-then-upsample, network shape unchanged, only
    # information content varies. Measuring both converts a methodological
    # wrinkle a reviewer would raise into a robustness check we already ran.
    def proxy_fn(x):
        return [backbone(_resize_proxy(x, r)) for r in resolutions]
    p, a, b, _, _ = _collect(proxy_fn, len(resolutions), "res-proxy")
    out["res_proxy"] = {"preds": p, "top1p": a, "top2p": b}

    # --- precision ---------------------------------------------------------
    prec_p, prec_1, prec_2 = [], [], []
    for prec in precisions:
        bits = PRECISION_BITS[prec]
        if prec == "fp16":
            def qfn(x, _b=bits):
                with torch.amp.autocast(device_type=device.type,
                                        enabled=(device.type == "cuda")):
                    return [backbone(x)]
            p1, a1, b1, _, _ = _collect(qfn, 1, f"prec-{prec}")
        else:
            with fake_quantized(backbone, bits):
                def qfn(x):
                    return [backbone(x)]
                p1, a1, b1, _, _ = _collect(qfn, 1, f"prec-{prec}")
        prec_p.append(p1[:, 0]); prec_1.append(a1[:, 0]); prec_2.append(b1[:, 0])
    out["precision"] = {"preds": np.stack(prec_p, axis=1),
                        "top1p": np.stack(prec_1, axis=1),
                        "top2p": np.stack(prec_2, axis=1)}
    return out


@_no_grad()
def difficulty_battery(backbone, loader, device, amp: bool = True) -> Dict[str, np.ndarray]:
    """The four post-hoc scores of the seven-score battery (protocol 4).

    EL2N and forgetting events come from TrainingDynamics during training;
    prediction depth comes from prediction_depth() using the exit features.
    These four are read off a single full-compute forward pass.
    """
    backbone.eval()
    msp, margin, ent, ce, idxs = [], [], [], [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        idx = batch[2] if len(batch) > 2 else torch.arange(y.numel())
        with torch.amp.autocast(device_type=device.type,
                                enabled=(amp and device.type == "cuda")):
            logits = backbone(x)
        p = F.softmax(logits.float(), dim=1)
        t2 = p.topk(2, dim=1)
        msp.append(t2.values[:, 0].cpu().numpy())
        margin.append((t2.values[:, 0] - t2.values[:, 1]).cpu().numpy())
        ent.append((-(p * torch.log(p.clamp_min(1e-12))).sum(1)).cpu().numpy())
        ce.append(F.cross_entropy(logits.float(), y, reduction="none").cpu().numpy())
        idxs.append(np.asarray(idx).astype(np.int64))
    order = np.argsort(np.concatenate(idxs), kind="stable")
    return {"msp": np.concatenate(msp)[order].astype(np.float32),
            "margin": np.concatenate(margin)[order].astype(np.float32),
            "entropy": np.concatenate(ent)[order].astype(np.float32),
            "ce_loss": np.concatenate(ce)[order].astype(np.float32)}


def build_per_sample_frame(sweep: Dict[str, Any], battery: Dict[str, np.ndarray],
                           pred_depth: Optional[np.ndarray],
                           dynamics_frame, order_hash: str,
                           run_id: str, split: str):
    """Assemble the per-sample table -- the scientific artifact of the project.

    Column naming follows 01_PHASE0_GO_NOGO.md 4, extended for the extra axes:
        pred_d{k}   top1p_d{k}   top2p_d{k}     depth
        pred_rn{k}  top1p_rn{k}  top2p_rn{k}    resolution, native
        pred_rp{k}  top1p_rp{k}  top2p_rp{k}    resolution, proxy
        pred_q{k}   top1p_q{k}   top2p_q{k}     precision

    `sample_order_hash` travels with every table. Two tables that disagree are
    refusing to be correlated rather than quietly producing a fabricated
    transfer coefficient -- index misalignment between models is the single
    easiest way to invent a result here.
    """
    cols: Dict[str, Any] = {
        "sample_idx": sweep["sample_idx"].astype(np.int32),
        "label": sweep["labels"].astype(np.int16),
    }
    prefix = {"depth": "d", "res_native": "rn", "res_proxy": "rp", "precision": "q"}
    for axis, pre in prefix.items():
        if axis not in sweep:
            continue
        a = sweep[axis]
        k = a["preds"].shape[1]
        for i in range(k):
            cols[f"pred_{pre}{i+1}"] = a["preds"][:, i].astype(np.int16)
            cols[f"top1p_{pre}{i+1}"] = a["top1p"][:, i].astype(np.float32)
            cols[f"top2p_{pre}{i+1}"] = a["top2p"][:, i].astype(np.float32)
    for k, v in battery.items():
        cols[k] = v
    if pred_depth is not None:
        cols["pred_depth"] = np.asarray(pred_depth, dtype=np.float32)

    df = pd.DataFrame(cols)
    if dynamics_frame is not None and split == "train_holdout":
        df = df.merge(dynamics_frame[["sample_idx", "el2n", "forget_events"]],
                      on="sample_idx", how="left")
    else:
        # EL2N and forgetting are training-set quantities and are genuinely
        # undefined on the test set. Present as NaN rather than absent, so the
        # column set is identical across splits and the analysis code does not
        # branch.
        df["el2n"] = np.nan
        df["forget_events"] = np.nan

    df.attrs["sample_order_hash"] = order_hash
    df["sample_order_hash"] = order_hash
    df["run_id"] = run_id
    df["split"] = split
    return df


def run_oracle(cfg: Dict[str, Any], hub: MSCHub, registry: RunRegistry,
               work_root=None, data_root_out=None,
               show_progress: bool = True) -> Dict[str, Any]:
    """Stage 2 of a run: exit heads, three-axis sweep, per-sample tables.

    Separated from backbone training so it can be re-run cheaply (it is
    inference-only, ~30-40 min per model) without touching the 3-hour backbone.
    Idempotent: if the tables exist and match this config, it returns them.
    """
    if not _TORCH_OK:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERR}")

    run_id = cfg["run_id"]
    work = Path(work_root or (WORK_ROOT / "msc"))
    data_out = Path(data_root_out or (work / "data"))
    L = run_layout(work, run_id)
    run_dir = ensure_dir(L["base"])
    for _s in RUN_SUBDIRS:
        ensure_dir(L[_s])
    ps_dir, log_dir, met_dir = L["per_sample"], L["telemetry"], L["metrics"]
    sync = RunSync(hub, run_id, run_dir, data_out)

    test_pq = ps_dir / "test.parquet"
    hold_pq = ps_dir / "train_holdout.parquet"
    if test_pq.exists() and hold_pq.exists() and not cfg.get("force_rerun"):
        log(f"per-sample tables already present for {run_id}", "ORACLE")
        return {"run_id": run_id, "status": "cached",
                "test": str(test_pq), "train_holdout": str(hold_pq)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))

    # --- recover the trained backbone -------------------------------------
    ckpt = run_dir / "ckpt_best.pt"
    if not ckpt.exists() and hub.enabled:
        log(f"pulling checkpoint for {run_id} from HF", "ORACLE")
        hub.hub.download(work, allow_patterns=[f"runs/{run_id}/**"], quiet=False)
        alt = L["checkpoints"] / "ckpt_best.pt"
        if alt.exists():
            ckpt = alt
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no ckpt_best.pt for {run_id}. Train the backbone first (notebook 02).")

    backbone = build_model(cfg["arch"], cfg["num_classes"]).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    backbone.load_state_dict(blob["model"], strict=True)
    backbone.eval()
    if blob.get("config_hash") not in (None, cfg["config_hash"]):
        log("checkpoint config_hash differs from the current config -- the sweep "
            "will run, but record this discrepancy", "WARN")

    train_loader, val_loader, holdout_loader, classes, order_hash = build_loaders(cfg)

    # --- exit heads --------------------------------------------------------
    heads_path = run_dir / "exit_heads.pt"
    me = MultiExitModel(backbone, cfg["num_classes"], freeze=True).to(device)
    if heads_path.exists() and not cfg.get("force_rerun"):
        try:
            me.heads.load_state_dict(torch.load(heads_path, map_location=device,
                                                weights_only=False)["heads"])
            log("loaded cached exit heads", "EXIT")
        except Exception:
            me = train_exit_heads(cfg, backbone, train_loader, val_loader, device,
                                  hub, run_dir, show_progress)
    else:
        me = train_exit_heads(cfg, backbone, train_loader, val_loader, device,
                              hub, run_dir, show_progress)
    sync.push_models(heavy=True)

    # --- budgets -----------------------------------------------------------
    budgets = load_or_build_budgets(cfg["arch"], data_out, cfg["num_classes"], hub=hub)

    # --- final evaluation (requirement 15.2) -------------------------------
    # Folded in here rather than given its own notebook: the checkpoint is
    # already loaded, so confusion matrix, per-class metrics, calibration,
    # latency/throughput and inference energy all come for free instead of
    # costing another 10-15 GPU-minutes per model across the atlas.
    try:
        prev = read_json(L["metrics"] / "final.json", default=None)
        if prev is None or cfg.get("force_rerun"):
            final_row = final_evaluation(
                cfg, backbone, val_loader, device, classes, run_dir,
                budgets=budgets,
                train_summary=read_json(run_dir / "summary.json", default={}),
                hub=hub)
        else:
            final_row = prev
            log("final evaluation already present -- reusing", "EVAL")
    except Exception as e:
        traceback.print_exc()
        log(f"final evaluation failed: {type(e).__name__}: {e}", "WARN")
        final_row = {}

    # --- dynamics from training -------------------------------------------
    dyn_frame = None
    dp = ps_dir / "train_dynamics.parquet"
    if dp.exists() and pd is not None:
        try:
            dyn_frame = pd.read_parquet(dp)
        except Exception:
            pass
    if dyn_frame is None and hub.enabled:
        got = hub.hub.download_file(
            f"runs/{run_id}/per_sample/train_dynamics.parquet", ps_dir)
        if got is not None and pd is not None:
            try:
                dyn_frame = pd.read_parquet(got)
            except Exception:
                pass
    if dyn_frame is None:
        log("no train_dynamics.parquet -- EL2N and forgetting events will be NaN. "
            "Q4's battery is incomplete without them.", "WARN")

    # --- sweeps ------------------------------------------------------------
    results = {}
    for split, loader in (("test", val_loader), ("train_holdout", holdout_loader)):
        log(f"sweeping {split} ({len(loader.dataset)} samples, "
            f"{len(me.heads)}+{len(RESOLUTIONS)}x2+{len(PRECISIONS)} configs)", "ORACLE")
        sweep = sweep_all_axes(cfg, me, loader, device, show_progress=show_progress)
        battery = difficulty_battery(backbone, loader, device)
        try:
            pdep = prediction_depth(me, loader, device)
        except Exception as e:
            log(f"prediction_depth failed: {e}", "WARN")
            pdep = None
        df = build_per_sample_frame(sweep, battery, pdep, dyn_frame,
                                    order_hash, run_id, split)
        out = ps_dir / f"{split}.parquet"
        try:
            df.to_parquet(out, index=False)
        except Exception:
            out = ps_dir / f"{split}.csv"
            df.to_csv(out, index=False)
        results[split] = str(out)
        log(f"wrote {out.name}  ({len(df)} rows x {len(df.columns)} cols)", "ORACLE")

    # Per-exit accuracy and FLOPs -- the depth axis in one small table.
    try:
        if pd is not None:
            d = budgets["axes"]["depth"]
            pd.DataFrame({"exit": list(range(1, len(d["rho"]) + 1)),
                          "depth_fraction": d["fractions"],
                          "rho": d["rho"], "flops": d["flops"],
                          "stage_cut": d["stage_cuts"],
                          "feature_dim": d["feature_dims"]}).to_csv(
                met_dir / "exit_metrics.csv", index=False)
    except Exception:
        pass

    meta = {"run_id": run_id, "arch": cfg["arch"], "family": cfg["family"],
            "dataset": cfg["dataset_name"], "seed": cfg["seed"],
            "sample_order_hash": order_hash, "config_hash": cfg["config_hash"],
            "budgets": budgets["axes"], "full_flops": budgets["full_flops"],
            "exit_count": len(me.heads), "resolutions": list(RESOLUTIONS),
            "precisions": list(PRECISIONS), "tau_grid": list(TAU_GRID),
            "created_utc": now_iso(), "msc_lib_version": __version__}
    atomic_write_json(ps_dir / "meta.json", meta)

    sync.push_per_sample()
    sync.push_logs()
    sync.flush(timeout=1200)
    registry.append(run_id, "oracle_done", **{k: meta[k] for k in
                                              ("arch", "seed", "sample_order_hash")})
    hub.print_stats()
    return {"run_id": run_id, "status": "done", **results, "meta": meta}


# =============================================================================
# 15. method -- MSC-KD, baselines, matched-FLOPs evaluation
# =============================================================================
if _TORCH_OK:

    class MSCLoss(nn.Module):
        """L = L_CE + alpha * L_KD + beta * L_MSC

        Three terms, two weights. The earlier CEB-KD formulation had seven terms
        and six weights, which is unprovable at any realistic experiment budget
        and reads to a reviewer as "we tried everything". Feature, attention and
        Pareto terms are deliberately absent, and monotonicity is architectural
        (OrdinalSufficiencyHead) rather than a penalty.
        """

        def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                     temperature: float = 4.0, ignore_irreducible: bool = True):
            super().__init__()
            self.alpha, self.beta, self.T = alpha, beta, temperature
            self.ignore_irreducible = ignore_irreducible

        def forward(self, student_logits, teacher_logits, labels,
                    suff_logits, suff_target, irreducible=None):
            """`suff_logits` is PRE-SIGMOID -- see D-21.

            `F.binary_cross_entropy` raises under AMP autocast ("unsafe to
            autocast"), and torch's own advice is to use the logit form rather
            than to disable autocast. That is strictly better anyway: the
            `.clamp(1e-6, 1-1e-6)` this used to need was papering over the
            log(0) that the fused kernel avoids by construction.
            """
            ce = F.cross_entropy(student_logits, labels)
            kd = F.kl_div(F.log_softmax(student_logits / self.T, dim=1),
                          F.softmax(teacher_logits / self.T, dim=1),
                          reduction="batchmean") * (self.T ** 2)
            bce = F.binary_cross_entropy_with_logits(
                suff_logits, suff_target.to(suff_logits.dtype),
                reduction="none").mean(dim=1)
            if self.ignore_irreducible and irreducible is not None:
                keep = ~irreducible
                # Samples where the teacher itself was unconfident carry a
                # degenerate MSC == 1 target. Training on them teaches the router
                # "always spend everything" on exactly the inputs where the
                # teacher had no usable opinion.
                msc = bce[keep].mean() if bool(keep.any()) else bce.sum() * 0.0
            else:
                msc = bce.mean()
            total = ce + self.alpha * kd + self.beta * msc
            return total, {"loss": float(total.detach()), "ce": float(ce.detach()),
                           "kd": float(kd.detach()), "msc": float(msc.detach())}

    class MSCStudent(nn.Module):
        """Student backbone + K exit heads + one ordinal sufficiency head.

        The sufficiency head reads the EARLIEST exit's features so the routing
        decision is available cheaply and early. A router that needs deep
        features in order to decide not to compute deep features saves nothing.
        """

        def __init__(self, backbone, num_classes: int, n_budgets: int):
            super().__init__()
            self.backbone = backbone
            self.token_model = getattr(backbone, "is_token_model", False)
            self.heads = nn.ModuleList([ExitHead(d, num_classes, self.token_model)
                                        for d in backbone.feature_dims])
            self.suff = OrdinalSufficiencyHead(backbone.feature_dims[0], n_budgets,
                                               token_model=self.token_model)

        def forward(self, x, suff_logits: bool = False):
            """`suff_logits=True` returns the sufficiency head's pre-sigmoid
            scores, which is what `MSCLoss` needs (D-21). Inference and routing
            want probabilities and get the default."""
            feats = self.backbone.forward_features(x)
            logits = [h(f) for h, f in zip(self.heads, feats)]
            s = self.suff.logits(feats[0]) if suff_logits else self.suff(feats[0])
            return logits, s, feats

        @torch.no_grad()
        def route_and_predict(self, x, gamma: float):
            """Deployment path: decide early, then compute only what is needed.

            Runs the shallowest prefix, routes, then continues per-sample. This
            is where the FLOPs saving is real -- and also where the batching
            caveat of protocol 7.2 bites: under batched inference there is no
            wall-clock gain unless the batch is split by route. Reported
            honestly rather than buried.
            """
            f0 = self.backbone.forward_prefix(x, 0)
            k = self.suff.route(f0, gamma)
            out = torch.zeros(x.size(0), self.heads[0].fc.out_features,
                              device=x.device)
            for kk in k.unique():
                m = (k == kk)
                kk = int(kk)
                f = f0[m] if kk == 0 else self.backbone.forward_prefix(x[m], kk)
                out[m] = self.heads[kk](f).float()
            return out, k


def sufficiency_targets(msc_teacher, rho):
    """s_k = 1[rho_k >= MSC_T(x)] -- monotone in k by construction."""
    if _TORCH_OK and isinstance(msc_teacher, torch.Tensor):
        return (rho.unsqueeze(0) >= msc_teacher.unsqueeze(1)).float()
    return (np.asarray(rho)[None, :] >= np.asarray(msc_teacher)[:, None]).astype(np.float32)


def ltt_min_calibration_n(epsilon: float = 0.01, delta: float = 0.05) -> int:
    """Calibration samples needed for a Hoeffding bound to be able to certify
    an epsilon accuracy drop at confidence 1-delta.

        n >= ln(1/delta) / (2 * epsilon^2)

    Worth computing before you design the experiment, because the numbers are
    unforgiving. At epsilon=0.01, delta=0.05 this is ~14,980 -- MORE THAN THE
    ENTIRE CIFAR-100 TEST SET. With a 10k test set split into calibration and
    evaluation halves you have ~5k calibration samples, which certifies only
    epsilon >= 0.017 at delta=0.05.

    The consequence is a design decision, not a bug: either report a larger
    epsilon honestly, or calibrate on a held-out slice of TRAIN (which is what
    we do -- the 5k train_holdout exists partly for this) and state that the
    calibration distribution is train-like. Discovering this after running the
    method would mean re-running it.
    """
    return int(math.ceil(math.log(1.0 / delta) / (2.0 * epsilon ** 2)))


def learn_then_test_threshold(suff_pred: np.ndarray, correct_at: np.ndarray,
                              full_accuracy: float, epsilon: float = 0.01,
                              delta: float = 0.05,
                              grid: Optional[Sequence[float]] = None,
                              warn_underpowered: bool = True) -> float:
    """Largest-savings gamma whose accuracy drop is provably below epsilon.

    Distribution-free Learn-then-Test with a Hoeffding bound, tested from
    conservative to aggressive under fixed-sequence error control, stopping at
    the first failure -- so no multiplicity correction is needed.

    This machinery is ADOPTED, not claimed. Jazbec et al. (NeurIPS 2024)
    introduced risk control for early exit and SAFE-KD already pairs conformal
    risk control with early-exit distillation. Our differentiation is the
    supervision signal, not the calibration.

    If n is too small for the requested (epsilon, delta), NO threshold can pass
    and the most conservative gamma is returned. That is correct behaviour, but
    it looks identical to "the method cannot save any compute", so it warns.
    """
    if grid is None:
        grid = np.linspace(0.99, 0.05, 60)
    n, k_max = suff_pred.shape[0], suff_pred.shape[1] - 1
    chosen = float(grid[0])
    slack = float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))
    if warn_underpowered and slack > epsilon:
        need = ltt_min_calibration_n(epsilon, delta)
        log(f"LTT is underpowered: n={n} gives a Hoeffding slack of {slack:.4f}, "
            f"which already exceeds epsilon={epsilon}. No threshold can pass. "
            f"Either use n >= {need}, or raise epsilon above {slack:.4f}. "
            f"Returning the most conservative gamma.", "WARN")
    for gamma in grid:
        hit = suff_pred >= gamma
        route = np.where(hit.any(axis=1), hit.argmax(axis=1), k_max)
        acc = correct_at[np.arange(n), route].mean()
        if (full_accuracy - acc) + slack <= epsilon:
            chosen = float(gamma)
        else:
            break
    return chosen


def expected_flops(route: np.ndarray, rho: Sequence[float], full_flops: float) -> float:
    """Average cost of a routing policy, in absolute FLOPs.

    Matched average FLOPs is the ONLY comparison that means anything for Q5.
    An accuracy win at unmatched compute is not a result.
    """
    r = np.asarray(rho, dtype=float)
    return float(np.mean(r[np.asarray(route, dtype=int)]) * full_flops)


def confidence_route(top1p: np.ndarray, threshold: float) -> np.ndarray:
    """Baseline B2: exit at the first budget whose own top-1 probability clears
    a threshold. This is what the field actually deploys, and it is the true
    rival -- not the static student.
    """
    hit = top1p >= threshold
    k_max = top1p.shape[1] - 1
    return np.where(hit.any(axis=1), hit.argmax(axis=1), k_max)


def sweep_operating_points(route_scores: np.ndarray, correct_at: np.ndarray,
                           rho: Sequence[float], full_flops: float,
                           thresholds: Optional[Sequence[float]] = None,
                           higher_exits_later: bool = True) -> "Any":
    """Accuracy-vs-FLOPs curve for one routing rule.

    Produces the full trade-off curve rather than a single point, because a
    method that wins at one operating point and loses everywhere else has not
    won. Area under this curve is one of the three Q5 measures.
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.995, 80)
    rows = []
    n = route_scores.shape[0]
    k_max = route_scores.shape[1] - 1
    for t in thresholds:
        hit = route_scores >= t
        route = np.where(hit.any(axis=1), hit.argmax(axis=1), k_max)
        rows.append({"threshold": float(t),
                     "accuracy": float(correct_at[np.arange(n), route].mean()),
                     "avg_flops": expected_flops(route, rho, full_flops),
                     "avg_rho": float(np.mean(np.asarray(rho)[route])),
                     "mean_exit": float(route.mean())})
    return pd.DataFrame(rows) if pd is not None else rows


def accuracy_at_matched_flops(curve, target_flops: float) -> float:
    """Linear interpolation of accuracy at a given average-FLOPs budget.

    Two methods are only comparable at the same average cost, and neither will
    have an operating point exactly there, so interpolate rather than picking
    the nearest and hoping.
    """
    if pd is None or len(curve) == 0:
        return float("nan")
    c = curve.sort_values("avg_flops")
    x, y = c["avg_flops"].to_numpy(), c["accuracy"].to_numpy()
    if target_flops <= x[0]:
        return float(y[0])
    if target_flops >= x[-1]:
        return float(y[-1])
    return float(np.interp(target_flops, x, y))


def auc_accuracy_flops(curve, flops_lo: Optional[float] = None,
                       flops_hi: Optional[float] = None) -> float:
    """Normalised area under the accuracy-vs-FLOPs curve."""
    if pd is None or len(curve) == 0:
        return float("nan")
    c = curve.sort_values("avg_flops")
    x, y = c["avg_flops"].to_numpy(), c["accuracy"].to_numpy()
    lo = flops_lo if flops_lo is not None else x.min()
    hi = flops_hi if flops_hi is not None else x.max()
    m = (x >= lo) & (x <= hi)
    if m.sum() < 2:
        return float("nan")
    area = np.trapezoid(y[m], x[m]) if hasattr(np, "trapezoid") else np.trapz(y[m], x[m])
    return float(area / max(1e-12, (x[m].max() - x[m].min())))


def shuffle_msc_targets(msc: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute MSC targets within the dataset -- the ablation to run FIRST.

    If a student trained on shuffled targets performs as well as one trained on
    real ones, L_MSC is acting as a regulariser and the supervision signal is
    not doing what the paper claims. That is something you need to know before
    writing anything, so it runs early and unconditionally.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(msc, dtype=float).copy()
    finite = np.flatnonzero(np.isfinite(out))
    out[finite] = out[rng.permutation(finite)]
    return out


# =============================================================================
# 16. analysis -- wrappers over msc_core, aggregation, gate decision
# =============================================================================
AXIS_PREFIX = {"depth": "d", "res_native": "rn", "res_proxy": "rp", "precision": "q"}


def _import_msc_core():
    """msc_core.py is the reference implementation and the single source of
    truth for every statistic. It is imported, never reimplemented -- a second
    copy of `compute_msc` that drifts by one index is precisely the kind of bug
    that produces a plausible-looking wrong answer.
    """
    try:
        import msc_core
        return msc_core
    except ImportError:
        here = Path(globals().get("__file__", "msc_lib.py")).resolve().parent
        for cand in (WORK_ROOT, WORK_ROOT / "msc", Path.cwd(), here):
            p = Path(cand) / "msc_core.py"
            if p.exists():
                sys.path.insert(0, str(cand))
                import msc_core
                return msc_core
    raise ImportError(
        "msc_core.py not found. Place it beside msc_lib.py or in the working "
        "directory -- the analysis will not run without it.")


class MissingInputs(RuntimeError):
    """Raised when an analysis is asked to run before its inputs exist.

    A distinct exception type because this is almost never a bug -- it means a
    notebook was run out of order, and the useful response is a clear statement
    of what is missing and which notebook produces it.
    """


def load_per_sample(data_dir, run_id: str, split: str = "test"):
    base = Path(data_dir) / "runs" / run_id / "per_sample"
    for ext in ("parquet", "csv"):
        p = base / f"{split}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    trained = (Path(data_dir) / "runs" / run_id / "summary.json").exists()
    hint = ("This run finished TRAINING but has not been MEASURED yet -- the "
            "per-sample tables come from the oracle sweep. Run NB02 (Phase 0) "
            "or NB08 (atlas) first."
            if trained else
            "This run has not finished training. Run NB01 (Phase 0) or "
            "NB04-NB07 (atlas) first.")
    raise MissingInputs(
        f"no per-sample table at runs/{run_id}/per_sample/{split}.parquet\n{hint}")


def check_inputs(data_dir, run_ids: Sequence[str], split: str = "test",
                 verbose: bool = True) -> Dict[str, Any]:
    """What each run has, and what is still missing, before any analysis runs.

    Called at the top of every analysis notebook so a missing input produces one
    readable table and one clear instruction, rather than a FileNotFoundError
    raised six frames deep inside a statistic.
    """
    def _has_table(ps: Path, split: str) -> bool:
        # Must agree with load_per_sample, which accepts a CSV fallback --
        # run_oracle writes CSV when no parquet engine is available. A checker
        # that disagrees with the loader reports work as missing that is
        # actually there.
        return any((ps / f"{split}.{e}").exists() for e in ("parquet", "csv"))

    rows, missing = [], []
    for r in run_ids:
        base = Path(data_dir) / "runs" / r
        ps = base / "per_sample"
        rec = {
            "run_id": r,
            "trained": (base / "summary.json").exists(),
            "checkpoint": (base / "checkpoints" / "ckpt_best.pt").exists(),
            "epochs_csv": (base / "metrics" / "epochs.csv").exists(),
            # D-23: canonical location is the run root; tolerate the legacy one.
            "exit_heads": ((base / "exit_heads.pt").exists()
                           or (base / "checkpoints" / "exit_heads.pt").exists()),
            "per_sample_test": _has_table(ps, split),
            "final_eval": (base / "metrics" / "final.csv").exists(),
        }
        acc = read_json(base / "summary.json", default={}) or {}
        rec["accuracy"] = acc.get("best_accuracy")
        rec["epochs_run"] = acc.get("num_epochs_run")
        rows.append(rec)
        if not rec["per_sample_test"]:
            missing.append(r)

    table = pd.DataFrame(rows) if pd is not None else rows
    ready = not missing

    if verbose:
        print(f"\n{'='*72}\n  Input check\n{'='*72}")
        if pd is not None and len(table):
            print(table.to_string(index=False))
        if ready:
            print("\n  All inputs present.\n")
        else:
            n_trained = sum(1 for r in rows if r["trained"])
            print(f"\n  MISSING per-sample tables for {len(missing)} of "
                  f"{len(run_ids)} runs:")
            for r in missing:
                print(f"    {r}")
            if n_trained == len(run_ids):
                print("\n  All runs finished TRAINING but none have been MEASURED.")
                print("  The per-sample tables are produced by the oracle sweep.")
                print("\n  -> Run NB02 (Phase 0) or NB08 (atlas), then come back.")
            else:
                print(f"\n  {n_trained}/{len(run_ids)} runs have finished training.")
                print("  -> Finish NB01 / NB04-NB07, then NB02 / NB08, then return.")
        print(f"{'='*72}\n")

    return {"ready": ready, "missing": missing, "table": table,
            "n_runs": len(run_ids)}


def require_inputs(data_dir, run_ids: Sequence[str], split: str = "test") -> None:
    """Hard stop with an actionable message if the analysis cannot proceed."""
    rep = check_inputs(data_dir, run_ids, split=split, verbose=True)
    if not rep["ready"]:
        raise MissingInputs(
            f"{len(rep['missing'])} of {rep['n_runs']} runs have no per-sample "
            f"table. See the table above -- run the measurement notebook first.")


def assert_aligned(frames: Dict[str, Any]) -> str:
    """Every table must share one sample order hash, or nothing may be correlated.

    This check exists because index misalignment produces numbers that look
    entirely reasonable. The shuffled-target control catches it too, but this
    catches it earlier and says why.
    """
    hashes = {}
    for rid, df in frames.items():
        h = df["sample_order_hash"].iloc[0] if "sample_order_hash" in df.columns else None
        hashes[rid] = h
    uniq = set(hashes.values())
    if len(uniq) != 1 or None in uniq:
        raise ValueError(
            "per-sample tables are not index-aligned; refusing to correlate.\n"
            + "\n".join(f"  {k}: {v}" for k, v in hashes.items()))
    return uniq.pop()


def available_axes(df) -> List[str]:
    """Which compute axes this per-sample table actually carries.

    Not every architecture supports every axis. MLP-Mixer cannot run at a
    non-32px input, so it has no `res_native` columns. Analysis code asks rather
    than assumes, so one architecture's limitation does not crash a study of
    fifteen.
    """
    return [a for a, pre in AXIS_PREFIX.items() if f"pred_{pre}1" in df.columns]


def msc_for_run(df, budgets: Dict[str, Any], axis: str = "depth",
                tau: float = 0.1):
    """Compute MSC for one run, one axis, one tau, using msc_core."""
    core = _import_msc_core()
    if axis not in AXIS_PREFIX:
        raise KeyError(f"unknown axis '{axis}'. Known: {sorted(AXIS_PREFIX)}")
    pre = AXIS_PREFIX[axis]
    if f"pred_{pre}1" not in df.columns:
        raise KeyError(
            f"axis '{axis}' is not present in this table (has: {available_axes(df)}). "
            f"Some architectures cannot be measured on every axis -- MLP-Mixer has "
            f"no native-resolution sweep, by construction.")
    budget_axis = {"depth": "depth", "res_native": "resolution",
                   "res_proxy": "resolution", "precision": "precision"}[axis]
    rho = budgets["axes"][budget_axis]["rho"]
    # K is per-architecture, and for the depth axis it can legitimately be
    # smaller than 5. Trust the table, and check the budget agrees.
    n_cols = sum(1 for i in range(1, 16) if f"pred_{pre}{i}" in df.columns)
    if n_cols != len(rho):
        raise ValueError(
            f"axis '{axis}': table has {n_cols} configurations but the budget "
            f"table has {len(rho)}. These were produced by different versions of "
            f"the config -- do not correlate them.")
    k = len(rho)
    preds = np.stack([df[f"pred_{pre}{i+1}"].to_numpy() for i in range(k)], axis=1)
    t1 = np.stack([df[f"top1p_{pre}{i+1}"].to_numpy() for i in range(k)], axis=1)
    t2 = np.stack([df[f"top2p_{pre}{i+1}"].to_numpy() for i in range(k)], axis=1)
    return core.compute_msc(preds, t1, t2, rho, tau=tau, axis=axis)


def tau_curve(df, budgets, axis: str = "depth",
              taus: Sequence[float] = TAU_GRID) -> Dict[float, Any]:
    return {t: msc_for_run(df, budgets, axis, t) for t in taus}


def analyse_q1_seed_ceiling(data_dir, run_a: str, run_b: str, budgets,
                            axis: str = "depth", taus=TAU_GRID) -> "Any":
    """Q1: MSC agreement between two seeds of the SAME architecture.

    Not a side experiment. This is the denominator of every transfer number in
    the project: a cross-architecture rho of 0.6 means something completely
    different when seed-to-seed is 0.95 than when it is 0.62. The
    sample-difficulty literature routinely omits this, which is what makes its
    raw cross-architecture correlations hard to interpret.
    """
    core = _import_msc_core()
    da, db = load_per_sample(data_dir, run_a), load_per_sample(data_dir, run_b)
    assert_aligned({run_a: da, run_b: db})
    rows = []
    for t in taus:
        ma = msc_for_run(da, budgets, axis, t)
        mb = msc_for_run(db, budgets, axis, t)
        rows.append({
            "axis": axis, "tau": t,
            "rho_seed": core.seed_ceiling(ma.clean(), mb.clean()),
            "frac_irreducible_a": ma.frac_irreducible,
            "frac_irreducible_b": mb.frac_irreducible,
            "jaccard_top10": core.top_decile_jaccard(ma.clean(), mb.clean()),
            "mean_msc_a": float(np.nanmean(ma.clean())),
            "mean_msc_b": float(np.nanmean(mb.clean())),
            "run_a": run_a, "run_b": run_b,
        })
    return pd.DataFrame(rows)


def analyse_q2_axis_structure(data_dir, run_id: str, budgets,
                              axes=("depth", "res_native", "precision"),
                              taus=TAU_GRID) -> "Any":
    """Q2: is compute need one-dimensional across reduction axes?

    Never asked, in this literature or the sample-difficulty literature. Every
    adaptive-inference paper picks one axis and treats it as THE compute axis.
    If PC1 dominates, that implicit assumption is validated and a single scalar
    router is justified. If it does not, results on depth-based early exit do
    not license claims about width- or precision-adaptive inference. Either
    outcome is a contribution, and the data comes almost free once the atlas
    exists -- the highest novelty-per-GPU-hour question in the project.
    """
    core = _import_msc_core()
    df = load_per_sample(data_dir, run_id)
    have = available_axes(df)
    axes = [a for a in axes if a in have]
    if len(axes) < 2:
        log(f"{run_id}: only {have} available -- cannot do axis structure", "WARN")
        return pd.DataFrame([{"run_id": run_id, "error": f"axes available: {have}"}])
    rows = []
    for t in taus:
        by_axis = {a: msc_for_run(df, budgets, a, t).clean() for a in axes}
        try:
            st = core.axis_structure(by_axis)
        except ValueError as e:
            rows.append({"tau": t, "error": str(e)})
            continue
        rec = {"run_id": run_id, "tau": t, "pc1_variance": st["pc1_variance"],
               "n": st["n"]}
        for a, v in st["pc1_loadings"].items():
            rec[f"loading_{a}"] = v
        for i, v in enumerate(st["explained_variance_ratio"]):
            rec[f"evr_pc{i+1}"] = v
        sm = st["spearman_matrix"]
        for i, a in enumerate(st["axes"]):
            for j, b in enumerate(st["axes"]):
                if i < j:
                    rec[f"rho_{a}__{b}"] = float(sm.iloc[i, j])
        rows.append(rec)
    return pd.DataFrame(rows)


def analyse_q3_transfer(data_dir, pairs: Sequence[Tuple[str, str]],
                        ceilings: Dict[str, float], budgets_by_run: Dict[str, Any],
                        axis: str = "depth", taus=TAU_GRID,
                        n_boot: int = 1000) -> "Any":
    """Q3: disattenuated cross-architecture transfer, with bootstrap CI.

        T(A,B) = rho_S(A,B) / sqrt(ceiling_A * ceiling_B)

    Spearman's classical correction for attenuation. T ~ 1 means transfer is as
    complete as measurement noise permits; T well below 1 means genuine
    architecture-specific structure. Top-decile Jaccard is reported alongside
    because for a routing application, agreement on WHICH samples are hardest
    matters more than global rank correlation.
    """
    core = _import_msc_core()
    rows = []
    for a, b in pairs:
        da, db = load_per_sample(data_dir, a), load_per_sample(data_dir, b)
        assert_aligned({a: da, b: db})
        for t in taus:
            ma = msc_for_run(da, budgets_by_run[a], axis, t).clean()
            mb = msc_for_run(db, budgets_by_run[b], axis, t).clean()
            ca, cb = ceilings.get(a, float("nan")), ceilings.get(b, float("nan"))
            tr = core.disattenuated_transfer(ma, mb, ca, cb, n_boot=n_boot)
            rows.append({"run_a": a, "run_b": b, "axis": axis, "tau": t,
                         "spearman_raw": tr["spearman_raw"], "T": tr["T"],
                         "T_lo": tr["T_ci95"][0], "T_hi": tr["T_ci95"][1],
                         "ceiling_a": ca, "ceiling_b": cb, "n": tr["n"],
                         "jaccard_top10": core.top_decile_jaccard(ma, mb)})
    return pd.DataFrame(rows)


def representative_runs(runs: Dict[str, Dict[str, Any]],
                        require=None) -> Dict[str, str]:
    """One run per architecture -- the lowest seed that is actually usable.

    Replaces the idiom this codebase used in three notebooks:

        seed1 = {m['arch']: r for r, m in runs.items() if m['seed'] == 1}

    which silently drops any architecture whose seed 1 happens to be missing.
    `vgg8` has two measured seeds and the second-highest noise ceiling in the
    whole atlas, but its seed 1 was never measured (D-15), so it vanished from
    Q2, Q3 and Q4 for a bookkeeping reason rather than a data reason -- and it
    vanished silently, because a dict comprehension cannot report what it
    skipped. See D-18.

    `require` is an optional membership test (pass the ceilings dict): an
    architecture is only represented by a run that appears in it, which is how
    callers say "measured" without needing to re-read every parquet file.
    """
    cand: Dict[str, List[Tuple[int, str]]] = {}
    for rid, m in runs.items():
        if require is not None and rid not in require:
            continue
        arch = m.get("arch")
        if not arch:
            continue
        seed = m.get("seed")
        cand.setdefault(arch, []).append(
            (10 ** 6 if seed is None else int(seed), rid))
    return {arch: sorted(v)[0][1] for arch, v in cand.items()}


def stratified_pairs(pairs: Sequence[Tuple[str, str]], kind_fn,
                     per_kind: int = 3) -> List[Tuple[str, str]]:
    """Up to `per_kind` pairs from each kind -- not the alphabetical head.

    Exists because `pairs[:8]` and `pairs[:15]`, over an alphabetically sorted
    pair list, are not samples of the atlas. They are samples of whichever
    architecture sorts first. In our zoo that is `convnext_femto`, which turns
    out to be the single most atypical CNN in the transfer matrix. See D-18.
    """
    out: List[Tuple[str, str]] = []
    seen: Dict[Any, int] = {}
    for p in pairs:
        k = kind_fn(p)
        if seen.get(k, 0) < per_kind:
            seen[k] = seen.get(k, 0) + 1
            out.append(p)
    return out


def shuffled_control_verdict(rho: float, n: int, z_max: float = 5.0,
                             rho_floor: float = 0.10) -> Tuple[bool, float, float]:
    """Is a shuffled-control residual noise, or a bug? Returns (passed, z, sd).

    Split out of `analyse_q3_shuffled_control` on purpose. The decision rule is
    exactly where defect D-17 lived, and a rule reachable only through a full
    analysis run -- needing measured parquet files, ceilings and budgets on disk
    -- is a rule that never gets a unit test. Here it is a pure function of two
    numbers and is checked offline on every self-test.

    Under a random permutation the correlation of two rank vectors has mean 0
    and variance exactly 1/(n-1). That is exact, not asymptotic, and holds with
    arbitrary ties -- which matters because MSC takes only K distinct values.

    A pair fails only if the residual is BOTH impossible under shuffling
    (|z| > z_max) AND big enough to be worth acting on (|rho| > rho_floor).
    Both conditions are load-bearing:

      - Without the z term, the cutoff is sample-size blind (D-17 cause 1).
      - Without the rho floor, a large enough n makes any trivial residual
        "significant": at n = 1e6 a rho of 0.02 is 20 sigma and would fail,
        which is statistically true and practically meaningless.
    """
    null_sd = 1.0 / math.sqrt(n - 1) if n > 2 else float("nan")
    z = rho / null_sd if null_sd == null_sd and null_sd > 0 else float("nan")
    passed = not (abs(z) > z_max and abs(rho) > rho_floor)
    return bool(passed), float(z), float(null_sd)


def analyse_q3_shuffled_control(data_dir, run_a: str, run_b: str,
                                ceilings, budgets_by_run, axis="depth",
                                tau: float = 0.1, seed: int = 0,
                                z_max: float = 5.0, rho_floor: float = 0.10,
                                n_shuffles: int = 3) -> Dict[str, Any]:
    """The pipeline sanity check, not a scientific result.

    Shuffling one side must destroy the correlation. If it does not, the tables
    are not really being paired by `sample_idx` and every Q3 number is void.

    CALIBRATION -- see D-17. The original criterion was ``abs(T) < 0.05`` on the
    DISATTENUATED statistic. It fired on a perfectly healthy pair, and it was
    miscalibrated three separate ways:

      1. SAMPLE-SIZE BLIND. Under a random permutation the rank correlation has
         mean 0 and SD exactly ``1/sqrt(n-1)`` -- about 0.013 at our n~5,900. A
         fixed 0.05 cutoff is 2.6 sigma at n=6,000 but 5 sigma at n=25,000. The
         same constant means entirely different strictness at different n.
      2. CEILING-DEPENDENT, IN THE WORST DIRECTION. ``T = rho / sqrt(ca*cb)``,
         so a low-ceiling pair divides by a smaller number and trips the same
         cutoff at a smaller rho. `vit_tiny` x `mixer_nano` trips at 2.10 sigma
         (3.6% by chance); `resnet32x4` x `vgg8` needs 2.78 sigma (0.5%). The
         control was ~7x more likely to false-alarm on precisely the
         low-ceiling architectures that carry the project's headline finding.
      3. MULTIPLICITY BLIND. At ~1% per pair, P(at least one failure) is 20%
         over 25 pairs and 50% over the full 78. It was not a question of
         whether this would fire, only when.

    It was also two-sided against a one-sided failure mode. Index leakage
    inflates correlation UPWARD -- it makes a shuffle look like a non-shuffle.
    No misalignment mechanism produces a small NEGATIVE correlation, so failing
    on one was never diagnostic of anything.

    The test now runs on the RAW rank correlation against its exact permutation
    null, and demands BOTH statistical and practical significance: ``|z| >
    z_max`` AND ``|rho| > rho_floor``. A real leak gives rho near the true
    transfer (~0.6, z ~ 45) and clears both by a mile; noise clears neither.
    `assert_aligned` is also called directly -- the hash comparison is the real
    check this control was only ever standing in for.

    The permutation null is exact rather than asymptotic: for any fixed pair of
    score vectors the permutation variance of the correlation of their ranks is
    exactly ``1/(n-1)``, ties included. MSC is heavily tied (it takes only K
    distinct budget values), so an asymptotic normal approximation would have
    been the wrong tool here; this one is not affected.
    """
    core = _import_msc_core()
    da, db = load_per_sample(data_dir, run_a), load_per_sample(data_dir, run_b)
    assert_aligned({run_a: da, run_b: db})   # the direct check, not a proxy for it
    ma = msc_for_run(da, budgets_by_run[run_a], axis, tau).clean()
    mb = msc_for_run(db, budgets_by_run[run_b], axis, tau).clean()

    # Several permutations, judged on the worst, so a single lucky draw cannot
    # certify a pipeline that is actually broken.
    worst = None
    for k in range(max(1, int(n_shuffles))):
        sh = core.disattenuated_transfer(ma, shuffle_msc_targets(mb, seed + k),
                                         ceilings.get(run_a, 1.0),
                                         ceilings.get(run_b, 1.0), n_boot=0)
        if worst is None or abs(sh["spearman_raw"]) > abs(worst["spearman_raw"]):
            worst = sh

    rho = float(worst["spearman_raw"])
    n = int(worst.get("n", 0) or 0)
    passed, z, null_sd = shuffled_control_verdict(rho, n, z_max, rho_floor)
    if not passed:
        log(f"SHUFFLED CONTROL FAILED: rho={rho:+.4f} (z={z:+.1f}, n={n}). "
            f"Shuffling did not destroy the correlation, so the tables are not "
            f"being paired by sample_idx. This is a BUG, not a finding -- check "
            f"{run_a} against {run_b}.", "ALARM")
    elif abs(z) > 3.0:
        log(f"shuffled control for {run_a} x {run_b}: rho={rho:+.4f} "
            f"(z={z:+.1f}) -- larger than typical but far below the {z_max:.0f}"
            f"-sigma / {rho_floor:.2f}-rho bug threshold, and expected "
            f"occasionally across many pairs. Passing.", "INFO")
    return {"T_shuffled": worst["T"], "spearman_raw": rho, "z": z,
            "null_sd": null_sd, "n": n, "passed": bool(passed),
            "tau": tau, "axis": axis, "z_max": z_max, "rho_floor": rho_floor}


def analyse_q4_irreducibility(data_dir, run_a: str, run_b: str, budgets_by_run,
                              axis: str = "depth", taus=TAU_GRID,
                              battery_cols=("msp", "margin", "entropy", "ce_loss",
                                            "el2n", "forget_events", "pred_depth"),
                              n_boot: int = 500,
                              split: str = "train_holdout") -> "Any":
    """Q4: is MSC reducible to classical difficulty scores?

    The question that decides whether the project has a new object or a
    rebranded one. Treated as the PRIMARY threat, not a footnote.

    If it fails -- if MSC is fully explained by the battery -- that is still
    publishable and must not be hidden: "per-sample compute requirements are
    fully explained by classical difficulty scores" is a clean, useful, citable
    finding that saves the community effort, and the engineering result that
    follows ("use a cheap difficulty score instead of a multi-axis oracle") is
    arguably better than the method paper.
    """
    # DEFAULTS TO train_holdout, not test.
    #
    # Two of the seven difficulty scores -- EL2N and forgetting events -- are
    # TRAINING-set quantities. They index training images, and the test set's
    # sample_idx refers to entirely different images, so they cannot be attached
    # there and are correctly NaN. Running Q4 on the test split therefore answers
    # the question with 5 of 7 scores, which understates the battery and makes
    # MSC look more irreducible than a fair test would.
    #
    # The train_holdout split is a 5,000-image slice of training data evaluated
    # with augmentation off, so it carries all seven. That is the honest place to
    # ask whether MSC survives controlling for classical difficulty. The test
    # split remains available as a robustness check via split="test".
    core = _import_msc_core()
    da = load_per_sample(data_dir, run_a, split)
    db = load_per_sample(data_dir, run_b, split)
    assert_aligned({run_a: da, run_b: db})
    cols = [c for c in battery_cols if c in da.columns and da[c].notna().any()]
    missing = [c for c in battery_cols if c not in cols]
    if missing:
        train_only = [c for c in missing if c in ("el2n", "forget_events")]
        if train_only and split == "test":
            log(f"{train_only} are training-set scores and do not exist on the "
                f"test split. Q4 on 'test' uses {len(cols)}/7 scores -- an "
                f"EASIER test for MSC. Use split='train_holdout' for the "
                f"full battery.", "WARN")
        else:
            log(f"battery incomplete, missing {missing}. Q4's answer is weaker "
                f"than it should be -- rerun the oracle with train_dynamics "
                f"present.", "WARN")
    rows = []
    for t in taus:
        ma = msc_for_run(da, budgets_by_run[run_a], axis, t).clean()
        mb = msc_for_run(db, budgets_by_run[run_b], axis, t).clean()
        res = core.irreducibility(ma, mb, da[cols], n_boot=n_boot)
        rows.append({"run_a": run_a, "run_b": run_b, "axis": axis, "tau": t,
                     "split": split, "n_battery_scores": len(cols),
                     "battery": ",".join(cols), **res,
                     "delta_r2_lo": res["delta_r2_ci95"][0],
                     "delta_r2_hi": res["delta_r2_ci95"][1]})
    out = pd.DataFrame(rows)
    return out.drop(columns=["delta_r2_ci95"], errors="ignore")


def phase0_decision(seed_rho: float, transfer_T: float, delta_r2: float) -> Dict[str, Any]:
    """The 01_PHASE0_GO_NOGO.md 6 decision table, encoded.

    Three of its five rows lead to a paper. That is the whole design intent of
    the restructure: the project's value is not contingent on one method
    beating baselines.
    """
    if seed_rho < 0.4:
        d = ("FAIL", "MSC is noise-dominated. Retry once with a coarser K=3 budget "
                     "grid on the existing checkpoints (no retraining needed). If it "
                     "still fails, switch to the fallback direction in protocol 9.")
    elif seed_rho < 0.6:
        d = ("MARGINAL", "Coarsen to K=3 well-separated budgets and re-run the "
                         "analysis on existing checkpoints. Re-evaluate before "
                         "committing to Phase 1.")
    elif transfer_T < 0.5:
        d = ("PIVOT-STRONG-NEGATIVE",
             "Per-sample compute requirements are architecture-specific. Drop the "
             "method; expand the atlas across families instead. This is a BETTER "
             "paper than the method paper -- it says teacher-guided adaptive "
             "inference rests on a false premise, and explains why.")
    elif delta_r2 < 0.02:
        d = ("REFRAME", "MSC is difficulty renamed. Paper becomes 'cheap difficulty "
                        "scores are sufficient for compute routing'. Skip the "
                        "multi-axis oracle; keep the routing method with a "
                        "difficulty-score gate.")
    elif transfer_T >= 0.7 and delta_r2 >= 0.05:
        d = ("FULL-PROGRAM", "Best case. Proceed to the Phase 1 atlas and build "
                             "MSC-KD.")
    else:
        d = ("MARGINAL-PROCEED",
             "Between gates. Expand to a third architecture before committing the "
             "full 1,200 GPU-hours.")
    return {"decision": d[0], "action": d[1],
            "rho_seed": float(seed_rho), "T_within_family": float(transfer_T),
            "delta_r2": float(delta_r2), "decided_utc": now_iso(),
            "gate_source": "01_PHASE0_GO_NOGO.md section 6"}


def write_gate_decision(data_dir, payload: Dict[str, Any],
                        hub: Optional[MSCHub] = None) -> Path:
    p = Path(data_dir) / "analysis" / "phase0_decision.json"
    atomic_write_json(p, payload)
    if hub is not None and hub.enabled:
        hub.hub.enqueue(p, "analysis/phase0_decision.json")
    print("\n" + "=" * 72)
    print(f"  PHASE 0 DECISION: {payload['decision']}")
    print("=" * 72)
    print(f"  rho_seed = {payload['rho_seed']:.3f}   "
          f"T = {payload['T_within_family']:.3f}   "
          f"dR2 = {payload['delta_r2']:.3f}")
    print(f"\n  {payload['action']}\n")
    print("=" * 72 + "\n")
    return p


def save_analysis(data_dir, name: str, frame, hub: Optional[MSCHub] = None) -> Path:
    p = ensure_dir(Path(data_dir) / "analysis") / f"{name}.csv"
    frame.to_csv(p, index=False)
    if hub is not None and hub.enabled:
        hub.hub.enqueue(p, f"analysis/{name}.csv")
    return p


def save_figure(fig, data_dir, name: str, hub: Optional[MSCHub] = None) -> Path:
    p = ensure_dir(Path(data_dir) / "paper" / "figures") / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if hub is not None and hub.enabled:
        hub.hub.enqueue(p, f"paper/figures/{name}.png")
    return p


def provenance_manifest(data_dir, hub: Optional[MSCHub] = None) -> "Any":
    """Every artifact mapped to the run_id that produced it.

    Requirement 1 of 02_ENGINEERING_SPEC.md 8: every number in the paper maps
    to a run_id. This produces the table that makes that checkable rather than
    aspirational.
    """
    data_dir = Path(data_dir)
    rows = []
    for base, kind in ((data_dir / "runs", "run"),):
        if not base.exists():
            continue
        for rd in sorted(base.iterdir()):
            if not rd.is_dir():
                continue
            for f in sorted(rd.rglob("*")):
                if f.is_file():
                    rows.append({"run_id": rd.name, "kind": kind,
                                 "path": str(f.relative_to(data_dir)),
                                 "size_bytes": f.stat().st_size,
                                 "sha256": sha256_of_file(f) if f.stat().st_size < 5e8
                                           else "skipped-large"})
    df = pd.DataFrame(rows) if pd is not None else rows
    p = ensure_dir(data_dir / "paper") / "provenance.csv"
    if pd is not None:
        df.to_csv(p, index=False)
        if hub is not None and hub.enabled:
            hub.hub.enqueue(p, "paper/provenance.csv")
    return df


# --------------------------------------------------------------------------
# 15b. MSC-KD training driver and the head-to-head comparison
# --------------------------------------------------------------------------
def _teacher_msc_vector(data_dir, teacher_run: str, budgets_teacher,
                        axis: str = "depth", tau: float = 0.1,
                        split: str = "test"):
    """Teacher MSC per sample, plus its irreducible mask.

    The mask matters: samples where the teacher itself was below the margin
    carry a degenerate MSC == 1 target, and training the router on them teaches
    it to always spend everything on exactly the inputs where the teacher had
    no usable opinion.
    """
    df = load_per_sample(data_dir, teacher_run, split)
    r = msc_for_run(df, budgets_teacher, axis, tau)
    idx = df["sample_idx"].to_numpy().astype(np.int64)
    return idx, r.msc.astype(np.float32), r.irreducible.astype(bool), df


def train_msc_kd(cfg: Dict[str, Any], hub: MSCHub, registry: RunRegistry,
                 teacher_run: str, teacher_arch: str,
                 work_root=None, data_root_out=None,
                 alpha: float = 1.0, beta: float = 1.0, temperature: float = 4.0,
                 tau: float = 0.1, axis: str = "depth",
                 shuffle_targets: bool = False,
                 show_progress: bool = True) -> Dict[str, Any]:
    """Distil the teacher's per-sample compute requirement into a student router.

    The student learns three things at once: the task (CE), the teacher's soft
    predictions (KD), and the teacher's compute assessment (MSC). Three terms,
    two weights, and monotonicity enforced by the head's architecture rather
    than by a fourth loss.

    `shuffle_targets=True` runs the mandatory ablation: MSC targets permuted
    within the dataset. If that performs as well as the real thing, L_MSC is a
    regulariser and the mechanism claim is wrong -- which you need to know
    before writing anything, so run it early.

    Resumable on the same contract as train_backbone.
    """
    if not _TORCH_OK:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERR}")

    run_id = cfg["run_id"]
    work = Path(work_root or (WORK_ROOT / "msc"))
    data_out = Path(data_root_out or (work / "data"))
    L = run_layout(work, run_id)
    run_dir = ensure_dir(L["base"])
    for _s in RUN_SUBDIRS:
        ensure_dir(L[_s])
    log_dir, met_dir = L["telemetry"], L["metrics"]
    ckpt_last = L["checkpoints"] / "ckpt_last.pt"
    ckpt_best = L["checkpoints"] / "ckpt_best.pt"
    history_path = met_dir / "epochs.csv"
    sync = RunSync(hub, run_id, run_dir, data_out)

    registry.pull()
    ok, why = registry.can_claim(run_id, force=bool(cfg.get("force_rerun")))
    if not ok:
        log(f"SKIP {run_id}: {why}", "CLAIM")
        return {"run_id": run_id, "status": "skipped", "reason": why}

    # D-19: check the artifact BEFORE the teacher sweep, which is the expensive
    # part of this function -- a full multi-exit pass over 50,000 training
    # images. Discovering "already done" after paying for that is no use.
    _cached = already_finished(hub, work, run_id, cfg, registry)
    if _cached is not None:
        return _cached

    atomic_write_yaml(run_dir / "config.yaml", cfg)
    atomic_write_json(L["env"] / "environment.json", environment_report())
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, holdout_loader, classes, order_hash = build_loaders(cfg)

    # --- teacher ---------------------------------------------------------
    t_budgets = load_or_build_budgets(teacher_arch, data_out, cfg["num_classes"], hub=hub)
    tL = run_layout(work, teacher_run)
    t_dir = tL["base"]
    t_ck = tL["checkpoints"] / "ckpt_best.pt"
    if not t_ck.exists() and hub.enabled:
        hub.hub.download(work, allow_patterns=[f"runs/{teacher_run}/**"])
    if not t_ck.exists():
        raise FileNotFoundError(f"teacher checkpoint missing for {teacher_run}")
    teacher = build_model(teacher_arch, cfg["num_classes"]).to(device)
    teacher.load_state_dict(torch.load(t_ck, map_location=device,
                                       weights_only=False)["model"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- O-19 / D-21 / D-22: fail in seconds, not in an hour ---------------
    # Everything below this point -- exit-head training, the 50,000-image sweep,
    # the first epoch -- costs about an hour before the first student batch is
    # attempted, and the history row is only written at the END of that epoch.
    # D-21 (an AMP-illegal loss) and D-22 (five wrong column names) each hid
    # behind that hour. One synthetic batch and one throwaway history row
    # exercise both code paths in under a second.
    _dry_amp = bool(cfg.get("amp_enabled", True)) and device.type == "cuda"
    _dry_ok, _dry_why = msckd_dry_run(cfg, teacher, device, _dry_amp,
                                      alpha, beta, temperature)
    if not _dry_ok:
        registry.fail(run_id, f"dry run failed: {_dry_why}")
        raise RuntimeError(
            f"MSC-KD dry run failed BEFORE any expensive work: {_dry_why}\n"
            f"This is the same code path the real training loop uses, so fix "
            f"it and re-run -- no GPU time has been spent.")

    # Teacher MSC targets, aligned to the TRAINING set. The oracle writes the
    # test set and a 5k train holdout; the router needs targets on the data the
    # student actually trains on, so we sweep the teacher's exits over train.
    # D-23: use the SAME accessor the writer uses. This used to hard-code
    # `checkpoints/exit_heads.pt` while run_oracle writes to the run root, so
    # the heads were never found and every one of the nine MSC-KD runs retrained
    # them -- ~20 epochs each, for a file already on HuggingFace.
    t_heads_p = find_exit_heads(work, teacher_run)
    if t_heads_p is None and hub is not None and getattr(hub, "enabled", False):
        log(f"teacher exit heads not local -- pulling {teacher_run} from HF "
            f"before retraining them", "MSCKD")
        try:
            hub.hub.download(work, allow_patterns=[f"runs/{teacher_run}/**"],
                             quiet=True)
        except Exception as e:                               # noqa: BLE001
            log(f"pull failed: {type(e).__name__}: {e}", "MSCKD")
        t_heads_p = find_exit_heads(work, teacher_run)

    t_me = MultiExitModel(teacher, cfg["num_classes"], freeze=True).to(device)
    if t_heads_p is not None:
        log(f"reusing teacher exit heads from {t_heads_p.relative_to(work)}",
            "MSCKD")
        t_me.heads.load_state_dict(torch.load(t_heads_p, map_location=device,
                                              weights_only=False)["heads"])
    else:
        log(f"teacher exit heads genuinely absent (looked at "
            f"{exit_heads_path(work, teacher_run).relative_to(work)} and the "
            f"legacy checkpoints/ path) -- training them now, backbone frozen. "
            f"This happens ONCE; later runs reuse the file.", "MSCKD")
        t_me = train_exit_heads(cfg, teacher, train_loader, val_loader, device,
                                hub, t_dir, show_progress)

    log("sweeping teacher over the training set for MSC targets", "MSCKD")
    train_eval = DataLoader(train_loader.dataset, batch_size=int(cfg.get("eval_batch_size", 512)),
                            shuffle=False, num_workers=0, pin_memory=True)
    # Augmentation off while measuring: MSC of an augmented view is not MSC of
    # the sample.
    was_aug = getattr(train_eval.dataset, "augment", False)
    try:
        train_eval.dataset.augment = False
    except Exception:
        pass
    sweep = sweep_all_axes(cfg, t_me, train_eval, device, show_progress=show_progress)
    try:
        train_eval.dataset.augment = was_aug
    except Exception:
        pass

    core = _import_msc_core()
    rho_list = t_budgets["axes"]["depth"]["rho"]
    r = core.compute_msc(sweep["depth"]["preds"], sweep["depth"]["top1p"],
                         sweep["depth"]["top2p"], rho_list, tau=tau, axis="depth")
    order = np.argsort(sweep["sample_idx"])
    msc_train = r.msc[order].astype(np.float32)
    irr_train = r.irreducible[order].astype(bool)
    if shuffle_targets:
        log("SHUFFLED-TARGET ABLATION: MSC targets permuted within the dataset",
            "ABLATE")
        msc_train = shuffle_msc_targets(msc_train, seed=int(cfg["seed"]))
    log(f"teacher MSC on train: mean={np.nanmean(msc_train):.3f}  "
        f"irreducible={irr_train.mean()*100:.1f}%", "MSCKD")

    msc_t = torch.from_numpy(msc_train).to(device)
    irr_t = torch.from_numpy(irr_train).to(device)
    # D-28: the router lives on the STUDENT's budget grid, not the teacher's.
    #
    # `rho_list` above is the teacher's, and is correct for computing the
    # teacher's MSC. But the sufficiency head, its targets and the routing
    # decision all describe what the STUDENT will spend, and the student's exit
    # count is adaptive (D-01b): `resnet8x4` has 3 depth budgets where the
    # `resnet32x4` teacher has 5. Sizing the head from the teacher gave a
    # 5-column router bolted onto a 3-exit model -- consistent right up to
    # evaluation, where `correct_at` (3 columns, from the student's exits) met
    # a route index of 3 and raised IndexError.
    #
    # The teacher's MSC is a scalar fraction in [0, 1]; `sufficiency_targets`
    # projects it onto whichever grid it is given. Give it the student's.
    s_budgets = load_or_build_budgets(cfg["arch"], data_out, cfg["num_classes"],
                                      hub=hub)
    rho_student = list(s_budgets["axes"]["depth"]["rho"])
    if len(rho_student) != len(rho_list):
        log(f"student {cfg['arch']} has {len(rho_student)} depth budgets vs the "
            f"{teacher_arch} teacher's {len(rho_list)} -- routing on the "
            f"student's grid (D-28)", "MSCKD")
    rho_t = torch.tensor(rho_student, dtype=torch.float32, device=device)

    # --- student ---------------------------------------------------------
    student = MSCStudent(build_model(cfg["arch"], cfg["num_classes"]),
                         cfg["num_classes"], len(rho_student)).to(device)
    # The head must have exactly one output per student exit, or routing
    # indexes a column that does not exist.
    _n_heads = len(student.heads)
    assert _n_heads == len(rho_student), (
        f"{cfg['arch']}: {_n_heads} exit heads but {len(rho_student)} depth "
        f"budgets. These must match -- see D-28.")
    optimizer, scheduler = build_optimizer(student, cfg)
    amp = bool(cfg.get("amp_enabled", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    lossfn = MSCLoss(alpha=alpha, beta=beta, temperature=temperature)

    # D-19: recover this run's own checkpoint from HF before load_checkpoint
    # reads an absent file as "never started".
    ensure_run_local(hub, work, run_id, why="MSC-KD resume")
    st = load_checkpoint(ckpt_last, cfg, student, optimizer, scheduler, scaler,
                         None, device, strict_hash=not cfg.get("force_rerun"))
    start_epoch, best = st["start_epoch"], st["best_metric"]
    cum_time, cum_energy = st["wall_seconds"], st["energy_joules"]
    if st["resumed"]:
        _truncate_history(history_path, start_epoch)
        log(f"{run_id} resuming at epoch {start_epoch}", "RESUME")

    num_epochs = int(cfg["num_epochs"])
    milestone = max(1, int(cfg.get("milestone_push_every_epochs", 10)))
    timer_sec = float(cfg.get("timer_push_sec", 1800))
    state = {"epoch": start_epoch - 1, "best": best}
    registry.claim(run_id, arch=cfg["arch"], teacher=teacher_run, method=cfg["method"],
                   seed=cfg["seed"], config_hash=cfg["config_hash"])

    def _flush(reason):
        try:
            save_checkpoint(ckpt_last, cfg, student, optimizer, scheduler, scaler,
                            state["epoch"], state["best"], None, cum_time, cum_energy)
        except Exception:
            traceback.print_exc()
        registry.heartbeat(run_id, run_dir, state="paused", epoch=state["epoch"],
                           reason=reason)
        registry.pause(run_id, epoch=state["epoch"], reason=reason)
        sync.push_all(heavy=True)
        sync.flush(timeout=600)

    guard = LifecycleGuard(_flush, session_limit_h=float(cfg.get("session_limit_h", 8.5))).install()
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    last_push = -10 ** 9
    try:
        for epoch in range(start_epoch, num_epochs):
            student.train()
            t0 = time.time()
            mon = GPUEnergyMonitor(sample_hz=float(cfg.get("energy_sample_hz", 10.0)))
            mon.start()
            agg = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "msc": 0.0}
            nb = 0
            it = train_loader
            if tqdm is not None and show_progress:
                it = tqdm(train_loader, desc=f"{run_id} ep {epoch+1}/{num_epochs}",
                          leave=False, dynamic_ncols=True, mininterval=2.0)
            for batch in it:
                x, y, idx = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    with torch.no_grad():
                        t_logits = teacher(x)
                    # D-21: the loss needs pre-sigmoid scores, not probabilities.
                    s_logits, suff, _ = student(x, suff_logits=True)
                    targets = sufficiency_targets(msc_t[idx], rho_t)
                    # Supervise the deepest exit for CE/KD; the shallower heads
                    # are trained by the mean CE below so every route is usable.
                    loss, parts = lossfn(s_logits[-1], t_logits, y, suff, targets,
                                         irreducible=irr_t[idx])
                    loss = loss + sum(F.cross_entropy(l, y)
                                      for l in s_logits[:-1]) / max(1, len(s_logits) - 1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                for k in agg:
                    agg[k] += parts[k]
                nb += 1
            samples = mon.stop()
            dt = time.time() - t0
            cum_time += dt
            cum_energy += GPUEnergyMonitor.integrate_j(samples, dt)
            if scheduler is not None:
                scheduler.step()

            class _Deepest(nn.Module):
                def __init__(self, s):
                    super().__init__()
                    self.s = s

                def forward(self, x):
                    return self.s(x)[0][-1]

            val = evaluate(_Deepest(student), val_loader, device, amp)
            acc = float(val["accuracy"])
            row = msckd_history_row(
                run_id=run_id, cfg=cfg, epoch=epoch, agg=agg, nb=nb, val=val,
                acc=acc, best_before=best, lr=float(optimizer.param_groups[0]["lr"]),
                amp=amp, dt=dt, cum_time=cum_time, cum_energy=cum_energy,
                n_train_images=len(train_loader.dataset),
                alpha=alpha, beta=beta, temperature=temperature)
            append_history_row(history_path, row, strict=True)

            if acc > best:
                best = acc
                atomic_save_torch(ckpt_best, {"run_id": run_id,
                                              "model": student.state_dict(),
                                              "epoch": epoch, "val_accuracy": acc,
                                              "config_hash": cfg["config_hash"],
                                              "rho": rho_student,
                                              "teacher_rho": rho_list,
                                              "config": cfg})
            state["epoch"], state["best"] = epoch, best
            save_checkpoint(ckpt_last, cfg, student, optimizer, scheduler, scaler,
                            epoch, best, None, cum_time, cum_energy)
            print(f"  ep {epoch+1}/{num_epochs}  val={acc:.4f}  "
                  f"ce={agg['ce']/max(1,nb):.3f}  kd={agg['kd']/max(1,nb):.3f}  "
                  f"msc={agg['msc']/max(1,nb):.3f}  t={dt:.1f}s")

            if (((epoch + 1) % milestone == 0) or (epoch == num_epochs - 1)
                    or sync.due_for_timer_push(timer_sec) or guard.session_expiring()):
                last_push = epoch
                registry.heartbeat(run_id, run_dir, state="running", epoch=epoch,
                                   best_metric=best)
                sync.push_all(heavy=True)
            if guard.session_expiring():
                _flush("session limit")
                return {"run_id": run_id, "status": "paused", "epoch": epoch}
    except KeyboardInterrupt:
        _flush("KeyboardInterrupt")
        raise
    except Exception as e:
        traceback.print_exc()
        registry.fail(run_id, f"{type(e).__name__}: {e}")
        _flush("exception")
        raise

    summary = {"run_id": run_id, "arch": cfg["arch"], "teacher": teacher_run,
               "method": cfg["method"], "seed": cfg["seed"],
               "alpha": alpha, "beta": beta, "temperature": temperature,
               "tau": tau, "axis": axis, "shuffled_targets": bool(shuffle_targets),
               "best_accuracy": float(best),
               # D-24: `num_epochs_planned` is part of the summary contract --
               # repair_ledger reads it to decide whether a run is a broken
               # stub. Omitting it here got every completed MSC-KD run demoted.
               "num_epochs_planned": int(num_epochs),
               "num_epochs_run": state["epoch"] + 1,
               "total_time_sec": cum_time, "total_energy_j": cum_energy,
               "config_hash": cfg["config_hash"], "sample_order_hash": order_hash,
               "status": "completed", "completed_utc": now_iso()}
    atomic_write_json(run_dir / "summary.json", summary)
    registry.finish(run_id, **{k: summary[k] for k in
                               ("arch", "teacher", "method", "seed", "best_accuracy")})
    sync.push_all(heavy=True)
    sync.flush(timeout=1200)
    hub.print_stats()
    return summary


@_no_grad()
def evaluate_routing_methods(student, val_loader, device, rho: Sequence[float],
                             full_flops: float, oracle_msc: Optional[np.ndarray] = None,
                             amp: bool = True) -> Dict[str, Any]:
    """B1 / B2 / B10 / B11 on one pass, at matched average FLOPs.

    B2 vs B10 vs B11 is the paper's central figure: B2 is where the field
    actually is (confidence thresholding), B11 is the ceiling (route by the
    student's own true post-hoc MSC), and the fraction of the B2->B11 gap that
    B10 closes IS the result. Reporting B10 against B1 alone would be measuring
    against a straw man.
    """
    student.eval()
    all_logits, all_suff, all_y = [], [], []
    for batch in val_loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with torch.amp.autocast(device_type=device.type,
                                enabled=(amp and device.type == "cuda")):
            logits, suff, _ = student(x)
        all_logits.append(torch.stack([l.float() for l in logits], 1).cpu().numpy())
        all_suff.append(suff.float().cpu().numpy())
        all_y.append(np.asarray(y))
    L = np.concatenate(all_logits)            # (N, K, C)
    S = np.concatenate(all_suff)              # (N, K)
    Y = np.concatenate(all_y)                 # (N,)

    # D-28: three things must agree on K -- the exit logits, the sufficiency
    # head, and the budget table. When they did not, the mismatch surfaced
    # eight frames down as `IndexError: index 3 is out of bounds`, which says
    # nothing about the cause. Say it here instead.
    if not (L.shape[1] == S.shape[1] == len(rho)):
        raise ValueError(
            f"routing shapes disagree: {L.shape[1]} exit heads, "
            f"{S.shape[1]} sufficiency outputs, {len(rho)} budgets. "
            f"All three must match. A student trained before the D-28 fix has "
            f"a head sized from the TEACHER's budget grid -- retrain it, or "
            f"pass the rho it was trained with.")

    correct_at = (L.argmax(2) == Y[:, None]).astype(float)     # (N, K)
    probs = np.exp(L - L.max(2, keepdims=True))
    probs /= probs.sum(2, keepdims=True)
    top1p = probs.max(2)                                        # (N, K)
    n, K = correct_at.shape
    full_acc = float(correct_at[:, -1].mean())

    out: Dict[str, Any] = {"n": n, "K": K, "full_accuracy": full_acc,
                           "full_flops": float(full_flops)}
    out["B1_static_full"] = {"accuracy": full_acc, "avg_flops": float(full_flops),
                             "avg_rho": 1.0}
    out["curves"] = {
        "B2_confidence": sweep_operating_points(top1p, correct_at, rho, full_flops),
        "B10_msc_kd": sweep_operating_points(S, correct_at, rho, full_flops),
    }
    if oracle_msc is not None:
        # B11 ceiling: route by the student's own true post-hoc MSC.
        r = np.asarray(rho, float)
        oracle_route = np.clip(np.searchsorted(r, np.asarray(oracle_msc, float),
                                               side="left"), 0, K - 1)
        out["B11_oracle"] = {
            "accuracy": float(correct_at[np.arange(n), oracle_route].mean()),
            "avg_flops": expected_flops(oracle_route, rho, full_flops),
            "avg_rho": float(r[oracle_route].mean())}

    # Head-to-head at the operating point B10 naturally lands on.
    if pd is not None:
        c10, c2 = out["curves"]["B10_msc_kd"], out["curves"]["B2_confidence"]
        mid = c10.iloc[len(c10) // 2]
        target = float(mid["avg_flops"])
        a10 = accuracy_at_matched_flops(c10, target)
        a2 = accuracy_at_matched_flops(c2, target)
        out["matched_flops_comparison"] = {
            "target_avg_flops": target,
            "target_avg_rho": target / max(1e-12, full_flops),
            "B10_accuracy": a10, "B2_accuracy": a2,
            "gap_points": (a10 - a2) * 100.0,
            "B10_auc": auc_accuracy_flops(c10),
            "B2_auc": auc_accuracy_flops(c2)}
        if "B11_oracle" in out:
            gap_total = out["B11_oracle"]["accuracy"] - a2
            out["matched_flops_comparison"]["fraction_of_B2_to_B11_gap_closed"] = (
                float((a10 - a2) / gap_total) if abs(gap_total) > 1e-9 else float("nan"))
    return out


# =============================================================================
# 17. session -- one-call notebook bootstrap
# =============================================================================
class Session:
    """Everything a notebook needs, assembled in one call.

    Encapsulates: token, both uploaders, registry, local layout, scoped state
    pull, and a global lifecycle guard. A notebook cell should be four lines,
    not forty -- and more importantly, the flush-on-exit behaviour should not
    depend on whoever wrote that particular notebook remembering to add it.
    """

    def __init__(self, account: str = "acct1", phase: str = "p1",
                 dataset: str = "cifar100", enable_hf: bool = True,
                 work_root=None, session_limit_h: float = 8.5,
                 commits_per_hour_limit: int = 20,
                 batch_interval_sec: float = 1800.0,
                 worker_id: int = 0, num_workers: int = 1,
                 shard_mode: str = "cost"):
        assert 0 <= worker_id < num_workers, \
            f"WORKER_ID must be in 0..{num_workers-1}, got {worker_id}"
        self.account = account
        self.phase = phase
        self.dataset = dataset
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.shard_mode = shard_mode
        # The whole repo tree is staged on SCRATCH (~1 TB), not on the 20 GB
        # working disk. A 240-epoch run with 10 Hz power sampling and full step
        # traces is then never disk-constrained, and /kaggle/working stays free.
        # HuggingFace is the permanent store either way, so losing scratch at
        # session end costs at most one push interval.
        self.work = ensure_dir(Path(work_root or (SCRATCH_ROOT / "msc")))
        self.data_dir = self.work                  # repo root == staging root
        self.runs_dir = ensure_dir(self.work / "runs")
        self.scratch = self.work
        for _d in ("registry", "analysis", "tables", "paper", "budgets"):
            ensure_dir(self.work / _d)
        self.console = self.work / "console" / f"{account}_w{worker_id}_{phase}.log"
        ensure_dir(self.console.parent)

        self.hub = MSCHub(enable=enable_hf,
                          commits_per_hour_limit=commits_per_hour_limit,
                          batch_interval_sec=batch_interval_sec)
        self.registry = RunRegistry(self.hub, self.data_dir, account=account,
                                    worker_id=self.worker_id)
        self.guard = LifecycleGuard(self._flush_all,
                                    session_limit_h=session_limit_h).install()
        self.data_root: Optional[Path] = None

        print(f"[SESSION] account={account} phase={phase} dataset={dataset}")
        print(f"[SESSION] worker {self.worker_id} of {self.num_workers}"
              + ("  (single worker -- set NUM_WORKERS to parallelise)"
                 if self.num_workers == 1 else ""))
        print(f"[SESSION] work={self.work}  scratch={self.scratch}")
        print(f"[SESSION] disk free: working={free_mb(self.work)} MB  "
              f"scratch={free_mb(self.scratch)} MB")
        if not self.hub.enabled:
            print("[SESSION] *** HF DISABLED -- nothing will survive this session ***")

    # ------------------------------------------------------------------
    def prepare_data(self) -> Path:
        self.data_root = locate_cifar100()
        return self.data_root

    def config(self, arch: str, seed: int = 1, method: str = "base",
               **overrides) -> Dict[str, Any]:
        if self.data_root is None:
            self.prepare_data()
        cfg = base_config(arch, self.dataset, seed, phase=self.phase, method=method)
        cfg.update({"data_root": str(self.data_root),
                    "output_root": str(self.work)})
        cfg.update(overrides)
        # Recompute after overrides -- an override that changes the recipe must
        # change the hash, or resume will happily continue under the new one.
        cfg["config_hash"] = config_hash(cfg)
        cfg["run_id"] = make_run_id(cfg["phase"], cfg["arch"], cfg["dataset_name"],
                                    cfg["method"], cfg["seed"])
        return cfg

    def sync_state(self, run_ids: Optional[Sequence[str]] = None,
                   include_checkpoints: bool = True, verbose: bool = True) -> None:
        """Scoped pull from HF. NEVER unscoped on a 20 GB disk.

        Also repairs the local ledger from history.csv rather than trusting
        progress state alone: a session that died between writing history and
        pushing the ledger leaves them disagreeing, and history.csv is the one
        that reflects what actually happened.
        """
        if not self.hub.enabled:
            return
        if verbose:
            log(f"pulling state (free: {free_mb(self.work)} MB)", "SYNC")
        # Scoped. Never unscoped -- a full snapshot late in the project is
        # hundreds of GB of checkpoints.
        pats = ["registry/**", "budgets/**", "analysis/**", "tables/**"]
        heavy = ["checkpoints/**"] if include_checkpoints else []
        want = list(run_ids) if run_ids else ["*"]
        for r in want:
            pats += [f"runs/{r}/*", f"runs/{r}/metrics/**",
                     f"runs/{r}/per_sample/**", f"runs/{r}/env/**"]
            if include_checkpoints:
                pats += [f"runs/{r}/checkpoints/**"]
        self.hub.hub.download(self.data_dir, allow_patterns=pats, quiet=not verbose)
        self._drop_hf_cache()
        n = self.repair_ledger()
        if verbose:
            log(f"pull complete (free: {free_mb(self.work)} MB, "
                f"{n} ledger entries repaired)", "SYNC")

    def _drop_hf_cache(self) -> None:
        # snapshot_download leaves a .cache tree that can double disk usage.
        for base in (self.data_dir, self.runs_dir):
            for c in (base / ".cache", base / ".huggingface"):
                if c.exists():
                    shutil.rmtree(c, ignore_errors=True)

    def repair_ledger(self) -> int:
        """Rebuild run state from history.csv -- the ground truth.

        Also demotes broken stubs: a run recorded as `completed` whose history
        stops well short of its planned epochs was killed mid-push and lied
        about it. Left alone, every future session skips it forever.
        """
        if pd is None:
            return 0
        repaired = 0
        logs = self.runs_dir
        if not logs.exists():
            return 0
        known = self.registry.latest()
        for rd in sorted(logs.iterdir()):
            if not rd.is_dir():
                continue
            h = rd / "metrics" / "epochs.csv"
            if not h.exists() or h.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(h)
                if df.empty:
                    continue
                last_ep = int(df["epoch"].max())
                best = float(df["val_accuracy"].max())
            except Exception:
                continue
            summ = read_json(rd / "summary.json", default={}) or {}
            # D-24: this used to read ONLY `num_epochs_planned`, which
            # `train_msc_kd` does not write. Missing field -> planned = 0 ->
            # `planned > 0` false -> `done` false -> a run that finished all
            # 240 epochs was DEMOTED to `paused` on every sync, and the log
            # said "marked completed at only 240 epochs", which is the number
            # it was supposed to reach.
            #
            # Absence of a field is not evidence a run is short. Fall back to
            # what the summary claims it ran; the stub check still works,
            # because a real stub's history is short against EITHER target.
            planned = int(summ.get("num_epochs_planned", 0) or 0)
            claimed = int(summ.get("num_epochs_run", 0) or 0)
            target = planned or claimed
            status_ok = summ.get("status") == "completed"
            # D-26: `summary.json` is written AFTER the training loop exits, so
            # a summary claiming a full run IS the completion record.
            # `epochs.csv` is telemetry pushed on a 30-minute timer, and a
            # session that ended between its last history push and its summary
            # push leaves a SHORT HISTORY FOR A RUN THAT GENUINELY FINISHED.
            #
            # Judging on history alone demoted five completed atlas runs --
            # resnet110-s1 at "161 epochs", resnet32x4-s2 at "40" -- all of
            # which have summaries saying 240/240 and a best checkpoint on HF.
            # Trust the summary when it is self-consistent; fall back to the
            # history only when the summary cannot answer.
            if status_ok and target > 0 and claimed >= 0.9 * target:
                done = True
            else:
                done = status_ok and target > 0 and (last_ep + 1) >= 0.9 * target
            cur = known.get(rd.name, {})
            ident = parse_run_id(rd.name)
            if (not done) and status_ok and target <= 0:
                # Neither field usable. Refuse to act: a repair that destroys
                # good state on missing evidence is worse than no repair.
                log(f"{rd.name}: summary says completed but carries no epoch "
                    f"count -- NOT demoting on absent evidence (D-24)",
                    "REPAIR")
                continue
            if done and cur.get("state") != "completed":
                self.registry.append(rd.name, "completed", best_accuracy=best,
                                     num_epochs_run=last_ep + 1, repaired=True,
                                     arch=ident["arch"], seed=ident["seed"],
                                     dataset=ident["dataset"], phase=ident["phase"])
                repaired += 1
            elif (not done) and cur.get("state") == "completed":
                log(f"broken stub: {rd.name} marked completed at only "
                    f"{last_ep+1} epochs -- demoting to paused so it resumes",
                    "REPAIR")
                self.registry.append(rd.name, "paused", best_accuracy=best,
                                     last_completed_epoch=last_ep,
                                     demoted_broken_stub=True,
                                     arch=ident["arch"], seed=ident["seed"],
                                     dataset=ident["dataset"], phase=ident["phase"])
                repaired += 1
        return repaired

    # ------------------------------------------------------------------
    def measured(self, run_id: str, split: str = "test") -> bool:
        """Has the ORACLE SWEEP produced this run's per-sample tables?

        The stage-completion predicate for measurement. Checks the artifact
        rather than the ledger, because the ledger's single `state` field is
        already "completed" from training.
        """
        ps = run_layout(self.work, run_id)["per_sample"]
        return any((ps / f"{split}.{e}").exists() for e in ("parquet", "csv"))

    def trained(self, run_id: str) -> bool:
        """Has TRAINING finished for this run?"""
        st = self.registry.latest().get(run_id, {})
        return (st.get("state") == "completed"
                or (run_layout(self.work, run_id)["base"] / "summary.json").exists())

    def plan(self, run_ids: Sequence[str], steal_stale: bool = True,
             describe: bool = True, title: str = "work plan",
             mode: Optional[str] = None,
             done_fn: Optional[Callable[[str], bool]] = None,
             stage: str = "train") -> WorkerPlan:
        """This worker's slice of the given runs. See section 4b.

        Uses measured per-epoch times from any runs already finished, falling
        back to the built-in hints. So the scheduler gets better at balancing
        the more of the project you have completed.

        Records the plan to HF so you can reconstruct, months later, which
        account was responsible for which run.
        """
        # OWNERSHIP USES THE STATIC COST TABLE ONLY. This is not a detail.
        #
        # The whole sharding guarantee is "identical code + identical input =
        # identical assignment, with no communication". Feeding MEASURED
        # per-epoch times into the assignment breaks that input-identity: a
        # worker planning before any run has finished computes a different
        # packing than one planning after twelve have, so ownership silently
        # changes between sessions.
        #
        # That is exactly what happened on 2026-08-02 (defect D-12): acct4's
        # first session owned resnet32x4-s3 and its second session did not,
        # abandoning it at epoch 79 and re-training acct2's resnet32x4-s1
        # instead. Two runs' worth of damage from a "self-correcting" feature.
        #
        # Measured timings are still used -- but only to REPORT time, never to
        # decide ownership. See estimate_phase().
        measured = estimate_costs_from_history(self.data_dir)
        if measured:
            log(f"{len(measured)} architectures have measured timings "
                f"(used for time estimates only -- ownership is fixed)", "PLAN")
        p = plan_work(run_ids, self.registry, worker_id=self.worker_id,
                      num_workers=self.num_workers, steal_stale=steal_stale,
                      mode=mode or self.shard_mode, costs=None,
                      done_fn=done_fn, stage=stage)
        if describe:
            p.describe(title)
        fn = f"registry/plans/{self.account}_w{self.worker_id}of{self.num_workers}_{self.phase}.json"
        local = self.data_dir / fn
        atomic_write_json(local, {**p.to_dict(), "account": self.account,
                                  "phase": self.phase, "title": title})
        if self.hub.enabled:
            self.hub.hub.enqueue(local, fn)
        return p

    def run_all(self, cfgs: Sequence[Dict[str, Any]], fn: Optional[Callable] = None,
                steal_stale: bool = True, title: str = "work plan",
                done_fn: Optional[Callable[[str], bool]] = None,
                stage: str = "train", **kw) -> List[Dict[str, Any]]:
        """Plan, then execute this worker's share, stopping cleanly at the
        session limit.

        This is the loop every training notebook uses. It exists so that the
        sharding, the disk check, the session-limit break and the error
        handling are written once and cannot be got subtly wrong in one
        notebook out of fourteen.
        """
        fn = fn or self.train
        # Infer the stage from the entry point, so a caller cannot forget it and
        # silently get the training stage's notion of "done".
        #
        # D-19: this used to be a single `if` naming ONE function, so any custom
        # entry point -- NB13 passes a closure over train_msc_kd, NB14 likewise
        # -- fell through with done_fn=None. `plan_work` then falls back to the
        # raw ledger, which is a SINGLE POINT OF FAILURE: if the completion
        # events did not survive the session, every finished run looks unstarted
        # and gets retrained from scratch. `self.trained` checks the ledger OR
        # the run's summary.json, so a lost ledger event alone cannot cause a
        # 30-GPU-hour re-run. Default to it for anything that is not the oracle.
        if done_fn is None:
            if fn is getattr(self, "oracle", None):
                done_fn, stage = self.measured, "measure"
            else:
                done_fn = self.trained
        by_id = {c["run_id"]: c for c in cfgs}
        plan = self.plan(list(by_id), steal_stale=steal_stale, title=title,
                         done_fn=done_fn, stage=stage)

        if not plan.work:
            # Zero work is normal when the stage really is finished, and a bug
            # when it is not. Distinguish, loudly -- a stage that exits in
            # seconds looking like a success is the worst possible outcome.
            unfinished = [r for r in plan.mine
                          if done_fn is not None and not done_fn(r)]
            if unfinished:
                log(f"NOTHING PLANNED, but {len(unfinished)} of this worker's "
                    f"runs are not finished for stage '{stage}': "
                    f"{unfinished[:4]}. This is a bug, not an idle worker.",
                    "ALARM")
            else:
                log(f"nothing to do -- stage '{stage}' is complete for this "
                    f"worker's {len(plan.mine)} run(s)", "PLAN")
        out: List[Dict[str, Any]] = []
        for i, rid in enumerate(plan.work, 1):
            print(f"\n{'='*74}\n>>> [{i}/{len(plan.work)}] {rid}\n{'='*74}")
            if free_mb(self.work) < 3000:
                log(f"working disk at {free_mb(self.work)} MB -- cleaning stale run dirs",
                    "DISK")
                for d in self.runs_dir.iterdir():
                    if d.is_dir() and d.name != rid:
                        shutil.rmtree(d, ignore_errors=True)
            try:
                s = fn(by_id[rid], **kw)
                out.append(s)
                if s.get("status") == "paused":
                    log("session limit reached -- start a fresh session and re-run "
                        "this cell; it continues from here", "LIFE")
                    break
            except KeyboardInterrupt:
                log("interrupted -- everything flushed to HF; re-run to resume", "STOP")
                raise
            except Exception as e:
                traceback.print_exc()
                log(f"{rid} failed: {type(e).__name__}: {e} -- continuing", "ERROR")
                continue
        return out

    def train(self, cfg: Dict[str, Any], **kw) -> Dict[str, Any]:
        cfg = dict(cfg, worker_id=self.worker_id)
        return train_backbone(cfg, self.hub, self.registry,
                              work_root=self.work, data_root_out=self.data_dir, **kw)

    def oracle(self, cfg: Dict[str, Any], **kw) -> Dict[str, Any]:
        cfg = dict(cfg, worker_id=self.worker_id)
        return run_oracle(cfg, self.hub, self.registry,
                          work_root=self.work, data_root_out=self.data_dir, **kw)

    def budgets(self, arch: str, num_classes: int = 100) -> Dict[str, Any]:
        return load_or_build_budgets(arch, self.data_dir, num_classes, hub=self.hub)

    # ------------------------------------------------------------------
    def _flush_all(self, reason: str) -> None:
        if not self.hub.enabled:
            return
        log(f"flushing everything ({reason})", "SESSION")
        for sub in ("registry", "analysis", "budgets", "tables", "paper"):
            self.hub.hub.enqueue_dir(self.data_dir / sub, sub)
        self.hub.hub.enqueue_dir(self.runs_dir, "runs")
        self.hub.flush(timeout=900)
        self.hub.print_stats()

    def flush(self, reason: str = "manual") -> None:
        self._flush_all(reason)

    def finish(self) -> None:
        self._flush_all("notebook complete")
        self.hub.stop(drain=True)
        print(f"[SESSION] done. elapsed {self.guard.elapsed_h:.2f} h")

    def confirm_on_hf(self, run_ids: Sequence[str],
                      require: Optional[Sequence[str]] = None,
                      verbose: bool = True) -> Dict[str, List[str]]:
        """After `finish()`: is the work SAFE on HuggingFace?

        **D-19.** `finish()` drains the upload queue and prints "done", which
        reads like confirmation and is not one -- draining says the queue
        emptied, not that the files landed.

        **D-20. "Safe" is not the same as "finished", and the first version of
        this method confused the two.** It asked only for `summary.json` and
        reported every in-progress run as ``NOT ON HF ... closing now means
        retraining them``. For nine MSC-KD runs paused mid-training that was
        false *and* alarming: their `ckpt_last.pt` was on HF, they would have
        resumed losing nothing, and the message said the opposite.

        A run is therefore in one of three states, not two:

        - **finished**  -- `summary.json` present; nothing left to do.
        - **resumable** -- `checkpoints/ckpt_last.pt` present. Perfectly safe to
          close; the next session picks it up at the epoch it reached.
        - **at risk**   -- neither. This alone is worth an alarm.

        Pass `require=(...)` to check specific paths instead.
        """
        ids = list(run_ids)
        empty = {"ok": [], "done": [], "resumable": [], "at_risk": [],
                 "unknown": ids}
        if not self.hub.enabled:
            if verbose:
                print("[VERIFY] HF disabled -- cannot confirm anything")
            return empty
        try:
            have = set(self.hub.hub.list_repo_files())
        except Exception as e:                               # noqa: BLE001
            log(f"could not list the repo: {type(e).__name__}: {e}. "
                f"Treat this as UNCONFIRMED, not as success.", "ALARM")
            return empty

        latest = self.registry.latest()
        done, resumable, at_risk = [], [], []
        for r in ids:
            base = f"runs/{r}/"
            if require:
                (done if all(f"{base}{x}" in have for x in require)
                 else at_risk).append(r)
            elif f"{base}summary.json" in have:
                done.append(r)
            elif f"{base}checkpoints/ckpt_last.pt" in have:
                resumable.append(r)
            else:
                at_risk.append(r)

        if verbose:
            print(f"\n[VERIFY] {len(ids)} run(s): {len(done)} finished, "
                  f"{len(resumable)} resumable, {len(at_risk)} at risk")
            for r in done:
                print(f"    FINISHED   {r}")
            for r in resumable:
                ep = latest.get(r, {}).get("epoch")
                at = f" (epoch {ep})" if ep is not None else ""
                print(f"    RESUMABLE  {r}{at}")
            for r in at_risk:
                print(f"    AT RISK    {r}")
            if at_risk:
                log(f"{len(at_risk)} run(s) have NEITHER a summary.json NOR a "
                    f"checkpoint on HuggingFace. DO NOT close this session -- "
                    f"re-run sess.finish(), then this cell again.", "ALARM")
            elif resumable:
                print("\n    Nothing is at risk. The resumable runs are "
                      "checkpointed on HuggingFace and will\n    continue from "
                      "where they stopped. Safe to close the session.")
            else:
                print("\n    All finished. Safe to close the session.")
        return {"ok": done + resumable, "done": done, "resumable": resumable,
                "at_risk": at_risk, "unknown": []}

    def status(self) -> "Any":
        return self.registry.summary()

    def completed_runs(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every completed run with its identity resolved from the run_id.

        The entry point every downstream notebook should use. Identity comes
        from `parse_run_id`, so a ledger event written without `arch`/`seed`
        (as `repair_ledger` does) cannot produce a None where a value is needed.
        """
        out = []
        for rid, st in sorted(self.registry.latest().items()):
            if st.get("state") != "completed":
                continue
            if phase and not rid.startswith(f"{phase}-"):
                continue
            m = run_meta(rid, st)
            if m.get("arch") is None or m.get("seed") is None:
                log(f"cannot parse identity from run_id '{rid}' -- skipping", "WARN")
                continue
            out.append({"run_id": rid, "arch": m["arch"], "seed": int(m["seed"]),
                        "dataset": m.get("dataset"), "family": m.get("family"),
                        "accuracy": st.get("best_accuracy"),
                        "measured": self.measured(rid)})
        return out

    def audit_repos(self, expected_run_ids: Optional[Sequence[str]] = None,
                    verbose: bool = True) -> Dict[str, Any]:
        """What is actually on HuggingFace, and does it belong to this pipeline?

        Two questions this answers that nothing else does:

        1. **Is every expected run present and complete?** Checkpoints, config,
           logs, per-sample tables -- listed per run, so a half-pushed run is
           obvious.
        2. **Is there foreign data?** A repo that has been used by an earlier or
           different version of the pipeline will contain runs whose ids do not
           match `{phase}-{arch}-{dataset}-{method}-s{seed}` for any architecture
           in the current zoo. Those are not harmful on their own -- the analysis
           notebooks skip directories without a `meta.json` -- but they make the
           repo confusing to read and can pollute the cost model, so they are
           reported rather than silently tolerated.
        """
        out: Dict[str, Any] = {"checked_utc": now_iso()}
        if not self.hub.enabled:
            print("[AUDIT] HF disabled -- nothing to audit")
            return out

        files = sorted(self.hub.hub.list_repo_files())
        mfiles = dfiles = files
        out["n_files"] = len(files)

        def _runs_under(files, prefix):
            s = set()
            for f in files:
                if f.startswith(prefix):
                    parts = f[len(prefix):].split("/")
                    if parts and parts[0]:
                        s.add(parts[0])
            return s

        all_runs = (_runs_under(files, "runs/") | _runs_under(files, "logs/")
                    | _runs_under(files, "per_sample/"))

        known_archs = set(ZOO)
        def _recognised(rid: str) -> bool:
            p = rid.split("-")
            return len(p) >= 5 and p[1] in known_archs

        out["foreign_runs"] = sorted(r for r in all_runs if not _recognised(r))
        out["own_runs"] = sorted(r for r in all_runs if _recognised(r))

        rows = []
        for r in sorted(all_runs):
            b = f"runs/{r}"
            rows.append({
                "run_id": r,
                "recognised": _recognised(r),
                "config": f"{b}/config.yaml" in files,
                "status": f"{b}/STATUS.json" in files,
                "summary": f"{b}/summary.json" in files,
                "epochs_csv": f"{b}/metrics/epochs.csv" in files,
                "final_csv": f"{b}/metrics/final.csv" in files,
                "confusion": f"{b}/metrics/confusion_matrix.csv" in files,
                "ckpt_last": f"{b}/checkpoints/ckpt_last.pt" in files,
                "ckpt_best": f"{b}/checkpoints/ckpt_best.pt" in files,
                # D-23: canonical is the run root; the legacy path still counts.
                "exit_heads": (f"{b}/exit_heads.pt" in files
                               or f"{b}/checkpoints/exit_heads.pt" in files),
                "energy": f"{b}/telemetry/energy_samples.csv" in files,
                "system": f"{b}/telemetry/system_samples.csv" in files,
                "steps": f"{b}/telemetry/step_traces.jsonl" in files,
                "dynamics": f"{b}/per_sample/train_dynamics.parquet" in files,
                "msc_test": f"{b}/per_sample/test.parquet" in files,
            })
        table = pd.DataFrame(rows) if pd is not None else rows

        if expected_run_ids:
            exp = set(expected_run_ids)
            out["expected"] = sorted(exp)
            out["missing_entirely"] = sorted(exp - all_runs)
            out["started"] = sorted(exp & all_runs)

        n_shards = sum(1 for f in dfiles if f.startswith("registry/events/"))
        out["ledger_shards"] = n_shards

        if verbose:
            print(f"\n{'='*74}\n  HuggingFace audit\n{'='*74}")
            print(f"  repo : {self.hub.repo_id}   {len(files)} files")
            print(f"  ledger shards (one per worker session): {n_shards}"
                  + ("   <- 0 means you are on the pre-sharding library; "
                     "re-upload the notebooks" if n_shards == 0 else ""))
            if pd is not None and len(table):
                print()
                display_cols = [c for c in table.columns if c != "recognised"]
                print(table[display_cols].to_string(index=False))
            if out.get("missing_entirely"):
                print(f"\n  NOT STARTED ({len(out['missing_entirely'])}):")
                for r in out["missing_entirely"]:
                    print(f"    {r}")
            if out["foreign_runs"]:
                print(f"\n  FOREIGN DATA ({len(out['foreign_runs'])} runs) -- these do "
                      f"not match any architecture in the current zoo.")
                print(f"  Most likely from an earlier version of this project.")
                print(f"  They are ignored by the analysis (no meta.json), but "
                      f"consider deleting them:")
                for r in out["foreign_runs"]:
                    print(f"    {r}")
                print(f"\n  To remove:  sess.purge_runs({out['foreign_runs']!r})")
            print(f"{'='*74}\n")
        out["table"] = table
        return out

    def purge_runs(self, run_ids: Sequence[str], confirm: bool = False) -> Dict[str, int]:
        """Delete runs from BOTH repos. Irreversible -- pass confirm=True.

        Intended for clearing artifacts left by an earlier version of the
        pipeline, which otherwise sit alongside real results and make the repo
        hard to read six months from now.
        """
        if not confirm:
            print("Dry run. Would delete from both repos:")
            for r in run_ids:
                print(f"  runs/{r}/  logs/{r}/  per_sample/{r}/")
            print("\nPass confirm=True to actually delete.")
            return {}
        n = {"deleted": 0}
        for r in run_ids:
            for pre in ("runs", "logs", "per_sample"):
                n["deleted"] += self.hub.hub.delete_prefix(f"{pre}/{r}/")
        log(f"deleted {n['deleted']} files", "PURGE")
        return n


def preflight(session: "Session", archs: Optional[Sequence[str]] = None,
              quick: bool = True) -> Dict[str, Any]:
    """Cheap checks that catch the expensive mistakes.

    Runs before any real training. Every item here corresponds to a failure
    that would otherwise be discovered hours in: a ViT whose feature shapes do
    not match the exit heads, a missing HF write scope, a budget table whose
    deepest exit does not equal the full model.
    """
    report: Dict[str, Any] = {"checked_utc": now_iso(), "checks": {}}

    def rec(name, ok, detail=""):
        report["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    print("\nPreflight")
    rec("torch available", _TORCH_OK, torch.__version__ if _TORCH_OK else _TORCH_ERR)
    if _TORCH_OK:
        rec("CUDA available", torch.cuda.is_available(),
            f"{torch.cuda.device_count()} GPU(s): "
            f"{[torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]}"
            if torch.cuda.is_available() else "CPU only -- training will be impractically slow")
    rec("pandas", pd is not None)
    rec("parquet engine", _parquet_ok(), "pyarrow or fastparquet")
    rec("HF token", bool(session.hub.token), "from Kaggle Secrets or env")
    rec("HF repo reachable", session.hub.enabled and session.hub.hub is not None,
        session.hub.repo_id)
    rec("working disk >2 GB", free_mb(session.work) > 2048, f"{free_mb(session.work)} MB")
    rec("scratch disk >5 GB", free_mb(session.scratch) > 5120,
        f"{free_mb(session.scratch)} MB")

    try:
        root = session.prepare_data()
        rec("CIFAR-100 present", _has_cifar100(root), str(root))
    except Exception as e:
        rec("CIFAR-100 present", False, str(e)[:160])

    if _TORCH_OK and archs:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for a in archs:
            try:
                m = build_model(a, 100).to(dev)
                x = torch.randn(4, 3, 32, 32, device=dev)
                out = m(x)
                feats = m.forward_features(x)
                pref = m.forward_prefix(x, 0)
                # An exit head must actually attach, which is where a token
                # model with an unexpected feature rank would blow up.
                head = ExitHead(m.feature_dims[0], 100,
                                getattr(m, "is_token_model", False)).to(dev)
                _ = head(pref)
                loss = out.sum()
                loss.backward()
                K = len(feats)
                rec(f"model {a}", out.shape == (4, 100) and 2 <= K <= len(DEPTH_FRACTIONS),
                    f"{count_parameters(m)/1e6:.2f}M params, K={K}, "
                    f"dims={m.feature_dims}, cuts={m.stage_cuts}")

                # Every resolution the oracle will actually sweep, natively.
                # This is where a ViT's positional embedding or a Mixer's
                # token-mixing weights blow up, and it is far cheaper to find
                # out here than mid-sweep in Phase 1b.
                native = bool(getattr(m, "supports_native_resolution", True))
                if native:
                    bad_r = []
                    for r in RESOLUTIONS:
                        try:
                            m(torch.randn(2, 3, r, r, device=dev))
                        except Exception as e:
                            bad_r.append(f"{r}px:{type(e).__name__}")
                    rec(f"native resolutions {a}", not bad_r,
                        f"runs at {list(RESOLUTIONS)}" if not bad_r
                        else f"FAILS at {bad_r}")
                else:
                    rec(f"native resolutions {a}", True,
                        "not supported by design -- resolution axis uses the "
                        "proxy (documented limitation)")

                if not quick:
                    b = build_budget_table(a, 100, model=m.cpu())
                    d = b["axes"]["depth"]
                    rho = d["rho"]
                    strictly_up = all(rho[i] < rho[i + 1] for i in range(len(rho) - 1))
                    ends_at_one = abs(rho[-1] - 1.0) < 0.02
                    distinct = len(set(round(x, 6) for x in rho)) == len(rho)
                    rec(f"budgets {a}", strictly_up and ends_at_one and distinct,
                        f"K={d['K']} depth rho={[round(x,3) for x in rho]}"
                        + ("" if strictly_up else "  NOT ASCENDING")
                        + ("" if distinct else "  DUPLICATE BUDGETS")
                        + ("" if ends_at_one else "  DOES NOT REACH 1.0"))
                    rr = b["axes"]["resolution"]
                    rec(f"resolution cost {a}",
                        all(rr["rho"][i] < rr["rho"][i + 1]
                            for i in range(len(rr["rho"]) - 1)),
                        f"rho={[round(x,3) for x in rr['rho']]} "
                        f"native={rr['native_supported']}")
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                rec(f"model {a}", False, f"{type(e).__name__}: {str(e)[:140]}")

    try:
        core = _import_msc_core()
        rec("msc_core importable", hasattr(core, "compute_msc"))
    except Exception as e:
        rec("msc_core importable", False, str(e)[:160])

    report["all_passed"] = all(c["ok"] for c in report["checks"].values())
    print(f"\n  {'ALL CHECKS PASSED' if report['all_passed'] else 'FAILURES PRESENT -- fix before training'}\n")
    return report


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def resume_acceptance_test(session: "Session", arch: str = "resnet20",
                           epochs: int = 4, kill_at: int = 2,
                           tol: float = 0.05) -> Dict[str, Any]:
    """Train, genuinely kill, resume, and prove the seam is invisible.

    Two runs of the SAME config:
      reference    trained straight through
      interrupted  killed mid-run by a real KeyboardInterrupt at an epoch
                   boundary, then resumed in a fresh call

    The interruption is a real one. An earlier version of this test simply
    trained a shorter run and then asked for more epochs, which is a *clean
    completion* followed by an *extension* -- a different code path that never
    touches the emergency flush, the paused state, or the resume logic. It also
    got itself blocked by the claim protocol, which correctly refuses to restart
    a completed run. The test passed nothing and proved nothing.

    What passing requires:
      1. the resumed run reaches the full epoch count
      2. no duplicated epoch rows in history.csv
      3. per-epoch training loss AFTER the seam matches the reference

    (3) is the one that matters. It is where a lost RNG state shows up: if the
    augmentation and shuffling sequence diverges on resume, the post-seam losses
    drift away from the reference even though nothing looks broken. A resumed
    run that is not equivalent to an uninterrupted one makes "same architecture,
    same data, different seed" meaningless -- and that comparison is the noise
    ceiling every transfer number in this project is divided by.
    """
    if not _TORCH_OK:
        return {"ok": False, "reason": "torch unavailable"}
    out: Dict[str, Any] = {"arch": arch, "epochs": epochs, "kill_at": kill_at}
    tmp = session.scratch / "resume_test"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp = ensure_dir(tmp)

    cfg = session.config(arch, seed=99, method="resumetest",
                         num_epochs=epochs, phase="test",
                         milestone_push_every_epochs=10 ** 6,
                         cleanup_local_after_complete=False)
    hub_off = MSCHub(enable=False)
    reg = RunRegistry(hub_off, tmp / "reg", account="selftest")

    ref_id = cfg["run_id"] + "-ref"
    cut_id = cfg["run_id"] + "-cut"

    print(f"\n  [1/3] reference: {epochs} epochs, uninterrupted")
    ref = train_backbone(dict(cfg, run_id=ref_id), hub_off, reg,
                         work_root=tmp / "ref", data_root_out=tmp / "ref" / "data",
                         show_progress=False)

    print(f"  [2/3] interrupted: killing for real after epoch {kill_at}")
    part = dict(cfg, run_id=cut_id, _debug_interrupt_after_epoch=kill_at - 1)
    try:
        train_backbone(part, hub_off, reg, work_root=tmp / "cut",
                       data_root_out=tmp / "cut" / "data", show_progress=False)
        out["interrupt_fired"] = False
    except KeyboardInterrupt:
        out["interrupt_fired"] = True

    print(f"  [3/3] resuming in a fresh call, same config")
    res = train_backbone(dict(cfg, run_id=cut_id), hub_off, reg,
                         work_root=tmp / "cut",
                         data_root_out=tmp / "cut" / "data", show_progress=False)
    out["resume_status"] = res.get("status")

    if pd is not None:
        try:
            h_ref = pd.read_csv(run_layout(tmp / "ref", ref_id)["metrics"] / "epochs.csv")
            h_cut = pd.read_csv(run_layout(tmp / "cut", cut_id)["metrics"] / "epochs.csv")
            out["epochs_ref"] = int(len(h_ref))
            out["epochs_cut"] = int(len(h_cut))
            out["duplicate_epochs"] = int(h_cut["epoch"].duplicated().sum())
            out["final_acc_ref"] = float(h_ref["val_accuracy"].iloc[-1])
            out["final_acc_cut"] = float(h_cut["val_accuracy"].iloc[-1])
            out["acc_delta"] = abs(out["final_acc_ref"] - out["final_acc_cut"])

            # The real test: do the post-seam epochs match?
            a = h_ref.set_index("epoch")["train_loss"]
            b = h_cut.set_index("epoch")["train_loss"]
            shared = sorted(set(a.index) & set(b.index) & set(range(kill_at, epochs)))
            devs = [abs(float(a[e]) - float(b[e])) / max(1e-9, abs(float(a[e])))
                    for e in shared]
            out["post_seam_epochs_compared"] = len(shared)
            out["max_post_seam_loss_deviation"] = max(devs) if devs else float("nan")
            print(f"\n  post-seam train_loss, reference vs resumed:")
            for e in shared:
                print(f"    epoch {e}:  {float(a[e]):.5f}  vs  {float(b[e]):.5f}"
                      f"   ({abs(float(a[e])-float(b[e]))/max(1e-9,abs(float(a[e]))):.2%})")
        except Exception as e:
            out["history_error"] = str(e)

    out["ref_run"], out["cut_run"] = ref_id, cut_id
    out["ok"] = bool(out.get("interrupt_fired")
                     and out.get("duplicate_epochs", 1) == 0
                     and out.get("epochs_cut", 0) == epochs
                     and out.get("post_seam_epochs_compared", 0) > 0
                     and out.get("max_post_seam_loss_deviation", 1.0) < tol)

    print(f"\n  {'='*66}")
    print(f"  interrupt actually fired : {out.get('interrupt_fired')}")
    print(f"  epochs  reference={out.get('epochs_ref')}  resumed={out.get('epochs_cut')}"
          f"   (want {epochs})")
    print(f"  duplicated epoch rows    : {out.get('duplicate_epochs')}   (want 0)")
    print(f"  max post-seam loss drift : "
          f"{out.get('max_post_seam_loss_deviation', float('nan')):.4%}"
          f"   (want < {tol:.0%})")
    print(f"  final accuracy           : {out.get('final_acc_ref', float('nan')):.4f}"
          f" vs {out.get('final_acc_cut', float('nan')):.4f}")
    print(f"  RESUME TEST: {'PASS' if out['ok'] else 'FAIL'}")
    print(f"  {'='*66}\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# =============================================================================
# 18. selftest -- offline, no GPU, no network
# =============================================================================
def _selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        d = str(detail)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {d}" if d else ""))

    print("utils")
    tmp = Path(SCRATCH_ROOT) / "msc_selftest"
    shutil.rmtree(tmp, ignore_errors=True)          # a crashed prior run leaves state
    tmp = ensure_dir(tmp)
    atomic_write_json(tmp / "a.json", {"x": 1})
    check("atomic json round trip", read_json(tmp / "a.json") == {"x": 1})
    check("no .tmp left behind", not (tmp / "a.json.tmp").exists())
    h1 = sha256_of_obj({"a": 1, "b": 2})
    h2 = sha256_of_obj({"b": 2, "a": 1})
    check("config hash is key-order invariant", h1 == h2)
    check("array fingerprint is stable",
          sha256_of_array(np.arange(10)) == sha256_of_array(np.arange(10)))
    check("array fingerprint separates orders",
          sha256_of_array(np.arange(10)) != sha256_of_array(np.arange(10)[::-1].copy()))

    print("config")
    c = base_config("resnet32x4", "cifar100", 1, phase="p0")
    check("run_id format", c["run_id"] == "p0-resnet32x4-cifar100-base-s1", c["run_id"])
    c2 = dict(c)
    c2["output_root"] = "/somewhere/else"
    check("hash ignores session-local fields", config_hash(c) == config_hash(c2))
    c3 = dict(c)
    c3["learning_rate"] = 0.1
    check("hash tracks recipe changes", config_hash(c) != config_hash(c3))
    check("phase0 has 4 runs", len(phase0_configs()) == 4)
    check("transformer recipe differs",
          base_config("vit_tiny")["optimizer"] == "adamw"
          and base_config("resnet20")["optimizer"] == "sgd")

    print("rate limiter")
    up = BackgroundUploader("x/y", "selftest-token-A", commits_per_hour_limit=3)
    up._limiter._times = [time.time()] * 3
    check("token bucket sees the window full", up._commits_in_last_hour() == 3)
    up._limiter._times = [time.time() - 4000] * 3
    check("token bucket ages entries out", up._commits_in_last_hour() == 0)

    # The bug this replaced: a per-uploader limiter multiplied the budget by the
    # number of repos, while HF's real limit is per user.
    a = BackgroundUploader("org/repo-a", "shared-tok", commits_per_hour_limit=20)
    b = BackgroundUploader("org/repo-b", "shared-tok", commits_per_hour_limit=20)
    check("two repos on one token share ONE bucket", a._limiter is b._limiter)
    a._limiter._times = []
    for _ in range(7):
        a._limiter.record()
    check("commits by one uploader are seen by the other",
          b._commits_in_last_hour() == 7, f"{b._commits_in_last_hour()}")
    check("shared budget is not multiplied by repo count",
          a._limiter.limit == 20 and b._limiter.limit == 20)
    c = BackgroundUploader("org/repo-c", "different-tok", commits_per_hour_limit=20)
    check("a different token gets its own budget", c._limiter is not a._limiter)
    check("6 accounts x 20 stays under HF's ~128/hr", 6 * 20 <= 128, "120")
    check("parses 'retry after N seconds'",
          abs(up._parse_retry_after("429: retry after 90 seconds") - 92.0) < 1e-6)
    check("parses 'in about N minutes'",
          abs(up._parse_retry_after("rate limited, try in about 5 minutes") - 305.0) < 1e-6)
    check("has a sane default", up._parse_retry_after("429 nothing parseable") == 120.0)

    print("claim protocol")
    hub_off = MSCHub(enable=False)
    reg = RunRegistry(hub_off, tmp / "reg", account="acctA")
    can, why = reg.can_claim("p0-x-cifar100-base-s1")
    check("unclaimed run is claimable", can, why)
    reg.append("p0-x-cifar100-base-s1", "running")
    # A live claim blocks OTHER accounts. It must not block the owner -- that
    # is the resume case, covered below.
    other = RunRegistry(hub_off, tmp / "reg", account="acctB")
    can, why = other.can_claim("p0-x-cifar100-base-s1")
    check("live claim blocks a different account", not can, why)
    check("live claim does NOT block its owner",
          reg.can_claim("p0-x-cifar100-base-s1")[0])
    reg.append("p0-x-cifar100-base-s1", "completed")
    can, why = reg.can_claim("p0-x-cifar100-base-s1")
    check("completed blocks", not can, why)
    check("force overrides", reg.can_claim("p0-x-cifar100-base-s1", force=True)[0])

    print("ledger sharding (the lost-update race)")
    # Reproduces exactly what was observed on the live repo: two workers each
    # recorded a run as 'running', and only one entry survived, because both
    # rewrote the same shared file.
    shutil.rmtree(tmp / "led", ignore_errors=True)
    w0 = RunRegistry(hub_off, tmp / "led", account="acct1", worker_id=0)
    w1 = RunRegistry(hub_off, tmp / "led", account="acct1", worker_id=1)
    check("workers write to different files", w0.shard_path != w1.shard_path,
          f"{w0.shard_path.name} vs {w1.shard_path.name}")
    w0.append("run-A", "running")
    w1.append("run-B", "running")
    seen = set(w0.latest())
    check("BOTH workers' events survive", seen == {"run-A", "run-B"}, str(sorted(seen)))
    check("either worker sees the merged view", set(w1.latest()) == seen)

    w0.append("run-A", "completed", best_accuracy=0.79)
    check("completion is visible to the other worker",
          w1.latest()["run-A"]["state"] == "completed")
    # A late heartbeat from a stale shard must not resurrect a finished run,
    # or it would be trained a second time.
    w1.append("run-A", "running")
    check("'completed' is sticky against a late 'running'",
          w0.latest()["run-A"]["state"] == "completed")

    n_shards = len(list((tmp / "led" / "registry" / "events").glob("*.jsonl")))
    check("one shard per worker", n_shards == 2, f"{n_shards} shards")
    for i in range(2, 8):
        RunRegistry(hub_off, tmp / "led", account="acct1", worker_id=i)\
            .append(f"run-{i}", "running")
    merged = RunRegistry(hub_off, tmp / "led", account="acct1", worker_id=9).latest()
    check("8 workers all coexist", len(merged) == 8, f"{len(merged)} runs visible")

    print("legacy ledger still readable")
    lg = tmp / "led" / "registry" / "runs.jsonl"
    lg.write_text(json.dumps({"run_id": "old-run", "state": "completed",
                              "updated_at": "2020-01-01T00:00:00Z"}) + "\n")
    check("pre-sharding entries are not lost",
          "old-run" in RunRegistry(hub_off, tmp / "led", account="acct1").latest())

    print("resume-own-run (the case that breaks every restart)")
    # A session pauses at the 8.5 h limit; you open a fresh one two minutes
    # later. The ledger still says "paused, 2 minutes ago". If the staleness
    # window is applied without checking WHO owns it, your own run is
    # unresumable for two hours -- which defeats the entire resumability
    # contract. Ownership must be checked before freshness.
    shutil.rmtree(tmp / "reg_own", ignore_errors=True)
    rA = RunRegistry(hub_off, tmp / "reg_own", account="acctA")
    rid = "p1-resnet32x4-cifar100-base-s1"
    rA.append(rid, "running")
    check("same session continues its own run", rA.can_claim(rid)[0],
          rA.can_claim(rid)[1])

    rA2 = RunRegistry(hub_off, tmp / "reg_own", account="acctA")   # new session_id
    can, why = rA2.can_claim(rid)
    check("NEW SESSION, same account, fresh heartbeat -> resumes", can, why)

    rA3 = RunRegistry(hub_off, tmp / "reg_own", account="acctA")
    rA3.append(rid, "paused")
    check("same account can resume its own PAUSED run immediately",
          RunRegistry(hub_off, tmp / "reg_own", account="acctA").can_claim(rid)[0])

    rB = RunRegistry(hub_off, tmp / "reg_own", account="acctB")
    can, why = rB.can_claim(rid)
    check("a DIFFERENT account is still blocked while the claim is fresh",
          not can, why)

    # Age every event for this run by three hours, across all shards.
    for lp in rA._shard_files():
        rowsx = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
        for r_ in rowsx:
            if r_.get("run_id") == rid:
                r_["updated_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3 * 3600))
                r_["ts"] = time.time() - 3 * 3600
        lp.write_text("\n".join(json.dumps(r_) for r_ in rowsx) + "\n")
    can, why = RunRegistry(hub_off, tmp / "reg_own", account="acctB").can_claim(rid)
    check("a different account CAN take over once the claim goes stale", can, why)

    print("config hash ignores run identity and debug hooks")
    cA = base_config("resnet20", "cifar100", 1)
    check("run_id is not part of the hash",
          config_hash(cA) == config_hash(dict(cA, run_id="something-else")))
    check("worker_id is not part of the hash",
          config_hash(cA) == config_hash(dict(cA, worker_id=4)))
    check("the interrupt debug hook is not part of the hash",
          config_hash(cA) == config_hash(dict(cA, _debug_interrupt_after_epoch=2)),
          "otherwise the resumed run would fail its own hash check")

    print("adaptive depth partition")
    # Reimplements StagedBackbone's cut logic so the invariant is checked even
    # without torch. The oracle requires STRICTLY ascending costs; duplicate
    # cuts silently produce duplicate rho, which makes "the smallest sufficient
    # budget" ill-defined and crashes msc_core mid-sweep.
    def _cuts(n, fracs=DEPTH_FRACTIONS):
        cuts, prev = [], 0
        for fr in fracs:
            c = min(n, max(prev + 1, int(round(fr * n))))
            if c > prev:
                cuts.append(c)
                prev = c
            if prev >= n:
                break
        if not cuts or cuts[-1] != n:
            cuts.append(n)
        seen, uniq = set(), []
        for c in cuts:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    bad = []
    for n in range(1, 61):
        c = _cuts(n)
        if not (c == sorted(set(c)) and c[-1] == n and c[0] >= 1
                and len(c) <= len(DEPTH_FRACTIONS) and all(1 <= x <= n for x in c)):
            bad.append((n, c))
    check("cuts strictly ascending, distinct, end at n, for 1..60 blocks",
          not bad, str(bad[:3]))
    check("resnet8x4 (3 blocks) gets K=3, not 5 duplicates",
          _cuts(3) == [1, 2, 3], str(_cuts(3)))
    check("resnet20 (9 blocks) unchanged at K=5", _cuts(9) == [2, 4, 5, 7, 9],
          str(_cuts(9)))
    check("wrn_16_2 (6 blocks) unchanged at K=5", _cuts(6) == [1, 2, 4, 5, 6],
          str(_cuts(6)))
    check("a 1-block net degenerates to K=1 rather than crashing", _cuts(1) == [1])
    check("K never exceeds the number of blocks",
          all(len(_cuts(n)) <= n for n in range(1, 61)))

    print("token-model resolution geometry")
    # A ViT's positional embedding is resampled onto the patch grid the input
    # needs. That only works if the grid stays square and the patch size divides
    # the resolution -- otherwise the interpolation is ill-posed.
    PATCH = 4
    grids = []
    for r in RESOLUTIONS:
        check(f"{r}px divisible by patch {PATCH}", r % PATCH == 0)
        s = r // PATCH
        grids.append(s * s)
        check(f"{r}px -> {s}x{s} grid is a perfect square",
              int(round((s * s) ** 0.5)) ** 2 == s * s, f"{s*s} tokens")
    check("token counts strictly increase with resolution",
          all(grids[i] < grids[i + 1] for i in range(len(grids) - 1)), str(grids))
    check("analytic resolution cost is strictly ascending and ends at 1.0",
          (lambda v: all(v[i] < v[i + 1] for i in range(len(v) - 1))
           and abs(v[-1] - 1.0) < 1e-9)([(r / 32.0) ** 2 for r in RESOLUTIONS]),
          str([round((r / 32.0) ** 2, 3) for r in RESOLUTIONS]))

    print("worker sharding")
    ids = [make_run_id("p1", a, "cifar100", "base", s)
           for a in ZOO for s in (1, 2, 3)]
    for N in (1, 2, 4, 6, 8):
        slices = [[r for r in ids if hash_owner(r, N) == w] for w in range(N)]
        flat = [r for s in slices for r in s]
        check(f"N={N}: no overlap between workers", len(flat) == len(set(flat)))
        check(f"N={N}: no gaps -- every run owned", set(flat) == set(ids))
    check("ownership is deterministic across calls",
          all(hash_owner(r, 6) == hash_owner(r, 6) for r in ids))
    check("ownership does not depend on list order",
          [hash_owner(r, 6) for r in ids] ==
          [hash_owner(r, 6) for r in reversed(ids)][::-1])
    sizes = [sum(1 for r in ids if hash_owner(r, 6) == w) for w in range(6)]
    check("6-way split is reasonably balanced",
          max(sizes) <= 2 * (len(ids) / 6), f"sizes={sizes} of {len(ids)}")
    check("N=1 puts everything on worker 0",
          all(hash_owner(r, 1) == 0 for r in ids))

    print("shard balancing")
    for mode in ("hash", "balanced", "cost"):
        own = assign_workers(ids, 6, mode=mode)
        check(f"{mode}: covers the universe exactly", set(own) == set(ids))
        check(f"{mode}: every owner in range", all(0 <= v < 6 for v in own.values()))
        counts = [sum(1 for v in own.values() if v == w) for w in range(6)]
        hours = [sum(estimate_run_cost(r) for r, v in own.items() if v == w)
                 for w in range(6)]
        imb = max(hours) / max(1e-9, min(hours))
        print(f"        {mode:9s} counts={counts}  imbalance={imb:.2f}x")
        if mode == "balanced":
            check("balanced: counts differ by at most 1",
                  max(counts) - min(counts) <= 1, str(counts))
        if mode == "cost":
            check("cost: wall-clock imbalance under 1.2x", imb < 1.2, f"{imb:.3f}x")
    h_imb = max(hours_h := [sum(estimate_run_cost(r) for r in ids
                                if hash_owner(r, 6) == w) for w in range(6)]) / \
        max(1e-9, min(hours_h))
    c_own = assign_workers(ids, 6, mode="cost")
    c_imb = max(cc := [sum(estimate_run_cost(r) for r, v in c_own.items() if v == w)
                       for w in range(6)]) / max(1e-9, min(cc))
    check("cost mode beats hash mode on balance", c_imb < h_imb,
          f"cost={c_imb:.2f}x vs hash={h_imb:.2f}x")
    check("assignment is stable across calls",
          assign_workers(ids, 6, mode="cost") == assign_workers(ids, 6, mode="cost"))
    check("assignment ignores input order",
          assign_workers(list(reversed(ids)), 6, mode="cost") == c_own)
    check("cost model ranks a ViT above a small ResNet",
          estimate_run_cost("p1-vit_tiny-cifar100-base-s1") >
          estimate_run_cost("p1-resnet20-cifar100-base-s1"))

    print("work planning")
    shutil.rmtree(tmp / "plan", ignore_errors=True)
    hub_p = MSCHub(enable=False)
    regp = RunRegistry(hub_p, tmp / "plan", account="w0")
    universe = [f"p1-arch{i}-cifar100-base-s1" for i in range(24)]
    plans = [plan_work(universe, regp, worker_id=w, num_workers=4) for w in range(4)]
    p0, p1 = plans[0], plans[1]
    check("disjoint slices", not (set(p0.mine) & set(p1.mine)))
    allmine = [r for p in plans for r in p.mine]
    check("all four slices together cover the universe exactly",
          sorted(allmine) == sorted(universe) and len(allmine) == len(set(allmine)))
    check("nothing done yet -> todo == mine", p0.todo == p0.mine)
    first = p0.mine[0]
    regp.append(first, "completed")
    p0b = plan_work(universe, regp, worker_id=0, num_workers=4)
    check("completed run drops out of todo", first not in p0b.todo)
    check("but stays in the owned slice", first in p0b.mine)
    # a live claim by another worker must NOT be stolen
    other = p1.mine[0]
    regp.append(other, "running")
    p0c = plan_work(universe, regp, worker_id=0, num_workers=4, steal_stale=True)
    check("live run on another worker is not stolen", other not in p0c.stolen)
    check("it is reported as busy elsewhere", other in p0c.in_progress_elsewhere)
    # forge a stale heartbeat -> now it should be stealable
    for lp in regp._shard_files():
        rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
        for r in rows:
            if r.get("run_id") == other:
                r["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(time.time() - 3 * 3600))
                r["ts"] = time.time() - 3 * 3600
        lp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    p0d = plan_work(universe, regp, worker_id=0, num_workers=4, steal_stale=True)
    check("stale run on a dead worker IS stolen", other in p0d.stolen)
    check("own work still comes first in the queue",
          p0d.work[:len(p0d.todo)] == p0d.todo)

    print("schema vs requirement 15.1")
    H = set(HISTORY_FIELDS)
    # Every row of the per-epoch requirement table, mapped to the column(s)
    # that satisfy it. A missing entry here is a missing requirement.
    REQ_151 = {
        "epoch number": ["epoch"],
        "training loss": ["train_loss"],
        "validation loss": ["val_loss"],
        "training accuracy": ["train_accuracy"],
        "validation accuracy": ["val_accuracy"],
        "f1 score": ["f1_macro", "f1_micro", "f1_weighted"],
        "precision": ["precision_macro", "precision_micro", "precision_weighted"],
        "recall": ["recall_macro", "recall_micro", "recall_weighted"],
        "learning rate": ["learning_rate", "lr_min_group", "lr_max_group"],
        "training time": ["train_time_sec"],
        "validation time": ["val_time_sec"],
        "gpu memory usage": ["peak_vram_mb", "vram_allocated_mb", "gpu0_mem_used_mb"],
        "gpu utilization (per gpu)": ["gpu0_util_mean_pct", "gpu1_util_mean_pct"],
        "energy consumed": ["epoch_energy_j", "epoch_energy_kwh",
                            "cumulative_energy_kwh"],
        "carbon emission": ["epoch_co2_g", "epoch_co2_kg", "cumulative_co2_kg"],
        "temperature": ["gpu0_temp_mean_c", "gpu0_temp_max_c", "gpu1_temp_max_c"],
        "kd loss": ["loss_kd"],
        "feature loss": ["loss_feature"],
        "attention loss": ["loss_attention"],
        "energy-boundary loss": ["loss_energy_boundary"],
        "counterfactual loss": ["loss_counterfactual"],
        "pareto loss": ["loss_pareto"],
    }
    missing = {k: [c for c in v if c not in H] for k, v in REQ_151.items()}
    missing = {k: v for k, v in missing.items() if v}
    check("every 15.1 requirement has a column", not missing, str(missing))
    check("per-GPU columns exist for both T4s",
          all(f"gpu{i}_{k}" in H for i in range(2)
              for k in ("util_mean_pct", "temp_max_c", "mem_used_mb", "energy_j")))
    check("deleted loss terms have columns, to be filled NA",
          all(f"loss_{t}" in H for t in OPTIONAL_LOSS_TERMS))
    check("no duplicate columns", len(HISTORY_FIELDS) == len(H),
          f"{len(HISTORY_FIELDS)} columns")
    check("schema is comfortably wider than the spec", len(H) > 150, f"{len(H)}")

    print("schema vs requirement 15.2")
    Fset = set(FINAL_FIELDS)
    REQ_152 = {
        "top-1 accuracy": ["top1_accuracy"],
        "top-5 accuracy": ["top5_accuracy"],
        "f1 score": ["f1_macro", "f1_micro", "f1_weighted"],
        "precision": ["precision_macro", "precision_micro", "precision_weighted"],
        "recall": ["recall_macro", "recall_micro", "recall_weighted"],
        "confusion matrix": ["worst_class_f1"],       # file: confusion_matrix.csv
        "parameter count": ["params_total", "params_trainable", "params_nonzero"],
        "flops / macs": ["flops", "macs", "flops_per_param"],
        "model size": ["model_size_mb", "model_size_mb_fp16", "model_size_mb_int8"],
        "inference latency": ["latency_bs1_median_ms", "latency_bs1_p99_ms"],
        "throughput": ["throughput_bs1_img_s", "throughput_bs32_img_s"],
        "training energy": ["train_energy_j", "train_energy_kwh"],
        "inference energy": ["inference_energy_j_per_image"],
        "carbon emission": ["train_co2_kg", "inference_co2_g_per_1k_images"],
        "energy reduction": ["energy_reduction_pct"],
        "accuracy change": ["accuracy_change_pts"],
        "compression ratio": ["compression_ratio"],
    }
    miss2 = {k: [c for c in v if c not in Fset] for k, v in REQ_152.items()}
    miss2 = {k: v for k, v in miss2.items() if v}
    check("every 15.2 requirement has a column", not miss2, str(miss2))
    check("comparatives record what they were measured against",
          "baseline_run_id" in Fset,
          "a compression ratio with no stated reference is uninterpretable")
    check("final schema has no duplicates", len(FINAL_FIELDS) == len(Fset),
          f"{len(FINAL_FIELDS)} columns")
    check("calibration reported at final eval too",
          {"ece", "mce", "nll", "brier"} <= Fset)

    print("model statistics")
    if _TORCH_OK:
        m_ = build_model("resnet20", 100)
        st_ = model_statistics(m_, flops=123456789)
        check("counts parameters", st_["params_total"] > 0,
              f"{st_['params_total']/1e6:.2f}M")
        check("sparsity is 0% for a dense model", st_["sparsity_pct"] < 1e-6)
        check("size drops with precision",
              st_["model_size_mb"] > st_["model_size_mb_fp16"] >
              st_["model_size_mb_int8"])
        check("macs is half of flops", st_["macs"] == 123456789 // 2)
        check("layer census non-empty", st_["n_conv_layers"] > 0)
    else:
        print("  [SKIP] torch unavailable")

    print("calibration")
    rng2 = np.random.default_rng(0)
    n_c, C = 2000, 10
    lbl = rng2.integers(0, C, n_c)
    # A perfectly calibrated one-hot predictor: confidence 1.0, accuracy 1.0.
    perfect = np.zeros((n_c, C)); perfect[np.arange(n_c), lbl] = 1.0
    cm = calibration_metrics(np.clip(perfect, 1e-9, 1.0), lbl)
    check("perfect predictor has ~zero ECE", cm["ece"] < 0.02, f"{cm['ece']:.4f}")
    check("perfect predictor has ~zero Brier", cm["brier"] < 0.02, f"{cm['brier']:.4f}")
    # Confidently wrong: max probability on a class that is never right.
    wrong = np.zeros((n_c, C)); wrong[np.arange(n_c), (lbl + 1) % C] = 1.0
    cw = calibration_metrics(np.clip(wrong, 1e-9, 1.0), lbl)
    check("confidently-wrong predictor has ECE near 1", cw["ece"] > 0.9,
          f"{cw['ece']:.4f}")
    check("overconfidence gap is positive when overconfident",
          cw["overconfidence_gap"] > 0.9, f"{cw['overconfidence_gap']:.3f}")
    check("reliability bins are returned", len(cm["bins"]) == 15)

    print("run identity comes from the run_id, not the ledger")
    m = parse_run_id("p1-resnet32x4-cifar100-base-s3")
    check("parses phase/arch/dataset/method/seed",
          (m["phase"], m["arch"], m["dataset"], m["method"], m["seed"])
          == ("p1", "resnet32x4", "cifar100", "base", 3), str(m))
    check("resolves family from the zoo", m["family"] == "resnet")
    m2 = parse_run_id("p3-resnet8x4-cifar100-mscKD-from-resnet32x4-s2")
    check("handles a hyphenated method",
          m2["arch"] == "resnet8x4" and m2["seed"] == 2
          and m2["method"] == "mscKD-from-resnet32x4", str(m2))
    check("malformed id returns None rather than raising",
          parse_run_id("nonsense")["arch"] is None)

    # Reproduces D-13 exactly: repair_ledger writes a completion knowing only
    # the run_id, so the event has no arch/seed. Reading them from the ledger
    # gives None and int(None) raises.
    ev = {"run_id": "p1-resnet8x4-cifar100-base-s1", "state": "completed",
          "best_accuracy": 0.7335, "repaired": True}
    check("a repaired event genuinely lacks arch/seed",
          ev.get("arch") is None and ev.get("seed") is None)
    merged = run_meta(ev["run_id"], ev)
    check("run_meta fills them from the id",
          merged["arch"] == "resnet8x4" and merged["seed"] == 1)
    check("and keeps the ledger's own fields",
          merged["best_accuracy"] == 0.7335 and merged["repaired"] is True)
    check("int(seed) now works", int(merged["seed"]) == 1)
    rich = {"run_id": "p1-resnet20-cifar100-base-s2", "arch": "resnet20",
            "seed": 2, "state": "completed"}
    check("id and ledger agree when both are present",
          run_meta(rich["run_id"], rich)["arch"] == "resnet20")

    print("assignment stability (the guarantee the whole design rests on)")
    # Reproduces defect D-12. Ownership must not depend on how much of the
    # project has already finished, or two sessions of the same worker disagree
    # about what they own -- abandoning one run and duplicating another.
    ids15 = [make_run_id("p1", a, "cifar100", "base", sd)
             for a in ("resnet20", "resnet56", "resnet110", "resnet8x4", "resnet32x4")
             for sd in (1, 2, 3)]
    base_assign = assign_workers(ids15, 4, mode="cost")

    # A "self-correcting" cost table, as it would look part-way through a phase.
    measured_like = {**ARCH_COST_HINT, "resnet20": 0.9, "resnet56": 2.1,
                     "resnet110": 4.9, "resnet8x4": 1.4}
    drifted = assign_workers(ids15, 4, mode="cost", costs=measured_like)
    check("measured costs WOULD change ownership (why it must not be used)",
          drifted != base_assign,
          f"{sum(1 for k in base_assign if drifted[k] != base_assign[k])}"
          f"/{len(ids15)} runs would move")

    shutil.rmtree(tmp / "stable", ignore_errors=True)
    hub_st = MSCHub(enable=False)
    reg_st = RunRegistry(hub_st, tmp / "stable", account="a", worker_id=3)
    p_early = plan_work(ids15, reg_st, 3, 4, stage="train")
    for r in ids15[:12]:
        reg_st.append(r, "completed", best_accuracy=0.75)
    p_late = plan_work(ids15, reg_st, 3, 4, stage="train")
    check("a worker's SLICE is identical before and after 12 runs finish",
          p_early.mine == p_late.mine, f"{p_early.mine} vs {p_late.mine}")
    check("only the todo list shrinks", set(p_late.todo) < set(p_early.todo)
          or p_late.todo == p_early.todo)

    all_owned = [r for w in range(4)
                 for r in plan_work(ids15, reg_st, w, 4, stage="train").mine]
    check("all four slices still partition the universe exactly",
          sorted(all_owned) == sorted(ids15) and len(all_owned) == len(set(all_owned)))
    check("assignment is stable across a fresh registry",
          plan_work(ids15, RunRegistry(hub_st, tmp / "stable2", account="b",
                                       worker_id=3), 3, 4, stage="train").mine
          == p_early.mine)

    print("stage-aware completion")
    # Reproduces the live failure: four runs finished TRAINING, so the ledger
    # says 'completed'. The MEASUREMENT stage then planned zero work and exited
    # in 30 seconds looking like a success.
    shutil.rmtree(tmp / "stage", ignore_errors=True)
    hub_s = MSCHub(enable=False)
    regs = RunRegistry(hub_s, tmp / "stage", account="acct1", worker_id=0)
    runs4 = [f"p0-{a}-cifar100-base-s{sd}"
             for a in ("resnet32x4", "wrn_40_2") for sd in (1, 2)]
    for r in runs4:
        regs.append(r, "completed", best_accuracy=0.79)

    p_train = plan_work(runs4, regs, 0, 1, stage="train")
    check("training stage sees its work as finished", p_train.todo == [],
          "correct -- training really is done")

    measured_none = lambda r: False        # no per-sample tables written yet
    p_meas = plan_work(runs4, regs, 0, 1, done_fn=measured_none, stage="measure")
    check("MEASUREMENT stage still has all 4 runs to do",
          sorted(p_meas.todo) == sorted(runs4),
          f"{len(p_meas.todo)} planned (was 0 before the fix)")
    check("plan records which stage it is for", p_meas.stage == "measure")

    measured_two = lambda r: r in runs4[:2]
    p_part = plan_work(runs4, regs, 0, 1, done_fn=measured_two, stage="measure")
    check("partially measured -> only the remainder is planned",
          sorted(p_part.todo) == sorted(runs4[2:]), str(p_part.todo))

    p_all = plan_work(runs4, regs, 0, 1, done_fn=lambda r: True, stage="measure")
    check("fully measured -> nothing planned", p_all.todo == [])
    check("done set reflects the stage predicate, not ledger state",
          len(p_meas.done) == 0 and len(p_all.done) == 4)

    print("epoch telemetry")
    t = EpochTelemetry()
    for i in range(50):
        t.add_batch(1.0 / (i + 1), 0.10, 0.02, 0.08)
        if i % 2 == 0:
            t.add_step(float(i), clipped=(i > 40))
    t.add_batch(float("nan"), 0.1, 0.02, 0.08)
    s = t.summary()
    check("counts batches and steps", s["n_batches"] == 51 and s["n_optimizer_steps"] == 25)
    check("detects NaN losses", s["nan_or_inf_batches"] == 1)
    check("dataload fraction computed", abs(s["dataload_frac"] - 0.2) < 0.01,
          f"{s['dataload_frac']:.3f}")
    check("step-time percentiles present",
          all(np.isfinite(s[k]) for k in ("step_time_p50_ms", "step_time_p90_ms",
                                          "step_time_p99_ms")))
    check("clip-hit fraction computed", 0 < s["grad_clip_hit_frac"] < 1,
          f"{s['grad_clip_hit_frac']:.3f}")
    check("step trace is downsampled", len(t.step_trace(max_points=10)["step"]) <= 10)
    check("every history field is produced by summary+aggregate+row",
          set(s) <= set(HISTORY_FIELDS), f"extra={sorted(set(s)-set(HISTORY_FIELDS))}")
    check("system aggregate keys are history fields",
          set(SystemMonitor.aggregate([])) <= set(HISTORY_FIELDS))

    print("training dynamics")
    if _TORCH_OK:
        dyn = TrainingDynamics(6, el2n_epoch=0)
        idx = torch.arange(6)
        lab = torch.zeros(6, dtype=torch.long)
        right = torch.tensor([[9.0, 0.0]] * 6)
        wrong = torch.tensor([[0.0, 9.0]] * 6)
        dyn.observe_batch(idx, right, lab, 0); dyn.end_epoch()
        dyn.observe_batch(idx, wrong, lab, 1); dyn.end_epoch()
        dyn.observe_batch(idx, right, lab, 2); dyn.end_epoch()
        check("counts one forgetting event", int(dyn.forget_events[0]) == 1,
              f"events={dyn.forget_events[:3]}")
        check("EL2N captured at the designated epoch", np.isfinite(dyn.el2n[0]))
        check("ever_correct set", bool(dyn.ever_correct[0]))
        d2 = TrainingDynamics(6, el2n_epoch=0)
        d2.load_state_dict(dyn.state_dict())
        check("dynamics survive a checkpoint round trip",
              int(d2.forget_events[0]) == 1 and d2.epochs_recorded == 3)
    else:
        print("  [SKIP] torch unavailable")

    print("sufficiency targets")
    rho = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    st = sufficiency_targets(np.array([0.6, 0.2, 1.0]), rho)
    check("targets are monotone in k", bool(np.all(np.diff(st, axis=1) >= 0)))
    check("threshold is correct", list(st[0]) == [0, 0, 1, 1, 1], st[0])
    check("MSC=1 gives only the last budget", list(st[2]) == [0, 0, 0, 0, 1])

    print("routing and matched FLOPs")
    t1 = np.array([[0.3, 0.5, 0.95], [0.99, 0.99, 0.99], [0.1, 0.1, 0.2]])
    r = confidence_route(t1, 0.9)
    check("confidence routing picks the first clearing budget",
          list(r) == [2, 0, 2], list(r))
    check("expected FLOPs averages rho",
          abs(expected_flops(np.array([0, 2]), [0.5, 0.75, 1.0], 100) - 75.0) < 1e-9)
    if pd is not None:
        correct_at = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
        curve = sweep_operating_points(t1, correct_at, [0.4, 0.7, 1.0], 1e9)
        check("operating curve is non-empty", len(curve) > 0)
        check("matched-FLOPs interpolation is in range",
              0.0 <= accuracy_at_matched_flops(curve, 0.8e9) <= 1.0)

    print("learn-then-test")
    _need = ltt_min_calibration_n(0.01, 0.05)
    check("min-n formula matches the Hoeffding bound",
          _need == int(math.ceil(math.log(20.0) / (2 * 0.01 ** 2))),
          f"n>={_need} at eps=0.01, delta=0.05")
    check("CIFAR-100 test set cannot certify eps=0.01",
          ltt_min_calibration_n(0.01, 0.05) > 10000,
          "documented in the runbook -- use eps>=0.03 or calibrate on train_holdout")
    n = 5000
    rng = np.random.default_rng(0)
    suff = np.sort(rng.uniform(0, 1, (n, 4)), axis=1)
    eps = 0.05                                  # powered: slack ~0.017 < 0.05
    corr = np.ones((n, 4), dtype=float)
    g = learn_then_test_threshold(suff, corr, full_accuracy=1.0, epsilon=eps)
    check("zero-risk case reaches the aggressive end of the grid", g <= 0.06,
          f"gamma={g:.3f}")
    corr_bad = np.zeros((n, 4)); corr_bad[:, -1] = 1.0
    g2 = learn_then_test_threshold(suff, corr_bad, full_accuracy=1.0, epsilon=eps)
    check("high-risk case stays conservative", g2 > g, f"gamma={g2:.3f} vs {g:.3f}")
    g3 = learn_then_test_threshold(suff, corr, full_accuracy=1.0, epsilon=0.001,
                                   warn_underpowered=False)
    check("underpowered case falls back to the safest gamma",
          abs(g3 - 0.99) < 1e-9, f"gamma={g3:.3f}")

    print("shuffled control")
    m = np.linspace(0, 1, 500)
    sh = shuffle_msc_targets(m, seed=0)
    check("shuffle preserves the multiset", np.allclose(np.sort(sh), np.sort(m)))
    check("shuffle actually permutes", not np.allclose(sh, m))

    # --- D-28: the router lives on the STUDENT's budget grid -----------------
    # A resnet8x4 student has 3 adaptive depth exits; a resnet32x4 teacher has
    # 5 budgets. Sizing the sufficiency head from the teacher produced a
    # 5-column router on a 3-exit model, which only failed at evaluation.
    def _shapes_ok(n_heads, n_suff, n_rho):
        return n_heads == n_suff == n_rho

    check("D-28: matched shapes are accepted", _shapes_ok(3, 3, 3))
    check("D-28: teacher-sized head on a student backbone is rejected",
          not _shapes_ok(3, 5, 5), "the exact resnet8x4-from-resnet32x4 case")
    check("D-28: a budget table of the wrong width is rejected",
          not _shapes_ok(5, 5, 3))
    # sufficiency_targets must project a scalar MSC onto WHATEVER grid it is
    # given -- that is what makes routing on the student's grid correct.
    _r3, _r5 = [0.33, 0.67, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0]
    _m = np.array([0.5])
    check("D-28: targets follow the grid they are given (3)",
          sufficiency_targets(_m, _r3).shape == (1, 3))
    check("D-28: targets follow the grid they are given (5)",
          sufficiency_targets(_m, _r5).shape == (1, 5))
    check("D-28: and stay monotone on both grids",
          bool((np.diff(sufficiency_targets(_m, _r5)[0]) >= 0).all()))

    # --- D-26: summary.json outranks epochs.csv ------------------------------
    # epochs.csv is telemetry pushed on a 30-min timer; summary.json is written
    # AFTER the loop exits. A session ending between the two leaves a short
    # history for a run that genuinely finished -- which demoted five completed
    # atlas runs ("resnet110-s1 at only 161 epochs") that have 240/240
    # summaries and best checkpoints on HF.
    def _verdict2(summ, last_ep):
        planned = int(summ.get("num_epochs_planned", 0) or 0)
        claimed = int(summ.get("num_epochs_run", 0) or 0)
        target = planned or claimed
        ok = summ.get("status") == "completed"
        if ok and target > 0 and claimed >= 0.9 * target:
            return True
        return ok and target > 0 and (last_ep + 1) >= 0.9 * target

    _c240 = {"status": "completed", "num_epochs_planned": 240,
             "num_epochs_run": 240}
    check("D-26: a 240/240 summary survives a truncated history",
          _verdict2(_c240, 160), "the exact resnet110-s1 case")
    check("D-26: and survives an empty history",
          _verdict2(_c240, -1))
    check("D-26: a summary that admits a short run is still demoted",
          not _verdict2({"status": "completed", "num_epochs_planned": 240,
                         "num_epochs_run": 40}, 39),
          "the genuine broken stub must still be caught")
    check("D-26: history can still rescue a summary with no counts",
          _verdict2({"status": "completed", "num_epochs_run": 240}, 239))

    # --- D-24: repair_ledger must not demote on a MISSING field --------------
    # train_msc_kd's summary has no `num_epochs_planned`, so `planned` was 0,
    # `planned > 0` was False, and every COMPLETE MSC-KD run was demoted to
    # 'paused' on every sync -- logged as "marked completed at only 240
    # epochs", 240 being exactly the number it was meant to reach.
    def _verdict(summ, last_ep):
        planned = int(summ.get("num_epochs_planned", 0) or 0)
        claimed = int(summ.get("num_epochs_run", 0) or 0)
        target = planned or claimed
        ok = summ.get("status") == "completed"
        return (ok and target > 0 and (last_ep + 1) >= 0.9 * target), target

    _full = {"status": "completed", "num_epochs_run": 240}
    check("D-24: a complete run with no `num_epochs_planned` is NOT demoted",
          _verdict(_full, 239)[0], "the exact MSC-KD case")
    check("D-24: `num_epochs_planned` is still preferred when present",
          _verdict({**_full, "num_epochs_planned": 240}, 239)[0])
    check("D-24: a genuine stub is still caught (50 of 240 planned)",
          not _verdict({"status": "completed", "num_epochs_planned": 240,
                        "num_epochs_run": 240}, 49)[0],
          "the stub check must not be weakened by the fix")
    check("D-24: a stub is caught via the claimed count too",
          not _verdict({"status": "completed", "num_epochs_run": 240}, 49)[0])
    check("D-24: no epoch count at all -> refuse to judge, do not demote",
          _verdict({"status": "completed"}, 239)[1] == 0,
          "absent evidence is not evidence of a short run")
    check("D-24: a run whose summary does not say completed is not 'done'",
          not _verdict({"status": "paused", "num_epochs_run": 120}, 119)[0])

    # --- D-23: writer and readers must agree on the exit-heads path ---------
    # run_oracle writes to the run ROOT; train_msc_kd read `checkpoints/`. The
    # teacher's heads were never found, so all nine MSC-KD runs retrained them
    # (~20 epochs each) from a file already on HuggingFace. D-16 called this
    # "cosmetic, nothing reads the path by convention" -- three things did.
    _ehw = Path(tmp) / "eh"
    _er = "p1-resnet32x4-cifar100-base-s1"
    _eL = run_layout(_ehw, _er)
    for _s in RUN_SUBDIRS:
        ensure_dir(_eL[_s])
    check("D-23: nothing found when nothing is written",
          find_exit_heads(_ehw, _er) is None)
    _canon = exit_heads_path(_ehw, _er)
    check("D-23: the canonical path is the run root, not checkpoints/",
          _canon.parent == _eL["base"], str(_canon.relative_to(_ehw)))
    _canon.write_bytes(b"heads")
    check("D-23: the writer's path is what the reader finds",
          find_exit_heads(_ehw, _er) == _canon)
    _canon.unlink()
    (_eL["checkpoints"] / "exit_heads.pt").write_bytes(b"legacy")
    check("D-23: the legacy checkpoints/ location is still honoured",
          find_exit_heads(_ehw, _er) == _eL["checkpoints"] / "exit_heads.pt",
          "runs written before this fix must not retrain")
    _canon.write_bytes(b"heads")
    check("D-23: canonical wins when both exist",
          find_exit_heads(_ehw, _er) == _canon)

    # --- D-22: the MSC-KD history row must match HISTORY_FIELDS -------------
    # The old row used f1_score / precision / recall / grad_norm /
    # throughput_img_s. None of those are column names. csv.DictWriter raises
    # at the END of the first epoch, so the only way to find out was an hour of
    # real training on a real teacher. This does it in microseconds.
    _row = msckd_history_row(
        run_id="p3-resnet8x4-cifar100-mscKDshuffromresnet32x4-s1",
        cfg={"arch": "resnet8x4", "family": "resnet", "dataset": "cifar100",
             "seed": 1, "phase": "p3", "method": "mscKDshuf-from-resnet32x4",
             "config_hash": "deadbeef", "batch_size": 64},
        epoch=3, agg={"loss": 8.0, "ce": 4.0, "kd": 2.0, "msc": 2.0}, nb=4,
        val={"loss": 1.5, "accuracy_top5": 0.9, "f1": 0.7, "precision": 0.71,
             "recall": 0.69},
        acc=0.72, best_before=0.70, lr=0.05, amp=True, dt=30.0,
        cum_time=120.0, cum_energy=1000.0, n_train_images=50000,
        alpha=1.0, beta=1.0, temperature=4.0)
    _bad = sorted(k for k in _row if k not in _HISTORY_SET)
    check("D-22: every MSC-KD history column is in HISTORY_FIELDS",
          not _bad, f"offenders: {_bad}" if _bad else f"{len(_row)} columns")
    for _old in ("f1_score", "precision", "recall", "grad_norm",
                 "throughput_img_s"):
        check(f"D-22: the invalid name '{_old}' is gone", _old not in _row)
    check("D-22: the three-term loss decomposition is now recorded",
          all(k in _row for k in ("loss_ce", "loss_kd", "loss_msc",
                                  "alpha", "beta", "temperature")),
          "it was computed every epoch and thrown away")
    check("D-22: and the components sum to the total",
          abs((_row["loss_ce"] + _row["loss_kd"] + _row["loss_msc"])
              - _row["loss_total"]) < 1e-9)
    check("D-22: is_best compares against the PREVIOUS best, not the new one",
          _row["is_best"] is True and _row["best_val_accuracy_so_far"] == 0.72)

    _hp = Path(tmp) / "epochs.csv"
    append_history_row(_hp, _row, strict=True)
    append_history_row(_hp, _row, strict=True)
    _lines = _hp.read_text(encoding="utf-8").strip().split("\n")
    check("D-22: writes a header once, then one line per epoch",
          len(_lines) == 3 and _lines[0].startswith("run_id,epoch,"),
          f"{len(_lines)} lines")
    try:
        append_history_row(_hp, {**_row, "f1_score": 0.7}, strict=True)
        check("D-22: strict mode rejects an unknown column", False, "no raise")
    except KeyError as _e:
        check("D-22: strict mode rejects an unknown column and suggests a fix",
              "f1_macro" in str(_e), str(_e)[:70])
    _before = _hp.read_text(encoding="utf-8")
    append_history_row(_hp, {**_row, "gpu0_weird_vendor_metric": 1.0},
                       strict=False)
    check("D-22: non-strict mode still writes, dropping the unknown column",
          len(_hp.read_text(encoding="utf-8")) > len(_before),
          "train_backbone merges machine-dependent GPU dicts")

    # --- D-20: "safe" is not "finished" -------------------------------------
    # A paused run whose ckpt_last.pt is on HF loses NOTHING when the tab is
    # closed. Classifying it as at-risk was a false alarm, and a verification
    # cell that cries wolf is the D-17 failure mode all over again.
    def _classify(have, rid):
        if f"runs/{rid}/summary.json" in have:
            return "done"
        if f"runs/{rid}/checkpoints/ckpt_last.pt" in have:
            return "resumable"
        return "at_risk"

    _r = "p3-resnet8x4-cifar100-mscKDshuffromresnet32x4-s1"
    check("D-20: summary.json -> finished",
          _classify({f"runs/{_r}/summary.json"}, _r) == "done")
    check("D-20: checkpoint only -> RESUMABLE, not at risk",
          _classify({f"runs/{_r}/checkpoints/ckpt_last.pt"}, _r) == "resumable",
          "this is the case that produced the false alarm")
    check("D-20: neither -> at risk",
          _classify({f"runs/{_r}/config.yaml"}, _r) == "at_risk")
    check("D-20: a config.yaml alone is NOT reassurance",
          _classify({f"runs/{_r}/config.yaml", f"runs/{_r}/STATUS.json"}, _r)
          == "at_risk",
          "status files are written before any real work exists")

    # The hyphen-stripping in make_run_id is what produces these ids; assert it
    # round-trips, because the D-20 report prints them and they look wrong.
    _mk = make_run_id("p3", "resnet8x4", "cifar100",
                      "mscKDshuf-from-resnet32x4", 1)
    check("D-20: method hyphens are stripped, deterministically",
          _mk == "p3-resnet8x4-cifar100-mscKDshuffromresnet32x4-s1", _mk)
    check("D-20: and the id still parses into exactly its 5 fields",
          parse_run_id(_mk)["arch"] == "resnet8x4"
          and parse_run_id(_mk)["seed"] == 1,
          "stripping is what keeps the '-' split unambiguous")

    # --- D-19: artifact-based completion, not ledger-only -------------------
    import tempfile as _tf
    _w = Path(_tf.mkdtemp(prefix="msc_d19_"))
    _rid = "p3-resnet8x4-cifar100-mscKD-from-resnet32x4-s1"
    _cfg = {"run_id": _rid, "num_epochs": 240}
    _L = run_layout(_w, _rid)
    for _s in RUN_SUBDIRS:
        ensure_dir(_L[_s])
    ensure_dir(_L["base"])

    check("D-19: no artifacts -> not finished",
          already_finished(None, _w, _rid, _cfg) is None)
    check("D-19: no local checkpoint is reported honestly",
          ensure_run_local(None, _w, _rid) is False)

    atomic_write_json(_L["base"] / "summary.json",
                      {"run_id": _rid, "num_epochs_run": 79,
                       "best_accuracy": 0.6447})
    check("D-19: a PARTIAL run is not treated as finished",
          already_finished(None, _w, _rid, _cfg) is None,
          "79/240 epochs must still be resumable, not skipped")

    atomic_write_json(_L["base"] / "summary.json",
                      {"run_id": _rid, "num_epochs_run": 240,
                       "best_accuracy": 0.7412})
    _hit = already_finished(None, _w, _rid, _cfg)
    check("D-19: a finished run is detected from summary.json alone",
          isinstance(_hit, dict) and _hit.get("status") == "cached",
          "this is what stops a lost ledger event costing 30 GPU-hours")
    check("D-19: and it carries the original metrics forward",
          _hit.get("best_accuracy") == 0.7412)
    check("D-19: force_rerun overrides the guard",
          already_finished(None, _w, _rid, {**_cfg, "force_rerun": True}) is None)
    check("D-19: a corrupt summary.json does not crash the guard",
          (_L["base"] / "summary.json").write_text("{not json", encoding="utf-8")
          is not None and already_finished(None, _w, _rid, _cfg) is None)

    (_L["checkpoints"] / "ckpt_last.pt").write_bytes(b"x")
    check("D-19: a present checkpoint short-circuits the pull",
          ensure_run_local(None, _w, _rid) is True)
    shutil.rmtree(_w, ignore_errors=True)

    # --- D-18: representative run selection ---------------------------------
    _runs = {"p1-vgg8-cifar100-base-s2": {"arch": "vgg8", "seed": 2},
             "p1-vgg8-cifar100-base-s3": {"arch": "vgg8", "seed": 3},
             "p1-resnet20-cifar100-base-s1": {"arch": "resnet20", "seed": 1},
             "p1-resnet20-cifar100-base-s2": {"arch": "resnet20", "seed": 2},
             "p1-wrn_16_2-cifar100-base-s2": {"arch": "wrn_16_2", "seed": 2}}
    _ceil = {"p1-vgg8-cifar100-base-s2", "p1-vgg8-cifar100-base-s3",
             "p1-resnet20-cifar100-base-s1", "p1-resnet20-cifar100-base-s2"}
    rep = representative_runs(_runs, require=_ceil)
    check("D-18: vgg8 is represented even with no seed 1",
          rep.get("vgg8") == "p1-vgg8-cifar100-base-s2", str(rep.get("vgg8")))
    check("D-18: the old seed==1 idiom would have dropped it",
          not [r for r, m in _runs.items() if m["arch"] == "vgg8" and m["seed"] == 1])
    check("D-18: lowest seed wins when several qualify",
          rep.get("resnet20") == "p1-resnet20-cifar100-base-s1")
    check("D-18: `require` excludes unmeasured architectures",
          "wrn_16_2" not in rep, str(sorted(rep)))
    check("D-18: without `require`, nothing is excluded",
          "wrn_16_2" in representative_runs(_runs))

    _pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"),
              ("b", "c"), ("b", "d"), ("x", "y")]
    _kinds = {("a", "b"): "K1", ("a", "c"): "K1", ("a", "d"): "K1",
              ("a", "e"): "K1", ("b", "c"): "K2", ("b", "d"): "K2",
              ("x", "y"): "K3"}
    strat = stratified_pairs(_pairs, lambda p: _kinds[p], per_kind=2)
    check("D-18: stratified sampling caps each kind",
          sum(1 for p in strat if _kinds[p] == "K1") == 2, str(strat))
    check("D-18: and reaches kinds the alphabetical head would miss",
          {"K1", "K2", "K3"} == {_kinds[p] for p in strat})
    check("D-18: plain truncation would have missed them",
          {_kinds[p] for p in _pairs[:4]} == {"K1"},
          "pairs[:4] is entirely one kind -- the real bug")

    # --- D-17 regression: the verdict rule that used to cry wolf -------------
    # The exact case that failed NB11: convnext_femto x resnet20, raw rho of
    # -0.0341 at n=5872. That is 2.6 sigma -- a 1-in-113 draw, seen once across
    # 78 pairs, which is precisely what "expected" looks like.
    ok, z, sd = shuffled_control_verdict(-0.0341, 5872)
    check("D-17: a healthy 2.6-sigma residual passes", ok, f"z={z:+.2f}")
    check("D-17: null SD matches 1/sqrt(n-1)", abs(sd - 1 / math.sqrt(5871)) < 1e-12)
    check("D-17: the old |T|<0.05 rule would have failed it",
          abs(-0.0341 / math.sqrt(0.7084 * 0.6425)) > 0.05,
          "this is the bug being regressed against")

    # A real index leak: shuffling leaves the true transfer intact.
    ok_leak, z_leak, _ = shuffled_control_verdict(0.60, 5872)
    check("a genuine leak fails", not ok_leak, f"z={z_leak:+.1f}")
    check("and fails by a wide margin, not marginally", abs(z_leak) > 40)

    # The rho floor: significance without magnitude must not fire.
    ok_big_n, z_big_n, _ = shuffled_control_verdict(0.02, 1_000_000)
    check("huge n + trivial rho passes despite significance",
          ok_big_n and abs(z_big_n) > 15, f"z={z_big_n:+.1f}, rho=0.02")

    # The z term: magnitude without significance must not fire either.
    ok_small_n, z_small_n, _ = shuffled_control_verdict(0.12, 30)
    check("tiny n + moderate rho passes (not yet distinguishable)",
          ok_small_n, f"z={z_small_n:+.2f}, rho=0.12")

    # Both conditions together.
    check("large rho at large n fails",
          not shuffled_control_verdict(0.15, 5872)[0])

    # Sample-size sensitivity -- the property the flat cutoff lacked.
    _, z_a, _ = shuffled_control_verdict(0.03, 6_000)
    _, z_b, _ = shuffled_control_verdict(0.03, 25_000)
    check("the same rho is judged differently at different n",
          abs(z_b) > 2 * abs(z_a), f"z(6k)={z_a:+.2f} vs z(25k)={z_b:+.2f}")

    # Ceiling independence -- D-17 cause 2. The verdict must not see ceilings.
    check("verdict is ceiling-independent by construction",
          shuffled_control_verdict(-0.0341, 5872)[0]
          is shuffled_control_verdict(-0.0341, 5872)[0],
          "operates on raw rho, ceilings never enter")

    # Symmetry: the rule is two-sided but a leak is one-sided; both must behave.
    check("verdict is symmetric in the sign of rho",
          shuffled_control_verdict(0.60, 5872)[0]
          == shuffled_control_verdict(-0.60, 5872)[0])

    print("gate decision table")
    check("noise-dominated -> FAIL",
          phase0_decision(0.3, 0.9, 0.9)["decision"] == "FAIL")
    check("marginal ceiling -> MARGINAL",
          phase0_decision(0.5, 0.9, 0.9)["decision"] == "MARGINAL")
    check("low transfer -> strong negative",
          phase0_decision(0.7, 0.3, 0.9)["decision"] == "PIVOT-STRONG-NEGATIVE")
    check("reducible to difficulty -> REFRAME",
          phase0_decision(0.7, 0.8, 0.01)["decision"] == "REFRAME")
    check("all gates clear -> full program",
          phase0_decision(0.7, 0.8, 0.1)["decision"] == "FULL-PROGRAM")

    print("zoo registry")
    check("15 architectures registered", len(ZOO) == 15, f"{len(ZOO)}")
    check("families cover the H3 ordering",
          {"resnet", "wrn", "vgg", "mobile", "vit", "mixer"}
          <= {v["family"] for v in ZOO.values()})
    if _TORCH_OK:
        for a in ("resnet20", "vgg8", "vit_tiny", "mixer_nano"):
            try:
                m = build_model(a, 10)
                x = torch.randn(2, 3, 32, 32)
                o, fs = m(x), m.forward_features(x)
                check(f"{a} builds and runs",
                      o.shape == (2, 10) and len(fs) == 5,
                      f"dims={m.feature_dims}")
            except Exception as e:
                check(f"{a} builds and runs", False, f"{type(e).__name__}: {e}")

        # --- D-21: the MSC-KD training step must survive AMP autocast -------
        # This is the loss the entire method rests on, and NO test had ever run
        # it under autocast -- the preflight built models and ran forward
        # passes, which is exactly the part that was fine. So
        # F.binary_cross_entropy, an op torch explicitly bans under autocast,
        # reached a real multi-account run and failed 1 hour in.
        #
        # CPU autocast enforces the same ban as CUDA, so this catches it with
        # no GPU.
        try:
            _st = MSCStudent(build_model("resnet20", 10), 10, n_budgets=5)
            _x = torch.randn(4, 3, 32, 32)
            _tl, _y = torch.randn(4, 10), torch.tensor([0, 1, 2, 3])
            _tg = torch.zeros(4, 5)
            _tg[:, 3:] = 1.0
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                _sl, _suff, _ = _st(_x, suff_logits=True)
                _loss, _ = MSCLoss()(_sl[-1], _tl, _y, _suff, _tg)
            _loss.backward()
            check("D-21: the MSC-KD loss runs under AMP autocast",
                  torch.isfinite(_loss).item(), f"loss={float(_loss):.4f}")
        except Exception as e:
            check("D-21: the MSC-KD loss runs under AMP autocast", False,
                  f"{type(e).__name__}: {e}")

        # The refactor must not have changed what the head computes.
        try:
            _st.eval()
            with torch.no_grad():
                _f = _st.backbone.forward_features(torch.randn(4, 3, 32, 32))[0]
                _p, _lg = _st.suff(_f), _st.suff.logits(_f)
            check("D-21: forward() is exactly sigmoid(logits())",
                  torch.allclose(_p, torch.sigmoid(_lg), atol=1e-6))
            check("D-21: the sufficiency curve is still monotone in k",
                  bool((_p[:, 1:] >= _p[:, :-1] - 1e-6).all()),
                  "architectural monotonicity must survive the logit split")
        except Exception as e:
            check("D-21: forward() is exactly sigmoid(logits())", False,
                  f"{type(e).__name__}: {e}")
    else:
        print("  [SKIP] torch unavailable -- model checks run in notebook 00")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(f"msc_lib v{__version__} -- run with --selftest for the offline checks")
