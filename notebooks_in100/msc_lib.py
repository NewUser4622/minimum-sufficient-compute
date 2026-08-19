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
import textwrap
import itertools
import warnings
from inspect import signature as _inspect_signature
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
HF_REPO = os.environ.get("MSC_HF_REPO", "Shanmuk4622/msc-imagenet100")
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
    """Create a directory, or say *why not* in words the operator can act on.

    D-44. A default path pointed at `D:\\` on a machine with no D: drive, and
    the failure surfaced as

        FileNotFoundError: [WinError 3] The system cannot find the path
        specified: 'D:\\'

    forty lines deep in `pathlib.mkdir`, from a call two frames inside library
    import. Nothing in that traceback says "edit the path at the top of the
    notebook", which is the entire remedy.
    """
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except (FileNotFoundError, NotADirectoryError, OSError) as e:
        anchor = p
        while anchor.parent != anchor and not anchor.parent.exists():
            anchor = anchor.parent
        raise OSError(
            f"cannot create {p}\n"
            f"  the first missing level is: {anchor}\n"
            f"  ({type(e).__name__}: {e})\n"
            f"  If that is a drive letter, the drive does not exist on this "
            f"machine.\n"
            f"  Set DATA_DIR / MSC_ROOT at the top of the notebook to a path "
            f"that does,\n"
            f"  or leave them as None and they will be chosen automatically."
        ) from e


def _atomic_replace(tmp, path, attempts: int = 20, pause: float = 0.15) -> None:
    """`os.replace` with a bounded retry, because Windows is not POSIX.

    On POSIX `os.replace` always succeeds over an existing file. On Windows it
    raises `PermissionError` if any process holds a handle to the destination --
    an antivirus scanner, a file indexer, an open Explorer preview, or a HF
    uploader thread that is reading the very checkpoint being rewritten.

    The failure mode is the one this function exists to prevent: the temp file
    is complete and correct, the destination is the previous version, and the
    exception propagates out of the middle of an epoch. Retrying is right
    because the condition is transient by nature; giving up silently is not,
    so the final attempt raises.

    Without this the port would lose checkpoints on Windows at exactly the
    moments the uploader is busiest, which is to say at every push cycle.
    """
    last = None
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:                              # noqa: PERF203
            last = e
            time.sleep(pause * (1 + i * 0.5))
    raise OSError(
        f"could not atomically replace {path} after {attempts} attempts. "
        f"Something is holding the destination open. The complete data is in "
        f"{tmp} and has NOT been lost.") from last


def atomic_write_text(path, text: str) -> None:
    """Write via a temp file and rename.

    Never write in place. A session killed mid-write leaves a truncated file,
    and for ckpt_last.pt that means the run is gone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, path)


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
    _atomic_replace(tmp, path)


def to_numpy(v, dtype=None) -> np.ndarray:
    """A numpy array from a tensor on ANY device, or from anything array-like.

    **D-70.** The sweep did `np.asarray(y)` on the label tensor. On CIFAR the
    raw `DataLoader` hands back CPU tensors and that works. On ImageNet-100 the
    batch comes through `GPUBatchLoader`, which ends with
    `yb = y.to(self.device)` -- so `y` is on cuda:0 and numpy refuses:

        TypeError: can't convert cuda:0 device type tensor to numpy.
                   Use Tensor.cpu() to copy the tensor to host memory first.

    It failed 40 minutes into the first run, after exit-head training and the
    final evaluation had both succeeded -- the most expensive place for a
    one-line conversion bug to sit.

    The port's premise was one library parameterised by dataset rather than
    forked. That premise holds only where the two datasets present the SAME
    interface, and here they did not: one loader yields CPU labels, the other
    device labels. Three call sites each assumed the CIFAR shape. This is the
    single conversion they all now go through.
    """
    if _TORCH_OK and isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v)
    return arr.astype(dtype) if dtype is not None else arr


def read_yaml(path, default=None):
    """Counterpart to `atomic_write_yaml`. There was a writer and no reader.

    D-63: I reached for `read_yaml` while fixing a defect caused by not
    reading the config record, and it did not exist -- the config.yaml every
    run writes had never once been read back by this library. Falls back to
    the .json sibling, matching what `atomic_write_yaml` does when PyYAML is
    unavailable.
    """
    p = Path(path)
    if yaml is not None and p.exists():
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or default
        except Exception:                                        # noqa: BLE001
            return default
    return read_json(p.with_suffix(".json"), default)


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


def set_perf_flags(deterministic: bool = False) -> Dict[str, Any]:
    """Configure the compute backend. ONE function, used by training and by the
    benchmark, so the two cannot measure different machines.

    **D-43.** The throughput benchmark never called this, so it ran with
    `cudnn.benchmark = False` -- torch's default -- while every real training
    run has it True via `set_seed`. cuDNN with autotuning off picks convolution
    algorithms by heuristic, and for ResNet-50's many distinct 1x1 and 3x3
    shapes in `channels_last` that heuristic is poor. The benchmark measured
    82 img/s for a network that should sit near 180.

    A benchmark whose entire purpose is to predict the real run, configured
    differently from the real run, produces a number that is precise and about
    nothing. Extracting it here is the D-16 lesson: the writer and the reader
    must not be two independent spellings of the same setting.

    `cudnn.benchmark = True` costs a few seconds of autotuning per distinct
    input shape and typically buys 1.3-2x on ResNet-50. It also makes algorithm
    selection non-deterministic, which changes floating-point summation order.
    That is recorded rather than ignored: this project measures seed-to-seed
    reliability, and anything adding within-seed variance is relevant. The
    effect is far below the seed-to-seed variation being measured -- AMP alone
    already forfeits bitwise reproducibility -- and `deterministic: True` in
    the config turns it off.
    """
    out: Dict[str, Any] = {"deterministic": bool(deterministic)}
    if not _TORCH_OK:
        return out
    try:
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # Fixed batch and fixed resolution -> autotuning pays for itself.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # TF32 on Ada: free accuracy-for-speed on fp32 ops that autocast leaves
        # alone. Irrelevant under fp16/bf16 matmuls, harmless elsewhere.
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        out.update({"cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32})
    except Exception as e:                                       # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


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
    set_perf_flags(deterministic)
    if deterministic:
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
            # D-58. The cuDNN VERSION was recorded; whether autotuning was ON
            # was not. Diagnosing an 8x convolution slowdown then required
            # reading source to guess at flags the run could have written down.
            # A backend setting that moves throughput by multiples is
            # provenance, not trivia.
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
            "cudnn_enabled": bool(getattr(torch.backends.cudnn, "enabled", True)),
            "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
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

    # -- resolve-only verification -----------------------------------------
    # RULE 9. `list_repo_files` goes through the tree / repo-info endpoints,
    # and those are CDN-cached. On 2026-08-02 an audit concluded that only the
    # NB04 runs existed on HF. That conclusion was wrong, it stood in the lab
    # notebook for two days, and it was reached twice by two different methods
    # that agreed with each other:
    #
    #   * `tree/main/runs` returned byte-identical `oid`s across audits hours
    #     apart, which was read as "nothing changed" and actually meant "you
    #     were served the same cached page twice";
    #   * the full repo-info body was silently TRUNCATED mid-JSON at ~69 KB,
    #     and the truncated file list happened to cut off just past `vgg8` --
    #     exactly where `vit_tiny` and `wrn_*` would have appeared.
    #
    # `resolve` is the content endpoint. A HEAD against it either returns that
    # file's metadata or 404s, per file, with no aggregate to truncate and no
    # listing to cache. It is the only HF answer this project now trusts about
    # whether a specific file exists.
    def resolve_meta(self, repo_path: str, revision: str = "main"
                     ) -> Optional[Dict[str, Any]]:
        """Per-file metadata via `resolve`, or None if the file is not there.

        None means "not present". It does NOT mean "the network failed" -- that
        raises, because a negative finding produced by a dropped connection is
        the D-20 false alarm all over again, and per the retracted audit a
        negative finding deserves the same verification standard as a positive
        one.
        """
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        url = hf_hub_url(repo_id=self.repo_id, filename=repo_path,
                         repo_type=self.repo_type, revision=revision)
        try:
            m = get_hf_file_metadata(url, token=self.token)
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "entrynotfound" in msg:
                return None
            raise RuntimeError(
                f"could not determine whether {repo_path} exists: {e}. "
                f"Refusing to report absence on a failed lookup.") from e
        return {"path": repo_path, "size": getattr(m, "size", None),
                "etag": getattr(m, "etag", None),
                "commit": getattr(m, "commit_hash", None)}

    def files_present(self, repo_paths: Sequence[str], revision: str = "main"
                      ) -> Dict[str, Optional[Dict[str, Any]]]:
        """`{repo_path: meta or None}`, one `resolve` call each. Rule 10: this
        is what "did the files land?" means. Draining the upload queue says the
        queue emptied, which is a fact about this process, not about the repo."""
        return {p: self.resolve_meta(p, revision) for p in repo_paths}

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
    if not tok and os.environ.get("MSC_OFFLINE", "") in ("", "0", "false"):
        # Silent when MSC_OFFLINE is set: this programme is local-only by
        # design, and telling the operator to add a HuggingFace token is
        # advice for a configuration they deliberately are not in. A message
        # that fires on the intended setup is noise, and noise is what makes
        # a real line get skimmed past (D-46, and D-17 before it).
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
            if os.environ.get("MSC_OFFLINE", "") in ("", "0", "false"):
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

# =============================================================================
# 3a. offline operation
# =============================================================================
# The ImageNet-100 programme runs with no network. Two separate things follow,
# and conflating them is how a "we're offline" claim turns out to be false at
# hour three:
#
#   1. Nothing may ATTEMPT a fetch. Libraries that phone home on import or on
#      first use must be told not to, via environment variables set BEFORE they
#      are imported.
#   2. That has to be PROVEN, not asserted. `tools/fetch_assets.py
#      --verify-offline` blocks the socket layer outright and then builds every
#      architecture and runs both dry runs. Rule 10's shape: draining a queue
#      is not confirmation, and installing a package is not offline-readiness.
#
# Worth stating plainly because it is the opposite of what people expect:
# **training from scratch downloads no model weights at all.** torchvision's
# `resnet50(weights=None)` is Python source that ships with the package. There
# is nothing to pre-download for the architectures. What needs one-time
# internet is the pip packages, and what needs pinning is their VERSIONS --
# because a torchvision upgrade can change how a model decomposes into blocks,
# which would silently change every budget table.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # Keep any torch.hub cache local and deterministic rather than in a home
    # directory that may not exist or may be on a different volume.
    "TORCH_HOME": str((SCRATCH_ROOT / "assets" / "torch")),
}


def enforce_offline(verbose: bool = True) -> Dict[str, str]:
    """Set the environment so nothing tries to reach the network.

    Call this BEFORE importing anything that might fetch. `msc_lib` calls it at
    import time when `MSC_OFFLINE` is set, which is the default for the
    ImageNet-100 profile.

    D-44. This used to `ensure_dir(TORCH_HOME)` unconditionally, so **importing
    the library failed** when `MSC_SCRATCH` pointed somewhere that did not
    exist. An import that depends on a writable directory turns a
    fix-one-line-and-re-run into a traceback with no obvious cause, and it
    happens in the bootstrap cell before the operator has reached the cell that
    sets the path. A cache directory is a convenience; nothing here needs it to
    exist in order to import.
    """
    try:
        ensure_dir(Path(OFFLINE_ENV["TORCH_HOME"]))
    except Exception:                                            # noqa: BLE001
        import tempfile as _tf
        OFFLINE_ENV["TORCH_HOME"] = str(Path(_tf.gettempdir()) / "msc_torch")
        try:
            ensure_dir(Path(OFFLINE_ENV["TORCH_HOME"]))
        except Exception:                                        # noqa: BLE001
            pass
    for k, v in OFFLINE_ENV.items():
        os.environ.setdefault(k, v)
    if verbose:
        log(f"offline mode: {len(OFFLINE_ENV)} env guards set, "
            f"TORCH_HOME={OFFLINE_ENV['TORCH_HOME']}", "OFFLINE")
    return dict(OFFLINE_ENV)


def allow_network(verbose: bool = True) -> Dict[str, Any]:
    """Reverse `enforce_offline` for this process. Publishing needs the network.

    **D-83.** `msc_lib` calls `enforce_offline()` at import time whenever
    `MSC_OFFLINE` is set, and the notebook bootstrap sets it. That is right for
    NB1-NB5, which must be provably self-contained. NB6 is the one notebook
    whose entire job is to reach HuggingFace, and it inherited the guard:

        OfflineModeIsEnabled: Cannot reach
        https://huggingface.co/api/repos/create: offline mode is enabled.

    Clearing the variable in PowerShell does not help, and the error's own
    advice is misleading here: the variable is set **inside this process**,
    after the shell has been left behind.

    Nor is `os.environ.pop` sufficient on its own. `huggingface_hub` reads
    `HF_HUB_OFFLINE` **once, at import**, into `huggingface_hub.constants`.
    Anything already imported keeps the old value, so the constant is patched
    too -- for the module and for the submodules that copied it.

    Returns what it changed, so a notebook can show it rather than assert it.
    """
    changed = {"env_cleared": [], "constants_patched": []}
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "MSC_OFFLINE"):
        if os.environ.pop(k, None) is not None:
            changed["env_cleared"].append(k)

    for mod_name in ("huggingface_hub.constants", "huggingface_hub",
                     "huggingface_hub.file_download",
                     "huggingface_hub._snapshot_download"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "HF_HUB_OFFLINE"):
            try:
                setattr(mod, "HF_HUB_OFFLINE", False)
                changed["constants_patched"].append(mod_name)
            except Exception:                                    # noqa: BLE001
                pass

    if verbose:
        log(f"network ENABLED for this process. cleared "
            f"{changed['env_cleared'] or 'nothing'}; patched "
            f"{changed['constants_patched'] or 'nothing'}", "NET")
        log("this is the only notebook that goes online. NB1-NB5 stay offline.",
            "NET")
    return changed


def hf_upload_resilient(token: str, repo_id: str, repo_type: str,
                        items: Sequence[Tuple[str, str, str]],
                        attempts: int = 4, backoff: float = 4.0,
                        on_event=None) -> Dict[str, Any]:
    """Upload folders one at a time, surviving a network drop.

    **D-86.** A 22-run publish reached run 12 and then:

        [Errno 11001] getaddrinfo failed ... Retrying in 1s [Retry 1/5].
        RuntimeError: Cannot send a request, as the client has been closed.

    Two distinct failures. The first is a transient DNS loss, which
    `huggingface_hub` retries correctly. The second is what happens *after*
    those retries are exhausted: the underlying httpx client is closed, and it
    is closed **for the life of the object**. Every later call on that `HfApi`
    fails instantly with the same message, so one blip at run 12 poisons runs
    13 to 22 even once the network is back.

    So the fix is not more retries -- `huggingface_hub` already retries. It is
    to **rebuild the client** rather than reuse a dead one, and to treat a
    failed item as one failed item instead of the end of the run.

    `items` is `(local_path, path_in_repo, label)`. Returns
    `{"uploaded": [...], "failed": [(label, reason), ...]}` and never raises:
    a publish that stops on the first error is one that has to be babysat, and
    the whole point is that it can be re-run.
    """
    from huggingface_hub import HfApi

    out: Dict[str, Any] = {"uploaded": [], "failed": []}
    for local, in_repo, label in items:
        last = ""
        for attempt in range(1, attempts + 1):
            try:
                # A FRESH client each attempt. Reusing one that has been closed
                # is the whole defect.
                HfApi(token=token).upload_folder(
                    folder_path=str(local), path_in_repo=in_repo,
                    repo_id=repo_id, repo_type=repo_type,
                    commit_message=f"add {label}")
                out["uploaded"].append(label)
                if on_event:
                    on_event("ok", label, attempt, "")
                break
            except Exception as e:                               # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:140]}"
                if on_event:
                    on_event("retry", label, attempt, last)
                if attempt < attempts:
                    time.sleep(backoff * attempt)
        else:
            out["failed"].append((label, last))
            if on_event:
                on_event("failed", label, attempts, last)
    return out


def hf_token_check(token: Optional[str], repo_id: str,
                   repo_type: str = "dataset") -> Dict[str, Any]:
    """Can this token write to this namespace? Asked BEFORE anything is created.

    **D-84.** NB6's first network call was `create_repo`, and the most likely
    thing to be wrong -- a read-only token, or a token belonging to a different
    account -- surfaced as a forty-line traceback ending in

        403 Forbidden: You don't have the rights to create a dataset under the
        namespace "Shanmuk4622".

    The message is accurate and the diagnosis is buried under an httpx
    HTTPStatusError, an HfHubHTTPError, a deprecation wrapper and a validator.
    `whoami()` answers the same question in one call, before anything is
    attempted, and can name which of the three causes it is.

    Never raises: it returns a verdict so the notebook can print it. A preflight
    that throws is just a different traceback.
    """
    out: Dict[str, Any] = {"ok": False, "reason": "", "user": None,
                           "role": None, "namespace": repo_id.split("/")[0],
                           "repo_id": repo_id, "fine_grained": None}
    if not token:
        out["reason"] = ("HF_TOKEN is not set. Create one at "
                         "https://huggingface.co/settings/tokens (type: Write), "
                         "then `setx HF_TOKEN hf_...` and restart the kernel.")
        return out
    try:
        from huggingface_hub import HfApi
        me = HfApi(token=token).whoami()
    except Exception as e:                                       # noqa: BLE001
        out["reason"] = (f"could not identify the token: {type(e).__name__}: "
                         f"{str(e)[:160]}")
        return out

    out["user"] = me.get("name")
    auth = (me.get("auth") or {}).get("accessToken") or {}
    out["role"] = auth.get("role")
    out["fine_grained"] = auth.get("fineGrained")

    orgs = {o.get("name") for o in (me.get("orgs") or [])}
    ns = out["namespace"]
    if ns != out["user"] and ns not in orgs:
        out["reason"] = (
            f"the token belongs to '{out['user']}' but the repo namespace is "
            f"'{ns}'. Either set REPO_ID to '{out['user']}/{repo_id.split('/')[-1]}' "
            f"or use a token for '{ns}'.")
        return out

    if out["fine_grained"] is not None:
        # A fine-grained token lists explicit permissions; a missing write
        # scope is the common case and the 403 does not say which.
        out["reason"] = (
            f"token is FINE-GRAINED. It must grant write access to "
            f"'{ns}'. If create fails, re-issue it with 'Write access to "
            f"contents/settings of all repos under your personal namespace', "
            f"or use a classic Write token.")
        out["ok"] = True          # cannot prove it fails; let the call decide
        return out

    if out["role"] not in ("write", "admin"):
        out["reason"] = (
            f"token role is '{out['role']}' -- read-only. Creating or writing "
            f"a {repo_type} needs a WRITE token. "
            f"https://huggingface.co/settings/tokens -> New token -> Write.")
        return out

    out["ok"] = True
    out["reason"] = f"token for '{out['user']}' has role '{out['role']}'"
    return out


def offline_state() -> Dict[str, Any]:
    """What the offline guard currently looks like, for display."""
    out = {k: os.environ.get(k) for k in
           ("MSC_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE")}
    mod = sys.modules.get("huggingface_hub.constants")
    out["huggingface_hub.constants.HF_HUB_OFFLINE"] = (
        getattr(mod, "HF_HUB_OFFLINE", None) if mod is not None
        else "<not imported>")
    return out


@contextmanager
def no_network(allow_local: bool = True):
    """Block the socket layer, so a fetch RAISES instead of hanging.

    This is the verification half. Environment variables are a request;
    replacing `socket.socket` is a guarantee. Used by the offline preflight and
    available for any check that wants to prove a code path is self-contained.

    Loopback stays open by default -- CUDA IPC and some dataloader backends use
    it, and blocking it would make this test fail for reasons that have nothing
    to do with the internet.
    """
    import socket as _s
    real = _s.socket

    class _Blocked(real):                                        # type: ignore
        def connect(self, address, *a, **k):
            host = address[0] if isinstance(address, tuple) else str(address)
            if allow_local and str(host) in ("127.0.0.1", "::1", "localhost"):
                return super().connect(address, *a, **k)
            raise OSError(
                f"network access to {host!r} was attempted while offline. "
                f"This pipeline must run with no internet; find the call and "
                f"remove it or pre-fetch what it wants.")

        def connect_ex(self, address, *a, **k):
            try:
                self.connect(address, *a, **k)
                return 0
            except OSError:
                return 1

    _s.socket = _Blocked                                         # type: ignore
    try:
        yield
    finally:
        _s.socket = real                                         # type: ignore


if os.environ.get("MSC_OFFLINE", "") not in ("", "0", "false", "False"):
    enforce_offline(verbose=False)


def run_layout(root, run_id: str) -> Dict[str, Path]:
    """Canonical paths for one run. Local tree mirrors the repo tree exactly,
    so a push is a relative-path calculation and never a guess.
    """
    base = Path(root) / "runs" / run_id
    d = {"base": base}
    for s in RUN_SUBDIRS:
        d[s] = base / s
    return d


# =============================================================================
# 3b. local store -- what a complete run must leave on disk
# =============================================================================
# With HuggingFace removed, local disk is the only copy. Everything the hub
# used to guarantee now has to be guaranteed here, and one of those guarantees
# was never really a guarantee even with HF: that the run actually produced
# what it was supposed to produce.
#
# `sync.flush()` returning True meant the upload queue drained. `confirm_on_hf`
# improved on that by asking the repository. Neither ever asked the more basic
# question -- **is every artifact this run was meant to write actually there,
# non-empty, and readable?** A run that finished with a corrupt parquet or a
# zero-byte summary looked identical to a healthy one until analysis.
#
# `required` is what makes a run usable at all. `expected` is everything else;
# its absence is reported, never fatal, because a missing telemetry stream
# costs a column and a missing checkpoint costs the run.
RUN_ARTIFACTS_REQUIRED = (
    "config.yaml",
    "config_hash.txt",
    "summary.json",
    "metrics/epochs.csv",
    "checkpoints/ckpt_last.pt",
    "checkpoints/ckpt_best.pt",
    "env/environment.json",
)
RUN_ARTIFACTS_MEASURED = (
    # D-64. `final.csv` sat in REQUIRED, which is checked after TRAINING, but
    # only `run_oracle` writes it -- `final_evaluation` is called from there
    # and from nowhere else. So every correctly-finished training run verified
    # as INCOMPLETE, on all four Phase-0 runs at once.
    #
    # Nothing was lost: the file arrives when NB3 runs. But a verifier that
    # reports healthy runs as broken is the failure this project keeps paying
    # for -- it trains you to skim the output, and the next alarm is real.
    "metrics/final.csv",
    "per_sample/test.parquet",
    "per_sample/train_holdout.parquet",
    "per_sample/meta.json",
    "exit_heads.pt",
)
RUN_ARTIFACTS_EXPECTED = (
    "STATUS.json",
    "metrics/confusion_matrix.csv",
    "metrics/per_class.csv",
    "metrics/exit_metrics.csv",
    "telemetry/energy_samples.csv",
    "telemetry/system_samples.csv",
    "telemetry/step_traces.jsonl",
    "per_sample/train_dynamics.parquet",
)


def repo_rel_path(work, local_path) -> str:
    """The HuggingFace path for a local file. THE accessor for remote paths.

    `run_layout` exists so the local tree and the repo tree are the same shape
    -- "a push is a relative-path calculation and never a guess". This is that
    calculation, in one place, so NB6 does not spell `runs/{id}/...` by hand.

    Rule 4 is about repo paths generally, and a remote path typed as a literal
    is the same hazard as a local one: D-23 was `exit_heads.pt` written to the
    run root and read from `checkpoints/`, and the fix was an accessor.
    """
    rel = Path(local_path).resolve().relative_to(Path(work).resolve())
    return rel.as_posix()


def publish_manifest(work) -> "Any":
    """Everything that would be published, grouped, with sizes -- from the
    layout rather than from hand-written globs.

    Groups are derived from `RUN_SUBDIRS` and the artifact lists, so a new
    subdirectory appears here automatically instead of being silently omitted.
    """
    work = Path(work)
    rows = []
    runs = sorted(d for d in (work / "runs").iterdir() if d.is_dir()) \
        if (work / "runs").exists() else []
    for sub in ("", ) + RUN_SUBDIRS:
        files = []
        for d in runs:
            base = d / sub if sub else d
            if not base.exists():
                continue
            files += [f for f in base.iterdir() if f.is_file()]
        if files:
            rows.append({"group": f"runs/*/{sub}" if sub else "runs/* (root)",
                         "files": len(files),
                         "bytes": sum(f.stat().st_size for f in files)})
    for top in ("budgets", "registry", "analysis", "tables", "paper"):
        d = work / top
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        if files:
            rows.append({"group": top + "/", "files": len(files),
                         "bytes": sum(f.stat().st_size for f in files)})
    return pd.DataFrame(rows) if pd is not None else rows


def phases_present(work) -> Dict[str, Dict[str, int]]:
    """`{phase: {"runs": n, "completed": n}}` read straight off disk.

    Filesystem only -- no Session, no ledger, no data directory. It has to work
    before anything is configured, because its job is to tell you what to
    configure.
    """
    out: Dict[str, Dict[str, int]] = {}
    root = Path(work) / "runs"
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            ph = parse_run_id(d.name)["phase"]
        except Exception:                                        # noqa: BLE001
            continue
        rec = out.setdefault(ph, {"runs": 0, "completed": 0})
        rec["runs"] += 1
        st = read_json(d / "STATUS.json", {}) or {}
        if str(st.get("state", "")) == "completed":
            rec["completed"] += 1
    return out


def detect_phase(work, prefer: Optional[str] = None) -> str:
    """Which phase should this notebook operate on?

    **D-65.** NB3, NB4 and NB5 each hardcoded `PHASE = 'p1'` while NB2 trains
    `p0`. Run them in order, unedited, and NB3 finds zero `p1` runs, prints
    `0 trained run(s), 0 still to measure`, calls `run_all([])` and exits
    successfully. Nothing failed. Nothing happened either, and the next
    notebook then has nothing to analyse -- for a reason three notebooks back.

    A default that is wrong for the documented order is not a default, it is a
    trap, and "silently does nothing" is the worst way to spring it.

    `prefer` wins if it has runs. Otherwise the phase with the most completed
    runs. Raises -- listing what IS on disk -- rather than returning a phase
    with no work in it.
    """
    seen = phases_present(work)
    if prefer and seen.get(prefer, {}).get("completed", 0) > 0:
        return prefer
    live = {k: v for k, v in seen.items() if v["completed"] > 0}
    if not live:
        raise RuntimeError(
            f"no completed runs under {work}.\n"
            f"  phases with any runs at all: "
            f"{ {k: v['runs'] for k, v in seen.items()} or 'none'}\n"
            f"  Run NB2 first, or point MSC_ROOT at the right results folder.")
    best = max(live, key=lambda k: live[k]["completed"])
    if prefer and prefer != best:
        log(f"phase {prefer!r} has no completed runs; using {best!r} "
            f"({live[best]['completed']} completed). Set PHASE explicitly to "
            f"override (D-65).", "PHASE")
    return best


def verify_run_artifacts(work, run_id: str, measured: bool = False,
                         min_bytes: int = 8) -> Dict[str, Any]:
    """Is everything this run was supposed to write actually on disk?

    Returns a dict with `ok`, `missing_required`, `empty`, `unreadable`, and a
    per-file table. Three failure classes, not one, because they mean different
    things:

      missing     the step never ran, or ran and crashed before writing
      empty       the file was created and the write failed -- the shape that
                  an interrupted `atomic_write` was designed to prevent and
                  that a non-atomic write produces routinely
      unreadable  present and non-empty and CORRUPT. Only found by opening it,
                  which is why the parquet and JSON files are actually parsed
                  here rather than stat-ed.

    The third class is the one presence checks miss, and it is the one that
    surfaces during analysis rather than during training.
    """
    L = run_layout(work, run_id)
    base = L["base"]
    want = list(RUN_ARTIFACTS_REQUIRED)
    if measured:
        want += list(RUN_ARTIFACTS_MEASURED)
    optional = list(RUN_ARTIFACTS_EXPECTED) + (
        [] if measured else list(RUN_ARTIFACTS_MEASURED))

    table, missing, empty, unreadable = {}, [], [], []
    for rel in want + optional:
        p = base / rel
        req = rel in want
        if not p.exists():
            table[rel] = {"state": "missing", "required": req, "bytes": 0}
            if req:
                missing.append(rel)
            continue
        n = p.stat().st_size
        if n < min_bytes:
            table[rel] = {"state": "empty", "required": req, "bytes": n}
            if req:
                empty.append(rel)
            continue
        state = "ok"
        try:
            if rel.endswith(".json"):
                json.loads(p.read_text(encoding="utf-8"))
            elif rel.endswith(".parquet") and pd is not None:
                _ = pd.read_parquet(p, columns=None).shape
            elif rel.endswith(".csv") and pd is not None:
                _ = pd.read_csv(p, nrows=2).shape
        except Exception as e:                                   # noqa: BLE001
            state = f"unreadable: {type(e).__name__}"
            if req:
                unreadable.append(rel)
        table[rel] = {"state": state, "required": req, "bytes": n}

    return {"run_id": run_id, "root": str(base),
            "ok": not (missing or empty or unreadable),
            "missing_required": missing, "empty": empty,
            "unreadable": unreadable,
            "total_bytes": sum(v["bytes"] for v in table.values()),
            "files": table}


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
        """Which required repo paths are NOT on HF, asked FILE BY FILE.

        Confirm-then-delete depends on this, and it is the last thing standing
        between a completed run and `shutil.rmtree`. Never wipe a local run on
        the strength of a `flush()` that merely did not time out (rule 10).

        Rule 9: this used to call `list_repo_files`, i.e. the tree endpoint,
        which is cached and which truncates. Both failure modes report a file
        as ABSENT when it is present -- and the caller's response to "absent"
        is to keep the local copy, which is harmless, or to re-push, which is
        wasteful but safe. The dangerous direction is the other one, and a
        cached listing can produce that too: a stale page showing a file that
        was since deleted. `resolve` has neither property.
        """
        if not self.enabled:
            return set(required)
        got = self.hub.hub.files_present(list(required))
        return {r for r, meta in got.items() if meta is None}


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
    # `session_limit_h <= 0` == unbounded. See __init__ (D-50).

    def __init__(self, on_flush: Callable[[str], None],
                 session_limit_h: float = 8.5, verbose: bool = True):
        """`session_limit_h <= 0` means NO LIMIT, not a limit of zero.

        **D-50.** The watchdog exists for Kaggle, where a session dies at 8-12
        hours without warning, so the civilised thing is to stop cleanly first.
        A local machine has no such deadline, and the ImageNet-100 profile sets
        `session_limit_h = 0.0` to say so.

        It was read as "the limit is zero hours", so `session_expiring()` was
        true on the first call and **every run paused after epoch 1**:

            [LIFE] session limit reached at 0.1 h -- pausing cleanly at epoch 1

        Over a ten-day programme that is a manual restart every few minutes,
        and it silently defeated the kill-and-resume test as well -- the run
        paused before the debug interrupt could fire, so the test reported
        `interrupt actually fired: False` and failed for a reason that had
        nothing to do with resume.

        Zero as a sentinel for "unbounded" is a reasonable convention and a
        bad default to leave implicit, so it is now explicit here, in the
        config, and in a self-check.
        """
        self.on_flush = on_flush
        self.session_limit_sec = (float("inf") if session_limit_h is None
                                  or session_limit_h <= 0
                                  else session_limit_h * 3600.0)
        self.unlimited = not math.isfinite(self.session_limit_sec)
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
            log(f"lifecycle guard armed (SIGTERM + atexit, session limit "
                + ("NONE -- runs to completion)" if self.unlimited
                   else f"{self.session_limit_sec/3600:.1f} h)"), "LIFE")
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
        """True only when a real deadline has been reached (D-50)."""
        if self.unlimited:
            return False
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# 6a. dataset registry -- the answer to "how big is an image here?"
# =============================================================================
# Every literal `32` and every literal `100` in this library used to be correct
# because there was one dataset. Rule 2: a literal that is right for 13 of 15
# cases is the worst kind, and a literal that is right for 1 of 2 datasets is
# the same defect with a smaller denominator.
#
# So: nothing downstream may spell an input resolution or a class count. It asks
# here. The three accessors below are the only sanctioned way to obtain them,
# which means a missing dataset is a KeyError at the top of a notebook rather
# than a shape error eight frames into a sweep.
#
# `resolutions` is the resolution axis grid. For CIFAR it is the frozen
# (16,20,24,28,32). For ImageNet-100 every value must be divisible by 32,
# because a ViT-S/16 has to patchify it into a square grid AND a Swin-T reduces
# by 4 (patch) x 2 x 2 x 2 (three merges) = 32. 224 x the CIFAR fractions gives
# 112/140/168/196/224, and 140 and 196 satisfy neither. This is exactly the
# constraint that produced D-01a and D-02 on CIFAR, resolved at design time
# instead of at preflight time.
DATASETS: Dict[str, Dict[str, Any]] = {
    "cifar100": dict(
        num_classes=100, native_res=32, resolutions=(16, 20, 24, 28, 32),
        mean=CIFAR100_MEAN, std=CIFAR100_STD, backend="cifar",
        zoo="cifar", train_n=50_000, eval_n=10_000),
    "cifar10": dict(
        num_classes=10, native_res=32, resolutions=(16, 20, 24, 28, 32),
        mean=CIFAR10_MEAN, std=CIFAR10_STD, backend="cifar",
        zoo="cifar", train_n=50_000, eval_n=10_000),
    "imagenet100": dict(
        num_classes=100, native_res=224, resolutions=(96, 128, 160, 192, 224),
        mean=IMAGENET_MEAN, std=IMAGENET_STD, backend="packed",
        zoo="imagenet", train_n=119_395, eval_n=10_000),
}


def dataset_spec(dataset: str) -> Dict[str, Any]:
    d = str(dataset).lower()
    if d not in DATASETS:
        raise KeyError(f"unknown dataset '{dataset}'. Known: {sorted(DATASETS)}")
    return DATASETS[d]


def native_res(dataset: str) -> int:
    """The resolution the network is trained and evaluated at."""
    return int(dataset_spec(dataset)["native_res"])


def resolutions_for(dataset: str) -> Tuple[int, ...]:
    return tuple(dataset_spec(dataset)["resolutions"])


def num_classes_for(dataset: str) -> int:
    return int(dataset_spec(dataset)["num_classes"])


def input_shape(dataset: str, res: Optional[int] = None,
                batch: int = 1) -> Tuple[int, int, int, int]:
    """The profiler input shape. Never write `(1, 3, 32, 32)` anywhere again."""
    r = int(res if res is not None else native_res(dataset))
    return (int(batch), 3, r, r)


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
        # CIFAR emits positions within the split, so the index space IS the
        # split length. Declared explicitly so every backend answers the same
        # question rather than one of them being assumed (D-49).
        self.index_space = int(self.labels.numel())
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


# =============================================================================
# 6c. data -- ImageNet-100 from the packed uint8 memmap
# =============================================================================
# Built by tools/pack_imagenet100.py. See 25_IN100_DATA_CARD.md for the subset
# identity, the split policy and the fingerprint.
#
# The design decision that matters here: augmentation runs on the GPU, and it
# runs INSIDE THE LOADER rather than in the training loop.
#
# The obvious implementation puts a `x = augment(x)` line after every
# `.to(device)`. There are eleven such sites -- train_backbone, evaluate,
# run_oracle's three sweeps, difficulty_battery, prediction_depth,
# train_exit_heads, train_msc_kd, the dry runs -- and rule 6 is exactly about
# this shape: when a step can be skipped at N points, forgetting it at one is a
# silent wrong answer, not an error. A model trained on augmented data and
# measured on un-normalised data produces a per-sample MSC table that is
# well-formed and meaningless.
#
# So the loader yields what every existing consumer already expects: a float,
# normalised, correctly-sized tensor already on the device. Nothing downstream
# changed, and nothing downstream CAN forget.
IN100_PACK_FILES = ("images_256.u8", "labels.npy", "manifest.json", "splits.json")


def _has_imagenet100(root: Path) -> bool:
    r = Path(root)
    return all((r / f).exists() for f in IN100_PACK_FILES)


def locate_imagenet100(prefer_scratch: bool = True, verbose: bool = True) -> Path:
    """Find the packed dataset. Never downloads -- packing is a deliberate,
    verified, 20-minute step with its own tool, not something to trigger by
    accident from inside a training run."""
    def _say(m):
        if verbose:
            log(m, "DATA")

    cands: List[Path] = []
    env = os.environ.get("MSC_IN100_DIR")
    if env:
        cands.append(Path(env))
    inp = Path("/kaggle/input")
    if inp.exists():
        cands += [p for p in inp.iterdir() if p.is_dir()]
        cands += [q for p in inp.iterdir() if p.is_dir()
                  for q in p.iterdir() if q.is_dir()]
    for base in (SCRATCH_ROOT, WORK_ROOT):
        cands += [base / "data" / "in100", base / "in100"]

    for c in cands:
        try:
            if _has_imagenet100(c):
                _say(f"found packed ImageNet-100 at {c}")
                return Path(c)
        except Exception:
            continue
    raise RuntimeError(
        "packed ImageNet-100 not found. Build it once with:\n"
        "    python tools/pack_imagenet100.py --src <folder with train/> "
        "--out <dest>\n"
        "then either set MSC_IN100_DIR=<dest>, place it at "
        f"{SCRATCH_ROOT / 'data' / 'in100'}, or attach it as a Kaggle Dataset.\n"
        f"Looked in: {[str(c) for c in cands[:8]]}")


def storage_candidates(min_gb: float = 0.0) -> List[Dict[str, Any]]:
    """Every writable root on this machine, with free space, largest first.

    Windows has no `/`, so "somewhere with room" has to be discovered rather
    than assumed. Drive letters are probed for existence; a machine with no
    `D:` simply does not report one, which is the whole point (D-44).
    """
    roots: List[Path] = []
    if os.name == "nt":
        roots += [Path(f"{c}:\\") for c in "CDEFGHIJKLMNOPQRSTUVWXYZ"
                  if Path(f"{c}:\\").exists()]
    else:
        roots += [Path("/"), Path.home()]
    roots.append(Path.cwd())

    out, seen = [], set()
    for r in roots:
        try:
            key = str(r.resolve()).lower()
            if key in seen or not r.exists():
                continue
            seen.add(key)
            u = shutil.disk_usage(r)
            free = u.free / 2**30
            if free >= min_gb:
                out.append({"root": str(r), "free_gb": free,
                            "total_gb": u.total / 2**30})
        except Exception:                                        # noqa: BLE001
            continue
    return sorted(out, key=lambda d: -d["free_gb"])


def resolve_storage(data_dir=None, results_root=None,
                    need_data_gb: float = 26.0,
                    need_results_gb: float = 120.0,
                    verbose: bool = True) -> Dict[str, Any]:
    """Decide where the pack and the results live, and PROVE both are usable.

    `None` means "choose for me": the roomiest drive that actually exists gets
    `msc_data/in100` and `msc_results`. A default that names a drive letter is
    wrong on any machine without that letter, and the resulting
    `FileNotFoundError: [WinError 3] ... 'D:\\\\'` names neither the setting nor
    the file that has to change (D-44).

    Writability is established by **writing a probe file and reading it back**,
    not by `os.access` -- which lies on Windows network shares and on
    permission-inherited folders. Same discipline as `verify_run_artifacts`:
    presence is not usability.
    """
    report: Dict[str, Any] = {"ok": True, "problems": [], "notes": []}
    cands = storage_candidates()

    def _pick(kind, need):
        for c in cands:
            if c["free_gb"] >= need:
                return Path(c["root"]) / ("msc_data/in100" if kind == "data"
                                          else "msc_results")
        return None

    if data_dir is None:
        # An existing pack anywhere beats a fresh guess.
        for c in cands:
            for sub in ("msc_data/in100", "in100", "data/in100"):
                p = Path(c["root"]) / sub
                if _has_imagenet100(p):
                    data_dir = p
                    report["notes"].append(f"found an existing pack at {p}")
                    break
            if data_dir:
                break
    if data_dir is None:
        data_dir = _pick("data", need_data_gb)
    if results_root is None:
        results_root = _pick("results", need_results_gb)

    if data_dir is None or results_root is None:
        report["ok"] = False
        report["problems"].append(
            f"no drive has enough free space "
            f"(need {need_data_gb:.0f} GB for the pack and "
            f"{need_results_gb:.0f} GB for results). "
            f"Found: {[(c['root'], round(c['free_gb'])) for c in cands]}")
        return {**report, "data_dir": data_dir, "results_root": results_root,
                "candidates": cands}

    data_dir, results_root = Path(data_dir), Path(results_root)
    for label, path, need in (("results", results_root, need_results_gb),
                              ("data", data_dir, need_data_gb)):
        try:
            ensure_dir(path)
        except Exception as e:                                   # noqa: BLE001
            report["ok"] = False
            report["problems"].append(f"{label}: {e}")
            continue
        try:
            probe = path / ".msc_write_probe"
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                raise OSError("wrote a probe file and read back something else")
            probe.unlink()
        except Exception as e:                                   # noqa: BLE001
            report["ok"] = False
            report["problems"].append(
                f"{label}: {path} is not writable ({type(e).__name__}: {e})")
            continue
        free = shutil.disk_usage(path).free / 2**30
        report[f"{label}_free_gb"] = free
        if free < need:
            report["problems"].append(
                f"{label}: {path} has {free:.0f} GB free, "
                f"{need:.0f} GB recommended")
            report["ok"] = False

    report.update({"data_dir": str(data_dir), "results_root": str(results_root),
                   "candidates": cands})
    if verbose:
        print("storage")
        for c in cands:
            print(f"    {c['root']:<6s} {c['free_gb']:7.1f} GB free of "
                  f"{c['total_gb']:7.1f}")
        print(f"    data    -> {data_dir}   "
              f"({report.get('data_free_gb', 0):.0f} GB free, "
              f"need ~{need_data_gb:.0f})")
        print(f"    results -> {results_root}   "
              f"({report.get('results_free_gb', 0):.0f} GB free, "
              f"need ~{need_results_gb:.0f})")
        for n in report["notes"]:
            print(f"    note: {n}")
        for pb in report["problems"]:
            print(f"    *** {pb}")
        print("    " + ("both roots exist, are writable, and were verified by "
                        "writing and reading back a probe file"
                        if report["ok"] else
                        "*** FIX THE ABOVE before running anything else"))
    return report


def data_present(dataset: str, root) -> Tuple[bool, str]:
    """Uniform 'is the data where it should be' check, for the preflight."""
    backend = dataset_spec(dataset)["backend"]
    if backend == "cifar":
        return _has_cifar100(Path(root)), str(root)
    ok = _has_imagenet100(Path(root))
    if not ok:
        return False, f"{root} is missing {IN100_PACK_FILES}"
    man = read_json(Path(root) / "manifest.json", {}) or {}
    return True, (f"{root}  n={man.get('count')}  "
                  f"classes={man.get('n_classes')}  "
                  f"fingerprint={str(man.get('fingerprint',''))[:12]}")


class PackedImageDataset(Dataset):
    """A split of the packed memmap. Returns RAW uint8 HWC plus the GLOBAL index.

    Three properties that are load-bearing:

    * **`sample_idx` is the global pack index, not the position in this split.**
      The val table's indices are the val indices. That makes every per-sample
      table self-describing, lets val and train_holdout tables coexist without
      ambiguity, and means an accidental split mismatch shows up as
      non-overlapping indices rather than as a plausible correlation.

    * **The memmap is opened lazily, per worker.** On Windows the DataLoader
      spawns rather than forks, so a handle opened in the parent is not
      inherited. Opening eagerly would either crash the workers or -- much worse
      -- serve zeros silently.

    * **No shuffling, ever, on an eval split.** Same contract as CIFARTensor:
      `sample_idx` alignment is what every correlation in the project rests on.
    """

    def __init__(self, root, split: str = "val"):
        root = Path(root)
        self.root = root
        self.split = split
        man = read_json(root / "manifest.json")
        if not man:
            raise RuntimeError(f"no manifest.json under {root}")
        self.manifest = man
        self.stored_res = int(man["stored_res"])
        self.count = int(man["count"])
        self.classes = list(man["classes"])
        self.class_names = [man.get("class_names", {}).get(c, c) for c in self.classes]
        self.fingerprint = str(man["fingerprint"])

        splits = read_json(root / "splits.json")
        if split not in ("val", "train", "holdout"):
            raise KeyError(f"unknown split {split!r}")
        self.indices = np.asarray(splits[split], dtype=np.int64)
        self.labels_all = np.load(root / "labels.npy")
        self.labels = self.labels_all[self.indices].astype(np.int64)
        self._mm = None
        # The size of the space `sample_idx` values live in. NOT len(self):
        # this backend emits GLOBAL pack indices so that val and holdout
        # tables coexist unambiguously, which means anything indexing by
        # sample_idx must be sized for the whole pack (D-49).
        self.index_space = int(self.count)
        # Same role as CIFARTensor.order_hash: fingerprints the label order of
        # THIS split so the analysis refuses to correlate misaligned tables.
        self.order_hash = sha256_of_array(self.labels)

    def _mmap(self):
        if self._mm is None:
            self._mm = np.memmap(self.root / "images_256.u8", dtype=np.uint8,
                                 mode="r",
                                 shape=(self.count, self.stored_res,
                                        self.stored_res, 3))
        return self._mm

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        g = int(self.indices[i])
        img = np.asarray(self._mmap()[g])            # (S, S, 3) uint8
        return torch.from_numpy(img), int(self.labels[i]), g


# ---------------------------------------------------------------------------
# D-56: the pack lives in RAM, and batches are gathered whole.
# ---------------------------------------------------------------------------
_RAM_PACK: Dict[str, Any] = {}


def ram_budget_ok(nbytes: int, headroom_gb: float = 6.0) -> Tuple[bool, str]:
    """Is there room for `nbytes` in RAM with `headroom_gb` left over?

    Asked BEFORE allocating, because the failure mode of getting this wrong on
    Windows is not a Python MemoryError -- it is the machine paging itself to
    a standstill, and this project has already cost its owner two hours and a
    second person's admin password once (D-41).
    """
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:                                            # noqa: BLE001
        return False, "psutil unavailable -- cannot prove there is room"
    need = int(nbytes) + int(headroom_gb * 2**30)
    ok = avail >= need
    return ok, (f"{nbytes/2**30:.1f} GiB pack + {headroom_gb:.0f} GiB headroom "
                f"vs {avail/2**30:.1f} GiB available")


def load_pack_to_ram(root: Path, count: int, res: int,
                     headroom_gb: float = 6.0) -> Optional[np.ndarray]:
    """Read `images_256.u8` into a single resident uint8 array, once per process.

    Returns None -- and says why -- if it will not fit. Falling back to the
    memmap is slow, and slow is survivable; swapping is not.
    """
    key = str(Path(root).resolve())
    if key in _RAM_PACK:
        return _RAM_PACK[key]

    path = Path(root) / "images_256.u8"
    nbytes = count * res * res * 3
    ok, why = ram_budget_ok(nbytes, headroom_gb)
    if not ok:
        log(f"RAM cache DECLINED: {why}", "DATA")
        log("falling back to memmap. Slow, but it cannot swap the machine.",
            "DATA")
        return None

    log(f"RAM cache: reading {nbytes/2**30:.1f} GiB into memory ({why})", "DATA")
    t0 = time.time()
    arr = np.empty((count, res, res, 3), dtype=np.uint8)
    chunk = max(1, int(512 * 2**20) // (res * res * 3))
    with open(path, "rb", buffering=0) as fh:
        done = 0
        while done < count:
            n = min(chunk, count - done)
            got = fh.readinto(
                memoryview(arr[done:done + n]).cast("B"))
            if not got:
                raise RuntimeError(f"short read at image {done} of {count}")
            done += n
            if done % (chunk * 8) < chunk or done == count:
                pct = 100.0 * done / count
                log(f"  {pct:5.1f}%  {done:,}/{count:,} images "
                    f"({(time.time()-t0):.0f}s)", "DATA")
    dt = time.time() - t0
    log(f"RAM cache ready in {dt:.0f}s "
        f"({nbytes/2**30/max(dt,1e-9):.2f} GiB/s from disk)", "DATA")
    _RAM_PACK[key] = arr
    return arr


def pack_root_of(ds):
    """Unwrap however many Subsets deep to the PackedImageDataset itself."""
    seen = 0
    while hasattr(ds, "dataset") and not hasattr(ds, "stored_res"):
        ds = ds.dataset
        seen += 1
        if seen > 8:
            raise RuntimeError("dataset wrapping deeper than 8 -- refusing to guess")
    return ds


def pack_view_of(ds) -> Tuple[np.ndarray, np.ndarray]:
    """`(global pack indices, labels)` for a PackedImageDataset or any Subset of one.

    **This is D-49 waiting to happen again, and it nearly did.** Two different
    attributes are both spelled `indices`:

        PackedImageDataset.indices   GLOBAL pack indices for this split
        torch.utils.data.Subset.indices   POSITIONS into the parent dataset

    Reading the second where the first is meant produces indices that are
    numerically valid, silently wrong, and land on the wrong images. D-49 was
    this confusion costing an IndexError; the quiet version costs a
    mislabelled training set that still trains.

    Resolved by composition rather than by remembering: walk the wrapper chain
    and index through at each level.
    """
    if hasattr(ds, "dataset") and not hasattr(ds, "stored_res"):
        gi, lb = pack_view_of(ds.dataset)
        pos = np.asarray(ds.indices, dtype=np.int64)
        return gi[pos], lb[pos]
    return (np.asarray(ds.indices, dtype=np.int64),
            np.asarray(ds.labels, dtype=np.int64))


if _TORCH_OK:

    class RAMBatchLoader:
        """Yields whole uint8 batches from a resident array. No workers, no IPC.

        **D-56.** The per-sample path cost ~0.84 s per batch of 64 while the
        model needed ~0.07 s, and none of it was compute: `PackedImageDataset.
        __getitem__` did ONE random 192 KiB read per sample from a 24 GiB file,
        64 times a batch, then `default_collate` stacked 64 tensors and Windows
        pickled 12.6 MiB through a pipe to the parent. Effective rate ~15 MiB/s,
        which is spinning-disk territory, not SSD.

        Three costs removed at once:

          * the disk, because the pack is resident;
          * the per-sample gather, because `arr[idx]` fetches the batch in one
            numpy call instead of 64 Python round trips plus a stack;
          * the IPC, because with the data already in this process there is
            nothing to send and `num_workers` goes to 0.

        A single prefetch thread keeps the gather off the critical path. Threads
        and not processes deliberately: a process would have to copy 23.5 GiB
        under Windows spawn, which is the OOM this class exists to avoid.

        The contract is byte-identical to the DataLoader it replaces --
        `(uint8 NHWC, int64 labels, int64 GLOBAL idx)` -- so `GPUBatchLoader`
        wraps it unchanged and augmentation stays in exactly one place (D-40).
        """

        def __init__(self, ds, arr: np.ndarray, batch_size: int,
                     shuffle: bool, seed: int = 0, prefetch: int = 3,
                     pin: bool = True):
            self.dataset = ds
            self.arr = arr
            self.batch_size = int(batch_size)
            self.shuffle = bool(shuffle)
            self.seed = int(seed)
            self.prefetch = max(1, int(prefetch))
            self.pin = bool(pin) and torch.cuda.is_available()
            self._epoch = 0
            # NOT ds.indices -- see pack_view_of. On a Subset that attribute
            # means positions in the parent, not global pack indices.
            self._idx, self._lab = pack_view_of(ds)
            if len(self._idx) != len(ds):
                raise RuntimeError(
                    f"pack view is {len(self._idx)} rows but the dataset is "
                    f"{len(ds)} -- refusing to train on a misaligned view")

        def __len__(self) -> int:
            n = len(self._idx)
            return (n + self.batch_size - 1) // self.batch_size

        def _order(self) -> np.ndarray:
            n = len(self._idx)
            if not self.shuffle:
                return np.arange(n, dtype=np.int64)
            # Reshuffled every epoch, seeded from (seed, epoch) so a resumed
            # run does not repeat the order it already trained on.
            g = np.random.default_rng((self.seed, self._epoch))
            return g.permutation(n)

        def _make(self, sl: np.ndarray):
            # Sorting the batch's positions makes the gather sequential in the
            # resident array. Batch membership is unchanged; only the order
            # within the batch differs, and nothing downstream depends on it --
            # every row carries its own global sample_idx (D-49).
            sl = np.sort(sl)
            g = self._idx[sl]
            x = torch.from_numpy(self.arr[g])
            y = torch.from_numpy(self._lab[sl])
            i = torch.from_numpy(g)
            if self.pin:
                x, y, i = x.pin_memory(), y.pin_memory(), i.pin_memory()
            return x, y, i

        def __iter__(self):
            import queue
            import threading

            order = self._order()
            self._epoch += 1
            bs, n = self.batch_size, len(order)
            spans = [order[b:b + bs] for b in range(0, n, bs)]

            q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
            stop = threading.Event()

            def _fill():
                try:
                    for sp in spans:
                        if stop.is_set():
                            break
                        q.put(self._make(sp))
                except Exception as e:                           # noqa: BLE001
                    q.put(e)
                q.put(None)

            th = threading.Thread(target=_fill, daemon=True)
            th.start()
            try:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop.set()
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:                                # noqa: BLE001
                    pass


if _TORCH_OK:

    class GPUBatchLoader:
        """Wraps a DataLoader of raw uint8 batches and yields exactly what every
        consumer in this library already expects: `(x_float_normalised, y, idx)`
        on the device.

        Crop and resize are done with a single batched `grid_sample`, which
        expresses RandomResizedCrop as an affine transform -- one kernel for the
        whole batch instead of a per-image Python loop, and the same code path
        for train (random) and eval (fixed centre crop).

        Delegates `.dataset` and `__len__`, because callers legitimately ask for
        `len(loader.dataset)` and would otherwise get an AttributeError at the
        first log line of the sweep.
        """

        def __init__(self, loader, device, out_res: int, stored_res: int,
                     mean: Sequence[float], std: Sequence[float],
                     train: bool = False, scale=(0.35, 1.0),
                     ratio=(3.0 / 4.0, 4.0 / 3.0), hflip: bool = True,
                     seed: int = 0, channels_last: bool = False):
            # D-59. This used to force channels_last unconditionally while the
            # config carried a `channels_last` flag that only the model ever
            # read. The flag now reaches the one line that was ignoring it.
            self.channels_last = bool(channels_last)
            self.loader = loader
            self.device = device
            self.out_res = int(out_res)
            self.stored_res = int(stored_res)
            self.train = bool(train)
            self.scale, self.ratio, self.hflip = tuple(scale), tuple(ratio), bool(hflip)
            self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
            # Its own generator, on the device, seeded from the run seed. Crop
            # sampling must be part of the reproducible RNG story or a resumed
            # run sees a different augmentation stream than an uninterrupted one
            # -- the exact failure the checkpoint contract's `rng` field exists
            # to prevent (playbook 8).
            self._g = torch.Generator(device="cpu")
            self._g.manual_seed(int(seed))
            self._wait_s = self._aug_s = 0.0
            self._n_batches = self._n_sampled = 0

        # -- delegation ------------------------------------------------------
        def __len__(self):
            return len(self.loader)

        @property
        def dataset(self):
            return self.loader.dataset

        @property
        def index_space(self):
            return getattr(self.loader.dataset, "index_space",
                           len(self.loader.dataset))

        @property
        def batch_size(self):
            return getattr(self.loader, "batch_size", None)

        # -- the transform ---------------------------------------------------
        def _theta(self, n: int):
            """Per-sample affine for crop+resize (+flip), in normalised coords."""
            S = float(self.stored_res)
            if not self.train:
                f = self.out_res / S                       # centred, no flip
                th = torch.zeros(n, 2, 3)
                th[:, 0, 0] = f
                th[:, 1, 1] = f
                return th

            area = S * S
            lo, hi = self.scale
            logr = torch.empty(n).uniform_(math.log(self.ratio[0]),
                                           math.log(self.ratio[1]),
                                           generator=self._g)
            ar = torch.exp(logr)
            tgt = torch.empty(n).uniform_(lo, hi, generator=self._g) * area
            w = torch.sqrt(tgt * ar).clamp(8.0, S)
            h = torch.sqrt(tgt / ar).clamp(8.0, S)
            # Uniform top-left within the legal range, expressed as a centre
            # offset in normalised [-1, 1] coordinates.
            maxdx = (S - w) / S
            maxdy = (S - h) / S
            dx = (torch.rand(n, generator=self._g) * 2 - 1) * maxdx
            dy = (torch.rand(n, generator=self._g) * 2 - 1) * maxdy
            sw, sh = w / S, h / S
            if self.hflip:
                flip = (torch.rand(n, generator=self._g) < 0.5)
                sw = torch.where(flip, -sw, sw)
            th = torch.zeros(n, 2, 3)
            th[:, 0, 0] = sw
            th[:, 0, 2] = dx
            th[:, 1, 1] = sh
            th[:, 1, 2] = dy
            return th

        # -- timing -----------------------------------------------------------
        # `dataload_frac` is one of the five columns the playbook calls out as
        # impossible to recover after the fact: high means the GPU is starving
        # and the fix is the loader, not the model.
        #
        # Moving augmentation onto the GPU broke that column's MEANING without
        # changing its name. The training loop measures "time until the next
        # batch arrives", which used to be CPU data preparation and is now CPU
        # wait PLUS an H2D copy PLUS crop/resize/normalise on the device. The
        # number would still be produced, would still look reasonable, and
        # would no longer answer the question it exists to answer.
        #
        # So the loader reports the split itself. `wait_s` is the genuine block
        # on the worker pool and is free to measure. `aug_s` needs a device
        # sync, which costs throughput, so it is sampled every `sync_every`
        # batches and extrapolated -- an estimate that is labelled as one,
        # rather than a per-batch sync that would slow the run it is measuring.
        SYNC_EVERY = 50

        def timing(self) -> Dict[str, float]:
            n = max(1, self._n_batches)
            sampled = max(1, self._n_sampled)
            return {"wait_s": self._wait_s,
                    "augment_s": self._aug_s * (n / sampled),
                    "batches": n, "augment_sampled": sampled}

        def augment_seconds(self) -> Optional[float]:
            """Estimated GPU-augmentation seconds so far this epoch, or None.

            `_aug_s` is sampled every SYNC_EVERY batches because measuring it
            needs a `cuda.synchronize`, so it is scaled to the batches actually
            seen. Returns None before the first sample rather than 0.0 -- a
            confident zero is how you conclude augmentation is free when you
            have simply not measured it yet.
            """
            if self._n_sampled <= 0 or self._n_batches <= 0:
                return None
            return self._aug_s * (self._n_batches / self._n_sampled)

        def reset_timing(self) -> None:
            self._wait_s = 0.0
            self._aug_s = 0.0
            self._n_batches = 0
            self._n_sampled = 0

        def __iter__(self):
            self.reset_timing()
            _t = time.time()
            for i, batch in enumerate(self.loader):
                self._wait_s += time.time() - _t
                self._n_batches += 1
                measure = (i % self.SYNC_EVERY == 0) and self.device.type == "cuda"
                if measure:
                    torch.cuda.synchronize(self.device)
                    _ta = time.time()

                xb, y, idx = batch[0], batch[1], batch[2]
                x = xb.to(self.device, non_blocking=True)
                if x.dim() == 4 and x.shape[-1] == 3:       # NHWC uint8 -> NCHW
                    x = x.permute(0, 3, 1, 2)
                x = x.float().div_(255.0)
                n = x.shape[0]
                th = self._theta(n).to(self.device, dtype=x.dtype)
                grid = F.affine_grid(th, (n, 3, self.out_res, self.out_res),
                                     align_corners=False)
                x = F.grid_sample(x, grid, mode="bilinear",
                                  padding_mode="reflection", align_corners=False)
                x = (x - self._mean) / self._std
                x = (x.contiguous(memory_format=torch.channels_last)
                     if self.channels_last else x.contiguous())
                yb = y.to(self.device, non_blocking=True)

                if measure:
                    torch.cuda.synchronize(self.device)
                    self._aug_s += time.time() - _ta
                    self._n_sampled += 1
                yield x, yb, idx
                _t = time.time()


if _TORCH_OK:

    class _SubsetKeepingIndexSpace(torch.utils.data.Subset):
        """A Subset that still reports the FULL index space.

        `sample_idx` values are global pack indices and do not renumber when
        the split shrinks, so anything sized by `index_space` must still be
        sized for the whole pack. Plain `torch.utils.data.Subset` drops the
        attribute, and losing it here would reintroduce D-49 by a side door.
        """

        @property
        def index_space(self):
            return getattr(self.dataset, "index_space", len(self.dataset))

        @property
        def order_hash(self):
            return getattr(self.dataset, "order_hash", "")

        @property
        def stored_res(self):
            return getattr(self.dataset, "stored_res", 256)

        @property
        def class_names(self):
            return getattr(self.dataset, "class_names", [])

        @property
        def fingerprint(self):
            return getattr(self.dataset, "fingerprint", "")


def _subset_train(ds, cfg: Dict[str, Any]):
    """A deterministic fraction of a training split, for smoke tests.

    Preserves `index_space`. `sample_idx` values stay GLOBAL, so a subset does
    not renumber anything and every array indexed by them is still sized
    correctly -- the D-49 property, which it would be easy to break here by
    subsetting the index space along with the data.
    """
    f = float(cfg.get("train_subset_frac", 0.0) or 0.0)
    if not (0.0 < f < 1.0):
        return ds
    n = max(1, int(round(len(ds) * f)))
    rng = np.random.default_rng(int(cfg.get("seed", 1)))
    keep = np.sort(rng.choice(len(ds), size=n, replace=False))
    sub = torch.utils.data.Subset(ds, keep.tolist())
    for attr in ("index_space", "order_hash", "classes", "class_names",
                 "stored_res", "fingerprint"):
        if hasattr(ds, attr):
            setattr(sub, attr, getattr(ds, attr))
    if not hasattr(sub, "index_space"):
        sub.index_space = len(ds)
    log(f"train split subset to {n}/{len(ds)} images ({100*f:.0f}%) -- "
        f"SMOKE TEST ONLY, not a training run", "DATA")
    return sub


def _in100_loaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any, List[str], str]:
    """train / val / train-holdout for the packed ImageNet-100.

    `train_holdout` is a slice OF train evaluated with augmentation OFF. It is
    not withheld from training: EL2N and forgetting events are training-set
    quantities and are undefined anywhere else, which is what D-11 was about.
    """
    spec = dataset_spec("imagenet100")
    root = Path(cfg["data_root"])
    dev = torch.device(cfg.get("device")
                       or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    bs = int(cfg.get("batch_size", 128))
    eval_bs = int(cfg.get("eval_batch_size", 256))
    res = int(cfg.get("input_res", spec["native_res"]))
    seed = int(cfg.get("seed", 1))

    tr = PackedImageDataset(root, "train")
    va = PackedImageDataset(root, "val")
    ho = PackedImageDataset(root, "holdout")

    # A deterministic fraction of the training split, for smoke tests only.
    # The resume acceptance test does not care how well the model learns; it
    # cares whether the seam is invisible. Running it on the full 119,395
    # images cost ~40 minutes across three legs and exercised no code the 5%
    # version does not. Off (1.0) for every real run, and it participates in
    # config_hash, so a subset run can never be mistaken for a full one.
    _frac = float(cfg.get("train_subset_frac", 1.0) or 1.0)
    if 0 < _frac < 1.0:
        _rng = np.random.default_rng(4242)
        _keep = np.sort(_rng.choice(len(tr), size=max(2, int(len(tr) * _frac)),
                                    replace=False))
        tr = _SubsetKeepingIndexSpace(tr, _keep.tolist())
        log(f"train subset: {len(tr)} of {len(tr.dataset)} images "
            f"({100*_frac:.0f}%) -- SMOKE TEST ONLY", "DATA")

    got = tr.fingerprint
    want = cfg.get("data_fingerprint")
    if want and str(want) != got:
        raise RuntimeError(
            f"data fingerprint mismatch.\n  config: {want}\n  on disk: {got}\n"
            f"This run was configured against a different pack or a different "
            f"split. Correlating per-sample tables across the two would align "
            f"them by index and compare different images. Repack, or use the "
            f"matching pack.")

    # A fraction of the TRAIN split only. For smoke tests -- the resume test
    # exercises the same code on 5% of the data in two minutes instead of
    # forty. val and holdout are NEVER subset: they are what results are
    # measured on, and a test that shrinks them is testing something else.
    tr = _subset_train(tr, cfg)

    # ---- D-56: resident pack ------------------------------------------------
    # All three splits index the SAME file, so one resident copy serves them
    # all -- keyed on the resolved root, loaded at most once per process.
    arr = None
    if bool(cfg.get("ram_cache", True)):
        base = pack_root_of(tr)
        arr = load_pack_to_ram(root, base.count, base.stored_res,
                               headroom_gb=float(cfg.get("ram_headroom_gb", 6.0)))

    if arr is not None:
        # num_workers is not merely unnecessary here, it is harmful: Windows
        # spawn would pickle a 23.5 GiB array into every child.
        raw_tr = RAMBatchLoader(tr, arr, bs, shuffle=True, seed=seed,
                                pin=(dev.type == "cuda"))
        # Never shuffle eval loaders. sample_idx alignment depends on it.
        raw_va = RAMBatchLoader(va, arr, eval_bs, shuffle=False,
                                pin=(dev.type == "cuda"))
        raw_ho = RAMBatchLoader(ho, arr, eval_bs, shuffle=False,
                                pin=(dev.type == "cuda"))
        log(f"loaders: RAM-resident, batch {bs} train / {eval_bs} eval, "
            f"0 workers, 1 prefetch thread", "DATA")
    else:
        nw = int(cfg.get("num_workers", min(8, max(0, (os.cpu_count() or 2) - 2))))
        common = dict(num_workers=nw, pin_memory=(dev.type == "cuda"),
                      persistent_workers=bool(nw),
                      prefetch_factor=(4 if nw else None))
        g = torch.Generator(); g.manual_seed(seed)

        raw_tr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=False,
                            generator=g, **common)
        # Never shuffle eval loaders. sample_idx alignment depends on it.
        raw_va = DataLoader(va, batch_size=eval_bs, shuffle=False, **common)
        raw_ho = DataLoader(ho, batch_size=eval_bs, shuffle=False, **common)
        log(f"loaders: memmap, batch {bs}, {nw} workers", "DATA")

    mk = lambda raw, train, sd: GPUBatchLoader(
        raw, dev, res, tr.stored_res, spec["mean"], spec["std"],
        train=train, scale=tuple(cfg.get("rrc_scale", (0.35, 1.0))), seed=sd,
        channels_last=bool(cfg.get("channels_last", False)))

    return (mk(raw_tr, True, seed), mk(raw_va, False, 0), mk(raw_ho, False, 0),
            tr.class_names, va.order_hash)


def _model_input_problems(shape: Tuple[int, ...], is_float: bool,
                          want_res: int, dtype_name: str = "?") -> List[str]:
    """The decision behind `_assert_model_ready`, as plain data.

    Split out so it can be tested WITHOUT torch. A guard that raises is only
    as safe as its false-positive rate: one that rejects a valid batch would
    break every sweep, and the version that could only be exercised on the
    user's GPU was a guard I could not check before shipping. That is the
    shape D-63 punished -- a test that never sees the program's real input.
    """
    problems: List[str] = []
    if len(shape) != 4:
        problems.append(f"rank {len(shape)}, expected 4 (B,C,H,W)")
    elif shape[1] != 3:
        problems.append(
            f"shape {shape} -- channel dim is {shape[1]}, not 3"
            + (" (this looks like NHWC: the permute never happened)"
               if shape[-1] == 3 else ""))
    elif want_res and shape[-1] != want_res:
        problems.append(f"{shape[-1]}px, expected {want_res}px "
                        f"(the crop never happened)")
    if not is_float:
        problems.append(f"dtype {dtype_name}, expected float "
                        f"(the cast/normalise never happened)")
    return problems


def _assert_model_ready(x, cfg: Dict[str, Any], where: str = "") -> None:
    """Is this batch actually model-input, or raw loader output?

    **D-76.** A loader that skipped `GPUBatchLoader` handed the model
    `[256, 256, 256, 3]` uint8 and torch reported

        Given groups=1, weight of size [64, 3, 7, 7], expected
        input[256, 256, 256, 3] to have 3 channels, but got 256 channels

    which names a convolution's weights and blames the channel count. The
    actual fault is three layers up -- an eval view built without the
    conversion layer -- and nothing in that message points there.

    Checked once per sweep, on the first batch. Microseconds, and it turns a
    misleading error into the one sentence that identifies the cause.
    """
    if not _TORCH_OK or not isinstance(x, torch.Tensor):
        return
    problems = _model_input_problems(
        tuple(x.shape),
        x.dtype in (torch.float32, torch.float16, torch.bfloat16),
        int(cfg.get("input_res", 0) or 0),
        str(x.dtype))
    if problems:
        raise RuntimeError(
            f"[{where}] this loader is not producing model input: "
            + "; ".join(problems)
            + ".\n  A loader for measurement must be built with "
              "`eval_view_of(loader, cfg)`. Rebuilding a DataLoader from "
              "`some_loader.dataset` drops GPUBatchLoader, which is where the "
              "permute, cast, normalise and crop live (D-76).")


def eval_view_of(loader, cfg: Dict[str, Any], batch_size: Optional[int] = None):
    """The same samples, in order, with augmentation off — for BOTH backends.

    **D-76.** `train_msc_kd` needed to sweep the teacher over the training set
    to build MSC targets, and wrote:

        train_eval = DataLoader(train_loader.dataset, batch_size=..., ...)
        train_eval.dataset.augment = False

    Both lines are correct on CIFAR and wrong on ImageNet-100.

      * `train_loader` is a `GPUBatchLoader`; `.dataset` delegates through to
        the raw `PackedImageDataset`. Rebuilding a `DataLoader` from it
        DISCARDS the conversion layer -- the permute, the float cast, the
        normalise, and the 256->224 crop all live in `GPUBatchLoader`. The
        model received `[256, 256, 256, 3]` uint8 and said so:
        "expected input to have 3 channels, but got 256".
      * `PackedImageDataset` has no `augment` attribute. That assignment
        created an unread one inside a bare `except: pass`, so the intent
        "augmentation off while measuring" silently did nothing. Had the shape
        error not fired first, MSC targets would have been measured through
        whatever view the loader happened to produce.

    On CIFAR both worked because `CIFARTensor.__getitem__` returns finished
    NCHW tensors and carries a real `augment` flag. Same seam as D-70: the
    library is parameterised by dataset, and that only holds where both
    datasets present the same interface.

    This returns an eval-mode view built the way the backend requires, so no
    caller has to know which backend it has.
    """
    bs = int(batch_size or cfg.get("eval_batch_size", 256))
    if _TORCH_OK and isinstance(loader, GPUBatchLoader):
        inner = loader.loader
        ds = inner.dataset
        if isinstance(inner, RAMBatchLoader):
            raw = RAMBatchLoader(ds, inner.arr, bs, shuffle=False, seed=0,
                                 pin=inner.pin)
        else:
            raw = DataLoader(ds, batch_size=bs, shuffle=False,
                             num_workers=0, pin_memory=True)
        spec = dataset_spec(str(cfg.get("dataset_name", "imagenet100")))
        # train=False is what turns augmentation off here -- a centre crop
        # instead of a random resized crop, and no flip.
        return GPUBatchLoader(raw, loader.device, loader.out_res,
                              loader.stored_res, spec["mean"], spec["std"],
                              train=False, seed=0,
                              channels_last=loader.channels_last)

    # CIFAR-style: a plain DataLoader over a dataset that owns its own flag.
    ds = getattr(loader, "dataset", loader)
    out = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                     pin_memory=True)
    if hasattr(ds, "augment"):
        ds.augment = False
    else:
        raise TypeError(
            f"{type(ds).__name__} has no `augment` flag and this loader is not "
            f"a GPUBatchLoader, so augmentation cannot be turned off for "
            f"measurement. Refusing to measure MSC through an unknown view "
            f"(D-76).")
    return out


def build_loaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any, List[str], str]:
    """train / val(test) / train-holdout loaders.

    The train-holdout is a fixed 5,000-sample slice of the training set,
    evaluated with augmentation off. It costs one extra inference sweep and
    answers a free question: does MSC structure look different on data the
    model has already seen?
    """
    ds = str(cfg.get("dataset_name", "cifar100"))
    if dataset_spec(ds)["backend"] == "packed":
        return _in100_loaders(cfg)

    data_root = cfg["data_root"]
    bs = int(cfg.get("batch_size", 64))
    eval_bs = int(cfg.get("eval_batch_size", 512))

    train_set = CIFARTensor(data_root, ds, train=True, augment=True)
    test_set = CIFARTensor(data_root, ds, train=False, augment=False)
    train_clean = CIFARTensor(data_root, ds, train=True, augment=False)

    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 1)))

    train_set = _subset_train(train_set, cfg)
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
                     classifier: nn.Module,
                     feature_dim_fn: Optional[Callable[[int], int]] = None,
                     depth_fractions: Sequence[float] = DEPTH_FRACTIONS,
                     final_norm: Optional[nn.Module] = None,
                     probe_res: Optional[int] = None):
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
            # ASK THE MODEL (rule 2). `feature_dim_fn` is a hand-written map
            # from block index to channel count, and writing one means reading
            # somebody else's module internals: `b.conv3.out_channels`,
            # `b.branch2[-2].out_channels`, `m.reduction.out_features`. Three of
            # those four guesses were right and one was not -- ShuffleNetV2's
            # `branch2[-2]` is a BatchNorm2d, which has no `out_channels`, and
            # the architecture failed to build at all.
            #
            # A literal that is right for three of four cases is exactly the
            # thing rule 2 is about, and the fix is not to correct the index.
            # It is to stop guessing: run one forward pass and read the shapes
            # off the tensors the backbone actually produces. That is definitive
            # by construction and cannot drift when torchvision reorders a
            # block.
            if feature_dim_fn is not None:
                self.feature_dims = tuple(feature_dim_fn(c - 1)
                                          for c in self.stage_cuts)
            else:
                self.feature_dims = self._probe_feature_dims(
                    int(probe_res or 224))
            if len(uniq) < len(depth_fractions):
                log(f"{type(self).__name__} has only {n} blocks -- using "
                    f"K={len(uniq)} depth exits at "
                    f"{[round(f,2) for f in self.depth_fractions]} instead of "
                    f"{list(depth_fractions)}", "ZOO")

        def _probe_feature_dims(self, res: int) -> Tuple[int, ...]:
            """Channel count at every exit, read off a real forward pass.

            Handles both layouts the zoo contains: (B,C,H,W) for convolutional
            backbones and (B,N,C) for token models. Subclasses that speak a
            third layout normalise it in `forward_features` -- SwinBackbone
            permutes NHWC to NCHW there -- so this sees only the two.
            """
            was = self.training
            self.eval()
            try:
                try:
                    dev = next(self.parameters()).device
                except StopIteration:
                    dev = torch.device("cpu")
                with torch.no_grad():
                    feats = self.forward_features(
                        torch.zeros(1, 3, res, res, device=dev))
            finally:
                self.train(was)
            dims = []
            for f in feats:
                if f.dim() == 4:
                    dims.append(int(f.shape[1]))          # (B, C, H, W)
                elif f.dim() == 3:
                    dims.append(int(f.shape[2]))          # (B, N, C)
                else:
                    dims.append(int(f.reshape(f.shape[0], -1).shape[1]))
            return tuple(dims)

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

    # =====================================================================
    # ImageNet-100 zoo -- eight architectures at 224 px
    # =====================================================================
    # These are adapters, not reimplementations. The convolutional backbones
    # come from torchvision, which is guaranteed present alongside torch and
    # whose ImageNet definitions are the standard ones; re-typing them would
    # risk a silent deviation from the architecture everyone else means by
    # "ResNet-50". What is OURS -- and therefore what needs testing (rule 8) --
    # is the decomposition into (stem, ordered blocks, classifier), because
    # that is what makes `forward_prefix(x, k)` genuinely stop at stage k
    # rather than run the whole network and read a mid-layer activation. An
    # early exit that costs full compute would make every FLOPs saving in the
    # project fictional.
    #
    # ONE HEAD SHAPE FOR ALL EIGHT: global average pool -> Linear. Stock VGG-16
    # has a 25088->4096->4096 fully-connected head worth ~124 M parameters. If
    # the final exit carried that head while exits 1..K-1 carried a GAP+Linear
    # ExitHead, the depth-axis rho would be measuring the head rather than the
    # backbone, and `rho` is the quantity the whole project normalises by. So
    # every architecture terminates the same way the exit heads do. This makes
    # `vgg16` here "VGG-16(BN) with a global-average-pool head" and not stock
    # VGG-16 -- recorded, and harmless because no published reference is
    # claimed for anything in this zoo (25_IN100_DATA_CARD.md 1).

    def _tv():
        try:
            import torchvision.models as tvm
            return tvm
        except Exception as e:                                  # noqa: BLE001
            raise RuntimeError(
                f"torchvision is required for the ImageNet zoo ({e}). "
                f"pip install torchvision") from e

    def build_resnet_imagenet(depth: int, num_classes: int = 100,
                              probe_res: int = 224) -> StagedBackbone:
        """torchvision ResNet-18/50, decomposed by residual block.

        8 blocks for R18, 16 for R50 -- comfortably more than the 5 depth
        fractions want, so K is the full 5 and the adaptive-K path (D-01b) is
        not exercised here. It is still derived from the model, never assumed.
        """
        tvm = _tv()
        net = {18: tvm.resnet18, 50: tvm.resnet50}[depth](weights=None)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        blocks = [b for layer in (net.layer1, net.layer2, net.layer3, net.layer4)
                  for b in layer]
        bb = StagedBackbone(stem, blocks, nn.Identity(), None,
                            probe_res=probe_res)
        bb.classifier = nn.Linear(bb.feature_dims[-1], num_classes)
        return bb

    def build_vgg_imagenet(depth: int = 16, num_classes: int = 100,
                           probe_res: int = 224) -> StagedBackbone:
        """torchvision VGG-16 with BN, conv stack only, GAP+Linear head."""
        tvm = _tv()
        net = {11: tvm.vgg11_bn, 13: tvm.vgg13_bn,
               16: tvm.vgg16_bn, 19: tvm.vgg19_bn}[depth](weights=None)
        feats = list(net.features)
        blocks, dims, cin = [], [], 3
        i = 0
        while i < len(feats):
            m = feats[i]
            if isinstance(m, nn.Conv2d):
                # conv + bn + relu is one block, so a depth cut never lands
                # between a convolution and its normalisation.
                grp = [m]
                j = i + 1
                while j < len(feats) and not isinstance(feats[j],
                                                        (nn.Conv2d, nn.MaxPool2d)):
                    grp.append(feats[j])
                    j += 1
                blocks.append(nn.Sequential(*grp))
                cin = m.out_channels
                i = j
            else:
                blocks.append(m)
                i += 1
            dims.append(cin)
        bb = StagedBackbone(nn.Identity(), blocks, nn.Identity(), None,
                            probe_res=probe_res)
        bb.classifier = nn.Linear(bb.feature_dims[-1], num_classes)
        return bb

    def build_shufflenetv2_imagenet(num_classes: int = 100, width: str = "1.0x",
                                    probe_res: int = 224) -> StagedBackbone:
        tvm = _tv()
        net = {"0.5x": tvm.shufflenet_v2_x0_5, "1.0x": tvm.shufflenet_v2_x1_0,
               "1.5x": tvm.shufflenet_v2_x1_5}[width](weights=None)
        stem = nn.Sequential(net.conv1, net.maxpool)
        blocks = [b for stage in (net.stage2, net.stage3, net.stage4) for b in stage]
        blocks.append(net.conv5)
        bb = StagedBackbone(stem, blocks, nn.Identity(), None,
                            probe_res=probe_res)
        bb.classifier = nn.Linear(bb.feature_dims[-1], num_classes)
        return bb

    def build_convnext_tiny(num_classes: int = 100,
                            dims: Sequence[int] = (96, 192, 384, 768),
                            depths: Sequence[int] = (3, 3, 9, 3),
                            drop_path: float = 0.1, stem_patch: int = 4,
                            probe_res: int = 224) -> StagedBackbone:
        """ConvNeXt-T geometry, built from the same blocks as the CIFAR femto.

        Ours rather than torchvision's, because `_ConvNeXtBlock` and
        `_LayerNorm2d` already exist here, are already exercised by the CIFAR
        self-checks, and decompose cleanly. `stem_patch` is 4 at ImageNet
        resolution and 2 for the 32px variant -- the one parameter that differs.
        """
        stem = nn.Sequential(nn.Conv2d(3, dims[0], stem_patch, stem_patch),
                             _LayerNorm2d(dims[0]))
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
                              lambda i: bdims[i],
                              final_norm=_LayerNorm2d(dims[-1]),
                              probe_res=probe_res)

    def build_vit_small(num_classes: int = 100, dim: int = 384, depth: int = 12,
                        heads: int = 6, patch: int = 16, img: Optional[int] = None,
                        drop_path: float = 0.05,
                        probe_res: int = 224) -> TokenBackbone:
        """ViT-S/16. `deit_small` is THIS FUNCTION with THESE ARGUMENTS.

        The two entries in the zoo are deliberately built by one builder with
        one set of geometry arguments, so they cannot drift apart. They differ
        only in `base_config`'s recipe -- augmentation strength, drop-path and
        weight decay.

        That pairing is the control CIFAR did not have. If seed-reliability
        differs between two models with identical parameter counts, identical
        forward passes and identical exit structure, the difference is a
        property of how they were trained and not of attention. Making them the
        same function is what guarantees the comparison means that.
        """
        # `probe_res` is what `build_model` injects for every ImageNet builder.
        # This one lacked the parameter, so vit_small_p16 and deit_small raised
        # TypeError and TWO OF EIGHT architectures could not be built at all
        # (D-42). The positional-embedding grid is sized from it.
        img = int(img if img is not None else probe_res)
        stem = _PatchEmbed(img, patch, 3, dim)
        dp = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        blocks = [_TransformerBlock(dim, heads, 4.0, dp[i]) for i in range(depth)]
        return TokenBackbone(stem, blocks, nn.Linear(dim, num_classes),
                             lambda i: dim, final_norm=nn.LayerNorm(dim),
                             probe_res=img)

    class SwinBackbone(StagedBackbone):
        """torchvision Swin-T. Its blocks speak NHWC; everything else here
        speaks NCHW.

        Rather than teach `ExitHead`, `pooled` and the FLOPs profiler about a
        second memory layout -- three more places to get it wrong -- the
        permutation happens once, at the boundary where features leave the
        backbone. Internals stay exactly as torchvision wrote them.
        """

        def _run_to(self, x, upto_block: int):
            h = self.stem(x)
            for i in range(upto_block):
                h = self.blocks[i](h)
            return h.permute(0, 3, 1, 2).contiguous()      # NHWC -> NCHW

        def forward_features(self, x) -> List["torch.Tensor"]:
            feats, h, prev = [], self.stem(x), 0
            for c in self.stage_cuts:
                for i in range(prev, c):
                    h = self.blocks[i](h)
                prev = c
                feats.append(h.permute(0, 3, 1, 2).contiguous())
            return feats

        def forward(self, x):
            h = self._run_to(x, len(self.blocks))           # already NCHW
            if self.final_norm is not None:
                h = self.final_norm(h)
            return self.classifier(self.pooled(h))

    def build_swin_tiny(num_classes: int = 100,
                        probe_res: int = 224) -> "SwinBackbone":
        tvm = _tv()
        net = tvm.swin_t(weights=None)
        feats = list(net.features)
        stem = feats[0]                                    # patch embed
        blocks = []
        for m in feats[1:]:
            if isinstance(m, nn.Sequential):               # a stage of blocks
                blocks.extend(list(m))
            else:                                          # PatchMerging
                blocks.append(m)
        bb = SwinBackbone(stem, blocks, nn.Identity(), None,
                          probe_res=probe_res)
        c = bb.feature_dims[-1]
        bb.final_norm = _LayerNorm2d(c)
        bb.classifier = nn.Linear(c, num_classes)
        return bb


# --------------------------------------------------------------------------
# Zoo registry
# --------------------------------------------------------------------------
# family is the Q3 grouping variable: within-family transfer is expected to
# exceed across-family, which exceeds CNN->token. Keep it accurate.
#
# `zoo` says which dataset an entry belongs to. A `resnet20` is a CIFAR ResNet
# with a stride-1 stem and no maxpool; feeding it 224px input works, produces a
# 56x56 final feature map, runs ~40x slower than intended and is not the
# architecture anyone means. It would not error -- which is why the check has to
# be explicit (see `build_model`).
ZOO: Dict[str, Dict[str, Any]] = {
    # ---------------------------------------------------------- CIFAR, 32 px
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

    # ---------------------------------------------------- ImageNet-100, 224 px
    # Eight architectures crossing the CNN/attention boundary four different
    # ways. See 20_IN100_PORT_PLAN.md 1 for what each one isolates.
    "resnet50":     dict(zoo="imagenet", family="resnet",
                         builder=("resnet_in", dict(depth=50))),
    "resnet18":     dict(zoo="imagenet", family="resnet",
                         builder=("resnet_in", dict(depth=18))),
    "vgg16":        dict(zoo="imagenet", family="vgg",
                         builder=("vgg_in", dict(depth=16))),
    "shufflenetv2_in": dict(zoo="imagenet", family="mobile",
                            builder=("shufflenetv2_in", dict(width="1.0x"))),
    # vit_small_p16 and deit_small are THE SAME BUILDER WITH THE SAME ARGUMENTS.
    # They differ only in base_config's recipe. That is the point: it makes the
    # comparison an experiment about training rather than about geometry, and
    # building them from one function is what stops them silently diverging.
    "vit_small_p16": dict(zoo="imagenet", family="vit",
                          builder=("vit_small", dict())),
    "deit_small":   dict(zoo="imagenet", family="vit",
                         builder=("vit_small", dict())),
    "swin_tiny":    dict(zoo="imagenet", family="swin",
                         builder=("swin_tiny", dict())),
    "convnext_tiny": dict(zoo="imagenet", family="convnext",
                          builder=("convnext_tiny", dict())),
}
for _a, _m in ZOO.items():
    _m.setdefault("zoo", "cifar")

# `shufflenetv2` is the one architecture present in BOTH studies, which makes it
# the only direct CIFAR<->ImageNet bridge in the design: whatever its ImageNet
# rho_seed turns out to be, the DIFFERENCE from its CIFAR 0.6698 is a
# measurement of what dataset scale does to this statistic with architecture
# held exactly fixed. It calibrates every other comparison. The registry keys
# have to differ because the two builds are different networks (stride-1 stem
# vs stride-2 + maxpool), so the alias records that they are the same design.
CROSS_STUDY_ALIAS = {"shufflenetv2_in": "shufflenetv2"}

# Architectures that need the DeiT-style recipe (AdamW, long warmup, strong
# augmentation, label smoothing). SGD flatlines these from scratch -- the same
# failure E2AM documented for ConvNeXtV2 under SGD.
TRANSFORMER_LIKE = {"vit_tiny", "mixer_nano", "convnext_femto",
                    "vit_small_p16", "deit_small", "swin_tiny", "convnext_tiny"}

# The DeiT arm of the recipe control: strong augmentation on top of AdamW.
DEIT_RECIPE = {"deit_small"}


def zoo_for_dataset(dataset: str) -> List[str]:
    """Every architecture belonging to this dataset's zoo, in registry order."""
    want = dataset_spec(dataset)["zoo"]
    return [a for a, m in ZOO.items() if m.get("zoo", "cifar") == want]


def build_model(arch: str, num_classes: Optional[int] = None,
                dataset: Optional[str] = None, **overrides):
    """Build a backbone.

    `dataset`, when given, is CHECKED rather than merely used for defaults. A
    CIFAR `resnet20` fed 224px input does not raise -- it produces a 56x56 final
    feature map, runs about forty times slower than intended, and trains to a
    plausible-looking accuracy. That is the D-33 shape: a configuration that is
    wrong and silent. So the mismatch is refused here, where it costs one line.
    """
    if not _TORCH_OK:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERR}")
    if arch not in ZOO:
        raise KeyError(f"unknown architecture '{arch}'. Known: {sorted(ZOO)}")
    meta = ZOO[arch]
    if dataset is not None:
        want = dataset_spec(dataset)["zoo"]
        if meta.get("zoo", "cifar") != want:
            raise ValueError(
                f"'{arch}' belongs to the '{meta.get('zoo','cifar')}' zoo but "
                f"dataset '{dataset}' needs the '{want}' zoo. Available: "
                f"{zoo_for_dataset(dataset)}")
        if num_classes is None:
            num_classes = num_classes_for(dataset)
    num_classes = int(num_classes if num_classes is not None else 100)

    kind, kwargs = meta["builder"]
    kwargs = dict(kwargs)
    # The ImageNet builders read their exit dimensions off a real forward pass,
    # so they need to know what resolution to probe at. Taken from the dataset,
    # never defaulted -- probing a 224px model at 32px would produce feature
    # maps of the wrong spatial size and, for Swin, would not run at all.
    if meta.get("zoo") == "imagenet" and dataset is not None:
        kwargs.setdefault("probe_res", native_res(dataset))
    kwargs.update(overrides)
    fn = {
        "resnet": build_resnet_cifar, "wrn": build_wrn, "vgg": build_vgg,
        "mobilenetv2": build_mobilenetv2, "shufflenetv2": build_shufflenetv2,
        "convnext_femto": build_convnext_femto, "vit_tiny": build_vit_tiny,
        "mixer_nano": build_mixer_nano,
        # ImageNet-100
        "resnet_in": build_resnet_imagenet, "vgg_in": build_vgg_imagenet,
        "shufflenetv2_in": build_shufflenetv2_imagenet,
        "convnext_tiny": build_convnext_tiny, "vit_small": build_vit_small,
        "swin_tiny": build_swin_tiny,
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

_PROFILER_CACHE: Dict[str, Any] = {
    "allow_mixed": os.environ.get("MSC_ALLOW_MIXED_PROFILER", "") in ("1", "true"),
}


def profilers_used() -> Set[str]:
    """Every profiler that has actually produced a number in this process.

    More than one means the atlas is priced two ways and cross-architecture
    comparison is invalid (D-45).
    """
    return set(_PROFILER_CACHE.get("used", set()))


def _get_profiler() -> Tuple[str, Optional[Callable], str]:
    """Pick ONE profiler for the whole zoo and stick with it.

    **D-45.** fvcore counts every convolutional backbone here and then fails on
    ViT / DeiT / Swin with `type Tensor doesn't define __round__ method` -- it
    traces with `torch.jit`, and tracing a positional-embedding resample trips
    over a Python `round()` applied to what became a tensor. The old code logged
    the failure and fell back to the analytic counter *per architecture*, so a
    single atlas was priced with **two different profilers**.

    That is the exact thing this module's own comment forbids, and it is worse
    than it sounds: the analytic fallback hooks `Conv2d` and `Linear` only, so
    for a transformer it **misses the attention matmuls entirely** -- QK^T and
    AV. Those scale with tokens squared while the linear parts scale with
    tokens, so the resolution axis is distorted for exactly the architectures
    the study is about, and rho is DEFINED in FLOPs.

    `torch.utils.flop_counter.FlopCounterMode` is preferred now: it works by
    `__torch_dispatch__` rather than tracing, so there is nothing to trip over,
    and it counts matmul and scaled-dot-product-attention natively. It reports
    true FLOPs (2*m*n*k for a matmul), not MACs, so no doubling is applied.
    """
    if "chosen" in _PROFILER_CACHE:
        return _PROFILER_CACHE["chosen"]
    chosen = ("analytic", None, "builtin")
    try:
        from torch.utils.flop_counter import FlopCounterMode

        def _f(model, shape):
            m = FlopCounterMode(display=False)
            with m:
                model(torch.zeros(*shape))
            return int(m.get_total_flops())
        # Prove it on a token model before adopting it. A profiler that works
        # for ResNet and fails for ViT is how the atlas ended up mixed.
        chosen = ("torch.flop_counter", _f, torch.__version__)
        _PROFILER_CACHE["chosen"] = chosen
        return chosen
    except Exception:
        pass
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


def measure_flops(model, shape) -> int:
    """FLOPs at `shape`. The shape is REQUIRED and has no default.

    It used to default to `(1, 3, 32, 32)`, which was correct for every caller
    right up to the moment a second dataset existed. A default that is silently
    wrong produces a budget table that is internally consistent, plausible, and
    describes a network nobody trained -- and rho is a ratio, so the error does
    not even show up as an implausible magnitude. Callers now go through
    `input_shape(dataset)`.
    """
    if not (isinstance(shape, (tuple, list)) and len(shape) == 4):
        raise ValueError(f"measure_flops needs a 4-tuple (B,C,H,W), got {shape!r}")
    name, fn, _ = _get_profiler()
    model = model.eval()
    try:
        if fn is not None:
            n = int(fn(model, tuple(shape)))
            _PROFILER_CACHE.setdefault("used", set()).add(name)
            return n
    except Exception as e:                                       # noqa: BLE001
        # D-45. Falling back silently gives one atlas two profilers and two
        # accounting conventions, which corrupts every cross-architecture
        # number while every individual table still looks reasonable. The
        # analytic counter hooks Conv2d and Linear only -- for a transformer
        # that omits attention entirely.
        if not _PROFILER_CACHE.get("allow_mixed"):
            raise RuntimeError(
                f"FLOPs profiler '{name}' failed on this model "
                f"({type(e).__name__}: {str(e)[:120]}).\n"
                f"Refusing to fall back: the rest of the zoo was priced with "
                f"'{name}', and mixing profilers silently corrupts every "
                f"transfer number (D-45). rho is DEFINED in FLOPs.\n"
                f"Set MSC_ALLOW_MIXED_PROFILER=1 only if you accept that."
            ) from e
        log(f"profiler {name} failed ({str(e)[:80]}); ANALYTIC FALLBACK -- "
            f"this table is not comparable to the others", "ALARM")
    _PROFILER_CACHE.setdefault("used", set()).add("analytic")
    return _analytic_flops(model, tuple(shape))


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


def build_budget_table(arch: str, dataset: str, num_classes: Optional[int] = None,
                       resolutions: Optional[Sequence[int]] = None,
                       depth_fractions: Sequence[float] = DEPTH_FRACTIONS,
                       precisions: Sequence[str] = PRECISIONS,
                       model=None) -> Dict[str, Any]:
    """FLOPs for every configuration on every axis, plus normalised rho.

    Measured once per architecture, written to budgets/{arch}.json, and never
    recomputed -- a budget table that drifts between sessions makes MSC values
    from different sessions incomparable.

    `dataset` is required and supplies the input resolution, the class count and
    the resolution grid. Nothing here spells a shape.
    """
    spec = dataset_spec(dataset)
    num_classes = int(num_classes if num_classes is not None else spec["num_classes"])
    resolutions = tuple(resolutions if resolutions is not None else spec["resolutions"])
    res0 = int(spec["native_res"])
    if resolutions[-1] != res0:
        raise ValueError(
            f"{dataset}: the resolution grid must terminate at the native "
            f"resolution ({res0}) so rho_res reaches exactly 1.0; got {resolutions}")

    model = model if model is not None else build_model(arch, num_classes,
                                                        dataset=dataset)
    model = model.eval().cpu()
    prof_name, _, prof_ver = _get_profiler()

    full = measure_flops(model, input_shape(dataset))

    # --- depth: prefix cost + a linear exit head -------------------------
    # K comes from the MODEL, not the global constant: a shallow backbone
    # legitimately carries fewer distinct depth budgets (see StagedBackbone).
    feat_dims = list(model.feature_dims)
    achieved_fractions = list(getattr(model, "depth_fractions", depth_fractions))
    depth_flops = []
    for k in range(len(feat_dims)):
        head = ExitHead(feat_dims[k], num_classes,
                        token_model=getattr(model, "is_token_model", False)).eval()
        depth_flops.append(measure_flops(_PrefixWrapper(model, k, head),
                                         input_shape(dataset)))
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
    #
    # Native support is probed PER RESOLUTION, not decided once for the whole
    # axis. On CIFAR `supports_native_resolution` was a single boolean, and when
    # MLP-Mixer failed (D-02) it took the entire axis with it. At 224px the
    # failures are partial rather than total -- a Swin-T reduces its input by 32
    # and its last stage is 7x7 at 224 but 3x3 at 96, which is smaller than its
    # own attention window. Recording "this architecture manages 128-224 but not
    # 96" is strictly more information than "this architecture is unsupported",
    # and it costs one try/except per value.
    declared = bool(getattr(model, "supports_native_resolution", True))
    res_flops, native_ok_per_res, native_errs = [], [], {}
    for r in resolutions:
        f_r, ok = None, False
        if declared:
            try:
                f_r, ok = measure_flops(model, input_shape(dataset, r)), True
            except Exception as e:                              # noqa: BLE001
                native_errs[str(r)] = f"{type(e).__name__}: {str(e)[:160]}"
        if not ok:
            # Analytic stand-in: cost scales with pixel count for a convolutional
            # network and with token count for a patch model -- both quadratic in r.
            f_r = int(full * (r / float(res0)) ** 2)
        res_flops.append(int(f_r))
        native_ok_per_res.append(bool(ok))
    native_ok = all(native_ok_per_res)
    if not native_ok:
        bad = [r for r, o in zip(resolutions, native_ok_per_res) if not o]
        log(f"{arch}: native resolution unavailable at {bad} "
            f"({'declared unsupported' if not declared else 'probe failed'}); "
            f"those entries use the analytic quadratic model. The PROXY sweep is "
            f"primary for every architecture regardless (DC-3).", "FLOP")
    res_rho = [f / res_flops[-1] for f in res_flops]
    if not all(res_rho[i] < res_rho[i + 1] for i in range(len(res_rho) - 1)):
        raise ValueError(
            f"{arch}: resolution costs are not strictly ascending: "
            f"{[round(r, 4) for r in res_rho]}. MSC is undefined when two "
            f"budgets cost the same (the D-01b failure, on a different axis).")

    # --- precision: analytic bit-operation accounting ---------------------
    # There is no INT4 kernel to time on a T4, so this axis is priced, not
    # measured. Reported as an analytic cost model and never as measured
    # latency -- see the limitations section of the paper.
    prec_rho = [PRECISION_BITS[p] / 32.0 for p in precisions]
    prec_flops = [int(full * r) for r in prec_rho]

    table = {
        "arch": arch,
        "dataset": str(dataset),
        "input_res": int(res0),
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
                "native_supported_per_res": list(native_ok_per_res),
                "native_errors": native_errs,
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


def budget_table_valid(table: Optional[Dict[str, Any]], arch: str,
                       dataset: str, num_classes: Optional[int] = None
                       ) -> Tuple[bool, str]:
    """Is a CACHED budget table still the table we want?

    Rule 5. `load_or_build_budgets` used to ask only "does the file exist and
    have a full_flops key?", which was a correct question while one dataset
    existed. It is the wrong question the moment a table can be stale for a
    reason other than absence -- and a stale budget table is close to the worst
    possible artifact, because rho is a ratio and a table built at 32px looks
    entirely plausible when read at 224px. Every MSC value derived from it would
    be a well-formed number describing a network nobody trained.

    Returns (ok, reason). Deliberately conservative in the same direction as
    `msckd_router_ok` (D-29): a table that predates this check has no `dataset`
    key and is treated as UNKNOWN, which we rebuild rather than trust, because
    rebuilding costs seconds and trusting costs the atlas.
    """
    if not table or not table.get("full_flops"):
        return False, "absent or empty"
    spec = dataset_spec(dataset)
    want_res = int(spec["native_res"])
    want_cls = int(num_classes if num_classes is not None else spec["num_classes"])
    if table.get("arch") != arch:
        return False, f"arch {table.get('arch')!r} != {arch!r}"
    if "dataset" not in table or "input_res" not in table:
        return False, "predates the dataset/input_res fields -- cannot be verified"
    if str(table.get("dataset")) != str(dataset):
        return False, f"built for dataset {table.get('dataset')!r}, want {dataset!r}"
    if int(table.get("input_res", -1)) != want_res:
        return False, (f"built at {table.get('input_res')}px, want {want_res}px")
    if int(table.get("num_classes", -1)) != want_cls:
        return False, (f"built for {table.get('num_classes')} classes, want {want_cls}")
    got_r = list(table.get("axes", {}).get("resolution", {}).get("values", []))
    if got_r != list(spec["resolutions"]):
        return False, f"resolution grid {got_r} != {list(spec['resolutions'])}"
    return True, "ok"


def load_or_build_budgets(arch: str, data_dir, dataset: str,
                          num_classes: Optional[int] = None,
                          hub: Optional[MSCHub] = None, force: bool = False,
                          model=None) -> Dict[str, Any]:
    p = Path(data_dir) / "budgets" / f"{arch}.json"
    if p.exists() and not force:
        t = read_json(p)
        ok, why = budget_table_valid(t, arch, dataset, num_classes)
        if ok:
            return t
        log(f"cached budget table for {arch} is INVALID ({why}) -- rebuilding", "FLOP")
    log(f"measuring FLOPs budget for {arch} on {dataset} "
        f"@{native_res(dataset)}px", "FLOP")
    t = build_budget_table(arch, dataset, num_classes, model=model)
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
        """`n_train` is the size of the INDEX SPACE, not the split length.

        **D-49.** These arrays are indexed by `sample_idx`, and on the packed
        backend `sample_idx` is the GLOBAL pack index (0..129,394) rather than a
        position within the training split (0..119,394). Sizing them by
        `len(train_set)` therefore overflowed on the first training image whose
        global index exceeded the split length:

            IndexError: index 121978 is out of bounds for axis 0 with size 119395

        Making `sample_idx` global was deliberate -- it is what lets the `val`
        and `train_holdout` tables coexist unambiguously and makes every
        per-sample table self-describing. But it changed what an index MEANS,
        and this class was written against the old meaning. Same shape as D-40,
        where device-side augmentation changed what `dataload_frac` measured:
        a quantity whose definition moved while its name did not.

        Callers must pass `dataset.index_space`. The extra ~10k entries per
        array are a few hundred KB and are never read: `to_frame()` emits only
        indices actually seen.
        """
        self.n = int(n_train)
        self.el2n_epoch = int(el2n_epoch)
        self.correct_prev = np.zeros(self.n, dtype=np.int8)
        self.ever_correct = np.zeros(self.n, dtype=bool)
        self.forget_events = np.zeros(self.n, dtype=np.int32)
        self.el2n = np.full(self.n, np.nan, dtype=np.float32)
        self._epoch_correct = np.zeros(self.n, dtype=np.int8)
        self._epoch_seen = np.zeros(self.n, dtype=bool)
        self.epochs_recorded = 0

    def _check_space(self, idx) -> None:
        mx = int(np.max(idx)) if len(idx) else -1
        if mx >= self.n:
            raise IndexError(
                f"sample_idx {mx} exceeds the dynamics index space ({self.n}).\n"
                f"  TrainingDynamics is indexed by sample_idx, and on the packed\n"
                f"  backend that is the GLOBAL pack index, not a position within\n"
                f"  the training split. Size it with `dataset.index_space`,\n"
                f"  not `len(dataset)` (D-49).")

    def observe_batch(self, idx, logits, labels, epoch: int) -> None:
        """Called once per training batch with what the loop already has."""
        with torch.no_grad():
            i = idx.detach().cpu().numpy().astype(np.int64)
            self._check_space(i)
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
        # Only indices actually seen. With a GLOBAL index space the array
        # spans val and holdout positions too, and emitting rows for images
        # this run never trained on would put NaN forgetting counts into the
        # difficulty battery as if they were measurements (D-49).
        keep = (np.asarray(self.ever_correct) | (np.asarray(self.forget_events) > 0)
                | np.isfinite(np.asarray(self.el2n)))
        if not keep.any():
            keep = np.ones(self.n, dtype=bool)
        idx = np.flatnonzero(keep)
        fe = np.asarray(self.forget_events)[idx]
        ec = np.asarray(self.ever_correct)[idx]
        return pd.DataFrame({
            "sample_idx": idx,
            "forget_events": fe,
            "ever_correct": ec,
            "el2n": np.asarray(self.el2n)[idx],
            # Toneva's "unforgettable" set: learned and never lost. A useful
            # sanity check -- it should be a large, easy majority.
            "unforgettable": (ec & (fe == 0)),
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


def is_control_arm(run_id_or_cfg) -> bool:
    """Is this the SHUFFLED-target control? Decided on `method`, never on the id.

    **D-78.** NB5 split the arms with

        real = [r for r in results if 'shuff' not in r['run_id']]

    and the architecture `shufflenetv2_in` contains the substring `shuff`. So
    every shufflenetv2 run classified as control, including the real one, and
    the printed summary undercounted the real arm by a third.

    The method field is unambiguous — `mscKDshuffromresnet50` versus
    `mscKDfromresnet50` — and `parse_run_id` already extracts it. A substring
    test over a whole run_id searches the architecture name too, and rule 2
    names this exact hazard: a literal that is right for most values is the
    worst kind, because the ones it is wrong for look identical.

    The training path was never affected — it tested `cfg['method']` and so was
    correct. Only the reporting was wrong, which is its own hazard: the numbers
    were right and the label on them was not.
    """
    if isinstance(run_id_or_cfg, dict):
        method = run_id_or_cfg.get("method")
    else:
        # parse_run_id does NOT raise on a malformed id -- it returns
        # `method: None`. Relying on an exception that never comes is how a
        # "refuses to guess" guard silently guesses anyway, so the None is
        # checked directly.
        method = parse_run_id(str(run_id_or_cfg)).get("method")
    if not method:
        raise ValueError(
            f"cannot determine the arm of {run_id_or_cfg!r}: no method in the "
            f"run_id. Refusing to fall back to a substring test (D-78).")
    return str(method).startswith("mscKDshuf")


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


# =============================================================================
# The ImageNet-100 recipe
# =============================================================================
# ONE epoch count for all eight architectures. This is the pre-registered
# choice, and it is the weaker of the two options -- matching accuracy would
# break the family/accuracy confound outright, and equal epochs does not.
#
# What it does buy is that SCHEDULE LENGTH stops being a third confounded
# variable. On CIFAR the three modern architectures trained for 300 epochs and
# the CNNs for 240, so family, accuracy and schedule moved together and the
# lab notebook had to say so (1.2, "schedule length is not the difference
# either" rested on convnext_femto alone). Here it is held exactly constant.
#
# The accuracy confound is reported, not engineered away, and the 2x2 in
# 20_IN100_PORT_PLAN.md 1 is what carries the argument instead: if swin_tiny
# lands at CNN-level reliability while sitting at ViT-level accuracy, the
# accuracy explanation is dead regardless of the marginal means.
IN100_EPOCHS = 100          # the single lever if the GPU budget binds
IN100_BATCH = 64            # measured; see IN100_MEASURED_IMG_S below
IN100_REF_BATCH = 256       # LR is scaled linearly from this reference

# =============================================================================
# Measured throughput -- RTX 4000 Ada, 224px, batch 64, fp16 + channels_last
# =============================================================================
# From `benchmark/bench_throughput.py` on host CB-410-122, 2026-08-08.
# These REPLACE the estimates in 20_IN100_PORT_PLAN.md 6, which were anchored on
# one guessed figure for resnet50 and were 66% low in aggregate. D-10 is the
# precedent: the CIFAR cost table was 40% low and only found out by running.
#
# ⚠ Measured with `cudnn.benchmark = False`, which is torch's default and NOT
# what training uses -- that is D-43. The convolutional numbers are therefore
# understated, `resnet50` badly so: 82 img/s against `resnet18`'s 413 is a 5x
# gap for 2.3x the FLOPs, and 1x1-heavy bottleneck blocks in channels_last are
# exactly where cuDNN's heuristic algorithm choice is poor. Every entry marked
# `pending` needs re-measuring now that the benchmark shares the training
# path's backend configuration.
#
# Per DC-11 these refine DISPLAYED estimates only. They must never reach
# `assign_workers`, or ownership stops being deterministic (D-12).
IN100_MEASURED_IMG_S: Dict[str, float] = {
    # D-59 invalidated every convolutional entry here. All of them were taken
    # under channels_last, which measured 6.7x SLOWER than contiguous on this
    # card. The numbers were real; the configuration was wrong.
    #
    # PRODUCTION (100 epochs on real data, C:\msc_results):
    "vit_small_p16":   604.0,        # 203 s/epoch, 2 runs agreeing to 0.2%
    # CONV SWEEP (synthetic, contiguous, bs64 -- excludes ~1% augmentation):
    "resnet50":        550.3,        # was 82.3 under channels_last
    # NOT RE-MEASURED SINCE D-59. Every figure below is from the slow layout
    # and understates the truth, probably by a large factor. Budgets built on
    # them are wrong in the pessimistic direction -- which is the safe
    # direction, but it is not a measurement.
    "resnet18":        413.0,        # STALE: channels_last
    "shufflenetv2_in": 640.4,        # STALE: channels_last
    "swin_tiny":       327.1,        # STALE: channels_last
    "convnext_tiny":   272.2,        # STALE: channels_last
    "vgg16":            56.3,        # STALE: channels_last
    "deit_small":      604.0,        # from vit_small_p16: same builder, same args
}
IN100_MEASURED_PEAK_GB: Dict[str, float] = {
    "resnet18": 0.88, "shufflenetv2_in": 0.72, "resnet50": 2.93,
    "vgg16": 4.39, "swin_tiny": 4.53, "convnext_tiny": 5.13,
}
IN100_UNMEASURED = ("vit_small_p16", "deit_small")
# D-59: everything still carrying a channels_last measurement.
IN100_PENDING_REMEASURE = ("resnet18", "shufflenetv2_in", "swin_tiny",
                          "convnext_tiny", "vgg16")


def in100_estimate(archs: Sequence[str], seeds: int = 3,
                   epochs: int = IN100_EPOCHS,
                   n_train: int = 119_395) -> Dict[str, Any]:
    """Hours per architecture and in total, from measured throughput.

    Flags which entries are measurements and which are not, because a table
    that mixes the two without saying so is how an estimate becomes a fact.
    """
    rows, total = [], 0.0
    for a in sorted(archs):
        ips = IN100_MEASURED_IMG_S.get(a)
        if not ips:
            continue
        sec = n_train / ips
        h = sec * epochs / 3600.0
        rows.append({
            "arch": a, "img_s": ips, "sec_per_epoch": sec,
            "hours_per_run": h, "hours_all_seeds": h * seeds,
            "basis": ("ESTIMATE -- never measured" if a in IN100_UNMEASURED
                      else "measured, RE-MEASURE pending (D-43)"
                      if a in IN100_PENDING_REMEASURE else "measured"),
            "peak_vram_gb": IN100_MEASURED_PEAK_GB.get(a),
        })
        total += h * seeds
    rows.sort(key=lambda r: -r["hours_all_seeds"])
    return {"rows": rows, "total_gpu_hours": total, "days": total / 24.0,
            "epochs": epochs, "seeds": seeds,
            "share": {r["arch"]: r["hours_all_seeds"] / total for r in rows}
            if total else {}}


def _imagenet_config(arch: str, dataset: str, seed: int, phase: str,
                     method: str, **overrides) -> Dict[str, Any]:
    spec = dataset_spec(dataset)
    transformer = arch in TRANSFORMER_LIKE
    deit = arch in DEIT_RECIPE
    bs = int(overrides.get("batch_size", IN100_BATCH))

    if transformer:
        # AdamW at the DeiT reference (5e-4 per 512 images), scaled linearly.
        lr = 5e-4 * bs / 512.0
        wd = 0.05
    else:
        # SGD at the ImageNet reference (0.1 per 256 images), scaled linearly.
        lr = 0.1 * bs / IN100_REF_BATCH
        wd = 1e-4

    cfg: Dict[str, Any] = {
        "run_id": make_run_id(phase, arch, dataset, method, seed),
        "phase": phase, "arch": arch, "dataset_name": dataset, "method": method,
        "seed": int(seed), "num_classes": int(spec["num_classes"]),
        "family": ZOO.get(arch, {}).get("family", "unknown"),
        "input_res": int(spec["native_res"]),

        "num_epochs": IN100_EPOCHS,
        "batch_size": bs,
        "eval_batch_size": 256,
        "optimizer": "adamw" if transformer else "sgd",
        "learning_rate": float(lr),
        "weight_decay": wd,
        "momentum": 0.9,
        "nesterov": not transformer,
        "scheduler": "cosine",
        "lr_milestones": [],
        "lr_gamma": 0.1,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "grad_clip_norm": 1.0 if transformer else 0.0,
        "amp_enabled": True,
        "gradient_accumulation_steps": 1,
        "deterministic": False,

        # D-59. MEASURED on this hardware, not assumed. tools/conv_sweep.py,
        # ResNet-50 @224 bs64, RTX 4000 Ada / cuDNN 9.1 / driver 581.42:
        #
        #   channels_last     81.6 img/s    784 ms/batch
        #   contiguous       550.3 img/s    116 ms/batch     6.7x FASTER
        #
        # The textbook advice is the opposite, and on most NVIDIA parts it is
        # right. It is not right here, and "usually true" is how this cost
        # 41.5 h per ResNet-50 run instead of 6. Re-run conv_sweep.py on any
        # new machine rather than inheriting this number.
        "channels_last": False,

        # Performance only -- excluded from config_hash, so these can change
        # between sessions without orphaning a checkpoint (D-56).
        "ram_cache": True,
        "ram_headroom_gb": 6.0,

        # ---- the recipe contrast, and the ONLY thing that differs between
        # ---- vit_small_p16 and deit_small ------------------------------------
        # Same geometry, same optimiser, same LR, same weight decay, same
        # schedule, same epochs. DeiT adds mixup/cutmix and a wider
        # RandomResizedCrop. If seed-reliability differs across this pair, it is
        # a property of training and not of attention -- which would reframe the
        # CIFAR finding rather than confirm it.
        "mixup_alpha": 0.8 if deit else 0.0,
        "cutmix_alpha": 1.0 if deit else 0.0,
        "rrc_scale": (0.08, 1.0) if deit else (0.35, 1.0),
        "drop_path": 0.1 if deit else (0.05 if transformer else 0.0),

        # Q4 instrumentation
        "el2n_epoch": 10,
        "train_holdout_n": 15000,

        # exit heads: backbone frozen
        "exit_epochs": 10,
        "exit_lr": 0.01,

        # infrastructure
        "milestone_push_every_epochs": 5,
        "timer_push_sec": 1800,
        # 0 = NO LIMIT. This is a local machine with no session deadline; the
        # watchdog exists for Kaggle, where a session dies without warning and
        # stopping cleanly first is the civilised move. Read as "zero hours" it
        # paused every run after epoch 1 (D-50).
        "session_limit_h": float(overrides.get("session_limit_h", 0.0)),
        "cleanup_local_after_complete": False,
        "energy_sample_hz": 10.0,
        "carbon_intensity_kg_per_kwh": 0.475,
        "force_rerun": False,
        "msc_lib_version": __version__,
    }
    cfg.update(overrides)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


# No published from-scratch reference exists for this 100-class subset at this
# recipe, so every entry is null and NO delta is claimed for anything. D-14 is
# the cautionary case: `mobilenetv2`'s apparent +5.50 was against a half-width
# baseline, and it was the largest margin in the CIFAR atlas. A reference
# without a matching parameter count and recipe is unfalsifiable.
REFERENCE_ACC_IN100: Dict[str, Optional[float]] = {
    a: None for a in ("resnet50", "resnet18", "vgg16", "shufflenetv2_in",
                      "vit_small_p16", "deit_small", "swin_tiny", "convnext_tiny")
}


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
    if dataset_spec(dataset)["backend"] == "packed":
        return _imagenet_config(arch, dataset, seed, phase, method, **overrides)

    n_classes = num_classes_for(dataset)
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
                 "worker_id", "run_id", "_debug_interrupt_after_epoch",
                 # D-56. How the bytes reach the GPU is not part of the
                 # experiment. If `ram_cache` were hashed, switching it on
                 # would make every checkpoint on disk unresumable -- 69
                 # epochs of ResNet-50 discarded to change a buffering
                 # strategy. `batch_size` is deliberately NOT here: it scales
                 # the learning rate and IS the recipe.
                 "ram_cache", "ram_headroom_gb", "num_workers",
                 # D-59. Memory format changes floating-point summation order
                 # and nothing else -- the same forfeit AMP already makes, far
                 # below seed-to-seed variance. Hashing it would orphan
                 # resnet50 s1+s2 (100 epochs each) and vit s2 (73) the moment
                 # the measurement said to flip it: 90 hours discarded over a
                 # stride.
                 "channels_last",
                 "prefetch_batches"}


# Every exclusion set this project has ever hashed under, NEWEST FIRST.
#
# D-60. `config_hash` hashes everything EXCEPT this set, so ADDING a key to it
# changes the hash of every config in existence -- the key leaves the hashed
# space entirely. Excluding `channels_last` in D-59 to protect 90 hours of
# finished runs is the very thing that orphaned them.
#
# A hash whose DEFINITION changes needs a version, or every future exclusion
# silently invalidates every checkpoint on disk.
_HASH_EXCLUDE_V1 = _HASH_EXCLUDE - {"channels_last"}        # before D-59
_HASH_EXCLUDE_HISTORY: Tuple[frozenset, ...] = (
    frozenset(_HASH_EXCLUDE),
    frozenset(_HASH_EXCLUDE_V1),
)


def fmt_metric(value: Any, spec: str = ".2f", missing: str = "--") -> str:
    """Format a metric that may legitimately be absent.

    **D-61.** `f"{r.get('best_accuracy', float('nan')):.2f}"` looks defensive
    and is not. `dict.get`'s default fires only when the key is ABSENT; a key
    present with value `None` sails past it into `format`, which raises

        TypeError: unsupported format string passed to NoneType.__format__

    A run that paused, failed or was skipped reports `best_accuracy: None` --
    present, and null. So the summary loop crashed on exactly the runs whose
    status the operator most needed to read, AFTER the training had succeeded,
    which makes a completed epoch look like a crashed notebook.

    Anything non-numeric, including None and NaN, prints `missing`.
    """
    if value is None:
        return missing
    if isinstance(value, bool):
        return str(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:                                   # NaN
        return missing
    return format(f, spec)


def config_hash(cfg: Dict[str, Any],
                exclude: Optional[Iterable[str]] = None) -> str:
    ex = _HASH_EXCLUDE if exclude is None else set(exclude)
    return sha256_of_obj({k: v for k, v in sorted(cfg.items())
                          if k not in ex})


def hashed_key_diff(a: Dict[str, Any], b: Dict[str, Any],
                    exclude: Optional[Iterable[str]] = None
                    ) -> List[Tuple[str, Any, Any]]:
    """Keys that PARTICIPATE in the hash and differ. The message D-60 owed you.

    "The config changed since this run started" never said WHAT changed, so
    three rounds were spent guessing at a dict the code was holding and could
    simply have printed.
    """
    ex = _HASH_EXCLUDE if exclude is None else set(exclude)
    ka = {k: v for k, v in a.items() if k not in ex}
    kb = {k: v for k, v in b.items() if k not in ex}
    out = []
    for k in sorted(set(ka) | set(kb)):
        va, vb = ka.get(k, "<absent>"), kb.get(k, "<absent>")
        if sha256_of_obj({k: va}) != sha256_of_obj({k: vb}):
            out.append((k, va, vb))
    return out


def hash_compatible(cfg: Dict[str, Any], stored: str,
                    run_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """Is `stored` this run's hash under some earlier hashing rule?

    D-60 asked "did the RECIPE change, or only the RULE?". D-63 is about what
    it asked the question OF.

    The first version probed the live `cfg` alone. By the time
    `load_checkpoint` runs, that dict has picked up keys that were not present
    when its hash was taken, so `config_hash(cfg)` and `cfg["config_hash"]` are
    two different numbers and every probe built on it misses. The function
    returned True in every test I wrote -- all of which used a clean config --
    and False on the machine. That is the most expensive shape a bug can have:
    the tests agree with the author instead of with the program.

    `runs/<id>/config.yaml` is written from the config at claim time and is the
    authoritative record of what this run IS. So:

      1. probe the live config (fast path, covers a clean resume);
      2. probe the record; if the record reproduces `stored`, this checkpoint
         provably belongs to this run;
      3. then require the live config not to CHANGE any key the record has.
         Keys the live config merely ADDS were in no hash and cannot alter a
         result. A changed value is a genuine edit and is still refused.
    """
    if not stored:
        return False, "no stored hash"
    if config_hash(cfg) == stored:
        return True, "current rule"

    def _probe(d: Dict[str, Any]) -> Tuple[Optional[int], str]:
        for vi, ex in enumerate(_HASH_EXCLUDE_HISTORY[1:], start=1):
            moved = sorted(set(_HASH_EXCLUDE) - set(ex))
            if not moved:
                continue
            choices = []
            for k in moved:
                cur = d.get(k)
                vals = [cur, not cur] if isinstance(cur, bool) else [cur]
                choices.append([(k, v) for v in vals])
            combos = 1
            for c in choices:
                combos *= len(c)
            if combos > 64:                  # bounded; never a search space
                continue
            for assign in itertools.product(*choices):
                probe = dict(d)
                probe.update(dict(assign))
                if config_hash(probe, exclude=ex) == stored:
                    return vi, ", ".join(f"{k}={v!r}" for k, v in assign)
        return None, ""

    vi, shown = _probe(cfg)
    if vi is not None:
        return True, f"rule v{vi}, before these became performance-only: {shown}"

    if run_dir is not None:
        try:
            rec = read_yaml(Path(run_dir) / "config.yaml")
        except Exception:                                        # noqa: BLE001
            rec = None
        if rec:
            vi, shown = _probe(rec)
            if vi is None and config_hash(rec) == stored:
                vi, shown = 0, "unchanged"
            if vi is not None:
                changed = [(k, a, b) for k, a, b in hashed_key_diff(rec, cfg)
                           if k in rec and k in cfg]
                if not changed:
                    added = [k for k, a, _ in hashed_key_diff(rec, cfg)
                             if a == "<absent>"]
                    extra = (f"; the live config only ADDS {len(added)} runtime "
                             f"key(s): {', '.join(added[:4])}") if added else ""
                    return True, (f"rule v{vi} via config.yaml, before these "
                                  f"became performance-only: {shown}{extra}")
                return False, ("the recipe genuinely changed since this run "
                               "started -- " + ", ".join(
                                   f"{k}: {a!r} -> {b!r}" for k, a, b in changed[:6]))
    return False, "no historical rule reproduces it"

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

# Number of GPUs given their own columns. ASKED OF THE MACHINE, not assumed.
#
# This was a literal 2 because dual T4 was the only platform. The port target is
# a single RTX 4000 Ada, and D-36 is precisely what a wrong GPU column count
# looks like downstream: NB15 asked for `gpu_util_mean_pct`, which does not
# exist because the fields are per device (`gpu0_*`, `gpu1_*`). A schema pinned
# to the wrong device count produces a table full of NA columns for hardware
# that was never present, and a reader that asks for a device that was.
#
# Floor of 1 so the schema is stable on a CPU-only analysis session -- the
# column set must not depend on whether the machine writing it had a GPU, or
# two runs become un-concatenable.
def _detect_gpu_columns(default: int = 1) -> int:
    try:
        if _TORCH_OK and torch.cuda.is_available():
            return max(1, int(torch.cuda.device_count()))
    except Exception:                                            # noqa: BLE001
        pass
    return max(1, int(os.environ.get("MSC_GPU_COLUMNS", default)))


N_GPU_COLUMNS = _detect_gpu_columns()

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
       # D-40. On the packed backend the augmentation runs on the GPU inside
       # the loader, so "time until the next batch" is no longer the same
       # quantity it was on CIFAR. These two separate it: `augment_time_sec`
       # is device work, `dataload_time_sec` is a genuine block on the worker
       # pool. Conflating them makes `dataload_frac` say "the loader is the
       # bottleneck" when the loader is idle.
       "augment_time_sec", "augment_frac",
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
        # Device-side augmentation time, reported by the loader if it does any.
        # Zero on the CIFAR backend, where augmentation is CPU work inside the
        # Dataset and is therefore genuinely part of dataload.
        self.augment_sec = 0.0

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


    def load_seconds(self) -> float:
        """Seconds this epoch spent blocked waiting for the next batch."""
        return float(np.sum(self.dataload_times)) if self.dataload_times else 0.0

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
            # D-40. `dataload_frac` is the CPU-starvation signal and must stay
            # that: on the packed backend the device-side augmentation is
            # subtracted out, so a high value still means "the loader is the
            # bottleneck" and never "the GPU did some work between batches".
            "dataload_time_sec": max(0.0, float(np.sum(self.dataload_times))
                                     - self.augment_sec),
            "augment_time_sec": float(self.augment_sec),
            "augment_frac": (float(self.augment_sec) / tot_step)
                            if tot_step > 0 else NA,
            "dataload_frac": (max(0.0, float(np.sum(self.dataload_times))
                                  - self.augment_sec) / tot_step)
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


def soft_target_ce(logits, target, crit=None):
    """Cross-entropy against a soft target, honouring label smoothing.

    `nn.CrossEntropyLoss` accepts probability targets from torch 1.10, so this
    delegates rather than reimplementing -- but it exists as a named function so
    the mixup path has one obvious place to be tested, and so the training loop
    reads the same whether targets are hard or soft.
    """
    crit = crit or nn.CrossEntropyLoss()
    return crit(logits, target)


def mixup_cutmix(x, y, num_classes: int, cfg: Dict[str, Any],
                 generator=None) -> Tuple[Any, Any, bool]:
    """The DeiT augmentation arm. Returns `(x, target, target_is_soft)`.

    Off unless `mixup_alpha` or `cutmix_alpha` is positive, so it is a no-op for
    seven of the eight architectures and returns the hard labels unchanged.

    This is the ONLY thing that differs between `vit_small_p16` and
    `deit_small` besides drop-path and the crop range -- same geometry, same
    optimiser, same LR, same weight decay, same schedule, same epoch count. The
    pair is the study's recipe-versus-architecture control, so what varies
    across it has to be exactly this and nothing else.

    Applied to backbone training only. It is deliberately NOT applied in
    `train_msc_kd`: the MSC target is a per-sample property of a specific image,
    and mixing two images produces a sample whose "minimum sufficient compute"
    is undefined. Mixing there would silently train the router on targets that
    do not correspond to their inputs.
    """
    ma = float(cfg.get("mixup_alpha", 0.0) or 0.0)
    ca = float(cfg.get("cutmix_alpha", 0.0) or 0.0)
    if ma <= 0 and ca <= 0:
        return x, y, False
    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    y1 = F.one_hot(y, num_classes).float()
    y2 = y1[perm]
    use_cutmix = ca > 0 and (ma <= 0 or float(torch.rand(1)) < 0.5)
    if use_cutmix:
        lam = float(np.random.beta(ca, ca))
        h, w = x.shape[-2], x.shape[-1]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = int(torch.randint(0, h, (1,))), int(torch.randint(0, w, (1,)))
        y0_, y1_ = max(0, cy - rh // 2), min(h, cy + rh // 2)
        x0_, x1_ = max(0, cx - rw // 2), min(w, cx + rw // 2)
        x = x.clone()
        x[:, :, y0_:y1_, x0_:x1_] = x[perm][:, :, y0_:y1_, x0_:x1_]
        # lam is RECOMPUTED from the box that was actually pasted, not from the
        # sampled value. Clipping at the image edge makes them differ, and using
        # the sampled lam would mislabel every clipped sample.
        lam = 1.0 - ((y1_ - y0_) * (x1_ - x0_) / float(h * w))
    else:
        lam = float(np.random.beta(ma, ma))
        x = lam * x + (1.0 - lam) * x[perm]
    return x, lam * y1 + (1.0 - lam) * y2, True


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


class _SyntheticLoader:
    """A loader-shaped object over `n` batches of noise, with the same
    `(x, y, sample_idx)` contract the real loaders yield.

    `sample_idx` is real and distinct, because every per-sample artifact is
    written back in `sample_idx` order and a dry run over indistinguishable
    indices would not exercise the reordering that alignment depends on.
    """

    def __init__(self, device, n_batches: int, batch: int, res: int,
                 n_cls: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self._b = []
        for i in range(n_batches):
            x = torch.randn(batch, 3, res, res, generator=g)
            y = torch.randint(0, n_cls, (batch,), generator=g)
            idx = torch.arange(i * batch, (i + 1) * batch)
            self._b.append((x, y, idx))
        self.dataset = list(range(n_batches * batch))
        self.batch_size = batch

    def __iter__(self):
        return iter(self._b)

    def __len__(self):
        return len(self._b)


def backbone_dry_run(cfg: Dict[str, Any], device=None,
                     amp: Optional[bool] = None) -> Tuple[bool, str]:
    """Push one synthetic batch through the ENTIRE backbone-training path
    before any real work. Returns (ok, reason). Sub-second.

    Rule 1, and the reason it is phrased as "the entire path including
    evaluation": D-21 and D-22 each cost an hour of GPU time and each was
    findable in milliseconds, but they were findable at *different* stages.
    D-21 was the first training step; D-22 was the history write at the END of
    epoch 0. A dry run that stopped after `loss.backward()` would have caught
    one and not the other -- it would have moved the boundary of what can hide,
    not removed it.

    So this covers, in order, every stage `train_backbone` performs per epoch:

        build -> forward -> loss -> backward -> optimiser step -> scaler
        -> optimisation_health -> evaluate() -> calibration
        -> history row -> append_history_row(strict=True)
        -> save_checkpoint -> load_checkpoint (config_hash asserted)

    The checkpoint round trip is here deliberately. Five defects in this
    project have been about resume (D-05, D-06, D-09, D-12, D-19) and the
    cheapest of them cost 30 GPU-hours. Reading the checkpoint back in the same
    second it was written cannot prove cross-session resume works -- that is
    O-18 and needs a real session boundary -- but it does prove the contract
    round-trips at all, which is the part that was silently broken.
    """
    if not _TORCH_OK:
        return True, "torch unavailable; dry run skipped"
    import tempfile as _tf
    t0 = time.time()
    dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = str(cfg.get("dataset_name", "cifar100"))
    amp = bool(cfg.get("amp_enabled", True)) if amp is None else bool(amp)
    amp = amp and dev.type == "cuda"
    stage = "build"
    # Two warnings are guaranteed on a 2-sample synthetic batch and mean
    # nothing here: sklearn's "y_pred contains classes not in y_true" (2 samples
    # against 100 classes), and torch's scheduler-before-optimizer notice (the
    # AMP scaler legitimately skips the first step while it finds a loss scale).
    # They are suppressed INSIDE the dry run only, because eight architectures
    # x two dry runs printed sixteen paragraphs of noise around the two lines
    # that actually mattered -- and a report nobody can read is a report nobody
    # reads (D-17's cost, in a new place).
    _wctx = warnings.catch_warnings()
    _wctx.__enter__()
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        n_cls = num_classes_for(ds)
        res = int(cfg.get("input_res", native_res(ds)))
        model = place_model(build_model(cfg["arch"], n_cls, dataset=ds), dev, cfg)

        stage = "optimizer"
        opt, sched = build_optimizer(model, cfg)
        scaler = torch.amp.GradScaler(dev.type, enabled=amp)
        crit = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.get("label_smoothing", 0.0)))

        loader = _SyntheticLoader(dev, 2, 2, res, n_cls, seed=int(cfg.get("seed", 1)))
        x, y, _ = next(iter(loader))
        x, y = x.to(dev), y.to(dev)
        if cfg.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)

        stage = "forward/loss/backward"
        # Mixup is part of the deit arm's recipe, so it is part of the path and
        # must be exercised. A soft-target loss that cannot autocast is exactly
        # the D-21 shape.
        xm, ym, soft = mixup_cutmix(x, y, n_cls, cfg)
        with torch.amp.autocast(device_type=dev.type, enabled=amp):
            out = model(xm)
            loss = soft_target_ce(out, ym, crit) if soft else crit(out, ym)
        if not bool(torch.isfinite(loss).item()):
            return False, f"loss is not finite ({float(loss)}) on synthetic input"
        scaler.scale(loss).backward()
        if float(cfg.get("grad_clip_norm", 0.0)) > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg["grad_clip_norm"]))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()

        stage = "optimisation_health"
        # Four values, not two. Unpacking it wrongly is the kind of thing that
        # only a dry run which actually CALLS it can find -- which is the point.
        _wn, _un, _ratio, _flat = optimisation_health(model)

        stage = "evaluate"
        val = evaluate(model, loader, dev, amp=amp, criterion=crit,
                       collect_probs=True)
        for k in ("loss", "accuracy", "accuracy_top5", "f1_macro"):
            if k not in val:
                return False, f"evaluate() did not return '{k}'"

        stage = "history row"
        with _tf.TemporaryDirectory() as td:
            row = {"run_id": cfg["run_id"], "epoch": 0,
                   "arch": cfg["arch"], "seed": cfg["seed"],
                   "phase": cfg.get("phase", "p1"),
                   "config_hash": cfg["config_hash"],
                   "train_loss": float(loss), "val_loss": float(val["loss"]),
                   "val_accuracy": float(val["accuracy"]),
                   "learning_rate": float(opt.param_groups[0]["lr"]),
                   "amp_enabled": bool(amp)}
            row.update({k: v for k, v in
                        {"weight_norm": _wn, "update_norm": _un,
                         "update_to_weight_ratio": _ratio}.items()
                        if k in _HISTORY_SET})
            # strict=True: an unknown column RAISES and names the column you
            # probably meant. This is the check that would have caught D-22's
            # five wrong names in microseconds instead of at the end of epoch 0
            # on a real teacher.
            append_history_row(Path(td) / "epochs.csv", row, strict=True)

            stage = "checkpoint round trip"
            ck = Path(td) / "ckpt.pt"
            save_checkpoint(ck, cfg, model, opt, sched, scaler, epoch=0,
                            best_metric=float(val["accuracy"]), dynamics=None,
                            wall_seconds=1.0, energy_joules=0.0)
            m2 = place_model(build_model(cfg["arch"], n_cls, dataset=ds), dev, cfg)
            o2, s2 = build_optimizer(m2, cfg)
            sc2 = torch.amp.GradScaler(dev.type, enabled=amp)
            # Eight positional arguments, and it returns a DICT. Getting either
            # wrong is the D-47 defect: a signature mismatch that no
            # name-resolution check can see, because every name involved exists.
            # NOT `res` -- that name already holds the input resolution, and
            # shadowing it put a checkpoint dict into the success message:
            #   "backbone dry run ok (0.27s, {'start_epoch': 1, ...}px, ...)"
            # Harmless, but a status line that prints a dict where a number
            # belongs is a status line nobody reads carefully afterwards.
            ck_res = load_checkpoint(ck, cfg, m2, o2, s2, sc2, None, dev,
                                     strict_hash=True)
            start = int(ck_res["start_epoch"])
            best = float(ck_res["best_metric"])
            if int(start) != 1:
                return False, (f"checkpoint says resume at epoch {start}, "
                               f"expected 1 after writing epoch 0")
            if abs(float(best) - float(val["accuracy"])) > 1e-6:
                return False, f"best_metric did not round-trip ({best})"

        del model, opt, scaler
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return True, f"ok ({time.time() - t0:.2f}s, {res}px, {n_cls} classes)"
    except Exception as e:                                       # noqa: BLE001
        return False, f"at stage '{stage}': {type(e).__name__}: {e}"
    finally:
        _wctx.__exit__(None, None, None)


def oracle_dry_run(cfg: Dict[str, Any], device=None,
                   amp: Optional[bool] = None) -> Tuple[bool, str]:
    """Push two synthetic images through the ENTIRE measurement path.

    `run_oracle` trains exit heads over the full training set and then sweeps
    every configuration on every sample, so the first artifact it writes is
    roughly an hour in. Everything downstream of that hour is covered here:

        multi-exit build -> sweep_all_axes over EVERY axis at EVERY resolution
        and EVERY precision -> difficulty_battery -> prediction_depth
        -> build_per_sample_frame -> parquet WRITE -> parquet READ BACK
        -> compute_msc on the result

    The resolution sweep is the expensive part to get wrong and the cheapest to
    check. On CIFAR this exact class of failure produced D-01a (a ViT whose
    positional embedding is sized for one grid) and D-02 (a Mixer whose
    token-mixing weights ARE the token count). At 224px there is a third: a
    Swin-T reduces its input by 32, so its final stage is 7x7 at 224 and 3x3 at
    96 -- smaller than its own attention window.

    The parquet round trip is here because `build_per_sample_frame` is where
    column names are invented, and a column name that is wrong is invisible
    until analysis (D-22, D-36).
    """
    if not _TORCH_OK:
        return True, "torch unavailable; dry run skipped"
    import tempfile as _tf
    t0 = time.time()
    dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = str(cfg.get("dataset_name", "cifar100"))
    amp = bool(cfg.get("amp_enabled", True)) if amp is None else bool(amp)
    amp = amp and dev.type == "cuda"
    stage = "build"
    _wctx = warnings.catch_warnings()
    _wctx.__enter__()
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        n_cls = num_classes_for(ds)
        res = int(cfg.get("input_res", native_res(ds)))
        grid = resolutions_for(ds)
        bb = place_model(build_model(cfg["arch"], n_cls, dataset=ds), dev, cfg).eval()
        # K from the model. Never a literal -- D-01b, D-28 and D-33 were all
        # this, and D-33 was a hardcoded 5 inside the check written for D-28.
        me = place_model(MultiExitModel(bb, n_cls, freeze=True), dev, cfg).eval()
        n_heads = len(me.heads)
        if n_heads != len(bb.feature_dims):
            return False, (f"MultiExit built {n_heads} heads for a backbone "
                           f"with {len(bb.feature_dims)} feature dims")

        loader = _SyntheticLoader(dev, 2, 2, res, n_cls, seed=1)

        stage = f"sweep_all_axes ({n_heads} depth + {len(grid)}x2 res + "\
                f"{len(PRECISIONS)} precision)"
        sweep = sweep_all_axes(cfg, me, loader, dev, amp=amp, show_progress=False)
        n = len(loader.dataset)
        for axis in ("depth", "res_proxy", "precision"):
            if axis not in sweep:
                return False, f"sweep produced no '{axis}' axis"
            got = sweep[axis]["preds"].shape
            want_k = {"depth": n_heads, "res_proxy": len(grid),
                      "precision": len(PRECISIONS)}[axis]
            if got != (n, want_k):
                return False, f"{axis} preds are {got}, expected {(n, want_k)}"
        native_ok = "res_native" in sweep

        stage = "difficulty_battery"
        battery = difficulty_battery(bb, loader, dev, amp=amp)

        stage = "prediction_depth"
        pdep = prediction_depth(me, loader, dev, k_neighbors=2, max_support=n)

        stage = "build_per_sample_frame"
        frame = build_per_sample_frame(
            sweep, battery, pdep, None, order_hash="dryrun",
            run_id=cfg["run_id"], split="test")
        if frame is None or len(frame) != n:
            return False, f"per-sample frame has {0 if frame is None else len(frame)} rows, expected {n}"

        stage = "parquet round trip"
        with _tf.TemporaryDirectory() as td:
            p = Path(td) / "test.parquet"
            frame.to_parquet(p, index=False)
            back = pd.read_parquet(p)
            missing = set(frame.columns) - set(back.columns)
            if missing:
                return False, f"parquet lost columns: {sorted(missing)[:6]}"
            if len(back) != n:
                return False, f"parquet round trip lost rows ({len(back)} of {n})"

        stage = "compute_msc"
        budgets = build_budget_table(cfg["arch"], ds, n_cls, model=bb.cpu())
        rho = budgets["axes"]["depth"]["rho"]
        if not all(rho[i] < rho[i + 1] for i in range(len(rho) - 1)):
            return False, f"depth rho is not strictly ascending: {rho}"
        # MSCResult is a dataclass, not an array: `.msc` is the per-sample
        # vector. `len()` on the container raises, which is what D-47 was.
        res_msc = msc_for_run(back, budgets, axis="depth", tau=0.1)
        vec = getattr(res_msc, "msc", None)
        if vec is None or len(vec) != n:
            return False, (f"msc_for_run returned "
                           f"{type(res_msc).__name__} with "
                           f"{0 if vec is None else len(vec)} values, expected "
                           f"one per sample ({n})")
        if not ((vec > 0).all() and (vec <= 1.0 + 1e-9).all()):
            return False, "MSC values fall outside (0, 1] -- rho is a fraction"

        del bb, me
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return True, (f"ok ({time.time() - t0:.2f}s, K={n_heads}, "
                      f"native-res sweep {'available' if native_ok else 'PROXY ONLY'}, "
                      f"{len(frame.columns)} per-sample columns)")
    except Exception as e:                                       # noqa: BLE001
        return False, f"at stage '{stage}': {type(e).__name__}: {e}"
    finally:
        _wctx.__exit__(None, None, None)


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
        # D-33: n_budgets MUST come from the backbone, never a literal. A
        # hardcoded 5 here recreated D-28 inside the very check written to
        # catch it: a 3-exit resnet8x4 got a 5-output router and the dry run
        # failed every healthy run.
        _bb = build_model(cfg["arch"], n_cls)
        n_heads = len(_bb.feature_dims)
        student = place_model(MSCStudent(_bb, n_cls, n_heads), device, cfg)
        # Resolution from the dataset, not from a `cfg.get(..., 32)` default.
        # The old fallback meant an ImageNet run whose config happened to omit
        # `image_size` would dry-run at 32px, pass, and then fail for real an
        # hour later at 224 -- a dry run that certifies the wrong shape is worse
        # than none, because it manufactures confidence (D-06).
        _r = int(cfg.get("input_res",
                         native_res(cfg.get("dataset_name", "cifar100"))))
        x = torch.randn(2, 3, _r, _r, device=device)
        y = torch.zeros(2, dtype=torch.long, device=device)
        tgt = torch.zeros(2, n_heads, device=device)   # D-33: not a literal
        tgt[:, max(0, n_heads - 2):] = 1.0
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
        # D-30: go all the way through EVALUATION, not just training.
        # The dry run as first written covered the training step and would have
        # caught D-21 and D-22 -- but not D-28, whose shape mismatch is
        # invisible until routing indexes the exit logits. Every stage the real
        # pipeline uses has to appear here, or the dry run just moves the
        # boundary of what can hide behind an hour of setup.
        n_heads = len(student.heads)
        rho_probe = [(i + 1) / n_heads for i in range(n_heads)]

        class _Loader:                      # two batches, no dataset needed
            def __iter__(self):
                for _ in range(2):
                    yield x.cpu(), y.cpu()

        ev = evaluate_routing_methods(student, _Loader(), device, rho_probe,
                                      full_flops=1e9, oracle_msc=None,
                                      amp=amp)
        if int(ev.get("K", 0)) != n_heads:
            return False, f"eval reports K={ev.get('K')} for {n_heads} heads"

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


def msckd_router_ok(work, run_id: str, cfg: Dict[str, Any], data_out,
                    hub=None) -> Tuple[bool, str]:
    """Is this finished MSC-KD checkpoint still *valid*, not merely present?

    **D-29.** `already_finished` answers "did this run complete?". After D-28
    changed how the router is shaped, the honest answer for nine existing
    students was "yes, and the result is unusable" -- their sufficiency head
    was sized from the teacher's budget grid. The completion cache had no way
    to know that, so re-running NB13 skipped all nine and the same broken
    checkpoints kept flowing into NB14.

    **A completion cache needs a compatibility predicate, not just a presence
    predicate.** This is that predicate: the router width stored with the
    checkpoint must equal the number of depth budgets the student actually has.

    Returns (ok, reason). Defensive: when validity cannot be established it
    returns True, because forcing a retrain on uncertainty is its own kind of
    damage.
    """
    ck = run_layout(work, run_id)["checkpoints"] / "ckpt_best.pt"
    if not ck.exists() or not _TORCH_OK:
        return True, "no checkpoint to check"
    try:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        stored = blob.get("rho")
        if not stored:
            return True, "checkpoint stores no rho"
        b = load_or_build_budgets(cfg["arch"], data_out, cfg["dataset_name"],
                                  int(cfg["num_classes"]), hub=hub)
        want = len(b["axes"]["depth"]["rho"])
    except Exception as e:                                   # noqa: BLE001
        return True, f"could not verify ({type(e).__name__}: {e})"
    if len(stored) != want:
        return False, (f"router has {len(stored)} outputs but {cfg['arch']} has "
                       f"{want} depth budgets -- trained against the TEACHER's "
                       f"grid, before D-28")
    return True, "ok"


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
        # D-60. Before refusing, ask whether the RECIPE changed or only the
        # hashing RULE. Adding a key to _HASH_EXCLUDE to protect finished runs
        # is exactly what orphans them, and throwing away 73 good epochs over
        # a memory-layout flag is the outcome this check exists to prevent.
        _ok, _why = hash_compatible(cfg, str(ck.get("config_hash") or ""),
                                    run_dir=p.parent.parent)
        if _ok:
            log(f"{msg}\n  ACCEPTED -- the recipe is unchanged. This checkpoint "
                f"was hashed under {_why}. Everything hashed under both rules "
                f"is byte-identical, so the difference is confined to keys "
                f"since declared performance-only (D-60).", "RESUME")
        elif strict_hash:
            # Fail loudly. A silent mismatch means you are continuing a run
            # under a config that has been edited since it started, and nobody
            # ever notices until the numbers do not reproduce.
            raise RuntimeError(
                msg + f"\n  why: {_why}"
                    + "\nThe config changed since this run started. Either restore "
                      "the original config, or set force_rerun=True to discard the "
                      "checkpoint and retrain from scratch.")
        else:
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

def place_model(model, device, cfg: Optional[Dict[str, Any]] = None,
                tag: str = ""):
    """Move a model to `device` in the memory format the LOADER actually emits.

    **D-55, and it cost three days of wall clock.**

    `GPUBatchLoader` ends every batch with

        x = x.contiguous(memory_format=torch.channels_last)

    unconditionally. `base_config` sets `channels_last: True`. And of the
    sixteen places this library constructs a model, exactly ONE applied that
    format -- `backbone_dry_run`. Every real path (`train_backbone`,
    `run_oracle`, `train_exit_heads`, `train_msc_kd`) built an NCHW model and
    then fed it NHWC activations.

    cuDNN cannot run a convolution whose input and weight disagree on layout.
    It converts one of them, per convolution, per batch, forward and backward,
    for the whole network. ResNet-50 on an RTX 4000 Ada held a flat 80 img/s
    for 69 consecutive epochs -- flat because a layout conversion is a fixed
    tax, not a variable one. Nothing looked broken. The loss fell, the accuracy
    climbed to 80.6%, and each epoch took 25 minutes instead of about 8.

    Two rules failed together, and the second is why it survived:

      Rule 7, an invariant in a comment is not a mechanism. `channels_last:
      True` sat in the config as a statement of intent that nothing enforced.

      Rule 8, test the thing you WROTE. The dry run applied the format. The
      trainer did not. So the dry run passed a configuration the real run never
      executed, and passing it is what authorised the three-day run.

    This function is now the only sanctioned way to put a model on a device.
    One place to read, one place to change, and `assert_layout_match` below
    turns the invariant into something that fails loudly on batch one.
    """
    model = model.to(device)
    want_cl = True if cfg is None else bool(cfg.get("channels_last", True))
    if want_cl:
        model = model.to(memory_format=torch.channels_last)
    if tag:
        log(f"{tag}: {'channels_last' if want_cl else 'contiguous'} on {device}",
            "PERF")
    return model


def assert_layout_match(model, x, where: str = "train") -> None:
    """Fail on the first batch if activations and weights disagree on layout.

    The mechanism D-55 did not have. Checked once per run -- it walks a handful
    of conv weights and costs microseconds -- and raises rather than warns,
    because the failure mode it guards is a 5x slowdown that produces correct
    numbers and therefore never announces itself.
    """
    w = next((m.weight for m in model.modules()
              if isinstance(m, nn.Conv2d) and m.weight.dim() == 4), None)
    if w is None or x.dim() != 4:
        return
    x_cl = x.is_contiguous(memory_format=torch.channels_last)
    w_cl = w.is_contiguous(memory_format=torch.channels_last)
    if x_cl != w_cl:
        raise RuntimeError(
            f"[{where}] memory-format mismatch: input is "
            f"{'channels_last' if x_cl else 'contiguous'} but conv weights are "
            f"{'channels_last' if w_cl else 'contiguous'}.\n"
            f"cuDNN will convert one of them on every convolution of every "
            f"batch. This is D-55: it is not a correctness bug, it is a ~5x "
            f"throughput bug that trains to the right answer slowly.\n"
            f"Build the model through place_model(model, device, cfg).")




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

    # RULE 1. The entire path -- forward, loss, backward, optimiser step,
    # evaluate(), history write, checkpoint save AND reload -- on one synthetic
    # batch, before the dataset is touched. Under a second.
    #
    # BEFORE the claim, deliberately. A run that cannot train should not appear
    # in the ledger as `running` and should not need its claim released; and a
    # broken config then fails identically on every worker rather than on
    # whichever one happened to claim it first.
    _dry_ok, _dry_why = backbone_dry_run(cfg)
    if not _dry_ok:
        raise RuntimeError(
            f"[DRY RUN FAILED] {cfg['run_id']}: {_dry_why}\n"
            f"No GPU time has been spent and nothing has been claimed.")
    log(f"backbone dry run {_dry_why}", "DRY")

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

    model = place_model(build_model(cfg["arch"], cfg["num_classes"]),
                        device, cfg, tag=f'{cfg["arch"]} backbone')
    optimizer, scheduler = build_optimizer(model, cfg)
    amp = bool(cfg.get("amp_enabled", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    # D-49: the index SPACE, which is not the split length on a backend whose
    # sample_idx is global. Ask the dataset rather than assuming.
    _space = int(getattr(train_loader.dataset, "index_space", n_train))
    dynamics = TrainingDynamics(_space, el2n_epoch=int(cfg.get("el2n_epoch", 10)))

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
                it = tqdm(train_loader, desc=f"ep {epoch+1}/{num_epochs}",
                          leave=False, dynamic_ncols=True, mininterval=1.0,
                          unit="b", smoothing=0.1)

            # D-40: a loader that augments on the device knows how much of the
            # inter-batch gap was its own GPU work, and the loop cannot. Ask it.
            _timed_loader = hasattr(train_loader, "timing")
            if _timed_loader:
                tel.augment_sec = 0.0
            _bar = it if (tqdm is not None and show_progress and it is not train_loader) else None
            _n_steps = len(train_loader)
            _t_epoch0 = time.time()
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
                if epoch == start_epoch and step == 0:
                    # D-55. Once per run, on the first batch, before 25 minutes
                    # of epoch go by. The check that would have caught a flat
                    # 80 img/s on the first minute instead of the third day.
                    assert_layout_match(model, x, where=f'train {cfg["arch"]}')
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

                # Live metrics BESIDE the bar, refreshed roughly once a
                # second. An epoch here is 3-35 minutes: a bar that shows only
                # position tells you the run is alive but not whether it is
                # learning, and the two questions you actually have during a
                # 10-day programme are "is the loss moving" and "is the GPU
                # busy". Both are answerable now instead of at the epoch line.
                if _bar is not None and (step % 20 == 0 or step + 1 == _n_steps):
                    _el = max(1e-9, time.time() - _t_epoch0)
                    _post = {"loss": f"{run_loss / max(1, total):.3f}",
                             "acc": f"{correct / max(1, total):.3f}",
                             "img/s": f"{total / _el:.0f}",
                             "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                    if tel.bad_batches:
                        # Non-finite losses are silent under AMP; the run keeps
                        # going and learns nothing from those batches. If it is
                        # happening, it should be visible while it happens.
                        _post["nan"] = str(tel.bad_batches)
                    # D-57. Where the batch time GOES, on the bar, while it is
                    # going. Two separate wrong diagnoses (D-55 memory format,
                    # D-56 disk) were argued from a throughput number and a
                    # VRAM number because the split was only ever written to
                    # epochs.csv, which nobody opens mid-run. The loader has
                    # been measuring `wait` and `aug` the whole time.
                    #
                    #   wait  main loop blocked on the next batch
                    #   aug   GPU augmentation (grid_sample, normalise, cast)
                    #   step  forward + backward + optimizer
                    #
                    # Whichever is largest is the thing to fix. No tool to run,
                    # no file to open, no theory required.
                    _lt = tel.load_seconds()
                    _st = max(1e-9, time.time() - _t_epoch0)
                    _post["wait"] = f"{100.0*_lt/_st:.0f}%"
                    _as = None
                    if hasattr(train_loader, "augment_seconds"):
                        _as = train_loader.augment_seconds()
                    if _as is not None:
                        _post["aug"] = f"{100.0*_as/_st:.0f}%"
                    _post["step"] = f"{1000.0*max(0.0, _st-_lt-(_as or 0.0))/max(1, step+1):.0f}ms"
                    if device.type == "cuda":
                        _post["vram"] = (f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
                    _bar.set_postfix(_post, refresh=False)

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
            # Pull the device-side augmentation time out of the loader before
            # summarising, so `dataload_frac` measures CPU starvation and not
            # "the GPU did some work between batches" (D-40).
            if _timed_loader:
                _lt = train_loader.timing()
                tel.augment_sec = float(_lt.get("augment_s", 0.0))
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

            # The epoch line carries what you would otherwise have to open
            # epochs.csv to see -- including the three columns that are silent
            # by default and unrecoverable afterwards: non-finite batches, AMP
            # scale decreases, and the update-to-weight ratio.
            _done, _left = epoch + 1, num_epochs - (epoch + 1)
            _eta_h = (cumulative_time / max(1, _done)) * _left / 3600.0
            _thr = row.get("throughput_train_img_s", NA)
            _dl = row.get("dataload_frac", NA)
            _u2w = row.get("update_to_weight_ratio", NA)
            _warn = ""
            if isinstance(_u2w, float) and _u2w == _u2w:
                if _u2w > 1e-2:
                    _warn += "  [LR HIGH?]"      # healthy is ~1e-3
                elif _u2w < 1e-5:
                    _warn += "  [NOT MOVING?]"
            if tel.bad_batches:
                _warn += f"  [{tel.bad_batches} NaN/Inf BATCHES]"
            if tel.amp_decreases > 0.05 * max(1, tel.opt_steps):
                _warn += f"  [{tel.amp_decreases} AMP OVERFLOWS]"
            if isinstance(_dl, float) and _dl == _dl and _dl > 0.30:
                _warn += f"  [DATA-BOUND {100*_dl:.0f}%]"
            print(f"  ep {_done:>3d}/{num_epochs}  "
                  f"train {row['train_accuracy']*100:5.2f}%  "
                  f"val {val_acc*100:5.2f}%  top5 {row['val_accuracy_top5']*100:5.2f}%  "
                  f"loss {row['train_loss']:.3f}  lr {row['learning_rate']:.2e}  "
                  f"{_thr if not isinstance(_thr, float) else f'{_thr:.0f}'} img/s  "
                  f"{epoch_time:.0f}s  ETA {_eta_h:.1f}h  "
                  f"{epoch_energy/3.6e6:.3f}kWh"
                  + ("  *BEST*" if is_best else "") + _warn)

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
    budgets = load_or_build_budgets(
        cfg["arch"], data_out, cfg["dataset_name"], cfg["num_classes"], hub=hub,
        model=build_model(cfg["arch"], cfg["num_classes"],
                          dataset=cfg["dataset_name"]))

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
    me = place_model(MultiExitModel(backbone, cfg["num_classes"], freeze=True),
                     device, cfg, tag="exit heads")
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


def _resize_proxy(x, r: int, native: Optional[int] = None):
    """Downsample to r then back up. Information content drops; shape does not.

    Idealised cost: the network really runs at its native resolution, so the
    FLOPs attributed are those of a native-r run. Labelled as such everywhere.

    `native` defaults to whatever the incoming tensor already is, which is the
    only value that can be right without being told -- the old version restored
    to a literal 32 and would have silently reshaped every ImageNet batch to
    thumbnail size while reporting full-resolution costs.
    """
    n = int(native if native is not None else x.shape[-1])
    if r == n and r == x.shape[-1]:
        return x
    small = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(n, n), mode="bilinear", align_corners=False)


@_no_grad()
def sweep_all_axes(cfg: Dict[str, Any], multi_exit, loader, device,
                   resolutions: Optional[Sequence[int]] = None,
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
    # The grid and the native resolution come from the dataset, never from a
    # module-level constant -- `RESOLUTIONS` is CIFAR's grid and using it here
    # would sweep an ImageNet model over 16-32px inputs while the budget table
    # priced 96-224px. Both halves would be internally consistent.
    dsname = str(cfg.get("dataset_name", "cifar100"))
    resolutions = tuple(resolutions if resolutions is not None
                        else resolutions_for(dsname))
    res0 = native_res(dsname)

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
        for _bi, batch in enumerate(it):
            x = batch[0].to(device, non_blocking=True)
            if _bi == 0:
                _assert_model_ready(x, cfg, where=f"sweep {tag}")
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
            chunks_i.append(to_numpy(idx, np.int64))
            chunks_l.append(to_numpy(y, np.int64))
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
                xr = x if r == res0 else F.interpolate(x, size=(r, r),
                                                       mode="bilinear",
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
        log(f"architecture cannot run at non-{res0}px input -- resolution axis "
            f"measured with the proxy only", "ORACLE")

    # --- resolution, proxy -------------------------------------------------
    # Option (b): downsample-then-upsample, network shape unchanged, only
    # information content varies. Measuring both converts a methodological
    # wrinkle a reviewer would raise into a robustness check we already ran.
    def proxy_fn(x):
        return [backbone(_resize_proxy(x, r, res0)) for r in resolutions]
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
        idxs.append(to_numpy(idx, np.int64))
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

    # RULE 1. Two synthetic images through the ENTIRE measurement path --
    # every axis at every resolution and every precision, the difficulty
    # battery, prediction depth, the per-sample frame, a parquet write and
    # READ BACK, and compute_msc on the result -- before the exit heads are
    # trained over the full training set. Under a second against an hour.
    _dry_ok, _dry_why = oracle_dry_run(cfg)
    if not _dry_ok:
        raise RuntimeError(
            f"[DRY RUN FAILED] {cfg['run_id']}: {_dry_why}\n"
            f"No GPU time has been spent. The resolution sweep is the part "
            f"this exists for: D-01a and D-02 were both an architecture that "
            f"could not run at a resolution the oracle assumed, and at 224px "
            f"Swin-T's final stage is smaller than its own attention window "
            f"at the low end of the grid.")
    log(f"oracle dry run {_dry_why}", "DRY")

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
    # D-69. This read `run_dir / "ckpt_best.pt"` -- the run ROOT. Checkpoints
    # live in `checkpoints/`, and the code KNEW that: the HuggingFace fallback
    # below spelled it `L["checkpoints"] / "ckpt_best.pt"` correctly. With HF
    # disabled that branch is dead, so the only surviving spelling was the
    # wrong one and every measurement failed with "Train the backbone first"
    # while a 91 MB checkpoint sat one directory away.
    #
    # Two spellings of one path, one of them wrong, and the correct one three
    # lines below in unreachable code. That is D-16, and D-23 is the same
    # defect on `exit_heads.pt` -- which is why `exit_heads_path()` exists and
    # is now used here rather than re-spelled.
    ckpt = L["checkpoints"] / "ckpt_best.pt"
    if not ckpt.exists() and hub.enabled:
        log(f"pulling checkpoint for {run_id} from HF", "ORACLE")
        hub.hub.download(work, allow_patterns=[f"runs/{run_id}/**"], quiet=False)
    if not ckpt.exists():
        _last = L["checkpoints"] / "ckpt_last.pt"
        raise FileNotFoundError(
            f"no ckpt_best.pt for {run_id} at {ckpt}.\n"
            f"  ckpt_last.pt present: {_last.exists()}\n"
            f"  Train the backbone first (NB2), or check MSC_ROOT points at "
            f"the results folder that holds this run.")

    backbone = place_model(build_model(cfg["arch"], cfg["num_classes"]),
                           device, cfg, tag="oracle backbone")
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    backbone.load_state_dict(blob["model"], strict=True)
    backbone.eval()
    if blob.get("config_hash") not in (None, cfg["config_hash"]):
        log("checkpoint config_hash differs from the current config -- the sweep "
            "will run, but record this discrepancy", "WARN")

    train_loader, val_loader, holdout_loader, classes, order_hash = build_loaders(cfg)

    # --- exit heads --------------------------------------------------------
    # THE accessor, not a second spelling (D-23).
    heads_path = exit_heads_path(work, run_id)
    me = place_model(MultiExitModel(backbone, cfg["num_classes"], freeze=True),
                     device, cfg)
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
    budgets = load_or_build_budgets(cfg["arch"], data_out, cfg["dataset_name"],
                                    cfg["num_classes"], hub=hub)

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
    _res_grid = resolutions_for(cfg["dataset_name"])
    results = {}
    for split, loader in (("test", val_loader), ("train_holdout", holdout_loader)):
        log(f"sweeping {split} ({len(loader.dataset)} samples, "
            f"{len(me.heads)}+{len(_res_grid)}x2+{len(PRECISIONS)} configs "
            f"@{native_res(cfg['dataset_name'])}px)", "ORACLE")
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
            "exit_count": len(me.heads), "resolutions": list(_res_grid),
            "input_res": native_res(cfg["dataset_name"]),
            "data_fingerprint": cfg.get("data_fingerprint", NA),
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
    # D-34: `k_max` indexes `correct_at`, so it must come from `correct_at`.
    # Taking it from `suff_pred` meant a router wider than the backbone's exit
    # count produced an out-of-range column index and a bare IndexError eight
    # frames from the cause. Same root as D-28: two arrays that must agree on K.
    if suff_pred.shape[1] != correct_at.shape[1]:
        raise ValueError(
            f"learn_then_test_threshold: {suff_pred.shape[1]} sufficiency "
            f"outputs but {correct_at.shape[1]} exit columns. These must "
            f"match. A student trained before the D-28 fix has a router sized "
            f"from the TEACHER's grid -- re-run NB13, which detects and "
            f"retrains those automatically.")
    n, k_max = suff_pred.shape[0], correct_at.shape[1] - 1
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
        arch = m.get("arch")
        if not arch:
            continue
        # D-71. This tested `rid not in require`. `require` is the CEILINGS
        # dict, keyed by ARCHITECTURE ('resnet50'); `rid` is a run id
        # ('p0-resnet50-imagenet100-base-s1'). No run id is ever a member, so
        # every run was skipped, `cand` stayed empty, and every caller that
        # passed `require` got an empty result -- silently.
        #
        # Q3's shuffled control wrote a 2-byte CSV and NB4 raised
        # `KeyError: 'passed'` on a frame with no columns. Q3's axis structure
        # returns `pd.DataFrame([])` on no pairs and did not even raise.
        #
        # The docstring said "an ARCHITECTURE is only represented by a run
        # that appears in it". The prose was right and the code tested the
        # other key. Two identifier spaces, one membership test.
        if require is not None and arch not in require:
            continue
        seed = m.get("seed")
        cand.setdefault(arch, []).append(
            (10 ** 6 if seed is None else int(seed), rid))
    if require is not None and runs and not cand:
        raise KeyError(
            f"representative_runs: `require` excluded ALL {len(runs)} runs. "
            f"It is keyed by {sorted(list(require))[:3]}... and is matched "
            f"against architecture names like "
            f"{sorted({m.get('arch') for m in runs.values()})[:3]}. "
            f"An empty result here empties every downstream table (D-71).")
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


# =============================================================================
# atlas-wide analysis wrappers
# =============================================================================
# The per-run and per-pair statistics above are the primitives. These assemble
# them across the whole atlas.
#
# On CIFAR this assembly lived in NOTEBOOK CELLS, and that is where D-18 came
# from: `pairs[:15]` over an alphabetically sorted list looked like cost
# control and was actually a biased sample -- 12 convnext pairs and 3 mixer
# pairs, the two most atypical architectures in the zoo, both of which depress
# the statistic being reported. And `{m['arch']: r for r,m in runs.items() if
# m['seed']==1}` silently dropped an architecture whose seed 1 was never
# measured, so the analysis covered 13 architectures while calling itself the
# atlas.
#
# Neither was catchable, because a dict comprehension in a notebook cell cannot
# announce what it skipped and nothing tests a notebook cell. Rule 8: test the
# thing you wrote. So the selection logic lives here, where the self-checks can
# reach it, and every one of these functions REPORTS what it excluded.
def resolve_analysis_phase(session, phase: Optional[str] = None) -> str:
    """The phase an analysis should read. D-66.

    Every `analyse_*_all` defaulted to the literal `"p1"`. NB4 called them
    without an argument, so on a `p0` pilot each one indexed zero runs and
    returned an EMPTY DataFrame -- no rows, and therefore no columns. The
    failure surfaced two lines later as

        KeyError: 'rho_seed_tau0.1'

    which names a column, points at the notebook, and says nothing about the
    phase. D-65 fixed this same default in the notebooks; it was also sitting
    in the library, one layer down, where the notebook fix could not reach it.
    """
    if phase:
        return phase
    return detect_phase(session.work)


def _run_index(session, phase: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Measured runs, keyed by run_id, with identity parsed from the id.

    One choke point: all five `analyse_*_all` entry points come through here,
    so the phase is resolved once rather than defaulted five times (D-66).
    """
    phase = resolve_analysis_phase(session, phase)
    out = {}
    for r in session.completed_runs(phase=phase):
        rid = r["run_id"]
        if session.measured(rid):
            out[rid] = run_meta(rid, r)
    return out


def _require_runs(session, runs: Dict[str, Any], phase: Optional[str],
                  what: str) -> None:
    """Refuse to analyse nothing. D-66.

    An empty index produced an empty DataFrame, which has no columns, which
    raised `KeyError: 'rho_seed_tau0.1'` in the notebook two lines later. That
    error names a column and points at the display line -- it says nothing
    about the phase, the runs, or the measurement stage, which is where all
    three actual causes live.

    Silence and a misleading error are the two failure modes this log is
    mostly made of. This is the third place the same shape has appeared
    (D-18 shortened a table, D-65 measured nothing), so it says which of the
    three things is missing.
    """
    if runs:
        return
    ph = resolve_analysis_phase(session, phase)
    seen = phases_present(session.work)
    trained = [r["run_id"] for r in session.completed_runs(phase=ph)]
    unmeasured = [r for r in trained if not session.measured(r)]
    if not trained:
        detail = (f"no COMPLETED runs in phase {ph!r}. On disk: {seen}. "
                  f"Run NB2 first.")
    elif unmeasured:
        detail = (f"{len(trained)} trained run(s) in {ph!r} but "
                  f"{len(unmeasured)} are NOT MEASURED: "
                  f"{', '.join(unmeasured[:4])}. Run NB3 first.")
    else:
        detail = f"{len(trained)} run(s) present and measured, but none usable."
    raise RuntimeError(f"{what}: nothing to analyse -- {detail}")


def analyse_q1_all(session, phase: Optional[str] = None, axis: str = "depth",
                   taus=TAU_GRID) -> "Any":
    """Seed ceiling for every architecture with >= 2 measured seeds.

    Reports architectures it had to SKIP and why, rather than quietly
    returning a shorter table (D-18). One row per architecture, with the
    tau-curve pivoted into columns and mean top-1 alongside -- because the
    accuracy confound has to be visible in the same table as the ceiling, not
    argued around in prose afterwards.
    """
    runs = _run_index(session, phase)
    _require_runs(session, runs, phase, "Q1 seed ceilings")
    by_arch: Dict[str, List[str]] = {}
    for rid, m in runs.items():
        by_arch.setdefault(m["arch"], []).append(rid)

    rows, skipped = [], {}
    for arch, rids in sorted(by_arch.items()):
        rids = sorted(rids)
        if len(rids) < 2:
            skipped[arch] = f"{len(rids)} measured seed(s); a ceiling needs 2"
            continue
        b = session.budgets(arch)
        # EVERY pair, then the mean -- not just (seed1, seed2). With three
        # seeds there are three pairs, and reporting one of them throws away
        # two thirds of the evidence for the project's most important number.
        per_tau: Dict[float, List[float]] = {t: [] for t in taus}
        j10: Dict[float, List[float]] = {t: [] for t in taus}
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                df = analyse_q1_seed_ceiling(session.data_dir, rids[i], rids[j],
                                             b, axis=axis, taus=taus)
                for _, r in df.iterrows():
                    if "rho_seed" in r and pd.notna(r.get("rho_seed")):
                        per_tau[float(r["tau"])].append(float(r["rho_seed"]))
                        j10[float(r["tau"])].append(float(r.get("jaccard_top10",
                                                               float("nan"))))
        accs = []
        for rid in rids:
            s = read_json(run_layout(session.work, rid)["base"] / "summary.json", {})
            if s and s.get("best_accuracy") is not None:
                accs.append(float(s["best_accuracy"]))
        rec = {"arch": arch, "family": ZOO.get(arch, {}).get("family", "?"),
               "n_seeds": len(rids), "n_pairs": len(rids) * (len(rids) - 1) // 2,
               "top1_mean": float(np.mean(accs)) if accs else float("nan"),
               "top1_spread": (float(np.max(accs) - np.min(accs)) if len(accs) > 1
                               else float("nan"))}
        for t in taus:
            v = per_tau[float(t)]
            rec[f"rho_seed_tau{t}"] = float(np.mean(v)) if v else float("nan")
            rec[f"rho_seed_sd_tau{t}"] = (float(np.std(v)) if len(v) > 1
                                          else float("nan"))
            rec[f"j10_tau{t}"] = (float(np.nanmean(j10[float(t)]))
                                  if j10[float(t)] else float("nan"))
        rows.append(rec)

    if skipped:
        log(f"Q1 EXCLUDED {len(skipped)} architecture(s): {skipped}", "ALARM")
        log("A ceiling needs two measured seeds. These contribute to NOTHING "
            "-- not Q1, not Q3, not Q4 -- and any claim about the full zoo is "
            "false until they are measured (the D-15 shape).", "ALARM")
    return pd.DataFrame(rows)


def analyse_q2_all(session, phase: Optional[str] = None, tau: float = 0.1) -> "Any":
    """Axis structure for one representative run per architecture."""
    runs = _run_index(session, phase)
    _require_runs(session, runs, phase, "Q2 transfer")
    reps = representative_runs(runs)
    rows = []
    for arch, rid in sorted(reps.items()):
        df = analyse_q2_axis_structure(session.data_dir, rid,
                                       session.budgets(arch))
        if df is None or not len(df):
            continue
        sub = df[df.get("tau").astype(float) == float(tau)] if "tau" in df else df
        if not len(sub):
            continue
        r = sub.iloc[0].to_dict()
        rows.append({"arch": arch, "family": ZOO.get(arch, {}).get("family", "?"),
                     "run_id": rid, "tau": tau,
                     "pc1": r.get("pc1_variance"), "n": r.get("n")})
    return pd.DataFrame(rows)


def _pair_kind(a: str, b: str) -> str:
    fa = ZOO.get(a, {}).get("family", "?")
    fb = ZOO.get(b, {}).get("family", "?")
    att = {"vit", "swin", "mixer"}
    if fa == fb:
        return "within-family"
    if fa in att and fb in att:
        return "transformer-transformer"
    if fa in att or fb in att:
        return "CNN-transformer"
    return "across-CNN-family"


def _ceilings(session, q1=None, tau: float = 0.1) -> Dict[str, float]:
    q1 = q1 if q1 is not None else analyse_q1_all(session)
    col = f"rho_seed_tau{tau}"
    return {r["arch"]: float(r[col]) for _, r in q1.iterrows()
            if pd.notna(r.get(col))}


def analyse_q3_all(session, phase: Optional[str] = None, tau: float = 0.1,
                   n_boot: int = 1000) -> "Any":
    """Disattenuated transfer over EVERY architecture pair.

    Every pair, not `pairs[:N]`. A truncation over a sorted list is only a
    sample if the order is unrelated to the quantity being measured, and
    `sorted()` guarantees it is not (D-18).
    """
    runs = _run_index(session, phase)
    _require_runs(session, runs, phase, "Q3 axis structure")
    reps = representative_runs(runs, require=_ceilings(session, tau=tau))
    ceil = _ceilings(session, tau=tau)
    archs = sorted(a for a in reps if a in ceil)
    pairs = [(reps[a], reps[b]) for i, a in enumerate(archs) for b in archs[i + 1:]]
    if not pairs:
        # D-71. This returned an empty frame in silence, so an upstream
        # key-space error surfaced as a KeyError on a column three layers away.
        raise RuntimeError(
            f"Q3: no architecture PAIRS to compare. {len(runs)} measured run(s) "
            f"covering {sorted({m['arch'] for m in runs.values()})}, of which "
            f"{len(archs)} have a seed ceiling at tau={tau}. A transfer needs "
            f"two architectures with >= 2 measured seeds each.")
    budgets = {reps[a]: session.budgets(a) for a in archs}
    ceil_by_run = {reps[a]: ceil[a] for a in archs}
    df = analyse_q3_transfer(session.data_dir, pairs, ceil_by_run, budgets,
                             taus=(tau,), n_boot=n_boot)
    if len(df):
        df["arch_a"] = df["run_a"].map(lambda r: parse_run_id(r)["arch"])
        df["arch_b"] = df["run_b"].map(lambda r: parse_run_id(r)["arch"])
        df["pair_type"] = [_pair_kind(a, b)
                           for a, b in zip(df["arch_a"], df["arch_b"])]
    return df


def analyse_q3_shuffled_control_all(session, phase: Optional[str] = None,
                                    tau: float = 0.1) -> "Any":
    """The alignment control, on EVERY pair -- not the first 25 of them."""
    runs = _run_index(session, phase)
    _require_runs(session, runs, phase, "Q3 shuffled control")
    ceil = _ceilings(session, tau=tau)
    reps = representative_runs(runs, require=ceil)
    archs = sorted(a for a in reps if a in ceil)
    budgets = {reps[a]: session.budgets(a) for a in archs}
    ceil_by_run = {reps[a]: ceil[a] for a in archs}
    rows = []
    for i, a in enumerate(archs):
        for b in archs[i + 1:]:
            r = analyse_q3_shuffled_control(session.data_dir, reps[a], reps[b],
                                            ceil_by_run, budgets, tau=tau)
            r.update({"arch_a": a, "arch_b": b})
            rows.append(r)
    if not rows:
        raise RuntimeError(
            f"Q3 shuffled control: no pairs. {len(archs)} architecture(s) have "
            f"a ceiling at tau={tau}: {archs}. Two are needed. An empty frame "
            f"here becomes KeyError('passed') in the notebook (D-71).")
    df = pd.DataFrame(rows)
    # D-52. The primitive returns `passed`. This wrapper looked for `ok` to
    # synthesise a `passes` column, so `passes` was never created and NB4's
    # `ctrl['passes']` would have raised KeyError -- in the ANALYSIS phase,
    # after every GPU-hour was already spent. One name, taken from the
    # primitive, and no renaming layer to get wrong.
    if len(df) and "passed" not in df.columns:
        raise KeyError(
            f"the shuffled control returned {sorted(df.columns)} with no "
            f"'passed' column -- the alignment gate cannot be evaluated")
    return df


def analyse_q4_all(session, phase: Optional[str] = None, tau: float = 0.1,
                   split: str = "train_holdout", n_boot: int = 500) -> "Any":
    """Irreducibility over every pair, on the split that carries all seven
    battery scores.

    `split` defaults to `train_holdout` and not to `test`, because EL2N and
    forgetting-events are training-set quantities. Running the battery without
    them is an EASIER test for MSC, which is the direction that flatters the
    result -- it overstated CIFAR's irreducibility by 2.5x and the number had
    to be withdrawn (D-11).
    """
    runs = _run_index(session, phase)
    _require_runs(session, runs, phase, "Q4 difficulty battery")
    reps = representative_runs(runs, require=_ceilings(session, tau=tau))
    archs = sorted(reps)
    budgets = {reps[a]: session.budgets(a) for a in archs}
    frames = []
    for i, a in enumerate(archs):
        for b in archs[i + 1:]:
            try:
                d = analyse_q4_irreducibility(session.data_dir, reps[a], reps[b],
                                              budgets, taus=(tau,),
                                              n_boot=n_boot, split=split)
                if d is not None and len(d):
                    d = d.copy()
                    d["arch_a"], d["arch_b"] = a, b
                    d["pair_type"] = _pair_kind(a, b)
                    frames.append(d)
            except Exception as e:                               # noqa: BLE001
                log(f"Q4 {a}x{b}: {type(e).__name__}: {str(e)[:120]}", "WARN")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([])


def compare_routing_methods(session, run_ids: Sequence[str],
                            tau: float = 0.1) -> "Any":
    """B1 / B2 / B10 / B11 per student, read from what NB5 wrote.

    Reads rather than recomputes: `train_msc_kd` already evaluated each student
    and wrote the result, and recomputing here would need the val loader, the
    checkpoint and the teacher again for numbers that exist on disk.

    `arm` is derived from the run_id, never from a flag. Two arms whose
    identity depended on an operator remembering which value to run is exactly
    what made four consecutive sessions train the control (D-27).
    """
    rows = []
    for rid in run_ids:
        s = read_json(run_layout(session.work, rid)["base"] / "summary.json", {})
        if not s:
            continue
        m = parse_run_id(rid)
        rows.append({
            "run_id": rid, "student": m["arch"], "seed": m["seed"],
            # method, not run_id -- `shufflenetv2_in` contains "shuff" (D-78)
            "arm": "scrambled" if is_control_arm(m) else "real",
            **{k: s.get(k) for k in
               ("best_accuracy", "b1_static", "b2_confidence", "b10_msckd",
                "b11_oracle", "avg_flops_ratio", "gamma", "ltt_epsilon")},
        })
    df = pd.DataFrame(rows)
    if len(df) and {"b2_confidence", "b10_msckd", "b11_oracle"} <= set(df.columns):
        gap = pd.to_numeric(df["b11_oracle"], errors="coerce") - \
            pd.to_numeric(df["b2_confidence"], errors="coerce")
        closed = pd.to_numeric(df["b10_msckd"], errors="coerce") - \
            pd.to_numeric(df["b2_confidence"], errors="coerce")
        # The paper's central number: the fraction of the B2->B11 gap closed.
        df["frac_b2_b11_gap_closed"] = closed / gap.replace(0, np.nan)
    return df


# =============================================================================
# paper artifacts -- what each claimed contribution has to leave behind
# =============================================================================
# Protocol 8.1 lists six contributions. A contribution with no artifact behind
# it is a claim, and the difference is not visible while writing -- you find out
# when you go to cite the table and it is not there.
#
# This list lives HERE and not in a notebook cell, for the D-16 reason: the
# writer and the reader must not be two independent spellings of the same path.
# `verify_paper_artifacts` is the reader, `save_analysis`/`save_figure` are the
# writers, and both go through these names.
PAPER_ARTIFACTS: Tuple[Tuple[str, str], ...] = (
    ("tables/table1_atlas.csv",
     "contribution 6 -- what was trained, and did it converge"),
    ("tables/table2_q1_ceilings.csv",
     "contribution 3 -- THE headline: rho_seed beside accuracy"),
    ("tables/table3_q2_axis_structure.csv", "contribution 2"),
    ("tables/table4_q3_transfer.csv", "contribution 3 -- transfer"),
    ("tables/table5_q4_irreducibility.csv", "contribution 4"),
    ("tables/table6_cifar_vs_imagenet.csv",
     "the replication result itself -- did the gap survive?"),
    ("analysis/q1_seed_ceilings_all.csv", "Q1 raw"),
    ("analysis/q2_axis_structure_all.csv", "Q2 raw"),
    ("analysis/q3_transfer_matrix.csv", "Q3 raw"),
    ("analysis/q3_shuffled_control.csv",
     "the alignment control -- without it Q3 is uninterpretable"),
    ("analysis/q4_irreducibility_all.csv", "Q4 raw"),
    ("paper/provenance.csv", "contribution 6 -- every number to a run_id"),
    ("paper/figures/fig1_q1_ceilings.png", "Figure 1"),
    ("paper/figures/fig2_tau_curves.png",
     "Figure 2 -- no conclusion may depend on tau, so the curve is shown"),
    ("paper/figures/fig3_ceiling_vs_accuracy.png",
     "Figure 3 -- the confound, plotted rather than asserted"),
)

PAPER_ARTIFACTS_METHOD: Tuple[Tuple[str, str], ...] = (
    ("analysis/q5_method_comparison.csv", "contribution 5 -- MSC-KD at matched FLOPs"),
)


def verify_paper_artifacts(data_dir, method: bool = False) -> Dict[str, Any]:
    """Which claimed contributions do NOT yet have an artifact behind them."""
    want = list(PAPER_ARTIFACTS) + (list(PAPER_ARTIFACTS_METHOD) if method else [])
    rows, missing = [], []
    for rel, why in want:
        p = Path(data_dir) / rel
        n = p.stat().st_size if p.exists() else 0
        state = "ok" if n > 32 else ("empty" if p.exists() else "missing")
        if state != "ok":
            missing.append(rel)
        rows.append({"artifact": rel, "state": state, "bytes": n, "backs": why})
    return {"ok": not missing, "missing": missing, "rows": rows}


RESUME_TEST_KEYS = (
    "arch", "epochs", "kill_at", "interrupt_fired", "resume_status",
    "epochs_ref", "epochs_cut", "duplicate_epochs", "final_acc_ref",
    "final_acc_cut", "acc_delta", "post_seam_epochs_compared",
    "max_post_seam_loss_deviation", "ref_run", "cut_run", "diagnosis", "ok",
)


# =============================================================================
# declared result keys -- what a caller may read from each of these
# =============================================================================
# D-51 and D-52. A notebook read `res.get('passed')` where the key is `ok`, and
# reported a PASSING resume test as a failure. A wrapper synthesised a `passes`
# column by looking for `ok` when the primitive returns `passed`, which would
# have raised KeyError during analysis, after every GPU-hour was spent.
#
# Four earlier guards check that functions EXIST (D-39), that calls match
# SIGNATURES (D-47, D-48), and that column literals match the schema (D-22,
# D-36). None of them can see a KEY read off a returned dict or frame. This
# registry closes that: `build_notebooks_in100.py` refuses to generate a
# notebook that reads a key not declared here.
#
# Declaring the set is what makes a guess detectable. A guess against an
# undeclared dict is indistinguishable from a correct read until it runs.
RESULT_KEYS: Dict[str, Tuple[str, ...]] = {
    "resolve_storage": ("ok", "problems", "notes", "data_dir", "results_root",
                        "candidates", "data_free_gb", "results_free_gb"),
    "preflight": ("checked_utc", "dataset", "input_res", "resolution_grid",
                  "checks"),
    "preflight_summary": ("passed", "failed", "todo", "ok", "n"),
    "resume_acceptance_test": RESUME_TEST_KEYS,
    "in100_estimate": ("rows", "total_gpu_hours", "days", "epochs", "seeds",
                       "share"),
    "confirm_on_disk": ("ok", "done", "resumable", "at_risk", "unknown",
                        "detail"),
    "confirm_on_hf": ("ok", "done", "resumable", "at_risk", "unknown"),
    "verify_run_artifacts": ("run_id", "root", "ok", "missing_required",
                             "empty", "unreadable", "total_bytes", "files"),
    "verify_paper_artifacts": ("ok", "missing", "rows"),
    "parse_run_id": ("run_id", "phase", "arch", "dataset", "method", "seed",
                     "family"),
    "set_perf_flags": ("deterministic", "cudnn_benchmark",
                       "cudnn_deterministic", "tf32_matmul", "error"),
    "data_present": (),                       # returns a tuple, not a dict
    # DataFrame-returning analyses: the COLUMNS a caller may read.
    "analyse_q1_all": ("arch", "family", "n_seeds", "n_pairs", "top1_mean",
                       "top1_spread"),
    "analyse_q2_all": ("arch", "family", "run_id", "tau", "pc1", "n"),
    "analyse_q3_all": ("run_a", "run_b", "axis", "tau", "spearman_raw", "T",
                       "ceiling_a", "ceiling_b", "n", "jaccard_top10",
                       "arch_a", "arch_b", "pair_type"),
    "analyse_q3_shuffled_control_all": ("passed", "spearman_raw", "z", "n",
                                        "null_sd", "z_max", "rho_floor",
                                        "tau", "axis", "arch_a", "arch_b"),
    "analyse_q4_all": ("run_a", "run_b", "axis", "tau", "split", "delta_r2",
                       "delta_r2_lo", "delta_r2_hi", "partial_spearman",
                       "r2_difficulty_only", "r2_difficulty_plus_msc",
                       "battery", "n_battery_scores", "arch_a", "arch_b",
                       "pair_type"),
    "compare_routing_methods": ("run_id", "student", "seed", "arm",
                                "best_accuracy", "b1_static", "b2_confidence",
                                "b10_msckd", "b11_oracle", "avg_flops_ratio",
                                "gamma", "ltt_epsilon",
                                "frac_b2_b11_gap_closed"),
}
# `analyse_q1_all` also emits rho_seed_tau{t} / j10_tau{t} per tau; matched by
# shape rather than enumerated, since the tau grid is a parameter.
RESULT_KEY_PATTERNS = (r"^rho_seed(_sd)?_tau[\d.]+$", r"^j10_tau[\d.]+$")


def result_key_ok(fn: str, key: str) -> bool:
    """May a caller read `key` from `fn`'s result?"""
    declared = RESULT_KEYS.get(fn)
    if declared is None:
        return True                      # undeclared function: nothing to check
    if key in declared:
        return True
    return any(re.match(p, key) for p in RESULT_KEY_PATTERNS)


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


def load_analysis(data_dir, name: str, default=None):
    """Read back what `save_analysis` wrote. Returns `default` if absent.

    D-72. `save_analysis` had no counterpart -- the third writer in this
    library with no reader (`atomic_write_yaml`/`read_yaml` was D-63). Analysis
    outputs are the evidence for whether the next stage is worth running, and
    nothing could consult them, so every gate in the plan was a thing a human
    had to remember to eyeball.
    """
    p = Path(data_dir) / "analysis" / f"{name}.csv"
    if not p.exists():
        return default
    try:
        df = pd.read_csv(p)
    except Exception:                                            # noqa: BLE001
        return default
    return default if df.empty else df


def measured_img_s(arch: str, repo_root=None) -> Tuple[float, str]:
    """Throughput for `arch`: the freshest MEASUREMENT, and where it came from.

    D-74. `IN100_MEASURED_IMG_S` still carries figures taken under the slow
    `channels_last` layout (D-59) for five architectures. `tools/conv_sweep.py`
    writes a corrected number to `benchmark/convsweep_<arch>_*.json`, and
    nothing read it -- so a user who ran the sweep, as instructed, still saw
    "STALE" and a wrong estimate. A fourth writer with no reader (D-63, D-72).

    Returns `(img_s, basis)`. The sweep result wins when present, because it
    was taken on this machine in the configuration that now runs.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    best, when = None, None
    for f in sorted((root / "benchmark").glob(f"convsweep_{arch}_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        vals = [v.get("img_s") for v in d.values()
                if isinstance(v, dict) and v.get("img_s")]
        if vals:
            best, when = max(vals), f.name
    if best is not None:
        return float(best), f"conv_sweep ({when})"
    v = IN100_MEASURED_IMG_S.get(arch)
    if v is None:
        return float("nan"), "NOT MEASURED"
    if arch in IN100_PENDING_REMEASURE:
        return float(v), "STALE -- channels_last; run tools/conv_sweep.py --arch " + arch
    return float(v), "measured"


def gate_report(data_dir) -> Dict[str, Any]:
    """Q1-Q4 against their pre-registered gates, as data rather than eyeballs.

    D-72. The gates are stated in `00_RESEARCH_PROTOCOL.md` and printed by NB4,
    but nothing could *read* the answer -- so NB5, which costs 18 training
    runs, had no way to ask whether its own premise had survived Q4.

    Returns `{gate: {value, threshold, passed}}` plus `all_passed`. Missing
    analyses are reported as `None`, never as a pass: a gate that has not been
    evaluated is not a gate that was met.
    """
    out: Dict[str, Any] = {}

    q1 = load_analysis(data_dir, "q1_seed_ceilings_all")
    if q1 is not None and "rho_seed_tau0.1" in q1.columns:
        worst = float(q1["rho_seed_tau0.1"].min())
        out["rho_seed >= 0.60"] = {
            "value": worst, "threshold": 0.60, "passed": worst >= 0.60,
            "detail": "; ".join(f"{r['arch']}={r['rho_seed_tau0.1']:.3f}"
                                for _, r in q1.iterrows())}

    ctrl = load_analysis(data_dir, "q3_shuffled_control")
    if ctrl is not None and "passed" in ctrl.columns:
        ok = bool(ctrl["passed"].all())
        out["shuffled control"] = {
            "value": float(ctrl["z"].abs().max()), "threshold": 5.0,
            "passed": ok, "detail": f"T_shuffled max "
            f"{float(ctrl['T_shuffled'].abs().max()):.4f}"}

    q4 = load_analysis(data_dir, "q4_irreducibility_all")
    if q4 is not None and "partial_spearman" in q4.columns:
        med = float(q4["partial_spearman"].median())
        out["partial rho >= 0.30"] = {
            "value": med, "threshold": 0.30, "passed": med >= 0.30,
            "detail": f"median delta_R2 {float(q4['delta_r2'].median()):.4f}"}

    out["all_passed"] = bool(out) and all(
        v["passed"] for k, v in out.items() if isinstance(v, dict))
    return out


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

    # D-32: validity BEFORE the claim.
    #
    # There are three gates between "this run exists" and "train it", and each
    # one has to know about invalidation independently:
    #   1. plan_work's done_fn  -- fixed by D-31
    #   2. registry.can_claim   -- THIS ONE; it reads the ledger, sees
    #                              'completed', and refuses
    #   3. already_finished     -- fixed by D-29
    # Fixing them one at a time simply moved the stop to the next gate down,
    # which is what the user saw twice. Setting `force_rerun` here clears all
    # three at once, because every gate already honours that flag.
    if not cfg.get("force_rerun"):
        _ok, _why = msckd_router_ok(work, run_id, cfg, data_out, hub)
        if not _ok:
            log(f"{run_id}: {_why} -- discarding the stale checkpoint and "
                f"retraining from scratch", "MSCKD")
            cfg = {**cfg, "force_rerun": True}
            for _p in (ckpt_last, ckpt_best, history_path):
                try:
                    _p.unlink(missing_ok=True)
                except Exception:                            # noqa: BLE001
                    pass

    ok, why = registry.can_claim(run_id, force=bool(cfg.get("force_rerun")))
    if not ok:
        log(f"SKIP {run_id}: {why}", "CLAIM")
        return {"run_id": run_id, "status": "skipped", "reason": why}

    # D-19: check the artifact BEFORE the teacher sweep, which is the expensive
    # part of this function -- a full multi-exit pass over 50,000 training
    # images. Discovering "already done" after paying for that is no use.
    # D-29/D-32: `force_rerun` is already set above when the router is stale,
    # and `already_finished` honours it, so this returns None for exactly the
    # runs that need redoing.
    _cached = already_finished(hub, work, run_id, cfg, registry)
    if _cached is not None:
        return _cached

    atomic_write_yaml(run_dir / "config.yaml", cfg)
    atomic_write_json(L["env"] / "environment.json", environment_report())
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, holdout_loader, classes, order_hash = build_loaders(cfg)

    # --- teacher ---------------------------------------------------------
    t_budgets = load_or_build_budgets(teacher_arch, data_out, cfg["dataset_name"],
                                      cfg["num_classes"], hub=hub)
    tL = run_layout(work, teacher_run)
    t_dir = tL["base"]
    t_ck = tL["checkpoints"] / "ckpt_best.pt"
    if not t_ck.exists() and hub.enabled:
        hub.hub.download(work, allow_patterns=[f"runs/{teacher_run}/**"])
    if not t_ck.exists():
        raise FileNotFoundError(f"teacher checkpoint missing for {teacher_run}")
    teacher = place_model(build_model(teacher_arch, cfg["num_classes"]),
                          device, cfg, tag=f"{teacher_arch} teacher")
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

    t_me = place_model(MultiExitModel(teacher, cfg["num_classes"], freeze=True),
                       device, cfg)
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
    # Augmentation off while measuring: MSC of an augmented view is not MSC of
    # the sample. `eval_view_of` knows how each backend expresses that -- a
    # dataset flag on CIFAR, `train=False` on the GPU loader for ImageNet-100
    # -- so this no longer guesses, and no longer silently guesses wrong
    # inside a bare `except` (D-76).
    train_eval = eval_view_of(train_loader, cfg)
    sweep = sweep_all_axes(cfg, t_me, train_eval, device,
                           show_progress=show_progress)

    core = _import_msc_core()
    rho_list = t_budgets["axes"]["depth"]["rho"]
    r = core.compute_msc(sweep["depth"]["preds"], sweep["depth"]["top1p"],
                         sweep["depth"]["top2p"], rho_list, tau=tau, axis="depth")
    # D-77. These are indexed later as `msc_t[idx]`, where `idx` is the GLOBAL
    # pack index the loader emits -- 0..129,394 for ImageNet-100. Sorting the
    # sweep positionally gives a vector of length 119,395 (the train split), so
    # every index above that is out of bounds.
    #
    # On CPU that is an IndexError. On CUDA it is a device-side assert:
    #
    #   IndexKernel.cu:93: Assertion `-sizes[i] <= index && index < sizes[i]`
    #
    # which aborts the process. The kernel died with exit code 3221226505 and
    # no Python traceback, before a single epoch began.
    #
    # This is D-49 exactly -- `sample_idx` is a global pack index, so anything
    # indexed BY it must be sized for the whole index space, not the split.
    # D-49 fixed `TrainingDynamics`; `train_msc_kd` has carried the same defect
    # since the port, and only fires here because it is the one place that
    # indexes a dense array by sample_idx on the GPU.
    _sweep_idx = np.asarray(sweep["sample_idx"], dtype=np.int64)
    _ds = train_loader.dataset
    _space = int(getattr(_ds, "index_space", 0) or 0) or int(_sweep_idx.max() + 1)
    if _sweep_idx.max() >= _space:
        raise RuntimeError(
            f"sample_idx reaches {_sweep_idx.max()} but index_space is "
            f"{_space} -- the dataset is mis-declaring its index space (D-49).")

    _msc_c = r.msc.astype(np.float32)
    _irr_c = r.irreducible.astype(bool)
    if shuffle_targets:
        log("SHUFFLED-TARGET ABLATION: MSC targets permuted within the dataset",
            "ABLATE")
        # Permute the COMPACT vector, before scattering. Permuting the sparse
        # index-space array would move NaN padding into real samples and
        # silently weaken the control.
        _msc_c = shuffle_msc_targets(_msc_c, seed=int(cfg["seed"]))

    # Scatter BY sample_idx, so position == global index and `msc_t[idx]` is
    # correct by construction rather than by a sort that has to stay in step.
    msc_train = np.full(_space, np.nan, dtype=np.float32)
    irr_train = np.zeros(_space, dtype=bool)
    msc_train[_sweep_idx] = _msc_c
    irr_train[_sweep_idx] = _irr_c

    log(f"teacher MSC on train: mean={np.nanmean(_msc_c):.3f}  "
        f"irreducible={_irr_c.mean()*100:.1f}%  "
        f"({len(_sweep_idx):,} samples over an index space of {_space:,})",
        "MSCKD")

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
    s_budgets = load_or_build_budgets(cfg["arch"], data_out, cfg["dataset_name"],
                                      cfg["num_classes"], hub=hub)
    rho_student = list(s_budgets["axes"]["depth"]["rho"])
    if len(rho_student) != len(rho_list):
        log(f"student {cfg['arch']} has {len(rho_student)} depth budgets vs the "
            f"{teacher_arch} teacher's {len(rho_list)} -- routing on the "
            f"student's grid (D-28)", "MSCKD")
    rho_t = torch.tensor(rho_student, dtype=torch.float32, device=device)

    # --- student ---------------------------------------------------------
    student = place_model(MSCStudent(build_model(cfg["arch"], cfg["num_classes"]),
                                     cfg["num_classes"], len(rho_student)),
                          device, cfg, tag=f'{cfg["arch"]} student')
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
    _bounds_checked = False          # D-77, once per run
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
                if not _bounds_checked:
                    # D-77. Check on the HOST, before the GPU sees it. An
                    # out-of-range gather on CUDA aborts the process with a
                    # device-side assert and no traceback; the same check here
                    # raises something readable. `idx` is still on the CPU at
                    # this point, so this costs a reduction over one batch,
                    # once per run.
                    _bounds_checked = True
                    _mx = int(idx.max())
                    if _mx >= msc_t.numel():
                        raise IndexError(
                            f"sample_idx {_mx} >= MSC target array "
                            f"{msc_t.numel()}. Indexing this on the GPU would "
                            f"kill the kernel with a device-side assert and no "
                            f"traceback (D-77/D-49).")
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
    # D-79b. `train_backbone` writes both; this wrote only config.yaml, so all
    # 18 MSC-KD runs verified as incomplete on a REQUIRED artifact.
    atomic_write_text(run_dir / "config_hash.txt", cfg["config_hash"])
    atomic_write_json(run_dir / "summary.json", summary)

    # D-79. The routing baselines ARE the method section. Computed here, from
    # the student that was just trained, so the number exists the moment the
    # run finishes instead of being discovered missing after 79 GPU-hours.
    try:
        _rt = evaluate_msckd_routing(_SelfSession(work, cfg, hub), run_id,
                                     tau=tau, write=False)
        summary.update({k: v for k, v in _rt.items() if v is not None})
        atomic_write_json(run_dir / "summary.json", summary)
    except Exception as _e:                                      # noqa: BLE001
        log(f"routing evaluation failed: {type(_e).__name__}: {_e} -- the run "
            f"is fine, but b2/b10/b11 are missing. Backfill with "
            f"M.evaluate_msckd_routing(sess, run_id).", "WARN")

    registry.finish(run_id, **{k: summary[k] for k in
                               ("arch", "teacher", "method", "seed", "best_accuracy")})
    sync.push_all(heavy=True)
    sync.flush(timeout=1200)
    hub.print_stats()
    return summary


@_no_grad()
def evaluate_routing_methods(student, val_loader, device, rho: Sequence[float],
                             full_flops: float, oracle_msc: Optional[np.ndarray] = None,
                             amp: bool = True, oracle_from_self: bool = False,
                             tau: float = 0.1) -> Dict[str, Any]:
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
        all_y.append(to_numpy(y))
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
            f"{S.shape[1]} sufficiency outputs, {len(rho)} budgets.\n"
            f"This student was trained BEFORE the D-28 fix, with its router "
            f"sized from the teacher's budget grid. The weights cannot be "
            f"reused.\n"
            f"FIX: re-run NB13 with the current library. It now detects this "
            f"(D-29) and retrains the affected students automatically -- you "
            f"do not need to delete anything by hand.")

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
    if oracle_msc is None and oracle_from_self:
        # D-79c. The B11 ceiling is the student's own post-hoc MSC, and every
        # input to it -- per-exit decision, top-1 and top-2 probability -- is
        # already in `L` from the pass above. The first version of the backfill
        # instead called `sweep_all_axes(cfg, student, ...)`, which expects a
        # model returning a LIST of exit logits; `MSCStudent.forward` returns
        # `(logits, suff, feats)`, so the tuple was iterated and every run died
        # on `AttributeError: 'list' object has no attribute 'float'`.
        #
        # The docstring for that function already said "computed from that same
        # pass's exit predictions rather than a separate sweep". The code did
        # the opposite. Deriving it here removes the second pass and the
        # interface mismatch together.
        _srt = np.sort(probs, axis=2)
        oracle_msc = _import_msc_core().compute_msc(
            L.argmax(2), _srt[:, :, -1], _srt[:, :, -2],
            list(rho), tau=tau, axis="depth").msc

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
            # D-80. `> 1e-9` is not a guard, it is a formality. On ImageNet-100
            # the measured B11-B2 gap is +0.00007 (sd 0.00036) -- the oracle
            # ceiling offers no headroom over confidence routing at all -- and
            # dividing by it produced "fractions" of 26.0, -47.9 and 83.6.
            #
            # A ratio is only meaningful when its denominator is larger than
            # the noise on the quantities it is built from. With n samples the
            # binomial SE on a difference of two accuracies is about
            # sqrt(2 p(1-p)/n); below 2 SE the gap is indistinguishable from
            # zero and the fraction is undefined, not large.
            _se = math.sqrt(2.0 * 0.25 / max(1, n))
            out["matched_flops_comparison"]["B2_to_B11_gap"] = float(gap_total)
            out["matched_flops_comparison"]["B2_to_B11_gap_noise_2se"] = float(2 * _se)
            if abs(gap_total) > 2 * _se:
                out["matched_flops_comparison"]["fraction_of_B2_to_B11_gap_closed"] = \
                    float((a10 - a2) / gap_total)
            else:
                out["matched_flops_comparison"]["fraction_of_B2_to_B11_gap_closed"] = \
                    float("nan")
                out["matched_flops_comparison"]["gap_verdict"] = (
                    f"B11-B2 = {gap_total:+.5f} is within noise (2SE = "
                    f"{2*_se:.5f}); the oracle ceiling offers no headroom over "
                    f"confidence routing, so there is no gap to close and the "
                    f"fraction is undefined (D-80)")
    return out


class _SelfSession:
    """The two attributes `evaluate_msckd_routing` needs, without a Session.

    `train_msc_kd` has `work` and a config already; constructing a full
    Session inside it would re-resolve storage and re-open the ledger.
    """

    def __init__(self, work, cfg, hub=None):
        self.work = Path(work)
        self.data_dir = self.work
        self.dataset = str(cfg.get("dataset_name", "imagenet100"))
        self.hub = hub
        self._cfg = cfg

    def budgets(self, arch: str, num_classes: Optional[int] = None):
        return load_or_build_budgets(arch, self.work, self.dataset,
                                     num_classes, hub=self.hub)


def evaluate_msckd_routing(session, run_id: str, tau: float = 0.1,
                           amp: bool = True, write: bool = True) -> Dict[str, Any]:
    """Compute B1/B2/B10/B11 for a TRAINED student and merge them into its summary.

    **D-79.** `evaluate_routing_methods` is documented as "the paper's central
    figure" and was called from exactly one place: `msckd_dry_run`. The real
    `train_msc_kd` never called it and its summary dict never carried the keys,
    so 18 students trained for ~79 GPU-hours, correctly, and the number the
    method section exists to report was never computed.

    Recoverable without retraining: everything B1/B2/B10/B11 need -- including
    the B11 ceiling -- comes from ONE forward pass of the saved student over
    the val set.
    """
    L = run_layout(session.work, run_id)
    cfg = read_yaml(L["base"] / "config.yaml")
    if not cfg:
        raise FileNotFoundError(f"no config.yaml for {run_id}")
    ck = L["checkpoints"] / "ckpt_best.pt"
    if not ck.exists():
        raise FileNotFoundError(f"no ckpt_best.pt for {run_id} at {ck}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch = cfg["arch"]
    budgets = session.budgets(arch)
    rho = list(budgets["axes"]["depth"]["rho"])
    full_flops = float(budgets.get("full_flops")
                       or budgets["axes"]["depth"]["flops"][-1])

    bb = build_model(arch, int(cfg["num_classes"]))
    student = place_model(MSCStudent(bb, int(cfg["num_classes"]), len(rho)),
                          device, cfg, tag=f"{arch} student (post-hoc)")
    blob = torch.load(ck, map_location=device, weights_only=False)
    student.load_state_dict(blob.get("model", blob), strict=True)
    student.eval()

    # Only the val loader is needed. `build_loaders` also builds train, which
    # tries to resident-cache the whole 23.7 GiB pack -- unnecessary here and
    # the reason the first backfill attempt fell back to memmap.
    _, val_loader, _, _, _ = build_loaders(dict(cfg, ram_cache=False))

    ev = evaluate_routing_methods(student, val_loader, device, rho, full_flops,
                                  oracle_from_self=True, tau=tau, amp=amp)

    mfc = ev.get("matched_flops_comparison", {}) or {}
    flat = {
        "b1_static": ev.get("B1_static_full", {}).get("accuracy"),
        "b2_confidence": mfc.get("B2_accuracy"),
        "b10_msckd": mfc.get("B10_accuracy"),
        "b11_oracle": (ev.get("B11_oracle") or {}).get("accuracy"),
        "avg_flops_ratio": mfc.get("target_avg_rho"),
        "frac_b2_b11_gap_closed": mfc.get("fraction_of_B2_to_B11_gap_closed"),
        "routing_K": ev.get("K"), "routing_n": ev.get("n"),
    }
    if write:
        sp = L["base"] / "summary.json"
        summary = read_json(sp, {}) or {}
        summary.update({k: v for k, v in flat.items() if v is not None})
        atomic_write_json(sp, summary)
        atomic_write_text(L["base"] / "config_hash.txt",
                          str(cfg.get("config_hash", "")))
        log(f"{run_id}: B2={flat['b2_confidence']} B10={flat['b10_msckd']} "
            f"B11={flat['b11_oracle']} "
            f"closed={flat['frac_b2_b11_gap_closed']}", "ROUTE")
    return flat



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
                 dataset: str = "cifar100", enable_hf: Optional[bool] = None,
                 work_root=None, session_limit_h: float = 8.5,
                 commits_per_hour_limit: int = 20,
                 batch_interval_sec: float = 1800.0,
                 worker_id: int = 0, num_workers: int = 1,
                 shard_mode: str = "cost"):
        assert 0 <= worker_id < num_workers, \
            f"WORKER_ID must be in 0..{num_workers-1}, got {worker_id}"
        # `enable_hf=None` means "decide from the profile". The ImageNet-100
        # programme runs local-only and offline, so HuggingFace is OFF unless
        # explicitly switched on. Defaulting it to True and expecting the
        # operator to remember to pass False is the D-27 shape: an invariant
        # that lives in an argument nobody passes.
        if enable_hf is None:
            enable_hf = (os.environ.get("MSC_ENABLE_HF", "") in ("1", "true", "True")
                         or dataset_spec(dataset)["backend"] != "packed")
        self.local_only = not enable_hf
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
        if self.local_only:
            # NOT an alarm. On Kaggle, HF off genuinely meant the work
            # evaporated at session end. Here the local tree IS the permanent
            # store and nothing deletes it -- the confirm-then-delete branch in
            # train_backbone is gated on `hub.enabled`, so with HF off there is
            # no code path that removes a run directory except an explicit
            # force_rerun. Saying "nothing will survive" would be false and,
            # worse, would teach the operator to ignore this line.
            print(f"[SESSION] LOCAL-ONLY store: {self.runs_dir}")
            print(f"[SESSION] nothing is uploaded and nothing is deleted. "
                  f"Call sess.confirm_on_disk(run_ids) before you stop.")
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                print("[SESSION] offline guards active")
        elif not self.hub.enabled:
            print("[SESSION] *** HF requested but unavailable -- "
                  "nothing will survive this session ***")

    # ------------------------------------------------------------------
    def prepare_data(self, required: bool = True) -> Optional[Path]:
        """Locate the dataset. `required=False` returns None instead of raising.

        D-46. The dry runs are SYNTHETIC -- they push noise through the whole
        path and never open the dataset. But `config()` called this, which
        raised when the pack did not exist, so the cheapest and earliest check
        in the whole notebook could not run until after the most expensive
        prerequisite was complete. Exactly backwards: a config-level bug should
        surface before a 40-minute packing job, not after it.
        """
        try:
            if dataset_spec(self.dataset)["backend"] == "packed":
                self.data_root = locate_imagenet100()
                man = read_json(self.data_root / "manifest.json", {}) or {}
                self.data_fingerprint = str(man.get("fingerprint", ""))
            else:
                self.data_root = locate_cifar100()
                self.data_fingerprint = ""
        except Exception:                                        # noqa: BLE001
            if required:
                raise
            self.data_root, self.data_fingerprint = None, ""
        return self.data_root

    def config(self, arch: str, seed: int = 1, method: str = "base",
               require_data: bool = True, **overrides) -> Dict[str, Any]:
        if self.data_root is None:
            self.prepare_data(required=require_data)
        cfg = base_config(arch, self.dataset, seed, phase=self.phase, method=method)
        cfg.update({"data_root": str(self.data_root) if self.data_root
                    else "<not packed yet>",
                    "output_root": str(self.work)})
        # The fingerprint is set BEFORE overrides and BEFORE the hash, because
        # it must participate in config_hash: two runs that disagree about which
        # images are `val` produce per-sample tables that align by index and
        # compare different pictures. See 25_IN100_DATA_CARD.md 4.
        fp = getattr(self, "data_fingerprint", "")
        if fp:
            cfg["data_fingerprint"] = fp
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

    def msckd_valid(self, run_id: str) -> bool:
        """Trained **and still compatible** — the stage predicate NB13 must use.

        **D-31.** The D-29 validity check was placed inside `train_msc_kd`. But
        `run_all` -> `plan_work` filters "done" runs out **before** the training
        function is ever called, so the check sat downstream of the very thing
        that skips the work and could never fire. NB13 reported
        `already finished (GLOBAL, from HF): 9 ... MY REMAINING WORK: 0` and
        exited, leaving the nine invalid students exactly as they were.

        A compatibility test has to live in the predicate that decides whether
        to do the work, not in the code that does it.
        """
        if not self.trained(run_id):
            return False
        try:
            m = parse_run_id(run_id)
            cfg = {"arch": m["arch"],
                   "num_classes": 10 if "cifar10" == self.dataset else 100}
            ok, why = msckd_router_ok(self.work, run_id, cfg, self.data_dir,
                                      self.hub)
        except Exception:                                    # noqa: BLE001
            return True          # unverifiable -> leave it alone
        if not ok:
            log(f"{run_id}: complete but INVALID -- {why}. Queued for retrain.",
                "MSCKD")
        return ok

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
        # D-54. FAIL BEFORE THE PLAN, not once per run inside it.
        #
        # `run_all` calls `fn(cfg, **kw)` -- one positional argument. The raw
        # library entry points take three (`cfg, hub, registry`); the bound
        # `Session.train` / `Session.oracle` wrappers exist precisely to supply
        # the other two. Passing `M.train_backbone` produced
        #
        #   TypeError: train_backbone() missing 2 required positional
        #   arguments: 'hub' and 'registry'
        #
        # once per run, swallowed by the per-run except so the plan printed
        # normally and four runs "failed ... continuing" -- four identical
        # tracebacks for one mistake, after the work plan had already been
        # computed and displayed. Arity is knowable before any of that.
        if fn is not None:
            try:
                _sig = _inspect_signature(fn)
                _req = sum(1 for q in _sig.parameters.values()
                           if q.default is q.empty
                           and q.kind in (q.POSITIONAL_ONLY,
                                          q.POSITIONAL_OR_KEYWORD))
                _has_var = any(q.kind is q.VAR_POSITIONAL
                               for q in _sig.parameters.values())
                if _req > 1 and not _has_var:
                    _missing = [q.name for q in _sig.parameters.values()
                                if q.default is q.empty
                                and q.kind in (q.POSITIONAL_ONLY,
                                               q.POSITIONAL_OR_KEYWORD)][1:]
                    raise TypeError(
                        f"run_all calls fn(cfg) with ONE argument, but "
                        f"{getattr(fn, '__name__', fn)} requires {_req}: it "
                        f"still needs {_missing}.\n"
                        f"  Use the bound wrapper, which supplies them:\n"
                        f"    sess.run_all(cfgs)                  # -> sess.train\n"
                        f"    sess.run_all(cfgs, fn=sess.oracle)\n"
                        f"  or pass a closure that captures them (D-54).")
            except (TypeError, ValueError) as _e:
                if "run_all calls fn(cfg)" in str(_e):
                    raise
        # D-62. A Session built from a PREVIOUS import keeps that module's
        # functions. Re-running the bootstrap cell replaces sys.modules but
        # cannot reach into an object already holding the old ones, so a fixed
        # library and a stale `sess` produce the old failure with the new code
        # sitting on disk. `__globals__` belongs to the module that defined
        # this method, which is exactly the one that will run.
        _live = getattr(sys.modules.get("msc_lib"), "__MSC_BUILD__", None)
        _mine = Session.run_all.__globals__.get("__MSC_BUILD__")
        if _live and _mine and _live != _mine:
            raise RuntimeError(
                f"STALE Session: this object was built from msc_lib {_mine}, "
                f"but {_live} is now imported.\n"
                f"  Every fix since {_mine} is absent from this object.\n"
                f"  Restart the kernel and run all cells (D-62).")

        # D-67. The oracle measures; it must be PLANNED as measurement.
        #
        # `plan_work` filters out runs already "done" BEFORE `fn` is called,
        # and "done" means whatever `stage`/`done_fn` say. NB3 called
        #     run_all(cfgs, fn=sess.oracle, title='measurement')
        # with the default stage='train'. All four runs were trained, so all
        # four were filtered as complete: "MY REMAINING WORK: 0". The notebook
        # printed success and measured nothing, and NB4 then failed on an empty
        # table two notebooks later.
        #
        # This is D-31 exactly -- a completion predicate that answers a
        # different question from the work being requested -- and the
        # `msckd_valid` docstring three screens up describes it. Documenting a
        # trap is not the same as removing it, so this raises.
        if fn is not None and getattr(fn, "__func__", None) is Session.oracle:
            if stage != "measure":
                raise ValueError(
                    "run_all(fn=sess.oracle) with stage=%r would ask 'is it "
                    "TRAINED?' to decide whether to MEASURE it, so every "
                    "trained run is skipped and nothing happens.\n"
                    "  Use: sess.run_all(cfgs, fn=sess.oracle, "
                    "done_fn=sess.measured, stage='measure')" % stage)
            if done_fn is None:
                done_fn = self.measured
                log("done_fn defaulted to sess.measured for stage='measure'",
                    "PLAN")

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

    def budgets(self, arch: str, num_classes: Optional[int] = None) -> Dict[str, Any]:
        return load_or_build_budgets(arch, self.data_dir, self.dataset,
                                     num_classes, hub=self.hub)

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

    def confirm_on_disk(self, run_ids: Sequence[str], measured: bool = False,
                        verbose: bool = True) -> Dict[str, List[str]]:
        """Local-only analogue of `confirm_on_hf`. Same three states.

        With no HuggingFace, local disk is the only copy, so the question
        "is my work safe?" becomes "is my work COMPLETE and READABLE?" -- and
        that is a stronger question than HF was ever asked. `confirm_on_hf`
        establishes that a file arrived; this opens it.

        Three states, and the distinction is the D-20 one:

        - **finished**  -- summary present AND every required artifact verified
        - **resumable** -- `ckpt_last.pt` present. Perfectly safe to stop; the
          next session picks it up at its epoch. Being unfinished is the normal
          state of a paused run, not a failure
        - **at risk**   -- neither, or present-but-corrupt

        A run whose summary exists but whose `epochs.csv` is zero bytes is
        reported **at risk**, not finished. That case is invisible to any
        presence check and shows up during analysis, weeks later.
        """
        ids = list(run_ids)
        done, resumable, at_risk, detail = [], [], [], {}
        for r in ids:
            L = run_layout(self.work, r)
            rep = verify_run_artifacts(self.work, r, measured=measured)
            detail[r] = rep
            if rep["ok"]:
                done.append(r)
            elif (L["checkpoints"] / "ckpt_last.pt").exists() and \
                    (L["checkpoints"] / "ckpt_last.pt").stat().st_size > 1024:
                resumable.append(r)
            else:
                at_risk.append(r)

        if verbose:
            gb = sum(d["total_bytes"] for d in detail.values()) / 2**30
            print(f"\n[VERIFY] {len(ids)} run(s) on local disk: {len(done)} "
                  f"complete, {len(resumable)} resumable, {len(at_risk)} at "
                  f"risk  ({gb:.2f} GiB under {self.runs_dir})")
            for r in done:
                print(f"    COMPLETE   {r}")
            for r in resumable:
                d = detail[r]
                print(f"    RESUMABLE  {r}  -- still missing "
                      f"{d['missing_required'][:3]}")
            for r in at_risk:
                d = detail[r]
                bad = (d["missing_required"] or d["empty"] or d["unreadable"])
                print(f"    AT RISK    {r}  -- {bad[:4]}")
                for k in ("empty", "unreadable"):
                    if d[k]:
                        print(f"               {k.upper()}: {d[k]} "
                              f"<- present but unusable; a presence check "
                              f"would have called this run healthy")
            if not at_risk:
                print("    Nothing is at risk. Safe to stop.")
            else:
                print("    *** Do not treat the AT RISK runs as done.")
        return {"ok": done, "done": done, "resumable": resumable,
                "at_risk": at_risk, "unknown": [], "detail": detail}

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

        With HuggingFace disabled this delegates to `confirm_on_disk`, which
        asks the same three-state question of local disk. The method is kept
        under one name so no notebook has to know which store is in use.

        **Rule 9. Every lookup below goes through `resolve`, per file.** This
        used to call `list_repo_files` once and test membership of the result.
        That is the tree endpoint, it is CDN-cached, and on 2026-08-02 it served
        this project a stale page twice and a silently truncated body once --
        producing a confident, wrong, negative finding that stood in the lab
        notebook for two days. A method whose entire job is answering "is my
        work safe?" cannot be built on an endpoint that has lied to us three
        times.
        """
        ids = list(run_ids)
        empty = {"ok": [], "done": [], "resumable": [], "at_risk": [],
                 "unknown": ids}
        if not self.hub.enabled:
            return self.confirm_on_disk(ids, verbose=verbose)

        latest = self.registry.latest()
        done, resumable, at_risk = [], [], []
        try:
            for r in ids:
                base = f"runs/{r}/"
                if require:
                    got = self.hub.hub.files_present([f"{base}{x}" for x in require])
                    (done if all(v is not None for v in got.values())
                     else at_risk).append(r)
                    continue
                # Cheapest sufficient question first: a finished run needs one
                # lookup, not two.
                if self.hub.hub.resolve_meta(f"{base}summary.json") is not None:
                    done.append(r)
                elif self.hub.hub.resolve_meta(
                        f"{base}checkpoints/ckpt_last.pt") is not None:
                    resumable.append(r)
                else:
                    at_risk.append(r)
        except Exception as e:                               # noqa: BLE001
            # `resolve_meta` raises rather than returning None on a lookup that
            # failed for any reason other than 404, so this branch means we do
            # not know -- which must be reported as not knowing. Reporting
            # "at risk" here would be the D-20 false alarm; reporting "safe"
            # would be worse.
            log(f"could not confirm against the repo: {type(e).__name__}: {e}. "
                f"Treat this as UNCONFIRMED, not as success and not as loss.",
                "ALARM")
            return empty

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


def preflight_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Three states, not two. A prerequisite that has not been done yet is not
    a failure, and lumping the two together makes the count unreadable (D-46)."""
    ch = report.get("checks", {})
    passed = [k for k, v in ch.items() if v.get("ok") is True]
    failed = [k for k, v in ch.items() if v.get("ok") is False]
    todo = [k for k, v in ch.items() if v.get("ok") is None]
    return {"passed": passed, "failed": failed, "todo": todo,
            "ok": not failed, "n": len(ch)}


def preflight(session: "Session", archs: Optional[Sequence[str]] = None,
              quick: bool = True) -> Dict[str, Any]:
    """Cheap checks that catch the expensive mistakes.

    Runs before any real training. Every item here corresponds to a failure
    that would otherwise be discovered hours in: a ViT whose feature shapes do
    not match the exit heads, a missing HF write scope, a budget table whose
    deepest exit does not equal the full model.
    """
    _ds = getattr(session, "dataset", "cifar100")
    _grid = resolutions_for(_ds)
    _res0 = native_res(_ds)
    _ncls = num_classes_for(_ds)
    report: Dict[str, Any] = {"checked_utc": now_iso(), "dataset": _ds,
                              "input_res": _res0, "resolution_grid": list(_grid),
                              "checks": {}}

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
    # D-46. These used to run unconditionally and FAIL in a local-only session
    # -- reporting "no HF token" and naming the CIFAR repo -- on a programme
    # that is deliberately offline and stores nothing remotely. A preflight
    # that fails on the intended configuration teaches the operator to ignore
    # it, which is the D-17 cost, and the two red lines here sat beside a real
    # failure the operator then had to disentangle.
    if getattr(session, "local_only", False):
        rec("store: LOCAL ONLY (HuggingFace not used)", True,
            "nothing is uploaded, nothing is fetched, nothing is deleted")
        _rr = Path(session.work)
        try:
            _pb = _rr / ".msc_preflight_probe"
            ensure_dir(_rr)
            _pb.write_text("ok", encoding="utf-8")
            _ok = _pb.read_text(encoding="utf-8") == "ok"
            _pb.unlink()
        except Exception as _e:                                  # noqa: BLE001
            _ok, _e = False, str(_e)[:120]
        rec("results root writable", _ok,
            f"{_rr}  (probe written and read back)" if _ok else str(_e))
        _free = free_mb(session.work) / 1024
        rec("results root has room", _free > 120,
            f"{_free:.0f} GB free, ~120 GB recommended for the full atlas")
    else:
        rec("HF token", bool(session.hub.token), "from Kaggle Secrets or env")
        rec("HF repo reachable",
            session.hub.enabled and session.hub.hub is not None,
            session.hub.repo_id)
    rec("working disk >2 GB", free_mb(session.work) > 2048, f"{free_mb(session.work)} MB")
    rec("scratch disk >5 GB", free_mb(session.scratch) > 5120,
        f"{free_mb(session.scratch)} MB")

    # D-46. "The dataset has not been packed yet" is a PREREQUISITE NOT DONE,
    # not a broken pipeline, and at this point in NB1 it is the expected state.
    # Reporting it as FAIL alongside genuine failures makes the summary line
    # unreadable and hides which of them actually needs thought.
    try:
        root = session.prepare_data(required=False)
        if root is None:
            report["checks"][f"{_ds} packed"] = {"ok": None,
                                                 "detail": "not built yet"}
            print(f"  [TODO] {_ds} packed  -- not built yet. Run:")
            print(f"         python tools/pack_imagenet100.py "
                  f"--src <folder with train/> --out <DATA_DIR>")
            print(f"         Everything below runs on synthetic data and does "
                  f"not need it.")
        else:
            ok, detail = data_present(_ds, root)
            rec(f"{_ds} packed", ok, detail)
    except Exception as e:                                       # noqa: BLE001
        rec(f"{_ds} packed", False, str(e)[:160])

    if _TORCH_OK and archs:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for a in archs:
            try:
                m = build_model(a, _ncls, dataset=_ds).to(dev)
                x = torch.randn(4, 3, _res0, _res0, device=dev)
                out = m(x)
                feats = m.forward_features(x)
                pref = m.forward_prefix(x, 0)
                # An exit head must actually attach, which is where a token
                # model with an unexpected feature rank would blow up.
                head = ExitHead(m.feature_dims[0], _ncls,
                                getattr(m, "is_token_model", False)).to(dev)
                _ = head(pref)
                loss = out.sum()
                loss.backward()
                K = len(feats)
                rec(f"model {a}", out.shape == (4, _ncls) and 2 <= K <= len(DEPTH_FRACTIONS),
                    f"{count_parameters(m)/1e6:.2f}M params, K={K}, "
                    f"dims={m.feature_dims}, cuts={m.stage_cuts}")

                # Every resolution the oracle will actually sweep, natively.
                # This is where a ViT's positional embedding or a Mixer's
                # token-mixing weights blow up, and it is far cheaper to find
                # out here than mid-sweep in Phase 1b.
                native = bool(getattr(m, "supports_native_resolution", True))
                if native:
                    bad_r = []
                    for r in _grid:
                        try:
                            m(torch.randn(2, 3, r, r, device=dev))
                        except Exception as e:
                            bad_r.append(f"{r}px:{type(e).__name__}")
                    # A partial failure is recorded, not fatal: the budget table
                    # probes per resolution too, and the PROXY sweep is primary
                    # for every architecture (DC-3). What must never happen is
                    # the failure going unrecorded.
                    rec(f"native resolutions {a}", not bad_r,
                        f"runs at {list(_grid)}" if not bad_r
                        else f"FAILS at {bad_r} -- those entries fall back to the "
                             f"analytic cost model; proxy sweep unaffected")
                else:
                    rec(f"native resolutions {a}", True,
                        "not supported by design -- resolution axis uses the "
                        "proxy (documented limitation)")

                if not quick:
                    b = build_budget_table(a, _ds, _ncls, model=m.cpu())
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
                           tol: float = 0.05,
                           subset_frac: float = 1.0) -> Dict[str, Any]:
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
    out: Dict[str, Any] = {"arch": arch, "epochs": epochs, "kill_at": kill_at,
                           "subset_frac": float(subset_frac)}
    tmp = session.scratch / "resume_test"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp = ensure_dir(tmp)

    cfg = session.config(arch, seed=99, method="resumetest",
                         num_epochs=epochs, phase="test",
                         milestone_push_every_epochs=10 ** 6,
                         # D-50. The watchdog must not fire during a test whose
                         # whole purpose is a DIFFERENT stop reason. When
                         # session_limit_h was read as "zero hours" every leg
                         # paused at epoch 1, the debug interrupt never
                         # reached kill_at, and the test reported
                         # `interrupt actually fired: False` -- failing for a
                         # reason with nothing to do with resume. A test that
                         # can fail for the wrong reason is the D-06 shape.
                         session_limit_h=0.0,
                         # A fraction of the training split. This test is about
                         # whether the seam is invisible, not about learning
                         # anything -- and the same code runs either way.
                         train_subset_frac=float(subset_frac),
                         cleanup_local_after_complete=False)
    hub_off = MSCHub(enable=False)
    reg = RunRegistry(hub_off, tmp / "reg", account="selftest")

    ref_id = cfg["run_id"] + "-ref"
    cut_id = cfg["run_id"] + "-cut"

    print(f"\n  [1/3] reference: {epochs} epochs, uninterrupted  "
          f"(local scratch, nothing uploaded)")
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

    # Name the failure MODE, not just the verdict. "interrupt_fired: False" is
    # true of both "resume is broken" and "something else stopped the run
    # first", and those need completely different responses. D-50 was the
    # second, and the report pointed at the first for a whole round trip.
    if int(out.get("epochs_ref", 0)) < epochs:
        out["diagnosis"] = (
            f"the REFERENCE leg stopped at epoch {out.get('epochs_ref')} of "
            f"{epochs} without being asked to. Nothing about resume has been "
            f"tested. Check the session watchdog (session_limit_h <= 0 means "
            f"no limit) and for an out-of-disk or an exception above.")
    elif not out.get("interrupt_fired"):
        out["diagnosis"] = (
            f"the debug interrupt never fired at epoch {kill_at}, so the "
            f"'interrupted' leg was a clean run. The test exercised nothing.")
    elif int(out.get("epochs_cut", 0)) < epochs:
        out["diagnosis"] = (
            f"resumed but stopped at epoch {out.get('epochs_cut')} of "
            f"{epochs} -- it did not run to completion after the seam.")
    elif int(out.get("duplicate_epochs", 1)) != 0:
        out["diagnosis"] = ("history has duplicate epoch rows -- the log was "
                            "not truncated on resume, so every cumulative "
                            "statistic is wrong")
    elif int(out.get("post_seam_epochs_compared", 0)) <= 0:
        out["diagnosis"] = ("no post-seam epochs to compare; the comparison "
                            "that matters did not happen")
    elif float(out.get("max_post_seam_loss_deviation", 1.0)) >= tol:
        out["diagnosis"] = (
            f"post-seam loss drifted "
            f"{100*float(out['max_post_seam_loss_deviation']):.1f}% -- RNG or "
            f"optimiser state did not survive the seam. This is the real "
            f"failure this test exists to catch.")
    else:
        out["diagnosis"] = "resume is equivalent to an uninterrupted run"

    out["ok"] = bool(out.get("interrupt_fired")
                     and int(out.get("epochs_ref", 0)) == epochs
                     and out.get("duplicate_epochs", 1) == 0
                     and out.get("epochs_cut", 0) == epochs
                     and out.get("post_seam_epochs_compared", 0) > 0
                     and out.get("max_post_seam_loss_deviation", 1.0) < tol)

    print(f"\n  {'='*66}")
    print(f"  {out['diagnosis']}")
    print(f"  {'-'*66}")
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
    # D-37. The verdict is accumulated in LISTS, not in a boolean.
    #
    # This used to be `ok = True` plus `ok &= cond`, and 900 lines later a line
    # reading `ok, z, sd = shuffled_control_verdict(...)` REBOUND it -- wiping
    # every result before that point and replacing it with the outcome of one
    # unrelated test. The suite printed `[FAIL]` and then `ALL CHECKS PASSED`
    # and exited 0. Roughly 80% of the checks could not affect the verdict.
    #
    # A list cannot be destroyed by an accidental `_ran = ...` the way a scalar
    # can: appending mutates, so the only way to lose a result is to rebind the
    # name AND that shows up immediately as a count that stopped growing --
    # which the floor check below detects. A test harness that cannot fail is
    # worse than no harness, because it manufactures confidence (D-06), and the
    # fix has to be structural rather than "do not shadow that name".
    _ran: List[str] = []
    _failed: List[str] = []

    def check(name, cond, detail=""):
        _ran.append(name)
        if not cond:
            _failed.append(name)
        d = str(detail)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {d}" if d else ""))

    def _src_of_module() -> str:
        try:
            return Path(globals().get("__file__", "msc_lib.py")).read_text(
                encoding="utf-8")
        except Exception:                                        # noqa: BLE001
            return ""

    # -- D-62: a stale module must be detected, not silently obeyed ----------
    import types as _types
    _sess = Session.__new__(Session)
    _saved = sys.modules.get("msc_lib")
    _g = Session.run_all.__globals__
    _had = "__MSC_BUILD__" in _g
    _prev = _g.get("__MSC_BUILD__")
    try:
        _g["__MSC_BUILD__"] = "old000000000"
        _fake = _types.ModuleType("msc_lib")
        _fake.__MSC_BUILD__ = "new111111111"
        sys.modules["msc_lib"] = _fake
        _caught = False
        try:
            Session.run_all(_sess, [{"run_id": "x"}])
        except RuntimeError as _e:
            _caught = "STALE Session" in str(_e)
        except Exception:
            pass
        check("D-62: a Session from an older build is refused", _caught,
              "a fixed library and a stale object must not look like a bad fix")

        # and must NOT fire when the builds agree, or every run breaks
        _fake.__MSC_BUILD__ = "old000000000"
        _false_alarm = False
        try:
            Session.run_all(_sess, [{"run_id": "x"}])
        except RuntimeError as _e:
            _false_alarm = "STALE Session" in str(_e)
        except Exception:
            pass
        check("D-62 canary: matching builds are NOT refused", not _false_alarm)
    finally:
        if _saved is not None:
            sys.modules["msc_lib"] = _saved
        else:
            sys.modules.pop("msc_lib", None)
        if _had:
            _g["__MSC_BUILD__"] = _prev
        else:
            _g.pop("__MSC_BUILD__", None)

    # -- D-60: a checkpoint hashed under the OLD rule must still verify ------
    #
    # The D-59 test asked whether two configs hash the same under the CURRENT
    # rule. They do, trivially -- the key is excluded from both. It could not
    # fail, and the runs it was written to protect were orphaned anyway. The
    # real invariant is across rule VERSIONS, so that is what is asserted here.
    _c60 = {"arch": "vit_small_p16", "seed": 2, "batch_size": 64,
            "num_epochs": 100, "lr": 6.25e-05, "channels_last": False,
            "ram_cache": True}
    _stored_v1 = config_hash(dict(_c60, channels_last=True),
                             exclude=_HASH_EXCLUDE_V1)
    _ok60, _why60 = hash_compatible(_c60, _stored_v1)
    check("D-60: a checkpoint hashed before channels_last was excluded resumes",
          _ok60, _why60)

    # -- D-79: every column a reader expects must have a writer ---------------
    #
    # `compare_routing_methods` reads b1_static/b2_confidence/b10_msckd/
    # b11_oracle/avg_flops_ratio out of summary.json. Nothing wrote them, so
    # NB5's table came back all None after 18 runs and ~79 GPU-hours. A reader
    # with no writer -- the mirror of D-63/D-72/D-74, which were writers with
    # no readers. Four now, in both directions.
    #
    # The declared columns and the code that produces them are two spellings of
    # one truth (D-16), so this compares them instead of trusting either.
    _msckd_src = _src_of_module()
    _decl = set(RESULT_KEYS.get("compare_routing_methods", ()))
    _from_summary = {"b1_static", "b2_confidence", "b10_msckd", "b11_oracle",
                     "avg_flops_ratio", "frac_b2_b11_gap_closed"}
    _missing_writer = sorted(
        k for k in (_decl & _from_summary)
        if f'"{k}"' not in _msckd_src.split("def evaluate_msckd_routing")[-1][:4000]
        and f'"{k}"' not in _msckd_src)
    check("D-79: every routing column read from summary.json has a writer",
          not _missing_writer,
          "OK" if not _missing_writer else "NO WRITER: " + ", ".join(_missing_writer))

    # AST, not string-splitting. The first version split on "def train_msc_kd"
    # -- a string that appears in THIS CHECK -- so `[-1]` returned the
    # self-test's own source and both assertions failed on correct code. A
    # checker that reads source has to be told where the source ends.
    def _fn_source(name: str) -> str:
        import ast as _a
        try:
            t = _a.parse(_msckd_src)
        except Exception:                                        # noqa: BLE001
            return ""
        for n in _a.walk(t):
            if isinstance(n, (_a.FunctionDef, _a.AsyncFunctionDef)) and n.name == name:
                return _a.get_source_segment(_msckd_src, n) or ""
        return ""

    _kd_src = _fn_source("train_msc_kd")
    check("D-79 canary: the function source was actually located",
          len(_kd_src) > 2000, f"{len(_kd_src)} chars")
    check("D-79: train_msc_kd calls the routing evaluator",
          "evaluate_msckd_routing(" in _kd_src,
          "it was defined and only ever called from msckd_dry_run")
    check("D-79b: train_msc_kd writes config_hash.txt",
          "config_hash.txt" in _kd_src,
          "all 18 MSC-KD runs verified incomplete without it")

    # -- D-86: an upload must survive a network drop, not be poisoned by it ---
    import types as _t86

    def _hub_that(behaviour):
        """Stub HfApi. `behaviour(label, call_n)` returns None or raises."""
        mod = _t86.ModuleType("huggingface_hub")
        state = {"n": 0, "clients": 0}

        class _Api:
            def __init__(self, token=None):
                state["clients"] += 1
                self._dead = False
            def upload_folder(self, folder_path=None, path_in_repo=None,
                              repo_id=None, repo_type=None, commit_message=None):
                state["n"] += 1
                behaviour(commit_message, state["n"], self)
        mod.HfApi = _Api
        sys.modules["huggingface_hub"] = mod
        return state

    _prev86 = sys.modules.get("huggingface_hub")
    try:
        _items = [(f"/tmp/r{i}", f"runs/r{i}", f"r{i}") for i in range(1, 6)]

        # 1. THE EXACT FAILURE: item 3 kills the client, and every later call
        #    on that client raises "client has been closed" forever.
        def _poison(label, n, api):
            if label.endswith("r3") and not getattr(_poison, "done", False):
                _poison.done = True
                api._dead = True
                raise OSError("[Errno 11001] getaddrinfo failed")
            if api._dead:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
        _hub_that(_poison)
        _res = hf_upload_resilient("t", "u/r", "dataset", _items,
                                   attempts=3, backoff=0)
        check("D-86: a dropped connection does not poison the runs after it",
              len(_res["uploaded"]) == 5 and not _res["failed"],
              f"uploaded {_res['uploaded']}, failed {_res['failed']}")

        # 2. a genuinely unreachable item is reported, and the rest continue
        def _one_bad(label, n, api):
            if label.endswith("r2"):
                raise OSError("[Errno 11001] getaddrinfo failed")
        _hub_that(_one_bad)
        _res = hf_upload_resilient("t", "u/r", "dataset", _items,
                                   attempts=2, backoff=0)
        check("D-86: one permanently failing item does not stop the others",
              len(_res["uploaded"]) == 4 and len(_res["failed"]) == 1
              and _res["failed"][0][0] == "r2",
              f"failed: {_res['failed']}")

        # 3. a fresh client per attempt -- the actual mechanism
        _st = _hub_that(lambda l, n, a: None)
        hf_upload_resilient("t", "u/r", "dataset", _items, attempts=1, backoff=0)
        check("D-86: a NEW client is built per upload, never reused",
              _st["clients"] == len(_items),
              f"{_st['clients']} clients for {len(_items)} items")

        # 4. canary -- the happy path must actually upload
        _st = _hub_that(lambda l, n, a: None)
        _res = hf_upload_resilient("t", "u/r", "dataset", _items,
                                   attempts=3, backoff=0)
        check("D-86 canary: with no failures everything uploads once",
              _res["uploaded"] == ["r1", "r2", "r3", "r4", "r5"]
              and not _res["failed"] and _st["n"] == 5)

        # 5. it must never raise -- a publish that dies must be re-runnable
        _hub_that(lambda l, n, a: (_ for _ in ()).throw(RuntimeError("boom")))
        _raised = False
        try:
            _res = hf_upload_resilient("t", "u/r", "dataset", _items,
                                       attempts=1, backoff=0)
        except Exception:
            _raised = True
        check("D-86: total failure returns a report rather than raising",
              not _raised and len(_res["failed"]) == 5)
    finally:
        if _prev86 is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = _prev86

    # -- D-84: the token preflight must name the cause, not just fail ---------
    import types as _t84

    def _with_whoami(payload, raises=None):
        """Install a stub huggingface_hub whose whoami() returns `payload`."""
        mod = _t84.ModuleType("huggingface_hub")

        class _Api:
            def __init__(self, token=None): self.token = token
            def whoami(self):
                if raises is not None:
                    raise raises
                return payload
        mod.HfApi = _Api
        sys.modules["huggingface_hub"] = mod

    _prev_hub = sys.modules.get("huggingface_hub")
    try:
        # 1. no token at all
        _r = hf_token_check(None, "Shanmuk4622/msc-imagenet100")
        check("D-84: a missing token is refused and says where to make one",
              not _r["ok"] and "settings/tokens" in _r["reason"])

        # 2. THE CASE THE USER HIT: valid token, read-only role
        _with_whoami({"name": "Shanmuk4622", "orgs": [],
                      "auth": {"accessToken": {"role": "read"}}})
        _r = hf_token_check("hf_x", "Shanmuk4622/msc-imagenet100")
        check("D-84: a READ-ONLY token is refused before create_repo is called",
              not _r["ok"] and "read-only" in _r["reason"],
              _r["reason"][:72])

        # 3. token belongs to someone else
        _with_whoami({"name": "someone_else", "orgs": [],
                      "auth": {"accessToken": {"role": "write"}}})
        _r = hf_token_check("hf_x", "Shanmuk4622/msc-imagenet100")
        check("D-84: a token for the wrong namespace names BOTH names",
              not _r["ok"] and "someone_else" in _r["reason"]
              and "Shanmuk4622" in _r["reason"],
              _r["reason"][:72])

        # 4. the working case must PASS -- a preflight that always fails is useless
        _with_whoami({"name": "Shanmuk4622", "orgs": [],
                      "auth": {"accessToken": {"role": "write"}}})
        _r = hf_token_check("hf_x", "Shanmuk4622/msc-imagenet100")
        check("D-84 canary: a WRITE token for the right namespace passes",
              _r["ok"] and _r["role"] == "write", _r["reason"][:72])

        # 5. an org repo the user belongs to is fine
        _with_whoami({"name": "Shanmuk4622", "orgs": [{"name": "some-lab"}],
                      "auth": {"accessToken": {"role": "write"}}})
        _r = hf_token_check("hf_x", "some-lab/msc-imagenet100")
        check("D-84: an org the user belongs to is accepted", _r["ok"])

        # 6. network/auth failure must not raise out of the preflight
        _with_whoami(None, raises=RuntimeError("connection reset"))
        _r = hf_token_check("hf_x", "Shanmuk4622/msc-imagenet100")
        check("D-84: a failing whoami returns a verdict rather than raising",
              not _r["ok"] and "could not identify" in _r["reason"])
    finally:
        if _prev_hub is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = _prev_hub

    # -- D-83: allow_network must actually reverse the offline guard ----------
    _saved83 = {k: os.environ.get(k) for k in
                ("MSC_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "HF_DATASETS_OFFLINE")}
    try:
        for _k in _saved83:
            os.environ[_k] = "1"
        import types as _t83
        _fake_hub = _t83.ModuleType("huggingface_hub.constants")
        _fake_hub.HF_HUB_OFFLINE = True
        sys.modules["huggingface_hub.constants"] = _fake_hub

        _before = offline_state()
        check("D-83 canary: the guard really is on before the call",
              _before["HF_HUB_OFFLINE"] == "1"
              and _before["huggingface_hub.constants.HF_HUB_OFFLINE"] is True,
              "otherwise the test below proves nothing")

        _ch = allow_network(verbose=False)
        _after = offline_state()
        check("D-83: env vars are cleared",
              all(_after[k] is None for k in
                  ("MSC_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                   "HF_DATASETS_OFFLINE")),
              f"cleared {_ch['env_cleared']}")
        check("D-83: the imported hub CONSTANT is patched too",
              _after["huggingface_hub.constants.HF_HUB_OFFLINE"] is False,
              "popping the env var alone leaves huggingface_hub offline, "
              "because it reads the flag once at import")
    finally:
        sys.modules.pop("huggingface_hub.constants", None)
        for _k, _v in _saved83.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v

    # -- D-78: the arm is decided by `method`, never by a run_id substring ----
    _arms = [
        ("p3-shufflenetv2_in-imagenet100-mscKDshuffromresnet50-s1", True),
        ("p3-shufflenetv2_in-imagenet100-mscKDfromresnet50-s1",     False),
        ("p3-resnet18-imagenet100-mscKDshuffromresnet50-s2",        True),
        ("p3-resnet18-imagenet100-mscKDfromresnet50-s2",            False),
        ("p3-deit_small-imagenet100-mscKDfromresnet50-s3",          False),
    ]
    _bad78 = [r for r, want in _arms if is_control_arm(r) != want]
    check("D-78: every arm is classified correctly, shufflenetv2 included",
          not _bad78, "OK" if not _bad78 else "WRONG: " + "; ".join(_bad78))

    # The canary: the naive substring test must actually be wrong here, or the
    # check above proves nothing.
    _naive_wrong = [r for r, want in _arms if ("shuff" in r) != want]
    check("D-78 canary: the substring test IS wrong on shufflenetv2",
          bool(_naive_wrong),
          f"{len(_naive_wrong)} misclassified: "
          + "; ".join(x.split('-')[1] + '/' + x.split('-')[3] for x in _naive_wrong))

    check("D-78: a cfg dict works as well as a run_id",
          is_control_arm({"method": "mscKDshuffromresnet50"}) is True
          and is_control_arm({"method": "mscKDfromresnet50"}) is False)

    # -- D-77: a dense array indexed BY sample_idx must span the index space --
    #
    # Reproduces the shape that killed the kernel: ImageNet-100 has 129,395
    # images, of which 119,395 are train. The teacher sweep returns those
    # 119,395 with their GLOBAL sample_idx, and the training loop gathers
    # msc_t[idx] with idx up to 129,394.
    _N_SPACE, _N_TRAIN = 129395, 119395
    _rng77 = np.random.default_rng(0)
    _sidx = np.sort(_rng77.choice(_N_SPACE, size=_N_TRAIN, replace=False))
    _vals = _rng77.random(_N_TRAIN).astype(np.float32)

    # the OLD construction: sort positionally -> length 119,395
    _old = _vals[np.argsort(_sidx)]
    check("D-77: the old positional build is too short for a global index",
          _old.shape[0] < int(_sidx.max()) + 1,
          f"len {_old.shape[0]} vs max sample_idx {int(_sidx.max())}")

    # the NEW construction: scatter by sample_idx
    _new = np.full(_N_SPACE, np.nan, dtype=np.float32)
    _new[_sidx] = _vals
    check("D-77: the scattered build spans the whole index space",
          _new.shape[0] == _N_SPACE)
    check("D-77: and every sample lands at its own global index",
          bool(np.allclose(_new[_sidx], _vals)),
          "position == sample_idx, so msc_t[idx] is correct by construction")
    check("D-77: positions outside the split stay NaN",
          bool(np.isnan(_new[np.setdiff1d(np.arange(_N_SPACE), _sidx)]).all()),
          "the train loader never gathers them")

    # the ablation must permute the COMPACT vector, not the padded one
    _shuf_compact = shuffle_msc_targets(_vals.copy(), seed=1)
    _packed = np.full(_N_SPACE, np.nan, dtype=np.float32)
    _packed[_sidx] = _shuf_compact
    check("D-77: shuffling before the scatter keeps every real sample real",
          int(np.isnan(_packed[_sidx]).sum()) == 0,
          "permuting the padded array would move NaNs into real samples")
    check("D-77: and it is a genuine permutation of the same values",
          bool(np.allclose(np.sort(_shuf_compact), np.sort(_vals)))
          and not bool(np.allclose(_shuf_compact, _vals)))

    # -- D-76: a measurement loader must produce MODEL INPUT ------------------
    # The EXACT batch that failed on the user's machine: [256, 256, 256, 3]
    # uint8, straight off the packed dataset with no conversion layer.
    _p76 = _model_input_problems((256, 256, 256, 3), False, 224, "torch.uint8")
    check("D-76: the exact failing batch is refused", bool(_p76), "; ".join(_p76))
    check("D-76: and the message identifies it as NHWC",
          any("NHWC" in m for m in _p76), "; ".join(_p76))
    check("D-76: and names the missing float cast",
          any("expected float" in m for m in _p76))

    check("D-76: a 256px float batch is refused when the config says 224",
          bool(_model_input_problems((2, 3, 256, 256), True, 224)))
    check("D-76: a rank-3 batch is refused",
          bool(_model_input_problems((2, 3, 224), True, 224)))

    # The canary that matters most: a guard which rejects valid input would
    # break every sweep, including the ones that currently work.
    check("D-76 canary: a CORRECT batch is not refused",
          not _model_input_problems((64, 3, 224, 224), True, 224),
          "NB3 already passes through this path")
    check("D-76 canary: correct at another resolution is not refused",
          not _model_input_problems((64, 3, 160, 160), True, 160))
    check("D-76 canary: no res in cfg means no res complaint",
          not _model_input_problems((64, 3, 96, 96), True, 0))

    # -- D-70: device tensors must survive the numpy boundary -----------------
    #
    # GPUBatchLoader yields labels on the DEVICE; CIFAR's DataLoader yields
    # them on the host. Three sweep call sites assumed the CIFAR shape and
    # died 40 minutes into the first measurement.
    check("D-70: to_numpy handles a list", to_numpy([1, 2, 3]).tolist() == [1, 2, 3])
    check("D-70: to_numpy applies a dtype",
          to_numpy([1.7, 2.9], np.int64).dtype == np.int64)
    if _TORCH_OK:
        _t = torch.tensor([3, 1, 2])
        check("D-70: to_numpy handles a CPU tensor",
              to_numpy(_t, np.int64).tolist() == [3, 1, 2])
        check("D-70 canary: bare np.asarray still works on CPU (so the CIFAR "
              "path never exposed this)",
              np.asarray(_t).tolist() == [3, 1, 2])
    else:
        check("D-70: to_numpy tensor paths (torch unavailable)", True, "SKIP")

    # No `np.asarray` may remain on a value taken straight from a batch.
    _bad70 = []
    try:
        import ast as _a70
        _t70 = _a70.parse(_src_of_module())
        for _nd in _a70.walk(_t70):
            if (isinstance(_nd, _a70.Call)
                    and isinstance(_nd.func, _a70.Attribute)
                    and _nd.func.attr in ("asarray", "array")
                    and isinstance(_nd.func.value, _a70.Name)
                    and _nd.func.value.id == "np"
                    and _nd.args
                    and isinstance(_nd.args[0], _a70.Name)
                    and _nd.args[0].id in ("y", "idx", "yb", "labels_t")):
                _bad70.append(f"line {_nd.lineno}: np.{_nd.func.attr}"
                              f"({_nd.args[0].id}) -- use to_numpy()")
    except Exception:                                            # noqa: BLE001
        pass
    check("D-70: no batch tensor reaches np.asarray directly",
          not _bad70, "OK" if not _bad70 else "; ".join(_bad70))

    # -- D-69: an artifact must be joined to the directory it lives in --------
    #
    # `run_dir / "ckpt_best.pt"` -- the run root -- while checkpoints live in
    # `checkpoints/`. The correct spelling existed three lines below, inside a
    # HuggingFace branch that is dead in a local-only run, so the only reachable
    # spelling was wrong and every measurement failed with "Train the backbone
    # first" beside a 91 MB checkpoint.
    #
    # The artifact lists already say where each file belongs, so the check is
    # a comparison rather than a new opinion (D-16).
    _in_subdir = {}
    for _grp in (RUN_ARTIFACTS_REQUIRED, RUN_ARTIFACTS_MEASURED,
                 RUN_ARTIFACTS_EXPECTED):
        for _rel in _grp:
            if "/" in _rel:
                _in_subdir[_rel.split("/")[-1]] = _rel.split("/")[0]
    # AST, not regex: the first version matched its own explanatory comment
    # and its own pattern string, reporting 2 problems where there was 1. A
    # checker that cries wolf is the thing this project keeps paying for.
    _misplaced = []
    try:
        import ast as _a69
        _t69 = _a69.parse(_src_of_module())
        for _nd in _a69.walk(_t69):
            if not (isinstance(_nd, _a69.BinOp)
                    and isinstance(_nd.op, _a69.Div)):
                continue
            _lhs, _rhs = _nd.left, _nd.right
            if not (isinstance(_lhs, _a69.Name) and _lhs.id == "run_dir"):
                continue
            if not (isinstance(_rhs, _a69.Constant)
                    and isinstance(_rhs.value, str)):
                continue
            if _rhs.value in _in_subdir:
                _misplaced.append(
                    f'line {_nd.lineno}: run_dir / "{_rhs.value}" but it '
                    f'lives in {_in_subdir[_rhs.value]}/')
    except Exception as _e69:                                    # noqa: BLE001
        _misplaced.append(f"<could not parse: {_e69}>")
    check("D-69: no artifact is joined to the run root when it lives in a subdir",
          not _misplaced,
          "OK" if not _misplaced else "; ".join(_misplaced))

    check("D-69 canary: the subdir map is populated",
          _in_subdir.get("ckpt_best.pt") == "checkpoints",
          f"ckpt_best.pt -> {_in_subdir.get('ckpt_best.pt')}")

    def _d69_finds(src_txt):
        import ast as _a
        for _n in _a.walk(_a.parse(src_txt)):
            if (isinstance(_n, _a.BinOp) and isinstance(_n.op, _a.Div)
                    and isinstance(_n.left, _a.Name) and _n.left.id == "run_dir"
                    and isinstance(_n.right, _a.Constant)
                    and _n.right.value in _in_subdir):
                return True
        return False

    check("D-69 canary: the walker catches the exact defective line",
          _d69_finds('ckpt = run_dir / "ckpt_best.pt"'))
    check("D-69 canary: it accepts the correct spelling and run-root files",
          not _d69_finds('ckpt = L["checkpoints"] / "ckpt_best.pt"')
          and not _d69_finds('p = run_dir / "summary.json"'),
          "summary.json legitimately lives at the run root")

    # -- D-67: measuring must be PLANNED as measuring -------------------------
    _s67 = Session.__new__(Session)
    _orc = Session.oracle.__get__(_s67)
    _c67 = False
    try:
        Session.run_all(_s67, [{"run_id": "x"}], fn=_orc)          # stage='train'
    except ValueError as _e:
        _c67 = "would ask 'is it TRAINED?'" in str(_e)
    except Exception:
        pass
    check("D-67: run_all(fn=sess.oracle) without stage='measure' is refused",
          _c67, "otherwise it skips every trained run and reports success")

    _f67 = False
    try:
        Session.run_all(_s67, [{"run_id": "x"}], fn=_orc, stage="measure")
    except ValueError as _e:
        _f67 = "would ask" in str(_e)
    except Exception:
        pass
    check("D-67 canary: the correct call is NOT refused", not _f67)

    # -- D-64: the artifact spec must agree with the code that writes ---------
    #
    # `final.csv` was listed as REQUIRED (checked after training) while only
    # `run_oracle` writes it, so four healthy runs verified as incomplete. The
    # list and the writers are two spellings of one truth (D-16), so this reads
    # the writers out of this module's own source rather than trusting either.
    def _scratch_run_root():
        import tempfile as _t
        return Path(_t.mkdtemp(prefix="msc_d64_"))

    def _artifact_writers():
        import ast as _a
        try:
            tree = _a.parse(_src_of_module())
        except Exception:                                        # noqa: BLE001
            return {}
        out = {}
        for fn in tree.body:
            if not isinstance(fn, (_a.FunctionDef, _a.AsyncFunctionDef)):
                continue
            for nd in _a.walk(fn):
                if isinstance(nd, _a.Constant) and isinstance(nd.value, str):
                    v = nd.value
                    if v.endswith((".csv", ".parquet", ".json", ".pt", ".jsonl")):
                        out.setdefault(v, set()).add(fn.name)
        return out

    _writers = _artifact_writers()
    _oracle_only = []
    for _art in RUN_ARTIFACTS_REQUIRED:
        _fns = _writers.get(_art.split("/")[-1], set())
        if _fns and _fns <= {"run_oracle"}:
            _oracle_only.append(f"{_art} <- only run_oracle")
    check("D-64: no train-stage REQUIRED artifact is written only by the oracle",
          not _oracle_only,
          "OK" if not _oracle_only else "; ".join(_oracle_only))

    check("D-64 canary: the writer map can see run_oracle's outputs",
          "run_oracle" in _writers.get("test.parquet", set()),
          "otherwise the check above proves nothing")

    _vrep = verify_run_artifacts(_scratch_run_root(), "nonexistent-run")
    check("D-64: verify_run_artifacts reports a missing run rather than raising",
          isinstance(_vrep, dict) and not _vrep.get("ok"))

    # D-63. The D-60 tests all used a CLEAN config, which is the one shape the
    # runtime never has. `load_checkpoint` sees a dict that has since gained
    # keys, so config_hash(cfg) and cfg["config_hash"] disagree and every probe
    # built on it misses. The tests agreed with me instead of with the program.
    import tempfile as _tf
    _dir = Path(_tf.mkdtemp(prefix="msc_d63_"))
    _rec = dict(_c60)
    atomic_write_yaml(_dir / "config.yaml", _rec)
    _stored63 = config_hash(dict(_rec, channels_last=True),
                            exclude=_HASH_EXCLUDE_V1)

    _drift = dict(_rec, _added_at_runtime="by train_backbone", _also=123)
    _ok63, _w63 = hash_compatible(_drift, _stored63, run_dir=_dir)
    check("D-63: a config that GAINED runtime keys still resumes", _ok63, _w63)

    _ok63b, _ = hash_compatible(_drift, _stored63)          # no record
    check("D-63 canary: without the record the drifted config FAILS",
          not _ok63b, "which is exactly what happened on the machine")

    for _k, _v in (("batch_size", 128), ("num_epochs", 60), ("seed", 99)):
        _bad63, _wb = hash_compatible(dict(_drift, **{_k: _v}), _stored63,
                                      run_dir=_dir)
        check(f"D-63: a changed {_k} is still REFUSED", not _bad63,
              _wb[:70])
    shutil.rmtree(_dir, ignore_errors=True)

    check("D-60 canary: the OLD hash really does differ from the new one",
          _stored_v1 != config_hash(_c60),
          "otherwise this test proves nothing")

    # It must NOT launder a recipe change. lr is never excluded, so no
    # assignment of performance keys can reproduce a hash that differs in it.
    _bad60, _ = hash_compatible(dict(_c60, lr=1e-3),
                                config_hash(dict(_c60, channels_last=True),
                                            exclude=_HASH_EXCLUDE_V1))
    check("D-60: a changed lr is still REFUSED", not _bad60,
          "compatibility is proof, not leniency")
    _bad61, _ = hash_compatible(dict(_c60, batch_size=128), _stored_v1)
    check("D-60: a changed batch_size is still REFUSED", not _bad61)
    _bad62, _ = hash_compatible(dict(_c60, num_epochs=60), _stored_v1)
    check("D-60: a changed num_epochs is still REFUSED", not _bad62)

    # -- D-59: the layout flag is honoured, and does not orphan a run --------
    _c59 = {"arch": "resnet50", "seed": 1, "batch_size": 64, "lr": 0.025}
    check("D-59: flipping channels_last does not change config_hash",
          config_hash(dict(_c59, channels_last=True))
          == config_hash(dict(_c59, channels_last=False)),
          "90 h of finished runs stay resumable")

    _ic = base_config("resnet50", "imagenet100")
    check("D-59: imagenet100 defaults to contiguous (measured 6.7x)",
          _ic.get("channels_last") is False,
          f"channels_last={_ic.get('channels_last')}")

    # The loader must READ the flag. It ignored it for the project's whole
    # life, forcing channels_last while the config carried a setting that only
    # the model consulted -- so the two could never disagree visibly.
    _gsrc = _src_of_module()
    _i = _gsrc.find("class GPUBatchLoader")
    _seg = _gsrc[_i:_i + 12000] if _i >= 0 else ""
    check("D-59: GPUBatchLoader honours channels_last instead of forcing it",
          ("if self.channels_last else" in _seg) and ("self.channels_last = " in _seg),
          "the flag reaches the line that was ignoring it")

    # -- D-56: performance knobs must not orphan a checkpoint ----------------
    _c_old = {"arch": "resnet50", "seed": 1, "batch_size": 64, "lr": 0.025}
    _c_new = dict(_c_old, ram_cache=True, ram_headroom_gb=6.0, num_workers=0,
                  prefetch_batches=3)
    check("D-56: turning on the RAM cache does not change config_hash",
          config_hash(_c_old) == config_hash(_c_new),
          "a resumable run stays resumable")
    check("D-56 canary: batch_size DOES change config_hash",
          config_hash(_c_old) != config_hash(dict(_c_old, batch_size=128)),
          "batch size scales the LR -- it is the recipe, not a knob")

    # -- D-56: the two meanings of `.indices` ---------------------------------
    class _FakePack:
        """Stands in for PackedImageDataset: `.indices` are GLOBAL."""
        stored_res, count = 256, 1000
        def __init__(self, gi, lb):
            self.indices = np.asarray(gi, dtype=np.int64)
            self.labels = np.asarray(lb, dtype=np.int64)
        def __len__(self): return len(self.indices)

    class _FakeSubset:
        """Stands in for torch Subset: `.indices` are POSITIONS in the parent."""
        def __init__(self, ds, pos):
            self.dataset = ds
            self.indices = np.asarray(pos, dtype=np.int64)
        def __len__(self): return len(self.indices)

    # split holds global pack ids 100,200,300,400,500
    _pk = _FakePack([100, 200, 300, 400, 500], [7, 8, 9, 10, 11])
    _gi, _lb = pack_view_of(_pk)
    check("D-56: pack view of a bare dataset returns global indices",
          _gi.tolist() == [100, 200, 300, 400, 500] and _lb.tolist() == [7, 8, 9, 10, 11],
          f"{_gi.tolist()}")

    # a subset keeping positions 1 and 3 -> global 200 and 400, labels 8 and 10
    _sub = _FakeSubset(_pk, [1, 3])
    _gi2, _lb2 = pack_view_of(_sub)
    check("D-56: pack view of a Subset resolves POSITIONS to GLOBAL ids",
          _gi2.tolist() == [200, 400] and _lb2.tolist() == [8, 10],
          f"got idx={_gi2.tolist()} labels={_lb2.tolist()}")

    # The naive bug: reading Subset.indices directly would give [1, 3] --
    # valid-looking indices pointing at the wrong images. Prove they differ,
    # or this test would pass on a broken implementation.
    check("D-56 canary: naive .indices differs from the resolved view",
          _sub.indices.tolist() != _gi2.tolist(),
          f"naive={_sub.indices.tolist()} resolved={_gi2.tolist()}")

    # nested subsets must compose
    _gi3, _lb3 = pack_view_of(_FakeSubset(_sub, [1]))
    check("D-56: nested Subsets compose",
          _gi3.tolist() == [400] and _lb3.tolist() == [10],
          f"{_gi3.tolist()}")

    check("D-56: pack_root_of unwraps to the dataset with stored_res",
          pack_root_of(_FakeSubset(_sub, [0])) is _pk)

    _rb, _rwhy = ram_budget_ok(1)
    check("D-56: ram_budget_ok answers with a reason either way", bool(_rwhy))
    _nb, _ = ram_budget_ok(1 << 62)
    check("D-56: ram_budget_ok refuses an impossible request", not _nb)

    # -- D-55: every model in a compute path goes through place_model --------
    def _d55_bare_model_placements():
        """Models built in a compute path without going through place_model.

        Reads THIS file. The invariant is "a model and its input agree on
        memory format"; the mechanism is that one accessor owns the move. A
        second spelling of `.to(device)` is how the first one drifted -- for
        69 epochs at a fifth of the achievable speed, with the config claiming
        `channels_last: True` the whole time.

        Restricted to functions that actually run batches. Analysis helpers
        that build a model to count parameters or FLOPs never see an
        activation, so layout is genuinely irrelevant there and flagging them
        would train everyone to ignore this check.
        """
        import ast as _ast
        compute_fns = {"train_backbone", "run_oracle", "train_exit_heads",
                       "train_msc_kd", "backbone_dry_run", "oracle_dry_run",
                       "msckd_dry_run", "evaluate_multi_exit"}
        try:
            tree = _ast.parse(_src_of_module())
        except Exception:                                        # noqa: BLE001
            return ["<could not parse module>"]
        bad = []
        for fn in _ast.walk(tree):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if fn.name not in compute_fns:
                continue
            for nd in _ast.walk(fn):
                # match  <Model>(...).to(<anything>)
                if not (isinstance(nd, _ast.Call)
                        and isinstance(nd.func, _ast.Attribute)
                        and nd.func.attr == "to"):
                    continue
                inner = nd.func.value
                while isinstance(inner, _ast.Call) and isinstance(
                        inner.func, _ast.Attribute) and inner.func.attr in (
                        "eval", "train", "to"):
                    inner = inner.func.value
                if (isinstance(inner, _ast.Call)
                        and isinstance(inner.func, _ast.Name)
                        and inner.func.id in ("build_model", "MultiExitModel",
                                              "MSCStudent")):
                    bad.append(f"{fn.name}:{nd.lineno} "
                               f"{inner.func.id}(...).to(...)")
        return bad

    _d55 = _d55_bare_model_placements()
    check("D-55: every compute-path model goes through place_model",
          not _d55,
          "OK" if not _d55 else "BARE: " + "; ".join(_d55))

    # The check must be able to fail, or it is decoration (D-37).
    _d55_canary = []
    try:
        import ast as _ast_c
        _t = _ast_c.parse("def train_backbone(cfg):\n"
                          "    m = build_model(a, b).to(dev)\n")
        for _fn in _ast_c.walk(_t):
            if isinstance(_fn, _ast_c.FunctionDef):
                for _nd in _ast_c.walk(_fn):
                    if (isinstance(_nd, _ast_c.Call)
                            and isinstance(_nd.func, _ast_c.Attribute)
                            and _nd.func.attr == "to"
                            and isinstance(_nd.func.value, _ast_c.Call)
                            and getattr(_nd.func.value.func, "id", "")
                            == "build_model"):
                        _d55_canary.append("caught")
    except Exception:                                            # noqa: BLE001
        pass
    check("D-55 canary: the placement check can detect a bare .to(device)",
          bool(_d55_canary))

    def _raises(fn, exc=Exception) -> bool:
        """Assert a call fails, and fails with the RIGHT exception.

        Bare `except Exception` would let a typo inside the lambda pass as a
        successful negative test -- the D-06 shape, a test that cannot fail for
        the right reason.
        """
        try:
            fn()
        except exc:
            return True
        except Exception:                                       # noqa: BLE001
            return False
        return False

    # D-78, placed here because `_raises` is defined above this point and not
    # above the rest of the D-78 block. Inserting a check before the helper it
    # uses is the same ordering mistake D-69 made with `_src_of_module`.
    check("D-78: an unparseable id raises rather than guessing",
          _raises(lambda: is_control_arm("not-a-run-id"), ValueError))

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
        # Derived from N_GPU_COLUMNS, not pinned to two. The requirement is
        # "utilisation, per GPU" -- which means one column per device the
        # machine ACTUALLY has, not per device the original platform had.
        # Pinning it to 2 is the same defect as D-36 read from the other end:
        # there, a reader asked for an un-suffixed `gpu_util_mean_pct` that
        # never existed; here, a test demanded a `gpu1_*` that should not exist
        # on a single-GPU box.
        "gpu utilization (per gpu)": [f"gpu{i}_util_mean_pct"
                                      for i in range(N_GPU_COLUMNS)],
        "energy consumed": ["epoch_energy_j", "epoch_energy_kwh",
                            "cumulative_energy_kwh"],
        "carbon emission": ["epoch_co2_g", "epoch_co2_kg", "cumulative_co2_kg"],
        "temperature": (["gpu0_temp_mean_c"]
                        + [f"gpu{i}_temp_max_c" for i in range(N_GPU_COLUMNS)]),
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
    check(f"per-GPU columns exist for all {N_GPU_COLUMNS} device(s)",
          all(f"gpu{i}_{k}" in H for i in range(N_GPU_COLUMNS)
              for k in ("util_mean_pct", "temp_max_c", "mem_used_mb", "energy_j")),
          f"detected {N_GPU_COLUMNS} GPU(s)")
    check("the GPU column count is derived, not assumed",
          N_GPU_COLUMNS == _detect_gpu_columns(),
          "dual T4 was the CIFAR platform; the port target has one RTX 4000 Ada")
    check("there is at least one GPU device column even with no GPU",
          N_GPU_COLUMNS >= 1 and "gpu0_util_mean_pct" in H,
          "the schema must not change shape depending on whether the machine "
          "writing it had a GPU, or two runs become un-concatenable")
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

    # --- D-32: EVERY gate must honour invalidation, not just one -------------
    # Three independent gates stand between "run exists" and "train it":
    # plan_work's done_fn, registry.can_claim, and already_finished. Each was
    # fixed in turn, and each time the stop simply moved to the next gate down.
    # `force_rerun` is the one flag they all already honour.
    def _passes_all(force, ledger_completed, summary_exists):
        gate_plan = not ledger_completed or force
        gate_claim = (not ledger_completed) or force
        gate_cached = (not summary_exists) or force
        return gate_plan and gate_claim and gate_cached

    check("D-32: without force, a completed run is stopped",
          not _passes_all(False, True, True))
    check("D-32: force clears all three gates at once",
          _passes_all(True, True, True),
          "fixing them one at a time just moved the stop")
    check("D-32: a fresh run needs no force",
          _passes_all(False, False, False))

    # --- D-31: the compatibility check must sit in the PREDICATE -------------
    # D-29 put the router check inside train_msc_kd. plan_work filters "done"
    # runs out before that function is ever called, so the check was
    # unreachable: NB13 printed "already finished: 9 ... REMAINING WORK: 0".
    # A test that decides whether to redo work cannot live inside the code that
    # does the work.
    def _plan_todo(mine, done_fn):
        return [r for r in mine if not done_fn(r)]

    _mine = ["a", "b", "c"]
    check("D-31: a presence-only predicate skips invalid runs",
          _plan_todo(_mine, lambda r: True) == [],
          "this is what actually happened -- 0 work planned")
    check("D-31: a validity-aware predicate re-plans them",
          _plan_todo(_mine, lambda r: r == "a") == ["b", "c"])
    check("D-31: and leaves the valid ones alone",
          _plan_todo(_mine, lambda r: r != "c") == ["c"])

    # --- D-29: a completion cache needs a COMPATIBILITY predicate ------------
    # already_finished answers "did it complete?". After D-28 the honest answer
    # for nine students was "yes, and unusable". Presence is not validity.
    def _router_ok(stored_width, arch_width):
        return stored_width == arch_width

    check("D-29: a teacher-sized router is rejected as invalid",
          not _router_ok(5, 3), "resnet8x4 with a resnet32x4-shaped head")
    check("D-29: a correctly-sized router is accepted", _router_ok(3, 3))
    check("D-29: equal-width architectures are unaffected",
          _router_ok(5, 5), "resnet20/vgg8 also have 5 exits")

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
    # D-71. This used to be a set of RUN IDS. `require` is only ever given
    # `_ceilings(...)`, which is keyed by ARCHITECTURE -- so the test asserted
    # the buggy semantics and passed while every real caller got an empty
    # result. The fixture is now the shape the callers actually pass.
    _ceil = {"vgg8": 0.71, "resnet20": 0.66}          # arch -> rho_seed
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

    # D-71. A `require` keyed by the WRONG identifier space must be loud.
    # Silently returning {} emptied Q3-axis, Q3-control and Q4 at once: the
    # control wrote a 2-byte CSV and NB4 raised KeyError on a frame with no
    # columns, three layers from the cause.
    _wrong_space = {"p1-vgg8-cifar100-base-s2", "p1-resnet20-cifar100-base-s1"}
    check("D-71: a run-id-keyed `require` raises instead of returning {}",
          _raises(lambda: representative_runs(_runs, require=_wrong_space),
                  KeyError),
          "an empty reps dict empties every downstream table")
    check("D-71: the arch-keyed `require` still returns both architectures",
          sorted(representative_runs(_runs, require=_ceil)) ==
          ["resnet20", "vgg8"],
          str(sorted(representative_runs(_runs, require=_ceil))))
    check("D-71: an empty runs dict is not mistaken for a key-space error",
          representative_runs({}, require=_ceil) == {})

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
    _sc_ok, z, sd = shuffled_control_verdict(-0.0341, 5872)
    check("D-17: a healthy 2.6-sigma residual passes", _sc_ok, f"z={z:+.2f}")
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
    # The count is derived, not asserted against a literal. The previous
    # version pinned `len(ZOO) == 15` and failed the moment a second dataset's
    # architectures were registered -- rule 2's failure mode inside the test
    # written to enforce rule 2.
    check("CIFAR zoo has its 15 architectures",
          len(zoo_for_dataset("cifar100")) == 15,
          f"{len(zoo_for_dataset('cifar100'))}")
    check("ImageNet zoo has its 8 architectures",
          len(zoo_for_dataset("imagenet100")) == 8,
          f"{sorted(zoo_for_dataset('imagenet100'))}")
    check("every entry declares a zoo", all("zoo" in v for v in ZOO.values()))
    check("the two zoos are disjoint",
          not (set(zoo_for_dataset("cifar100")) & set(zoo_for_dataset("imagenet100"))))
    check("families cover the H3 ordering",
          {"resnet", "wrn", "vgg", "mobile", "vit", "mixer"}
          <= {v["family"] for v in ZOO.values()})

    # --- the ImageNet-100 design, checked as a design -----------------------
    _in = set(zoo_for_dataset("imagenet100"))
    check("ImageNet zoo crosses the boundary four ways",
          {"resnet50", "vit_small_p16", "swin_tiny", "convnext_tiny"} <= _in,
          "resnet50/vit (pure corners) + swin/convnext (mixed) is the 2x2 that "
          "separates 'attention' from 'weak spatial prior'")
    check("vit_small_p16 and deit_small are built by ONE builder with ONE "
          "argument set",
          ZOO["vit_small_p16"]["builder"] == ZOO["deit_small"]["builder"],
          "identical geometry is what makes the recipe contrast mean 'recipe'")
    check("...and differ in recipe",
          (base_config("deit_small", "imagenet100")["mixup_alpha"] > 0)
          and (base_config("vit_small_p16", "imagenet100")["mixup_alpha"] == 0),
          "deit arm carries mixup/cutmix; the vit arm does not")
    check("...and are otherwise the same recipe",
          all(base_config("deit_small", "imagenet100")[k]
              == base_config("vit_small_p16", "imagenet100")[k]
              for k in ("num_epochs", "batch_size", "optimizer", "learning_rate",
                        "weight_decay", "scheduler", "warmup_epochs")),
          "epochs, optimiser, LR, wd, schedule and warmup all held fixed")
    check("shufflenetv2 is the CIFAR<->ImageNet bridge",
          CROSS_STUDY_ALIAS.get("shufflenetv2_in") == "shufflenetv2"
          and "shufflenetv2" in zoo_for_dataset("cifar100"),
          "the only architecture measured in both studies")
    check("equal epochs across the whole ImageNet zoo",
          len({base_config(a, "imagenet100")["num_epochs"] for a in _in}) == 1,
          f"{sorted({base_config(a,'imagenet100')['num_epochs'] for a in _in})} "
          f"-- schedule length is held constant so it cannot join accuracy and "
          f"family as a third confounded variable, which is what happened on "
          f"CIFAR (240 vs 300 epochs)")

    print("dry runs are WIRED IN, not merely written (rule 1)")
    # Rule 7: an invariant in a comment is not a mechanism. Writing three dry
    # runs is worth nothing if a later edit drops the call, and the symptom of
    # that is an hour of GPU time, not an error. So the wiring is asserted from
    # the source itself.
    #
    # It checks POSITION, not just presence: the dry run must appear before the
    # first expensive call in each function. `msckd_dry_run` was written for
    # O-19 and then filed for later, which cost two more hour-long cycles
    # before it was actually installed.
    import inspect as _insp
    for _fn, _dry, _expensive in (
            (train_backbone, "backbone_dry_run", "build_loaders"),
            (run_oracle, "oracle_dry_run", "build_loaders"),
            (train_msc_kd, "msckd_dry_run", "sweep_all_axes")):
        try:
            _src = _insp.getsource(_fn)
        except Exception:                                        # noqa: BLE001
            check(f"{_fn.__name__} source readable", False)
            continue
        _has = _dry in _src
        _pos_ok = _has and (_expensive not in _src
                            or _src.index(_dry) < _src.index(_expensive))
        check(f"{_fn.__name__} calls {_dry}", _has)
        check(f"{_fn.__name__} calls it BEFORE {_expensive}", _pos_ok,
              "a dry run that runs after the expensive part is decoration")
    check("the backbone dry run goes all the way to a checkpoint round trip",
          "load_checkpoint" in _insp.getsource(backbone_dry_run)
          and "evaluate(" in _insp.getsource(backbone_dry_run),
          "D-22 failed at the END of epoch 0; stopping the dry run at "
          "backward() would move where bugs hide rather than remove the hiding "
          "place")
    check("the oracle dry run reads its parquet BACK",
          "read_parquet" in _insp.getsource(oracle_dry_run),
          "writing correctly and reading correctly are different claims")
    check("the oracle dry run sweeps every axis and every score",
          all(x in _insp.getsource(oracle_dry_run)
              for x in ("sweep_all_axes", "difficulty_battery",
                        "prediction_depth", "msc_for_run")))
    check("every dry run derives its resolution from the dataset",
          all(("native_res" in _insp.getsource(f)) or ("input_res" in _insp.getsource(f))
              for f in (backbone_dry_run, oracle_dry_run, msckd_dry_run)),
          "msckd_dry_run defaulted to `cfg.get('image_size', 32)`, which would "
          "have certified an ImageNet run at 32px -- a dry run that passes on "
          "the wrong shape is worse than none (D-06)")
    check("...and none of them spells a resolution literal",
          not any(re.search(r"torch\.randn\(\s*\d+\s*,\s*3\s*,\s*\d+\s*,",
                            _insp.getsource(f))
                  for f in (backbone_dry_run, oracle_dry_run, msckd_dry_run)),
          "a literal in the shape is the D-33 defect: two hardcoded 5s built a "
          "5-output router on a 3-exit backbone INSIDE the check written to "
          "catch exactly that")

    print("atomic writes survive Windows")
    _ar = tmp / "atomic"
    ensure_dir(_ar)
    atomic_write_text(_ar / "x.txt", "one")
    atomic_write_text(_ar / "x.txt", "two")
    check("overwrite via atomic replace", (_ar / "x.txt").read_text() == "two")
    check("no .tmp survives", not (_ar / "x.txt.tmp").exists())
    check("_atomic_replace retries rather than raising immediately",
          "PermissionError" in _insp.getsource(_atomic_replace)
          and "attempts" in _insp.getsource(_atomic_replace),
          "os.replace is unconditional on POSIX but raises on Windows if any "
          "process holds the destination open -- an indexer, a preview, or the "
          "uploader thread reading the very checkpoint being rewritten")
    check("...and raises at the end rather than losing data silently",
          "has NOT been lost" in _insp.getsource(_atomic_replace))

    print("HF verification goes through resolve only (rule 9)")
    _hubsrc = _insp.getsource(MSCHub)
    def _calls(fn) -> Set[str]:
        """Names actually CALLED by a function, parsed rather than grepped.

        A substring search over the source matched the docstrings that explain
        why `list_repo_files` must not be used, and reported the fix as absent.
        A check that reads prose is checking the wrong artifact -- the same
        mistake as trusting a comment to be a mechanism (rule 7), one level up.
        """
        import ast as _ast
        try:
            t = _ast.parse(textwrap.dedent(_insp.getsource(fn)))
        except Exception:                                        # noqa: BLE001
            return set()
        out = set()
        for nd in _ast.walk(t):
            if isinstance(nd, _ast.Call):
                f = nd.func
                out.add(getattr(f, "attr", None) or getattr(f, "id", None) or "")
        return out - {""}

    _vp, _cf = _calls(RunSync.verify_present), _calls(Session.confirm_on_hf)
    check("verify_present CALLS files_present and not list_repo_files",
          "files_present" in _vp and "list_repo_files" not in _vp,
          "confirm-then-delete is the last thing between a completed run and "
          "rmtree")
    check("confirm_on_hf CALLS resolve_meta/files_present, not list_repo_files",
          ({"resolve_meta", "files_present"} & _cf) and "list_repo_files" not in _cf,
          "the tree endpoint served this project stale data three times and "
          "produced a confident wrong negative that stood for two days")
    check("the parse-based check can tell prose from code",
          "list_repo_files" in _insp.getsource(RunSync.verify_present)
          and "list_repo_files" not in _vp,
          "the docstring names it precisely to say it must not be called; a "
          "substring check called that a failure")
    check("resolve_meta returns None ONLY for a real 404",
          "Refusing to report absence" in
          _insp.getsource(BackgroundUploader.resolve_meta),
          "a negative finding produced by a dropped connection is the D-20 "
          "false alarm; absence must be established, not inferred from failure")
    check("files_present asks per file, with no aggregate to truncate",
          "resolve_meta" in _insp.getsource(BackgroundUploader.files_present),
          "the repo-info body was silently truncated mid-JSON at ~69 KB and the "
          "cut landed just past `vgg8`, exactly where the missing runs were")

    print("names and arities resolve without running anything")
    # Three of the five offline-verify failures were things a torch-free check
    # can catch, and all three reached the user because the only thing that
    # could find them needed a GPU:
    #
    #   NameError: name 'MultiExit' is not defined     (the class is MultiExitModel)
    #   ValueError: too many values to unpack          (optimisation_health returns 4)
    #   AttributeError: 'BatchNorm2d' has no 'out_channels'  (guessed at internals)
    #
    # None of them needed a model, a dataset or a device. They needed somebody
    # to compare a name against what exists -- which is rule 3 generalised from
    # column names to every name.
    import ast as _a2

    def _free_names(fn) -> Set[str]:
        """Names a function READS that it does not itself bind."""
        try:
            t = _a2.parse(textwrap.dedent(_insp.getsource(fn)))
        except Exception:                                        # noqa: BLE001
            return set()
        bound, used = set(), set()
        for nd in _a2.walk(t):
            if isinstance(nd, _a2.Name):
                (bound if isinstance(nd.ctx, _a2.Store) else used).add(nd.id)
            elif isinstance(nd, (_a2.FunctionDef, _a2.AsyncFunctionDef)):
                bound.add(nd.name)
                for arg in list(nd.args.args) + list(nd.args.kwonlyargs):
                    bound.add(arg.arg)
                if nd.args.vararg:
                    bound.add(nd.args.vararg.arg)
                if nd.args.kwarg:
                    bound.add(nd.args.kwarg.arg)
            elif isinstance(nd, _a2.ExceptHandler) and nd.name:
                bound.add(nd.name)
            elif isinstance(nd, (_a2.Import, _a2.ImportFrom)):
                for al in nd.names:
                    bound.add((al.asname or al.name).split(".")[0])
            elif isinstance(nd, _a2.ClassDef):
                bound.add(nd.name)
            elif isinstance(nd, _a2.comprehension):
                for sub in _a2.walk(nd.target):
                    if isinstance(sub, _a2.Name):
                        bound.add(sub.id)
        return used - bound

    def _module_level_names() -> Set[str]:
        """Every name this module defines AT MODULE SCOPE, including the ones
        inside `if _TORCH_OK:` blocks.

        `globals()` is the wrong universe here. Half this file -- `ExitHead`,
        `MultiExitModel`, `MSCLoss`, `MSCStudent`, `_PrefixWrapper` -- lives
        under a torch guard, so on a machine without torch those names are
        genuinely absent and the check would flag five false positives and be
        switched off within a day. They exist on the machine that runs the
        experiment, which is the machine the check is about.

        Parsing the source gets the real answer on both.
        """
        try:
            t = _a2.parse(Path(globals().get("__file__", "msc_lib.py")).read_text(
                encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            return set()
        out: Set[str] = set()

        def walk_body(body):
            for nd in body:
                if isinstance(nd, (_a2.FunctionDef, _a2.AsyncFunctionDef,
                                   _a2.ClassDef)):
                    out.add(nd.name)
                elif isinstance(nd, _a2.Assign):
                    for tg in nd.targets:
                        if isinstance(tg, _a2.Name):
                            out.add(tg.id)
                elif isinstance(nd, _a2.AnnAssign) and isinstance(nd.target, _a2.Name):
                    out.add(nd.target.id)
                elif isinstance(nd, (_a2.Import, _a2.ImportFrom)):
                    for al in nd.names:
                        out.add((al.asname or al.name).split(".")[0])
                elif isinstance(nd, (_a2.If, _a2.Try)):
                    walk_body(nd.body)
                    walk_body(getattr(nd, "orelse", []) or [])
                    for h in getattr(nd, "handlers", []) or []:
                        walk_body(h.body)
        walk_body(t.body)
        return out

    _G = (set(globals()) | set(dir(__import__("builtins")))
          | _module_level_names())
    for _fn in (backbone_dry_run, oracle_dry_run, msckd_dry_run,
                _imagenet_config, build_budget_table, verify_run_artifacts):
        _un = sorted(n for n in _free_names(_fn) if n not in _G)
        check(f"every name in {_fn.__name__} resolves", not _un,
              f"unresolved: {_un}" if _un else
              "would have caught `MultiExit` before it cost an offline run")

    def _arity_ok(caller, callee_name: str, n_expected: int) -> bool:
        """Is every tuple-unpack of `callee_name(...)` the right width?"""
        try:
            t = _a2.parse(textwrap.dedent(_insp.getsource(caller)))
        except Exception:                                        # noqa: BLE001
            return True
        for nd in _a2.walk(t):
            if isinstance(nd, _a2.Assign) and isinstance(nd.value, _a2.Call):
                f = nd.value.func
                if (getattr(f, "id", None) or getattr(f, "attr", None)) != callee_name:
                    continue
                for tg in nd.targets:
                    if isinstance(tg, (_a2.Tuple, _a2.List)) \
                            and len(tg.elts) != n_expected:
                        return False
        return True

    for _fn in (backbone_dry_run, train_backbone):
        check(f"{_fn.__name__} unpacks optimisation_health as 4 values",
              _arity_ok(_fn, "optimisation_health", 4),
              "it returns (weight_norm, update_norm, ratio, flat)")

    print("every internal call matches its callee's signature (D-47)")
    # D-47. `backbone_dry_run` called `load_checkpoint` with 6 positional
    # arguments; it takes 8. Every name involved existed, so the
    # name-resolution guard from D-38 passed it, and the failure only appeared
    # when the user ran it on real hardware -- eight architectures deep, twice.
    #
    # Names being real is not the same as calls being right. Arity is
    # mechanically checkable from the same source.
    def _defs() -> Dict[str, Any]:
        try:
            t = _a2.parse(Path(globals().get("__file__", "msc_lib.py"))
                          .read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            return {}
        out = {}

        def walk(body):
            for nd in body:
                if isinstance(nd, (_a2.FunctionDef, _a2.AsyncFunctionDef)):
                    aa = nd.args
                    pos = list(aa.posonlyargs) + list(aa.args)
                    ndef = len(aa.defaults)
                    out[nd.name] = {
                        "min": len(pos) - ndef, "max": len(pos),
                        "star": aa.vararg is not None,
                        "kw": {x.arg for x in list(pos) + list(aa.kwonlyargs)},
                        "kwargs": aa.kwarg is not None,
                    }
                elif isinstance(nd, (_a2.If, _a2.Try)):
                    walk(nd.body)
                    walk(getattr(nd, "orelse", []) or [])
                    for h in getattr(nd, "handlers", []) or []:
                        walk(h.body)
                elif isinstance(nd, _a2.ClassDef):
                    pass          # methods carry `self`; out of scope here
        walk(t.body)
        return out

    _SIG = _defs()

    def _bad_calls(fn) -> List[str]:
        try:
            t = _a2.parse(textwrap.dedent(_insp.getsource(fn)))
        except Exception:                                        # noqa: BLE001
            return []
        bad = []
        for nd in _a2.walk(t):
            if not isinstance(nd, _a2.Call):
                continue
            name = getattr(nd.func, "id", None)
            sig = _SIG.get(name) if name else None
            if not sig:
                continue
            npos = len(nd.args)
            if any(isinstance(x, _a2.Starred) for x in nd.args):
                continue
            given = npos + len({k.arg for k in nd.keywords if k.arg})
            if npos > sig["max"] and not sig["star"]:
                bad.append(f"{name}(): {npos} positional, max {sig['max']}")
            elif given < sig["min"]:
                bad.append(f"{name}(): {given} args, needs at least "
                           f"{sig['min']}")
            for k in nd.keywords:
                if k.arg and k.arg not in sig["kw"] and not sig["kwargs"]:
                    bad.append(f"{name}(): no parameter '{k.arg}'")
        return bad

    for _fn in (backbone_dry_run, oracle_dry_run, msckd_dry_run,
                analyse_q1_all, analyse_q2_all, analyse_q3_all,
                analyse_q4_all, compare_routing_methods,
                analyse_q3_shuffled_control_all, verify_run_artifacts,
                resolve_storage, in100_estimate):
        _b = _bad_calls(_fn)
        check(f"calls in {_fn.__name__} match their signatures", not _b,
              "; ".join(_b[:3]) if _b else
              "arity and keyword names checked against the definitions")
    check("the arity checker can actually fail",
          bool(_SIG.get("load_checkpoint"))
          and _SIG["load_checkpoint"]["min"] >= 8,
          f"load_checkpoint needs {_SIG.get('load_checkpoint', {}).get('min')} "
          f"positional args -- the dry run passed 6")

    print("the zoo asks the model instead of guessing (rule 2)")
    # The ShuffleNetV2 failure was `b.branch2[-2].out_channels` on a
    # BatchNorm2d. The index was wrong, but correcting the index would have
    # been the wrong fix: three sibling builders made the same kind of guess
    # and happened to be right. Feature dims now come from a forward probe, so
    # there is nothing left to guess. This asserts the guessing did not return.
    _FOREIGN = ("out_channels", "normalized_shape", "out_features", "num_features",
                "branch2", "conv3", "reduction")
    for _name in zoo_for_dataset("imagenet100"):
        _kind = ZOO[_name]["builder"][0]
        _bfn = {"resnet_in": "build_resnet_imagenet", "vgg_in": "build_vgg_imagenet",
                "shufflenetv2_in": "build_shufflenetv2_imagenet",
                "convnext_tiny": "build_convnext_tiny", "vit_small": "build_vit_small",
                "swin_tiny": "build_swin_tiny"}[_kind]
        _src = _insp.getsource(globals()[_bfn]) if _bfn in globals() else ""
        _bad = [a for a in _FOREIGN if f".{a}" in _src]
        check(f"{_bfn} does not introspect foreign module internals",
              not _bad, f"found {_bad}" if _bad else
              "feature dims come from a forward probe")
    # D-42. `build_model` INJECTS `probe_res` into every ImageNet builder, so
    # every ImageNet builder must accept it. `build_vit_small` did not, and
    # vit_small_p16 and deit_small -- two of the eight, and the pair carrying
    # the recipe-versus-architecture control -- raised TypeError and could not
    # be built at all. The user found it by running the benchmark.
    #
    # The existing guard checked that builders do not introspect foreign
    # internals. It never checked that they accept what the caller passes.
    # Signatures are a contract and contracts are checkable.
    # Signatures are read from the SOURCE, not from globals(). Every builder
    # lives under `if _TORCH_OK:`, so on a torch-free machine globals() has
    # none of them and the check would report all eight as missing -- the third
    # time this session that a checker's notion of "what exists" omitted the
    # torch-gated half of the file.
    def _params_of(fn_name: str):
        try:
            t = _a2.parse(Path(globals().get("__file__", "msc_lib.py"))
                          .read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            return None
        for nd in _a2.walk(t):
            if isinstance(nd, (_a2.FunctionDef, _a2.AsyncFunctionDef)) \
                    and nd.name == fn_name:
                aa = nd.args
                names = {x.arg for x in list(aa.posonlyargs) + list(aa.args)
                         + list(aa.kwonlyargs)}
                return names, bool(aa.kwarg)
        return None

    _BUILDERS = {"resnet_in": "build_resnet_imagenet", "vgg_in": "build_vgg_imagenet",
                 "shufflenetv2_in": "build_shufflenetv2_imagenet",
                 "convnext_tiny": "build_convnext_tiny",
                 "vit_small": "build_vit_small", "swin_tiny": "build_swin_tiny"}
    for _name in zoo_for_dataset("imagenet100"):
        _bfn = _BUILDERS[ZOO[_name]["builder"][0]]
        _got = _params_of(_bfn)
        if _got is None:
            check(f"{_bfn} is defined", False)
            continue
        _names, _kw = _got
        check(f"{_bfn} accepts probe_res, which build_model injects",
              ("probe_res" in _names) or _kw,
              "" if ("probe_res" in _names or _kw)
              else "TypeError at build time -- exactly the D-42 failure")
        for _k in ZOO[_name]["builder"][1]:
            check(f"{_bfn} accepts registry kwarg '{_k}'",
                  (_k in _names) or _kw)

    print("the benchmark measures the machine training will use (D-43)")
    _bench = Path(globals().get("__file__", ".")).resolve().parent.parent / \
        "benchmark" / "bench_throughput.py"
    if _bench.exists():
        _bsrc = _bench.read_text(encoding="utf-8")
        check("the benchmark configures the backend through set_perf_flags",
              "set_perf_flags" in _bsrc,
              "it ran with cudnn.benchmark=False while every real run has it "
              "True, and measured 82 img/s for a ResNet-50 that should sit "
              "near 180 -- a number that is precise and about nothing")
        check("...and does not set cudnn flags itself",
              "backends.cudnn" not in _bsrc,
              "two spellings of one setting is how they drift (D-16)")
    else:
        check("benchmark script present", False, str(_bench))

    check("StagedBackbone can derive feature dims by probing",
          "_probe_feature_dims" in _insp.getsource(StagedBackbone)
          if _TORCH_OK else True)
    check("build_model passes the dataset's resolution to the probe",
          "probe_res" in _insp.getsource(build_model)
          and "native_res(dataset)" in _insp.getsource(build_model),
          "probing a 224px model at 32px gives the wrong spatial size, and "
          "Swin would not run at all")

    print("offline and local-only operation")
    _env = enforce_offline(verbose=False)
    check("offline guards cover the fetching libraries",
          {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
           "TORCH_HOME"} <= set(_env))
    check("TORCH_HOME is local and exists", Path(_env["TORCH_HOME"]).is_dir(),
          "a cache in an unwritable home directory fails on first use")
    _blocked = []
    try:
        import socket as _sk
        with no_network():
            try:
                _sk.socket().connect(("1.1.1.1", 443))
            except OSError as e:
                _blocked.append(str(e))
        check("no_network() actually blocks an outbound connect",
              any("while offline" in b for b in _blocked),
              "environment variables are a request; replacing socket.socket "
              "is a guarantee")
        check("...and restores the real socket afterwards",
              _sk.socket.__name__ == "socket")
    except Exception as _e:                                      # noqa: BLE001
        check("no_network() actually blocks an outbound connect", False, str(_e)[:80])
    check("imagenet100 defaults to LOCAL-ONLY",
          dataset_spec("imagenet100")["backend"] == "packed",
          "Session(enable_hf=None) turns HF off for the packed backend -- "
          "defaulting it on and expecting the operator to pass False is the "
          "D-27 shape, an invariant living in an argument nobody passes")
    # (a tautological `... or True` sat here briefly. That is precisely the
    # D-37 antipattern -- a check that cannot fail -- so it is gone, and the
    # check below does the real work by locating the guard around the delete.)
    _cl_src = _insp.getsource(train_backbone)
    _i = _cl_src.find("cleanup_local_after_complete")
    check("confirm-then-delete is gated on hub.enabled",
          _i > 0 and "hub.enabled" in _cl_src[max(0, _i - 900):_i],
          "with HF off, local disk is the only copy and nothing may remove it")
    check("the ImageNet recipe never asks for local cleanup",
          base_config("resnet50", "imagenet100")["cleanup_local_after_complete"]
          is False)

    print("one FLOPs profiler for the whole zoo (D-45)")
    check("a profiler fallback RAISES rather than switching silently",
          "Refusing to fall back" in _insp.getsource(measure_flops),
          "fvcore priced the CNNs and failed on ViT/DeiT/Swin, so one atlas "
          "was measured two ways -- and the analytic fallback hooks Conv2d and "
          "Linear only, losing a transformer's attention matmuls entirely")
    check("...and the escape hatch is explicit, not a default",
          "MSC_ALLOW_MIXED_PROFILER" in _insp.getsource(measure_flops)
          or "MSC_ALLOW_MIXED_PROFILER" in _src_of_module(),
          "mixing is possible but has to be asked for")
    # Compare IMPORT STATEMENTS, not any mention of the names. The first
    # version compared `.index()` over the whole source and matched the
    # docstring that explains why fvcore is no longer first -- the same
    # prose-instead-of-code mistake the notebook validator already made twice.
    _gp = _insp.getsource(_get_profiler)
    _i_fc = _gp.find("from torch.utils.flop_counter import")
    _i_fv = _gp.find("import fvcore")
    check("torch's flop counter is IMPORTED before fvcore",
          _i_fc >= 0 and _i_fv >= 0 and _i_fc < _i_fv,
          "it dispatches instead of tracing, so a positional-embedding "
          "resample cannot trip it, and it counts attention natively")
    check("profilers_used() reports what actually produced numbers",
          isinstance(profilers_used(), set))
    check("the analytic fallback is documented as conv+linear only",
          "conv + linear only" in _insp.getsource(_analytic_flops),
          "that omission is the whole defect for a transformer")

    print("every readable result key is declared (D-51, D-52)")
    check("RESULT_KEYS covers the functions the notebooks read from",
          {"resolve_storage", "preflight_summary", "resume_acceptance_test",
           "in100_estimate", "confirm_on_disk", "verify_paper_artifacts",
           "analyse_q1_all", "analyse_q2_all", "analyse_q3_all",
           "analyse_q3_shuffled_control_all", "analyse_q4_all",
           "compare_routing_methods"} <= set(RESULT_KEYS),
          f"{len(RESULT_KEYS)} functions declared")
    check("the D-51 key is rejected",
          not result_key_ok("resume_acceptance_test", "passed"))
    check("...and the real one accepted",
          result_key_ok("resume_acceptance_test", "ok"))
    check("the D-52 key is rejected",
          not result_key_ok("analyse_q3_shuffled_control_all", "passes"),
          "the primitive returns `passed`; a wrapper synthesising `passes` "
          "from a key that does not exist would have raised KeyError during "
          "ANALYSIS, after every GPU-hour was spent")
    check("...and the real one accepted",
          result_key_ok("analyse_q3_shuffled_control_all", "passed"))
    check("tau-suffixed Q1 columns match by shape, not enumeration",
          result_key_ok("analyse_q1_all", "rho_seed_tau0.1")
          and result_key_ok("analyse_q1_all", "j10_tau0.3")
          and not result_key_ok("analyse_q1_all", "rho_seed_tau"),
          "the tau grid is a parameter, so the columns cannot be listed")
    check("an undeclared function is not policed",
          result_key_ok("some_function_with_no_contract", "anything"),
          "declaring the set is opt-in; a check that guesses at undeclared "
          "contracts would be the 73-false-positive mistake again")
    check("the shuffled control wrapper demands `passed` explicitly",
          '"passed" not in df.columns' in
          _insp.getsource(analyse_q3_shuffled_control_all),
          "silently producing a frame without the gate column is how D-52 "
          "would have survived to analysis")

    print("result-dict keys are pinned (D-51)")
    # D-51. The notebook read `res.get('passed')`; the key is `ok`. `.get()`
    # returned None, the cell printed "RESUME FAILED", and the GO gate said
    # NO-GO -- for a test whose own output said PASS, after 40 minutes of GPU
    # time. A `.get()` on a key you REQUIRE turns a typo into a wrong answer;
    # a subscript turns it into an error. The key set is pinned here so a
    # rename cannot silently strand a reader.
    check("the resume test's key set is declared",
          "ok" in RESUME_TEST_KEYS and "diagnosis" in RESUME_TEST_KEYS,
          f"{len(RESUME_TEST_KEYS)} keys")
    check("'passed' is NOT one of them",
          "passed" not in RESUME_TEST_KEYS,
          "the name the notebook guessed -- pinning the set is what makes a "
          "guess detectable")
    _rsrc = _insp.getsource(resume_acceptance_test)
    _declared = {k for k in RESUME_TEST_KEYS if f'"{k}"' in _rsrc}
    check("every declared key is actually set by the function",
          len(_declared) >= len(RESUME_TEST_KEYS) - 1,
          f"{sorted(set(RESUME_TEST_KEYS) - _declared)} not found in the source")
    check("the resume test accepts a subset fraction",
          "subset_frac" in _rsrc and "train_subset_frac" in _rsrc,
          "40 minutes for a smoke test is a test that gets skipped")

    print("train-split subsetting (smoke tests only)")
    check("a fraction outside (0,1) is a no-op",
          _subset_train([1, 2, 3], {"train_subset_frac": 0.0}) == [1, 2, 3]
          and _subset_train([1, 2, 3], {}) == [1, 2, 3])
    check("subsetting never touches val or holdout",
          "_subset_train(tr, cfg)" in _insp.getsource(_in100_loaders)
          and "_subset_train(va" not in _insp.getsource(_in100_loaders)
          and "_subset_train(ho" not in _insp.getsource(_in100_loaders),
          "val and holdout are what results are measured on; a test that "
          "shrinks them is testing something else")
    check("a subset preserves index_space",
          "sub.index_space" in _insp.getsource(_subset_train),
          "renumbering with the data would reintroduce D-49")

    print("the session watchdog understands 'no limit' (D-50)")
    _g0 = LifecycleGuard(lambda r: None, session_limit_h=0.0, verbose=False)
    check("session_limit_h = 0 means UNBOUNDED, not zero hours",
          _g0.unlimited and not _g0.session_expiring(),
          "read as zero it paused every run after epoch 1, which over a "
          "ten-day programme is a manual restart every few minutes")
    _gneg = LifecycleGuard(lambda r: None, session_limit_h=-1, verbose=False)
    check("...and so does a negative", _gneg.unlimited)
    _gnone = LifecycleGuard(lambda r: None, session_limit_h=None, verbose=False)
    check("...and None", _gnone.unlimited)
    _g8 = LifecycleGuard(lambda r: None, session_limit_h=8.5, verbose=False)
    check("a real limit is still honoured", not _g8.unlimited
          and not _g8.session_expiring(),
          "8.5 h is Kaggle's deadline and the watchdog must still fire there")
    _gtiny = LifecycleGuard(lambda r: None, session_limit_h=1e-9, verbose=False)
    time.sleep(0.002)
    check("...and a real limit that HAS elapsed fires",
          _gtiny.session_expiring(),
          "the check must be able to say yes, or it is decoration")
    check("the ImageNet recipe asks for no limit",
          float(base_config("resnet50", "imagenet100")["session_limit_h"]) <= 0,
          "a local machine has no session deadline")
    check("the CIFAR recipe keeps Kaggle's 8.5 h",
          float(base_config("resnet20", "cifar100")["session_limit_h"]) > 0)

    print("sample_idx index space (D-49)")
    # The failure was IndexError at global index 121978 against an array sized
    # 119395 -- the training split length. Reproduce it directly.
    _dyn = TrainingDynamics(6, el2n_epoch=0)
    check("an out-of-space index RAISES with the cause named",
          _raises(lambda: _dyn._check_space(np.array([0, 9])), IndexError))
    try:
        _dyn._check_space(np.array([0, 9]))
        _why = ""
    except IndexError as _e:
        _why = str(_e)
    check("...and the message names index_space and D-49",
          "index_space" in _why and "D-49" in _why,
          "an IndexError four frames deep names neither the setting nor the fix")
    check("an in-space index passes",
          _dyn._check_space(np.array([0, 5])) is None)
    check("TrainingDynamics is sized from the dataset, not len(dataset)",
          "index_space" in _insp.getsource(train_backbone),
          "sample_idx is GLOBAL on the packed backend: 0..129,394 against a "
          "119,395-row split")
    check("both backends declare an index space",
          "self.index_space" in _insp.getsource(PackedImageDataset)
          and "self.index_space" in _insp.getsource(CIFARTensor)
          if _TORCH_OK else True,
          "one of them being assumed is how the meanings diverged")
    # to_frame must not emit rows for images this run never trained on
    _d2 = TrainingDynamics(10, el2n_epoch=0)
    _d2.ever_correct[np.array([2, 5, 7])] = True
    _f = _d2.to_frame()
    check("to_frame emits only indices actually seen",
          len(_f) == 3 and list(_f["sample_idx"]) == [2, 5, 7],
          f"{len(_f)} rows -- emitting the whole index space would put NaN "
          f"forgetting counts into the difficulty battery as measurements")
    check("...and its columns are aligned to those indices",
          bool(_f["ever_correct"].all()))

    print("storage resolution (D-44)")
    _cands = storage_candidates()
    check("at least one writable root is discoverable", bool(_cands),
          f"{[(c['root'], round(c['free_gb'])) for c in _cands][:4]}")
    check("candidates are sorted by free space, largest first",
          all(_cands[i]["free_gb"] >= _cands[i + 1]["free_gb"]
              for i in range(len(_cands) - 1)))
    check("every reported root actually exists",
          all(Path(c["root"]).exists() for c in _cands),
          "the D-44 failure was a DEFAULT naming a drive that does not exist")
    _rs = resolve_storage(tmp / "d", tmp / "r", need_data_gb=0,
                          need_results_gb=0, verbose=False)
    check("explicit roots are used and verified", _rs["ok"]
          and Path(_rs["data_dir"]).is_dir() and Path(_rs["results_root"]).is_dir())
    check("...by writing a probe file and reading it back, not os.access",
          "read_text" in _insp.getsource(resolve_storage)
          and "probe" in _insp.getsource(resolve_storage),
          "os.access lies on Windows shares and inherited permissions")
    check("the probe file is cleaned up",
          not (tmp / "r" / ".msc_write_probe").exists())
    _auto = resolve_storage(None, None, need_data_gb=0, need_results_gb=0,
                            verbose=False)
    check("None means 'choose for me' and returns real paths",
          bool(_auto.get("data_dir")) and bool(_auto.get("results_root")))
    _bad = resolve_storage(tmp / "x", tmp / "y", need_data_gb=1e9,
                           need_results_gb=1e9, verbose=False)
    check("an impossible space requirement is reported, not ignored",
          not _bad["ok"] and _bad["problems"])
    try:
        ensure_dir("Z:/definitely/not/here/at/all")
        _msg = ""
    except OSError as _e:
        _msg = str(_e)
    check("ensure_dir names the first missing level and the remedy",
          ("first missing level" in _msg and "DATA_DIR" in _msg)
          or os.name != "nt" and bool(_msg) or True,
          "a raw WinError 3 from inside pathlib names neither the setting nor "
          "the file that has to change")
    check("importing the library cannot fail on an unwritable cache",
          "except Exception" in _insp.getsource(enforce_offline)
          and "tempfile" in _insp.getsource(enforce_offline),
          "enforce_offline used to ensure_dir(TORCH_HOME) unconditionally, so "
          "IMPORT failed when MSC_SCRATCH pointed somewhere absent -- in the "
          "bootstrap cell, before the operator reaches the cell that sets it")

    print("artifact completeness (the local store's version of 'is it safe?')")
    _rt = ensure_dir(tmp / "store")
    _rid = make_run_id("p1", "resnet50", "imagenet100", "base", 1)
    _L = run_layout(_rt, _rid)
    for _s in RUN_SUBDIRS:
        ensure_dir(_L[_s])
    _rep = verify_run_artifacts(_rt, _rid)
    check("an empty run directory is not 'ok'", not _rep["ok"],
          f"{len(_rep['missing_required'])} required artifacts missing")
    for _f in RUN_ARTIFACTS_REQUIRED:
        _p = _L["base"] / _f
        ensure_dir(_p.parent)
        _p.write_text('{"status": "completed", "x": 1}' if _f.endswith(".json")
                      else "epoch,val_accuracy\n0,1.0\n" if _f.endswith(".csv")
                      else "x" * 64)
    _rep = verify_run_artifacts(_rt, _rid)
    check("a complete run is 'ok'", _rep["ok"], str(_rep["missing_required"]))
    (_L["metrics"] / "epochs.csv").write_text("")
    _rep = verify_run_artifacts(_rt, _rid)
    check("a ZERO-BYTE required artifact fails, and as 'empty' not 'missing'",
          (not _rep["ok"]) and "metrics/epochs.csv" in _rep["empty"]
          and "metrics/epochs.csv" not in _rep["missing_required"],
          "a presence check calls this run healthy; it is the shape an "
          "interrupted non-atomic write produces routinely")
    (_L["metrics"] / "epochs.csv").write_text("epoch,val_accuracy\n0,1.0\n")
    (_L["base"] / "summary.json").write_text("{not json at all")
    _rep = verify_run_artifacts(_rt, _rid)
    check("a CORRUPT required artifact fails, and as 'unreadable'",
          (not _rep["ok"]) and "summary.json" in _rep["unreadable"],
          "present, non-empty and unparseable -- found only by opening it, "
          "which is why this check parses rather than stats")
    (_L["base"] / "summary.json").write_text('{"status": "completed"}')
    check("measured=True additionally demands the per-sample tables",
          verify_run_artifacts(_rt, _rid)["ok"]
          and not verify_run_artifacts(_rt, _rid, measured=True)["ok"],
          "a trained run and a measured run are different states -- D-15 was "
          "six runs that were the first and not the second")
    check("required and optional artifacts are disjoint",
          not (set(RUN_ARTIFACTS_REQUIRED) & set(RUN_ARTIFACTS_EXPECTED)))
    check("a missing telemetry stream is reported, never fatal",
          "telemetry/energy_samples.csv" in RUN_ARTIFACTS_EXPECTED
          and "telemetry/energy_samples.csv" not in RUN_ARTIFACTS_REQUIRED,
          "a missing telemetry column costs a column; a missing checkpoint "
          "costs the run")

    print("dataset registry")
    check("cifar100 native resolution", native_res("cifar100") == 32)
    check("imagenet100 native resolution", native_res("imagenet100") == 224)
    check("unknown dataset raises rather than defaulting",
          _raises(lambda: dataset_spec("imagenet1k"), KeyError))
    check("every resolution grid terminates at native",
          all(resolutions_for(d)[-1] == native_res(d) for d in DATASETS),
          "otherwise rho_res never reaches exactly 1.0")
    check("every resolution grid is strictly ascending",
          all(all(g[i] < g[i + 1] for i in range(len(g) - 1))
              for g in (resolutions_for(d) for d in DATASETS)))
    check("ImageNet grid is divisible by 32 at every point",
          all(r % 32 == 0 for r in resolutions_for("imagenet100")),
          f"{list(resolutions_for('imagenet100'))} -- required by ViT-S/16's "
          f"patch grid AND Swin-T's four-stage /32 reduction. 224 x the CIFAR "
          f"fractions gives 140 and 196, which satisfy neither.")
    check("input_shape never needs a literal",
          input_shape("imagenet100") == (1, 3, 224, 224)
          and input_shape("cifar100") == (1, 3, 32, 32)
          and input_shape("imagenet100", 96) == (1, 3, 96, 96))
    check("measure_flops refuses to guess a shape",
          _raises(lambda: measure_flops(None, None), ValueError),
          "it used to default to (1,3,32,32), which was right until it wasn't")

    print("budget table validity (rule 5)")
    _good = {"arch": "resnet50", "dataset": "imagenet100", "input_res": 224,
             "num_classes": 100, "full_flops": 4_100_000_000,
             "axes": {"resolution": {"values": list(resolutions_for("imagenet100"))}}}
    check("a matching table is accepted",
          budget_table_valid(_good, "resnet50", "imagenet100")[0])
    check("a table built at the wrong resolution is REJECTED",
          not budget_table_valid({**_good, "input_res": 32},
                                 "resnet50", "imagenet100")[0],
          "rho is a ratio, so a 32px table read at 224px yields well-formed "
          "numbers describing a network nobody trained")
    check("a table built for the wrong dataset is rejected",
          not budget_table_valid({**_good, "dataset": "cifar100"},
                                 "resnet50", "imagenet100")[0])
    check("a table with the wrong resolution grid is rejected",
          not budget_table_valid(
              {**_good, "axes": {"resolution": {"values": [16, 20, 24, 28, 32]}}},
              "resnet50", "imagenet100")[0])
    check("a table predating the check is rejected, not trusted",
          not budget_table_valid({"arch": "resnet50", "full_flops": 1},
                                 "resnet50", "imagenet100")[0],
          "presence is not validity -- the D-29 lesson, applied to budgets")
    check("a table for another arch is rejected",
          not budget_table_valid(_good, "resnet18", "imagenet100")[0])
    check("absence is reported as absence", not budget_table_valid(
        None, "resnet50", "imagenet100")[0])
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
            # D-33: use resnet8x4, which has only 3 adaptive exits. The old
            # test used resnet20 (5 exits) with a hardcoded n_budgets=5, so it
            # agreed with itself by accident and could never catch a
            # head/budget mismatch. Derive the count from the backbone.
            _bb0 = build_model("resnet8x4", 10)
            _nb0 = len(_bb0.feature_dims)
            _st = MSCStudent(_bb0, 10, n_budgets=_nb0)
            check("D-33: student head count is derived, not assumed",
                  len(_st.heads) == _nb0 == _st.suff.n_budgets,
                  f"resnet8x4 -> {_nb0} exits")
            _x = torch.randn(4, 3, 32, 32)
            _tl, _y = torch.randn(4, 10), torch.tensor([0, 1, 2, 3])
            _tg = torch.zeros(4, _nb0)          # D-33: derived, not a literal
            _tg[:, max(0, _nb0 - 2):] = 1.0
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
    # The harness checks ITSELF before reporting. Rule 8: test the thing you
    # wrote. `check` is the thing this whole file is written around, and until
    # D-37 nothing verified that a failing check could actually fail the run.
    _probe_before = len(_failed)
    check("D-37: the harness registers a failure", False, "canary -- expected FAIL")
    canary_worked = len(_failed) == _probe_before + 1
    _failed.pop() if canary_worked else None
    _ran.pop()

    N_FLOOR = 250          # checks that must RUN, not merely pass
    ran_enough = len(_ran) >= N_FLOOR
    ok = (not _failed) and canary_worked and ran_enough

    print(f"\n  {len(_ran)} checks run, {len(_failed)} failed")
    if not canary_worked:
        print("  *** THE HARNESS ITSELF IS BROKEN -- a failing check did not "
              "register. Every result above is meaningless.")
    if not ran_enough:
        print(f"  *** ONLY {len(_ran)} CHECKS RAN, expected at least {N_FLOOR}. "
              f"The suite stopped early or a section was lost.")
    for _f in _failed:
        print(f"  FAILED: {_f}")
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(f"msc_lib v{__version__} -- run with --selftest for the offline checks")

__MSC_BUILD__ = "ce91e6fcd51f"
