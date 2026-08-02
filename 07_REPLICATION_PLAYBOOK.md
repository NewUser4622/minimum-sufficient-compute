# Replication Playbook

**How to build this research infrastructure again, in any project.**

Project-agnostic. Nothing here is specific to MSC or CIFAR-100 — it's the pattern for running hundreds of GPU-hours of experiments across several free Kaggle accounts, with HuggingFace as the only permanent store, without losing work, colliding, or hitting rate limits.

Written after building it once and finding six real bugs in the process. Every one of those bugs is documented below with the symptom, the cause, and the fix — because they are not obvious, and you will hit all of them.

---

## Contents

1. [The constraints that shape everything](#1)
2. [Architecture in one page](#2)
3. [Component 1 — the base64 library bootstrap](#3)
4. [Component 2 — the batched uploader](#4)
5. [Component 3 — rate limiting](#5)
6. [Component 4 — the run registry](#6)
7. [Component 5 — work sharding across N accounts](#7)
8. [Component 6 — resumability](#8)
9. [Component 7 — lifecycle guards](#9)
10. [Component 8 — repository layout](#10)
11. [Component 9 — telemetry](#11)
12. [Component 10 — the preflight notebook](#12)
13. [The six bugs, and how to avoid them](#13)
14. [Build order](#14)
15. [Checklist](#15)

---

<a name="1"></a>
## 1. The constraints that shape everything

| Constraint | Consequence |
|---|---|
| Kaggle session dies at ~9–12 h, without warning | Every run must checkpoint and resume. Nothing may depend on finishing in one session. |
| `/kaggle/working` is 20 GB | Cannot hold datasets or large intermediates. |
| `/kaggle/temp` is ~1 TB, wiped at session end | Perfect staging area — use it for everything. |
| HuggingFace write limit ~128 commits/hour, **per user** | Push on a schedule, batch aggressively, and never let a limiter live on a per-repo object. |
| No shared filesystem between accounts | HF Hub is the only coordination substrate. |
| No locking primitive on HF Hub | Coordination must be lock-free. |
| HF has no append operation | Any shared append-only file will lose writes. |

Two of these — the per-*user* rate limit and the absence of append — are the ones that quietly break naive designs. Both are covered below.

---

<a name="2"></a>
## 2. Architecture in one page

```
 Kaggle account 1        Kaggle account 2        ...        account N
 ┌──────────────┐        ┌──────────────┐                  ┌──────────────┐
 │ NB (WORKER 0)│        │ NB (WORKER 1)│                  │ NB (WORKER N)│
 │              │        │              │                  │              │
 │ bootstrap    │        │ bootstrap    │                  │ bootstrap    │
 │ lib from b64 │        │ lib from b64 │                  │ lib from b64 │
 │      ↓       │        │      ↓       │                  │      ↓       │
 │ plan_work()  │        │ plan_work()  │                  │ plan_work()  │
 │ → my slice   │        │ → my slice   │                  │ → my slice   │
 │      ↓       │        │      ↓       │                  │      ↓       │
 │ /kaggle/temp │        │ /kaggle/temp │                  │ /kaggle/temp │
 │  staging     │        │  staging     │                  │  staging     │
 │      ↓       │        │      ↓       │                  │      ↓       │
 │ uploader     │        │ uploader     │                  │ uploader     │
 │ (30-min batch│        │              │                  │              │
 │  + shared    │        │              │                  │              │
 │  rate limit) │        │              │                  │              │
 └──────┬───────┘        └──────┬───────┘                  └──────┬───────┘
        └───────────────────────┴───────────  ...  ───────────────┘
                                ↓
                  ┌──────────────────────────────┐
                  │   HuggingFace  (ONE repo)    │
                  │                              │
                  │  runs/{run_id}/…             │
                  │  registry/events/*.jsonl     │  ← one shard per writer
                  │  registry/claims/*.json      │
                  │  tables/ analysis/ paper/    │
                  └──────────────────────────────┘
```

**The single most important idea:** workers never talk to each other. Ownership is decided by arithmetic every worker computes independently, and progress is read from HF. There is no negotiation, no locking, and therefore nothing to deadlock.

---

<a name="3"></a>
## 3. Component 1 — the base64 library bootstrap

**Problem.** You have a 5,000-line library. Kaggle notebooks are self-contained. `pip install` of a private package needs auth; `git clone` needs a token in the notebook; a Kaggle Dataset of code needs re-uploading on every edit.

**Solution.** Embed the library as base64 in cell 1. It writes itself to disk and imports.

```python
# generator (runs on your machine)
lib_b64 = base64.b64encode(Path("src/mylib.py").read_bytes()).decode()
chunks = [lib_b64[i:i+96] for i in range(0, len(lib_b64), 96)]
cell = "_LIB = (\n" + ",\n".join(f"    '{c}'" for c in chunks) + ",\n)\n"
```

```python
# cell 1 of every notebook
import base64, sys
from pathlib import Path
WORK = Path('/kaggle/working')
_LIB = ('...', '...', ...)
(WORK / 'mylib.py').write_bytes(base64.b64decode(''.join(_LIB)))
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
for _m in [m for m in sys.modules if m == 'mylib']:
    del sys.modules[_m]          # force reimport if the cell is re-run
import mylib
```

**Why chunk it.** A single 300 KB string literal makes notebook JSON unreadable and some editors choke. 96-char chunks stay diffable.

**Why delete from `sys.modules`.** Without it, re-running cell 1 after editing does nothing — Python returns the cached module and you debug a ghost.

**The critical discipline:** the `.py` file is the source of truth; the notebook is *generated*. Write `build_notebooks.py` that regenerates every notebook from the library. Edit Python, run the generator, re-upload. Never hand-edit the base64.

> Verify this: after generating, decode the blob out of the `.ipynb` and byte-compare against the source file. A silent truncation here means you debug the wrong code for an hour.

---

<a name="4"></a>
## 4. Component 2 — the batched uploader

**Problem.** Pushing per-file blows the rate limit. Pushing synchronously blocks training.

**Solution.** One background thread, one buffer keyed by repo path, one commit per cycle.

```python
class BackgroundUploader:
    BATCH_INTERVAL_SEC = 1800      # 30 min
    def enqueue(self, local_path, repo_path):
        fp = f"{repo_path}|{size}|{mtime}"
        if fp in self._pushed:          # dedup: unchanged file, skip
            return False
        self._buffer[repo_path] = ...   # newer supersedes pending
    def _loop(self):
        while not stop:
            self._wakeup.wait(timeout=BATCH_INTERVAL_SEC)
            batch, self._buffer = list(self._buffer.values()), {}
            self._rate_limiter.wait_for_slot()
            api.create_commit(operations=[CommitOperationAdd(...) for f in batch])
```

Four properties that matter:

- **Buffer keyed by repo path.** A rolling checkpoint enqueued five times in one window produces one file in one commit, not five.
- **Fingerprint dedup on `(path, size, mtime)`.** Config files never change; re-enqueueing them is free.
- **Failed batch goes back in the buffer**, using `setdefault` so a newer version that arrived meanwhile isn't clobbered.
- **A failed push never kills training.** Log it, retry next cycle.

**429 handling.** HF's response body carries a human-readable hint. Parse it and sleep exactly that long — it beats blind exponential backoff, which either wastes a window or hammers early.

```python
m = re.search(r"retry after (\d+)\s*second", err, re.I)
if m: return float(m.group(1)) + 2.0
m = re.search(r"in about (\d+)\s*minute", err, re.I)
if m: return float(m.group(1)) * 60.0 + 5.0
```

**Auth failures must break immediately**, not retry eight times — a read-only token will never become writable.

---

<a name="5"></a>
## 5. Component 3 — rate limiting

### ⚠ The bug that cost the most

HuggingFace's write limit is **per user**, not per repository. If your limiter lives on the uploader object, then N repos multiply your budget by N.

I had two repos, each capped at 20 commits/hour. One account could emit 40/hour. Six accounts: **240/hour against a real ceiling near 128.** The cap was decorative.

**Fix: one bucket per token, process-wide.**

```python
class _SharedRateLimiter:
    _buckets = {}                     # token hash -> bucket
    _registry_lock = threading.Lock()

    @classmethod
    def for_token(cls, token, limit):
        key = hashlib.sha256((token or "anon").encode()).hexdigest()[:16]
        with cls._registry_lock:
            b = cls._buckets.setdefault(key, cls(limit))
            b.limit = min(b.limit, limit)     # most conservative wins
            return b

    def wait_for_slot(self, stop):
        while not stop.is_set():
            now = time.time()
            with self._lock:
                self._times = [t for t in self._times if now - t < 3600]
                if len(self._times) < self.limit:
                    return
                oldest = self._times[0]
            stop.wait(max(1.0, 3600 - (now - oldest) + 2.0))
```

**Budget arithmetic.** Cap at `128 / n_accounts`, rounded down, with headroom. Six accounts → 20 each = 120. At a 30-minute cycle one 9-hour session makes ~18 commits, comfortably inside.

**When the cap is hit, sleep — don't fail.** Training continues; only the uploader waits.

---

<a name="6"></a>
## 6. Component 4 — the run registry

You need to know, across all accounts: what exists, what's running, what's finished, who owns what.

### ⚠ The second serious bug: lost updates

HF has no append. Every worker appending to a shared `runs.jsonl` and pushing it means **the last push silently destroys every other worker's lines.** No error. The file just forgets.

I caught this on a live repo: two runs were training, and the ledger listed one.

This is expensive, not cosmetic — work planning reads *completion* from the ledger, so a lost `completed` entry makes a finished 3-hour run look unfinished, and someone retrains it.

**Fix: shard by writer.**

```
registry/events/{account}_w{worker_id}_{session_id}.jsonl
```

Each writer owns one file nobody else touches. Reads merge every shard.

```python
def entries(self):
    out = []
    for p in self.events_dir.glob("*.jsonl"):
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    out.sort(key=lambda e: e.get("ts", 0.0))     # float clock, see below
    return out

def latest(self):
    st = {}
    for e in self.entries():
        rid = e["run_id"]
        # 'completed' is sticky: a late heartbeat from a stale shard must not
        # resurrect a finished run, or it gets trained twice.
        if st.get(rid, {}).get("state") == "completed" and e["state"] != "completed":
            continue
        st[rid] = e
    return st
```

**Two details that matter:**

- **Store a float epoch timestamp**, not just an ISO string. Second granularity means same-second events across shards sort ambiguously — exactly where ordering must be trustworthy.
- **Make terminal states sticky.** Otherwise out-of-order pushes resurrect finished work.

This is the same pattern as any collision-safe shard writer: unique filename per writer, reconcile on read.

---

<a name="7"></a>
## 7. Component 5 — work sharding across N accounts

Every worker runs the **same notebook**. One line differs:

```python
NUM_WORKERS = 6
WORKER_ID   = 0      # 0..N-1, different on each account
```

### The naive version: hashing

```python
def owner(key, n):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n
```

Three guarantees, free, with zero communication:

- **No overlap** — a hash has one value
- **No gaps** — everything hashes somewhere
- **Restart-proof** — depends only on the key, not on timing or crashes

Perfect when the universe is large and open-ended (10,000 images arriving over time).

### ⚠ Where hashing fails: small universes with uneven cost

45 training jobs into 6 buckets gave:

```
[11, 7, 4, 10, 3, 10]     →  4.91× imbalance
```

Worse, jobs differed 6× in cost (a small ResNet ~1 h, a ViT ~6 h). A phase ends when the **slowest** worker ends, so this is a direct multiplier on wall-clock.

**Fix: longest-processing-time-first bin packing.**

```python
def assign_workers(ids, n, mode="cost", costs=None):
    ids = sorted(ids)                    # canonical order on every machine
    if mode == "hash":
        return {r: owner(r, n) for r in ids}
    if mode == "balanced":
        return {r: i % n for i, r in enumerate(ids)}
    # cost: greedy LPT
    jobs = sorted(ids, key=lambda r: -estimate_cost(r, costs))
    load, out = [0.0] * n, {}
    for r in jobs:
        w = int(np.argmin(load))
        out[r] = w
        load[w] += estimate_cost(r, costs)
    return out
```

Result:

| Mode | Counts | Imbalance |
|---|---|---|
| hash | `[11, 7, 4, 10, 3, 10]` | 4.91× |
| balanced | `[8, 8, 8, 7, 7, 7]` | 1.31× |
| **cost** | `[7, 7, 7, 8, 8, 8]` | **1.02×** |

Still deterministic, still zero-communication — every worker computes the same packing from the same inputs.

**Make it self-correcting.** Seed with rough estimates, then replace them with measured per-unit times once any run has finished:

```python
measured = estimate_costs_from_history(data_dir)   # median sec/epoch per arch
costs = {**HINTS, **measured}
```

**Add work stealing as a safety net.** After a worker exhausts its own slice, let it pick up runs whose heartbeat has gone stale (>2 h):

```python
if steal_stale:
    for r in universe:
        if r in done or owner[r] == me: continue
        st = latest.get(r)
        if st and st["state"] in ("running", "paused"):
            if age(st) >= STALE_SEC: stolen.append(r)
            else:                    busy_elsewhere.append(r)
```

**Own work always first.** Two live workers then never fight.

**Print the plan before committing hours to it.** A shard report showing estimated hours per worker catches a bad split on minute one instead of day three.

---

<a name="8"></a>
## 8. Component 6 — resumability

### The checkpoint contract

Saving weights is not enough. Save all of:

```python
{
  "epoch": int,                  # last COMPLETED epoch
  "model": state_dict,
  "optimizer": state_dict,
  "scheduler": state_dict,
  "scaler": state_dict,          # AMP: omit and loss scale resets on resume
  "rng": {                       # ALL FOUR streams
      "python": random.getstate(),
      "numpy":  np.random.get_state(),
      "torch":  torch.get_rng_state(),
      "cuda":   torch.cuda.get_rng_state_all(),
  },
  "config_hash": str,            # asserted on resume
  "wall_seconds": float,         # cumulative, survives restarts
  "energy_joules": float,
  "custom_state": ...,           # anything you accumulate across epochs
}
```

Each field prevents a specific silent corruption:

| Omit | What breaks |
|---|---|
| `scaler` | AMP loss scale resets; post-resume steps behave differently |
| `rng` | Augmentation and shuffling order diverges → **a resumed run is not equivalent to an uninterrupted one** |
| `config_hash` | You resume under an edited config and never notice |
| `wall_seconds` / `energy_joules` | Cumulative totals restart at zero mid-run |

The RNG one is the subtle killer. If your science compares seeds, a resume that loses RNG state makes "same config, different seed" stop meaning what you think.

### Atomic writes, always

```python
torch.save(state, path.with_suffix(".tmp"))
os.replace(path.with_suffix(".tmp"), path)     # atomic on POSIX
```

A session killed mid-write leaves a truncated file and the run is gone.

### Checkpoint at epoch boundaries only

Mid-epoch resumption needs dataloader state, which isn't reliably serialisable with workers. Epoch granularity is fine.

### Rebuild progress from logs, not from status files

A session that died between writing a log and pushing its status leaves them disagreeing. **The log is the honest one.**

```python
def repair_ledger(self):
    for run_dir in runs.iterdir():
        df = pd.read_csv(run_dir / "metrics" / "epochs.csv")
        last_epoch, best = df.epoch.max(), df.val_accuracy.max()
        summary = read_json(run_dir / "summary.json")
        done = summary.get("status") == "completed" and last_epoch + 1 >= 0.9 * planned
        if not done and ledger_says_completed:
            # "broken stub": marked complete, truncated by a crash.
            # Left alone, every future session skips it forever.
            demote_to_paused()
```

### Truncate the log on resume

A milestone push can land after the checkpoint was written, so the log may contain epochs the checkpoint doesn't know about. Without truncation you append duplicate epoch numbers and every cumulative statistic is wrong.

```python
h = pd.read_csv(history_path)
h[h.epoch < start_epoch].to_csv(history_path, index=False)
```

---

<a name="9"></a>
## 9. Component 7 — lifecycle guards

Four ways a session ends. Handle all four:

```python
class LifecycleGuard:
    def install(self):
        signal.signal(signal.SIGTERM, self._handle)   # Kaggle sends this first
        atexit.register(self._atexit)

    def _fire(self, reason):
        if self._fired.is_set(): return               # exactly once
        self._fired.set()
        self.on_flush(reason)
```

| Exit | Handler |
|---|---|
| `KeyboardInterrupt` | try/except around the loop |
| `SIGTERM` | signal handler — **this is the common one on Kaggle** |
| Uncaught exception | except block, mark `failed` |
| Normal/abnormal shutdown | `atexit` |
| Approaching session limit | watchdog: at 8.5 h, push and mark `paused` |

Catching only `KeyboardInterrupt` misses the platform kill entirely — which is how you lose the last 30 minutes of a 3-hour run.

**The watchdog is the civilised one.** Detect the limit yourself and stop cleanly, rather than being killed mid-epoch.

---

<a name="10"></a>
## 10. Component 8 — repository layout

### One repo, one folder per run

```
runs/{run_id}/
├── config.yaml · config_hash.txt · STATUS.json · summary.json
├── metrics/      epochs.csv · final.csv · confusion_matrix.csv · …
├── telemetry/    energy_samples.csv · system_samples.csv · step_traces.jsonl
├── per_sample/   *.parquet
├── checkpoints/  ckpt_last.pt · ckpt_best.pt
└── env/          environment.json
registry/  events/ claims/ plans/
tables/    all_epochs.csv · all_final.csv · summary.csv
analysis/  paper/
```

**Why one repo:** the rate limit is per user, so two repos means two commits per cycle for no benefit. And a run's artifacts belong together.

**Why a *dataset* repo:** HF renders CSV and Parquet previews for datasets. Every metrics table becomes browsable in the browser without downloading. That is worth a lot when you want to glance at progress.

### Run IDs must be deterministic and readable

```
{phase}-{arch}-{dataset}-{method}-s{seed}
p0-resnet32x4-cifar100-base-s1
```

**Never auto-generate a UUID.** Six weeks from now you need to find a run by reading its name.

### Staging on scratch, not on the small disk

Put the whole tree on `/kaggle/temp` (~1 TB). The uploader reads from there directly. `/kaggle/working` stays nearly empty. You are then never disk-constrained, and losing scratch costs at most one push interval.

### Confirm-then-delete

```python
ok = sync.flush(timeout=1800)
missing = sync.verify_present([f"runs/{run_id}/checkpoints/ckpt_last.pt", ...])
if ok and not missing:
    shutil.rmtree(run_dir)
```

A flush that merely didn't time out is **not** evidence the files arrived. Re-list the repo.

### Push tiers

Not everything deserves the same cadence:

| Tier | Contents | When |
|---|---|---|
| light | config, status, metrics CSVs | every 30 min |
| heavy | checkpoints | every 30 min |
| bulk | raw telemetry, large parquet | every N epochs + at end |

Raw sample streams reach several MB; re-uploading them every half hour churns LFS storage for data nobody reads until the run finishes.

---

<a name="11"></a>
## 11. Component 9 — telemetry

**Principle: you train once. Record everything you could conceivably want, because re-running to recover a forgotten metric is unrecoverable time.**

### Per epoch (~170 columns)

| Group | Columns |
|---|---|
| Identity | run_id, epoch, global_step, timestamps, account, worker, session, host, config_hash |
| Learning | losses, accuracies, macro/micro/weighted P-R-F1, balanced acc, κ, MCC |
| Calibration | ECE, MCE, NLL, Brier, mean confidence |
| Loss parts | one column per term, `NA` when the term is not in the objective |
| Optimisation | LR per group, grad-norm mean/max/p50/p95/p99, clip-hit rate, weight norm, **update-to-weight ratio**, AMP scale, **AMP scale-decrease count**, **NaN/Inf batch count** |
| Timing | epoch/train/val, **dataload vs compute split**, step-time p50/p90/p99, throughput |
| GPU, **per device** | util, memory, temperature, clocks, power, energy, throttle reasons |
| Host | CPU %, RAM, process RSS, free disk |
| Energy | J / Wh / kWh and CO₂, per epoch and cumulative |
| Config echo | batch size, optimizer, scheduler, … so the CSV is self-describing |

### The five columns people forget, and what each catches

1. **`dataload_frac`** — time waiting for data ÷ total. High means the GPU is starving and the fix is the loader, not the model. Impossible to recover after the fact.
2. **`update_to_weight_ratio`** — ‖Δw‖/‖w‖. Healthy ≈ 1e-3. 1e-1 means the LR is far too high; 1e-6 means nothing is moving. Tells you before the loss curve does.
3. **`nan_or_inf_batches`** — under AMP, non-finite losses are silent. The run continues and learns nothing from those batches.
4. **`amp_scale_decreases`** — each one is a step whose gradients overflowed and were **discarded**. Invisible by default.
5. **`gpu{i}_throttle_reasons`** — non-zero means the card clocked down. Otherwise a slow epoch is a permanent mystery.

### Per-device, not aggregated

If the platform gives two GPUs and you train on one, an aggregate reports ~50% utilisation and hides that half the allocation is idle. Sample every device, write one row per device per sample.

### Three raw streams alongside the summary

- `energy_samples.csv` — power at 10 Hz, per GPU
- `system_samples.csv` — util/temp/clock/CPU/RAM at 1 Hz, per GPU
- `step_traces.jsonl` — per-step timing, downsampled to ≤2000 points per epoch

Downsample the step trace. Full resolution over 240 epochs is gigabytes; 2000 points per epoch is enough to see a within-epoch slowdown and stays a few MB.

### Schema discipline

- **Every column always present.** Missing quantity → `NA`, never omitted, never 0. "This term doesn't exist" and "this term was zero" are different facts.
- **Fill from the schema at the end**, so a forgotten key can't produce a ragged CSV:
  ```python
  for c in SCHEMA: row.setdefault(c, NA)
  ```
- **Test the schema against your requirements list**, programmatically. Map each requirement to the column(s) satisfying it and assert none are missing.

---

<a name="12"></a>
## 12. Component 10 — the preflight notebook

**Build this first and run it on every account before anything else.** It is the cheapest place to find expensive mistakes.

What it must check:

| Check | Failure it prevents |
|---|---|
| Secrets present **and writable** — push a probe, then **re-list the repo** | Nine hours of training with nowhere to save it |
| Dataset located | Silent fallback to a slow path |
| **Every model builds, forwards, backprops** | An architecture whose internals don't match your measurement code |
| Every model at **every input configuration** you'll sweep | Shape errors discovered mid-sweep |
| Cost tables sane — strictly ascending, no duplicates, correct endpoint | An entire axis being meaningless |
| **Kill-and-resume equivalence** | See below |
| Work split balance | One account working 3× longer than another |

### The resume test is the important one

Do not test resume by training a shorter run and asking for more epochs. That is a *clean completion* followed by an *extension* — a completely different code path that never touches your interrupt handler, emergency flush, or paused state. (It also gets blocked by your own claim protocol, which correctly refuses to restart a completed run.)

**Test it properly:**

```python
# 1. reference, uninterrupted
ref = train(cfg)

# 2. same config, killed for real mid-run
part = dict(cfg, _debug_interrupt_after_epoch=2)   # excluded from config_hash!
try: train(part)
except KeyboardInterrupt: pass

# 3. resume in a fresh call
res = train(cfg)

# 4. compare PER-EPOCH LOSS AFTER THE SEAM, not just final accuracy
```

Post-seam loss is where a lost RNG state shows up. Final accuracy can match by luck; the loss curve cannot.

Add the debug hook to your training loop and **exclude it from the config hash**, or the resumed run fails its own hash check.

---

<a name="13"></a>
## 13. The six bugs, and how to avoid them

Every one of these was found by running the thing, not by reading the code.

### Bug 1 — rate limiter on the wrong object
**Symptom:** none, until throttled.
**Cause:** limiter per repo; the real limit is per user.
**Fix:** one shared bucket keyed by token, process-wide.
**Lesson:** ask what the *provider* meters, not what your objects look like.

### Bug 2 — lost updates on the shared ledger
**Symptom:** two runs training, ledger lists one.
**Cause:** no append on HF; last push wins.
**Fix:** one shard file per writer, merge on read.
**Lesson:** any shared mutable file in an object store is a lost-update race.

### Bug 3 — claim protocol blocked self-resume
**Symptom:** `held by acct1 (5 min ago)` — refusing to resume *your own* run.
**Cause:** staleness window applied without checking ownership.
**Fix:** check owner before freshness. Same account → always allowed.
**Lesson:** the most common case (my session died, this is the new one) must be the easy path.

### Bug 4 — degenerate configurations from a fixed constant
**Symptom:** a cost table with three identical `1.0` entries.
**Cause:** requesting 5 exit points from a network with 3 blocks.
**Fix:** adaptive K; validate strict monotonicity and distinctness at build time.
**Lesson:** any "we always use K of these" constant needs a check that the object can supply K.

### Bug 5 — architecture couldn't do what the sweep assumed
**Symptom:** shape error at a non-default input size.
**Cause:** ViT positional embeddings are sized for one grid; MLP-Mixer's token-mixing layer *is* sized to the token count.
**Fix:** interpolate where principled (ViT); declare unsupported and fall back where not (Mixer), recording the limitation in the artifact.
**Lesson:** when one member of a family can't do something, uniform treatment across the whole family beats a better measurement on all but one.

### Bug 6 — a test that validated nothing
**Symptom:** resume test "failed" — but it had never exercised resume.
**Cause:** simulated a kill by training a shorter run.
**Fix:** real interrupt via a debug hook; compare post-seam losses.
**Lesson:** a test that can't fail for the right reason is worse than no test — it manufactures confidence.

---

<a name="14"></a>
## 14. Build order

Build and verify in this order. Each step depends on the last.

1. **Library skeleton** — utils, atomic IO, config hashing, seeding, RNG capture
2. **Offline self-test harness** — a `--selftest` flag with no GPU and no network. Everything below adds tests here.
3. **Uploader** — batching, dedup, backoff, 429 parsing, **shared** rate limiter
4. **Registry** — sharded events, claims, ownership-aware `can_claim`
5. **Sharding** — `assign_workers`, `plan_work`, cost model, shard report
6. **Lifecycle** — SIGTERM + atexit + interrupt + watchdog
7. **Telemetry** — schema, per-device monitors, epoch accumulator
8. **Training loop** — resumable, instrumented, tiered pushes
9. **Notebook generator** — base64 bootstrap, round-trip verification
10. **Preflight notebook** — including the real kill-and-resume test
11. **Everything else** — the actual science

**Steps 1–6 are infrastructure and are the same in every project.** Lift them wholesale.

---

<a name="15"></a>
## 15. Checklist

**Before the first real run**

- [ ] Library regenerates notebooks; base64 round-trips byte-identically
- [ ] `--selftest` passes offline, no GPU, no network
- [ ] Token has **write** scope — verified by pushing and re-listing
- [ ] Rate limit is per token, and `cap × n_accounts < provider_limit`
- [ ] Registry shards per writer; no shared mutable file anywhere
- [ ] `can_claim` lets a worker resume its own run immediately
- [ ] Work split printed with estimated hours per worker before starting
- [ ] Checkpoint contains optimizer, scheduler, scaler, **all RNG streams**, cumulative counters
- [ ] Resume asserts `config_hash`
- [ ] Kill-and-resume test compares **post-seam per-epoch losses**
- [ ] SIGTERM, atexit, KeyboardInterrupt, and session watchdog all flush
- [ ] Datasets and staging on the large scratch disk
- [ ] Confirm-then-delete verifies by re-listing the repo
- [ ] Schema tested against the requirements list programmatically
- [ ] Every schema column always present; `NA` where undefined

**Per notebook**

- [ ] `ACCOUNT` and `WORKER_ID` at the top, clearly marked
- [ ] Plan printed before work starts
- [ ] Safe to stop at any moment
- [ ] Safe to re-run from scratch
- [ ] Ends with a blocking flush
- [ ] Markdown explains *why*, not just *what*

**Recurring**

- [ ] Audit the repo: is every expected run present and complete?
- [ ] Sanity-check the science (a shuffled-control or equivalent)
- [ ] Watch `dataload_frac`, `nan_or_inf_batches`, `amp_scale_decreases`

---

## The two ideas worth carrying to every project

**1. Coordination by arithmetic, not negotiation.** Every worker computes the same assignment from the same inputs and keeps its slice. No locks, no leader, no messages, nothing to deadlock. Add stealing only as a recovery mechanism, never as the primary path.

**2. Record everything, because you train once.** The cost of an extra column is bytes. The cost of a missing one is a re-run you cannot afford. When a quantity doesn't exist, write `NA` — never omit it, never fake it.
