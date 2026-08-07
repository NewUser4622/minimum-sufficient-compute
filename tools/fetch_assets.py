#!/usr/bin/env python3
"""
fetch_assets.py -- run ONCE with internet. Everything after that runs offline.

Read this first, because it is the opposite of what you probably expect:

    TRAINING FROM SCRATCH DOWNLOADS NO MODEL WEIGHTS.

`torchvision.models.resnet50(weights=None)` is Python source that ships inside
the torchvision package. So does `swin_t`, so does `vgg16_bn`. There is no
architecture to fetch and no checkpoint to cache. A tool that pretended to
"download the models" would create a directory, put nothing useful in it, and
give you a false sense that offline operation had been arranged.

What actually needs one-time internet is **the pip packages**. And what
actually needs pinning is **their versions** -- because a torchvision upgrade
can change how a model decomposes into blocks, and this pipeline decomposes
every backbone into an ordered block list to make `forward_prefix(x, k)` stop
early. A different block count is a different budget table, a different rho,
and every MSC value silently shifts. That is a real hazard and it is the one
this tool exists to close.

So it does four things:

  1. **Records** the exact versions of everything imported, into a manifest.
  2. **Fingerprints all eight architectures** -- parameter count, FLOPs, exit
     count K, stage cuts, feature dims -- so a later environment change is
     caught by comparison rather than discovered in the results.
  3. **Optionally caches pretrained weights** into a local TORCH_HOME, for the
     one scenario where you would ever want them. Off by default: this study
     trains from scratch and pretrained initialisation would invalidate every
     seed-reliability measurement.
  4. **Proves the pipeline is offline-clean** by blocking the socket layer and
     then building every architecture and running both dry runs. Rule 10's
     shape: installing a package is not offline-readiness, in the same way that
     draining an upload queue is not confirmation.

Usage
-----
    python tools/fetch_assets.py                 # record + fingerprint (needs net once)
    python tools/fetch_assets.py --verify-offline    # prove it, sockets blocked
    python tools/fetch_assets.py --check         # compare against the manifest
    python tools/fetch_assets.py --pretrained    # also cache ImageNet weights
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ASSETS = Path(os.environ.get("MSC_ASSETS", ROOT / "assets"))
MANIFEST = ASSETS / "environment_manifest.json"

# Everything the pipeline imports, and what breaks without it. `required` is
# the difference between "cannot run" and "runs with a column missing".
PACKAGES = [
    ("torch",        True,  "everything"),
    ("torchvision",  True,  "resnet18/50, vgg16, shufflenetv2, swin_tiny"),
    ("numpy",        True,  "everything"),
    ("pandas",       True,  "every table, every analysis"),
    ("pyarrow",      True,  "per_sample/*.parquet -- the scientific artifact"),
    ("yaml",         True,  "config.yaml per run"),
    ("scipy",        True,  "Spearman correlations = Q1 and Q3"),
    ("sklearn",      True,  "Q4 nested-model delta-R^2, PCA for Q2"),
    ("PIL",          True,  "packing the dataset"),
    ("psutil",       True,  "host CPU/RAM telemetry columns"),
    ("pynvml",       True,  "GPU power at 10 Hz -- energy columns are NA without it"),
    ("fvcore",       True,  "FLOPs profiler. rho is defined in FLOPs, so this "
                            "is not optional decoration"),
    ("matplotlib",   False, "paper figures (NB15)"),
    ("tqdm",         False, "progress bars"),
    ("thop",         False, "fallback profiler if fvcore is absent"),
]


def _ver(mod) -> str:
    for a in ("__version__", "version", "VERSION"):
        v = getattr(mod, a, None)
        if isinstance(v, str):
            return v
    return "unknown"


def survey() -> dict:
    out, missing = {}, []
    for name, required, why in PACKAGES:
        try:
            m = importlib.import_module(name)
            out[name] = {"version": _ver(m), "required": required, "why": why}
        except Exception as e:                                   # noqa: BLE001
            out[name] = {"version": None, "required": required, "why": why,
                         "error": f"{type(e).__name__}: {str(e)[:90]}"}
            if required:
                missing.append(name)
    return {"packages": out, "missing_required": missing}


def fingerprint_zoo(dataset: str = "imagenet100") -> dict:
    """Parameter count, FLOPs, K, stage cuts and feature dims for every arch.

    This is the artifact that makes the version pin meaningful. If torchvision
    changes `resnet50`'s block structure, `n_blocks` or `feature_dims` moves and
    `--check` says so. Without it a changed environment produces a different
    budget table, which produces different rho, which shifts every MSC value --
    and every one of those numbers looks entirely reasonable.
    """
    import msc_lib as M
    out = {}
    for arch in M.zoo_for_dataset(dataset):
        t0 = time.time()
        try:
            m = M.build_model(arch, dataset=dataset).eval()
            tbl = M.build_budget_table(arch, dataset, model=m)
            d = tbl["axes"]["depth"]
            out[arch] = {
                "params": M.count_parameters(m),
                "full_flops": tbl["full_flops"],
                "K": d["K"],
                "n_blocks": d["n_blocks"],
                "stage_cuts": d["stage_cuts"],
                "feature_dims": d["feature_dims"],
                "depth_rho": [round(x, 6) for x in d["rho"]],
                "native_res_ok": tbl["axes"]["resolution"]["native_supported_per_res"],
                "profiler": tbl["profiler"]["name"],
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  {arch:16s} {out[arch]['params']/1e6:7.2f} M  "
                  f"{out[arch]['full_flops']/1e9:6.2f} GFLOPs  K={d['K']}  "
                  f"blocks={d['n_blocks']}  native={sum(out[arch]['native_res_ok'])}"
                  f"/{len(out[arch]['native_res_ok'])}")
        except Exception as e:                                   # noqa: BLE001
            out[arch] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  {arch:16s} FAILED  {out[arch]['error']}")
    return out


def cache_pretrained() -> dict:
    """Cache ImageNet-1k weights into the local TORCH_HOME.

    OFF by default and it should stay off for this study. Pretrained
    initialisation would make every seed share a starting point, which is
    exactly the variance the noise ceiling measures -- rho_seed would be
    inflated toward 1.0 for reasons that have nothing to do with architecture,
    and the CNN/transformer comparison would be meaningless.

    Provided because "I want to check something with a pretrained model" is a
    reasonable future need, and doing it later would require internet again.
    """
    import torchvision.models as tvm
    got = {}
    for name, fn, w in (("resnet18", tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
                        ("resnet50", tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
                        ("vgg16_bn", tvm.vgg16_bn, tvm.VGG16_BN_Weights.DEFAULT),
                        ("shufflenet_v2_x1_0", tvm.shufflenet_v2_x1_0,
                         tvm.ShuffleNet_V2_X1_0_Weights.DEFAULT),
                        ("swin_t", tvm.swin_t, tvm.Swin_T_Weights.DEFAULT)):
        try:
            fn(weights=w)
            got[name] = "cached"
            print(f"  cached {name}")
        except Exception as e:                                   # noqa: BLE001
            got[name] = f"{type(e).__name__}: {str(e)[:90]}"
            print(f"  FAILED {name}: {got[name]}")
    return got


def verify_offline(dataset: str = "imagenet100") -> int:
    """Block the socket layer, then run the pipeline. The proof, not the claim.

    Builds every architecture, measures every budget table, and runs both dry
    runs -- the whole training path and the whole measurement path -- with the
    network genuinely unavailable. Anything that tries to fetch raises with the
    host it wanted.
    """
    os.environ.setdefault("MSC_OFFLINE", "1")
    import msc_lib as M

    print("blocking the socket layer (loopback stays open for CUDA/dataloader)")
    fails = []
    with M.no_network():
        for arch in M.zoo_for_dataset(dataset):
            try:
                m = M.build_model(arch, dataset=dataset)
                M.build_budget_table(arch, dataset, model=m.eval())
                print(f"  [OK]   {arch} builds and prices offline")
            except Exception as e:                               # noqa: BLE001
                fails.append(f"{arch}: {type(e).__name__}: {str(e)[:160]}")
                print(f"  [FAIL] {arch}: {fails[-1]}")

        # The dry runs cover the two expensive entry points end to end, so if
        # anything anywhere in training or measurement reaches for the network,
        # this is where it surfaces.
        for arch in ("resnet50", "vit_small_p16"):
            cfg = M.base_config(arch, dataset, seed=1, phase="p0")
            cfg["data_root"] = "/nonexistent"
            for name, fn in (("backbone", M.backbone_dry_run),
                             ("oracle", M.oracle_dry_run)):
                ok, why = fn(cfg)
                print(f"  [{'OK' if ok else 'FAIL'}]   {name} dry run "
                      f"({arch}) offline: {why}")
                if not ok:
                    fails.append(f"{name} dry run {arch}: {why}")

    if fails:
        print(f"\n{len(fails)} offline failure(s). The pipeline is NOT "
              f"self-contained:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nOK -- every architecture builds, every budget table prices, and "
          "both dry runs complete with the network blocked.")
    return 0


def check_against_manifest(dataset: str = "imagenet100") -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}. Run without --check first.")
        return 1
    old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    new_pkgs = survey()["packages"]
    print("package versions")
    drift = []
    for name, meta in old.get("packages", {}).items():
        was, now = meta.get("version"), new_pkgs.get(name, {}).get("version")
        if was != now:
            drift.append(f"{name}: {was} -> {now}")
            print(f"  [DRIFT] {name}: {was} -> {now}")
    if not drift:
        print("  all versions unchanged")

    print("architecture fingerprints")
    new_zoo = fingerprint_zoo(dataset)
    arch_drift = []
    for arch, meta in old.get("zoo", {}).items():
        cur = new_zoo.get(arch, {})
        for k in ("params", "full_flops", "K", "n_blocks", "stage_cuts",
                  "feature_dims"):
            if meta.get(k) != cur.get(k):
                arch_drift.append(f"{arch}.{k}: {meta.get(k)} -> {cur.get(k)}")
    for d in arch_drift:
        print(f"  [DRIFT] {d}")
    if arch_drift:
        print(f"\n*** {len(arch_drift)} ARCHITECTURE CHANGE(S). Every budget "
              f"table built under the old environment is invalid, and so is "
              f"every MSC value derived from it. rho is a ratio, so the numbers "
              f"will look entirely reasonable. Rebuild budgets/ and re-measure, "
              f"or pin the old versions back.")
        return 1
    print("  all eight architectures identical to the manifest")
    return 1 if drift else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-offline", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--dataset", default="imagenet100")
    a = ap.parse_args()

    if a.verify_offline:
        return verify_offline(a.dataset)
    if a.check:
        return check_against_manifest(a.dataset)

    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"assets -> {ASSETS}\n")

    print("packages")
    s = survey()
    for name, meta in s["packages"].items():
        mark = "  " if meta["version"] else ("!!" if meta["required"] else " ~")
        print(f" {mark} {name:14s} {str(meta['version'] or 'MISSING'):12s} "
              f"{meta['why']}")
    if s["missing_required"]:
        print(f"\nMISSING REQUIRED: {s['missing_required']}")
        print("Install them, then re-run. See requirements.txt.")
        return 1

    print("\narchitecture fingerprints "
          "(this is the pin that makes the version record useful)")
    zoo = fingerprint_zoo(a.dataset)

    pre = {}
    if a.pretrained:
        print("\ncaching pretrained ImageNet-1k weights "
              "(NOT used by this study -- see the docstring)")
        pre = cache_pretrained()

    MANIFEST.write_text(json.dumps({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dataset": a.dataset,
        "packages": s["packages"],
        "zoo": zoo,
        "pretrained_cache": pre,
        "torch_home": os.environ.get("TORCH_HOME", ""),
        "note": "Training from scratch downloads no model weights. This "
                "manifest exists to detect environment drift, because a "
                "torchvision change can alter how a backbone decomposes into "
                "blocks and therefore silently change every budget table.",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {MANIFEST}")
    print("\nNow prove it:  python tools/fetch_assets.py --verify-offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
