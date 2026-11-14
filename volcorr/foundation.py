"""Zero-shot foundation forecasts. Input: log GK variance, context 512 days ending at the origin.
Output per origin: per-day median log forecasts for t+1..t+5, exponentiated back to variance.

Work is split into fixed chunks of origins; each chunk is one parquet file, so a crash loses at
most one chunk and a rerun skips everything already on disk."""
import argparse
import os
import time

try:  # Windows: the torch and JAX DLLs must load before pyarrow (pulled in by pandas) or jaxlib fails to initialise
    import torch  # noqa: F401
    import jax  # noqa: F401
except ImportError:  # only matters for inference; the pure helpers below still import
    pass
import numpy as np
import pandas as pd

from .common import FOUNDATION, MAX_H, RAW_DIR, load_cfg, load_panel, log_run, origin_range, raw_rows

CHUNK = 20  # origins per chunk (x 10 assets = 200 series per batch)


def contexts(y, idx, length):
    """Rows y[i-length+1 .. i] for each origin position i: nothing after the origin."""
    return np.stack([y[i - length + 1:i + 1] for i in idx])


def _snapshot(m, patterns=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(m["repo"], revision=m["revision"], allow_patterns=patterns)


def load_chronos(m, cfg, device):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(_snapshot(m), device_map=device)

    def predict(x):
        q, _ = pipe.predict_quantiles(torch.tensor(x), prediction_length=MAX_H, quantile_levels=[m["quantile"]])
        return q[:, :, 0].float().cpu().numpy()
    return predict, cfg["context"]


def load_timesfm(m, cfg, device):
    import timesfm
    path = os.path.join(_snapshot(m, ["torch_model.ckpt"]), "torch_model.ckpt")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="gpu" if device == "cuda" else "cpu", per_core_batch_size=CHUNK * 10,
                                       horizon_len=MAX_H, num_layers=50, use_positional_embedding=False,
                                       context_len=cfg["context"]),
        checkpoint=timesfm.TimesFmCheckpoint(path=path))

    def predict(x):
        point, _ = tfm.forecast(list(x), freq=[m["freq"]] * len(x))  # library default point forecast
        return point
    return predict, cfg["context"]


def load_moirai(m, cfg, device):
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    model = MoiraiForecast(module=MoiraiModule.from_pretrained(_snapshot(m)), prediction_length=MAX_H,
                           context_length=cfg["context"], num_samples=m["num_samples"], target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(device).eval()  # patch_size: library default

    def predict(x):
        t = torch.tensor(x, device=device)[:, :, None]
        with torch.no_grad():
            s = model(past_target=t, past_observed_target=torch.ones_like(t, dtype=torch.bool),
                      past_is_pad=torch.zeros(t.shape[:2], dtype=torch.bool, device=device))
        return s.median(dim=1).values.cpu().numpy()
    # the default "auto" patch size scores candidates on the last 5 observed days, so it needs context + 5
    return predict, model.past_length


LOADERS = {"chronos_bolt_base": load_chronos, "timesfm_2_0_500m": load_timesfm, "moirai_1_1_r_base": load_moirai}


def chunk_inputs(gk, first, last, k, length):
    """Origin positions and stacked contexts for chunk k, one block of origins per asset."""
    idx = np.arange(first + k * CHUNK, min(first + (k + 1) * CHUNK, last + 1))
    y = np.log(gk.values).astype(np.float32)
    x = np.concatenate([contexts(y[:, a], idx, length) for a in range(y.shape[1])])  # asset-major
    return idx, x


def run_chunk(predict, gk, first, last, k, length, seed):
    """Variance forecasts (n_series, 5) for chunk k. Seeded per chunk so results do not depend on resume point."""
    import torch
    idx, x = chunk_inputs(gk, first, last, k, length)
    out = None
    for _ in range(2):  # retry once with the same seed
        torch.manual_seed(seed + k)
        try:
            out = np.exp(np.asarray(predict(x), dtype=np.float64))
            if np.isfinite(out).all():
                break
        except Exception as e:  # noqa: BLE001
            print("chunk", k, "failed:", repr(e)[:200], flush=True)
            out = None
    if out is None:
        out = np.full((len(x), MAX_H), np.nan)
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return idx, out


def chunk_frame(model, gk, idx, out):
    n = len(idx)
    return pd.concat([raw_rows(model, a, gk.index[idx], out[j * n:(j + 1) * n]) for j, a in enumerate(gk.columns)],
                     ignore_index=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=FOUNDATION)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_cfg()
    fc = cfg["foundation"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log_run("forecast-foundation", device=args.device, revisions={k: v["revision"] for k, v in fc["models"].items()})
    gk, _ = load_panel()
    first, last = origin_range(cfg, gk.index)
    n_chunks = -(-(last - first + 1) // CHUNK)
    for model in args.models:
        out_dir = RAW_DIR / model
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [k for k in range(n_chunks) if not (out_dir / f"chunk_{k:04d}.parquet").exists()]
        if not todo:
            continue
        predict, length = LOADERS[model](fc["models"][model], fc, args.device)
        t0 = time.time()
        for n, k in enumerate(todo, 1):
            idx, out = run_chunk(predict, gk, first, last, k, length, fc["seed"])
            tmp = out_dir / f"chunk_{k:04d}.tmp"
            chunk_frame(model, gk, idx, out).to_parquet(tmp, index=False)
            tmp.replace(out_dir / f"chunk_{k:04d}.parquet")
            if n % 10 == 0 or n == len(todo):
                print(f"{model}: {n}/{len(todo)} chunks, {time.time() - t0:.0f}s", flush=True)
        del predict
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
