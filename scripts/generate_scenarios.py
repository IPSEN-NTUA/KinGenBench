#!/usr/bin/env python3
"""
Train the six conditional generative variants on one benchmark scenario and
write one synthetic K_m parameter CSV per variant (seeded; run once per seed).

Variants:
    cvae / gan / nf                                 binary (joint high-performance) label
    cvae_continuous / gan_continuous / nf_continuous  continuous standardized log-KPI vector

Scenarios (--scenario): pgi (I), pgi_pgl (II), all_mutants (III); see benchmark/config.py.

Outputs
    <out-dir>/benchmark_<variant>_<scenario-tag>_seed<seed>.csv  (default outputs/synthetic_parameters)
    <gen-dir>/<variant>_<scenario-tag>_seed<seed>/              (default outputs/generators)
        trained networks + scalers + config (disable with --no-save)

Continuous generation target (--kpi-target-mode), all inside the positive class:
    class       (default) one KPI vector per sample, drawn from the positive-class
                training models                                   file suffix _cls
    threshold   a single target holding each perturbation's KPI at its labelling
                threshold (the class boundary)                    no file suffix
    percentile  a single target at --kpi-target-pct of every KPI  file suffix _p<pct>
Publication: Scenarios I-II used 'threshold', Scenario III used 'class' (main)
and 'percentile' at 5% (targeted-generation supplement).

--from-checkpoint skips training and generates from the saved networks, e.g.
to draw both targets from the same trained continuous models.

Examples (run from the repository root):
    python scripts/generate_scenarios.py --scenario pgi --pgi-threshold 10 --seed 66 --kpi-target-mode threshold
    python scripts/generate_scenarios.py --scenario all_mutants --seed 65 --kpi-target-mode class
    python scripts/generate_scenarios.py --scenario all_mutants --seed 65 --from-checkpoint \
        --kpi-target-mode percentile --kpi-target-pct 5 \
        --variants cvae_continuous,gan_continuous,nf_continuous
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from benchmark import paths
from benchmark.config import SCENARIOS, scenario_tag, cutoffs as cfg_cutoffs
from benchmark.data import build_scenario, postprocess
from benchmark.ml.utils.env import reset_environment

import tensorflow as tf
from benchmark.ml.cvae.encoder import build_conditional_encoder
from benchmark.ml.cvae.decoder import build_conditional_decoder
from benchmark.ml.cvae.vae_model import ConditionalVAE, EpochSetter

import torch
from benchmark.ml.gans import build_conditional_generator, build_conditional_critic, ConditionalWGANGP
from benchmark.ml.nf import build_conditional_realnvp, ConditionalRealNVP


ALL_VARIANTS = ["cvae", "gan", "nf", "cvae_continuous", "nf_continuous", "gan_continuous"]


# ---------------------------------------------------------------------------
# Generation target for the continuous variants
# ---------------------------------------------------------------------------
def kpi_target(Y_raw, kpi_scaler, pct, log=True, eps=1e-8):
    """Scaled conditioning target = per-perturbation low percentile (good region)."""
    t = np.percentile(Y_raw, pct, axis=0)
    t = np.log(t + eps) if log else t
    return kpi_scaler.transform(t.reshape(1, -1)).astype("float32")


# ---------------------------------------------------------------------------
# Network builders (shared by training and --from-checkpoint loading, so the
# rebuilt architecture is always identical to the trained one)
# ---------------------------------------------------------------------------
def build_cvae(input_dim, label_dim, latent_dim):
    enc = build_conditional_encoder(input_dim, label_dim, latent_dim,
                                    hidden_layers=[256, 128, 128], activation="elu",
                                    use_label_embedding=True)
    dec = build_conditional_decoder(input_dim, label_dim, latent_dim,
                                    hidden_layers=[128, 128, 256], activation="gelu",
                                    output_activation=None, use_label_embedding=True)
    return enc, dec


def build_gan(input_dim, latent_dim, device, kpi_dim=None):
    """kpi_dim=None -> binary label + embedding; int -> continuous KPI context."""
    if kpi_dim is None:
        gen = build_conditional_generator(input_dim, latent_dim, hidden_layers=[128, 256],
                                          activation="relu", use_label_embedding=True,
                                          n_classes=2, label_embed_dim=32)
        cri = build_conditional_critic(input_dim, hidden_layers=[256, 128],
                                       activation="leaky_relu", use_label_embedding=True,
                                       n_classes=2, label_embed_dim=32)
        return ConditionalWGANGP(gen, cri, latent_dim, lr=5e-4, n_critic=5,
                                 lambda_gp=10.0, device=device)
    gen = build_conditional_generator(input_dim, latent_dim, hidden_layers=[128, 256],
                                      activation="relu", use_label_embedding=False,
                                      label_dim=kpi_dim)
    cri = build_conditional_critic(input_dim, hidden_layers=[256, 128],
                                   activation="leaky_relu", use_label_embedding=False,
                                   label_dim=kpi_dim)
    return ConditionalWGANGP(gen, cri, latent_dim, lr=5e-4, n_critic=5,
                             lambda_gp=10.0, device=device, continuous=True)


def build_nf(input_dim, context_dim, layers, hidden, device):
    flow = build_conditional_realnvp(input_dim, context_dim=context_dim,
                                     num_layers=layers, hidden_features=hidden,
                                     num_blocks=2, activation="relu")
    return ConditionalRealNVP(flow, lr=1e-3, grad_clip=5.0, device=device)


# ---------------------------------------------------------------------------
# Training (returns the trained objects; generation is done separately)
# ---------------------------------------------------------------------------
def train_cvae(X, y_cond, label_dim, latent_dim, epochs, batch, seed, verbose):
    reset_environment(seed)
    enc, dec = build_cvae(X.shape[1], label_dim, latent_dim)
    m = ConditionalVAE(enc, dec, beta=0.1, max_beta=1.0, warmup_epochs=80, recon_loss="mse")
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4))
    m.fit(x=(X, y_cond), y=X, epochs=epochs, batch_size=batch, shuffle=True,
          callbacks=[EpochSetter()], verbose=verbose)
    return enc, dec


def generate_cvae(dec, y_target, n, latent_dim, temperature):
    y_target = np.atleast_2d(np.asarray(y_target, dtype="float32"))
    if y_target.shape[0] == 1:                       # single target -> broadcast
        y_cond = tf.repeat(tf.constant(y_target), repeats=n, axis=0)
    else:                                            # one target row per sample
        y_cond = tf.constant(y_target[:n])
    z = temperature * tf.random.normal((n, latent_dim))
    return dec([z, y_cond], training=False)


def train_gan(X, y, latent_dim, epochs, batch, device, seed, verbose, kpi_dim=None):
    torch.manual_seed(seed); np.random.seed(seed)
    gan = build_gan(X.shape[1], latent_dim, device, kpi_dim=kpi_dim)
    gan.fit(X, y, epochs=epochs, batch_size=batch, verbose=verbose)
    return gan


def train_nf(X, y_cond, context_dim, layers, hidden, epochs, batch, device, seed, verbose):
    torch.manual_seed(seed); np.random.seed(seed)
    nf = build_nf(X.shape[1], context_dim, layers, hidden, device)
    nf.fit(X, y_cond, epochs=epochs, batch_size=batch, verbose=verbose)
    return nf


# ---------------------------------------------------------------------------
# Checkpoints:  <gen-dir>/<variant>_<scenario-tag>_seed<seed>/
#   config.json        architecture + training settings
#   preproc.pkl        x/kpi scalers, column names, positive-class KPI pool, Y_raw
#   cvae : encoder.weights.h5, decoder.weights.h5
#   gan  : gan.pt   (generator + critic state_dict)
#   nf   : flow.pt  (flow state_dict)
# Training is independent of the KPI generation target, so one continuous
# checkpoint serves both --kpi-target-mode class and percentile.
# ---------------------------------------------------------------------------
def family(variant):
    return variant.replace("_continuous", "")          # cvae | gan | nf


def save_checkpoint(d, variant, model, config, preproc):
    d.mkdir(parents=True, exist_ok=True)
    fam = family(variant)
    if fam == "cvae":
        enc, dec = model
        enc.save_weights(str(d / "encoder.weights.h5"))
        dec.save_weights(str(d / "decoder.weights.h5"))
    elif fam == "gan":
        torch.save({"generator": model.generator.state_dict(),
                    "critic": model.critic.state_dict()}, d / "gan.pt")
    elif fam == "nf":
        torch.save(model.flow.state_dict(), d / "flow.pt")
    with open(d / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    with open(d / "preproc.pkl", "wb") as fh:
        pickle.dump(preproc, fh)
    print(f"[checkpoint] saved {d}")


def load_checkpoint(d, variant, device):
    """Rebuild the exact architecture from config.json and load the weights."""
    if not d.exists():
        raise SystemExit(f"[error] checkpoint not found: {d}")
    with open(d / "config.json") as fh:
        cfg = json.load(fh)
    with open(d / "preproc.pkl", "rb") as fh:
        pp = pickle.load(fh)
    fam = family(variant)
    if fam == "cvae":
        enc, dec = build_cvae(cfg["input_dim"], cfg["cond_dim"], cfg["latent_dim"])
        enc.load_weights(str(d / "encoder.weights.h5"))
        dec.load_weights(str(d / "decoder.weights.h5"))
        model = (enc, dec)
    elif fam == "gan":
        kd = cfg["cond_dim"] if cfg["continuous"] else None
        model = build_gan(cfg["input_dim"], cfg["latent_dim"], device, kpi_dim=kd)
        sd = torch.load(d / "gan.pt", map_location=device)
        model.generator.load_state_dict(sd["generator"])
        model.critic.load_state_dict(sd["critic"])
    elif fam == "nf":
        model = build_nf(cfg["input_dim"], cfg["cond_dim"], cfg["nf_layers"],
                         cfg["nf_hidden"], device)
        model.flow.load_state_dict(torch.load(d / "flow.pt", map_location=device))
    print(f"[checkpoint] loaded {d}")
    return model, cfg, pp


def threshold_target(Y_raw, kpi_scaler, cutoffs, log=True, eps=1e-8):
    """Scaled target = each perturbation's KPI at its labelling percentile."""
    t = np.array([np.percentile(Y_raw[:, j], c) for j, c in enumerate(cutoffs)])
    t = np.log(t + eps) if log else t
    return kpi_scaler.transform(t.reshape(1, -1)).astype("float32")


def make_kpi_target(pp, mode, pct, n, seed):
    """Continuous generation target from the saved preprocessing state."""
    if mode == "threshold":
        return threshold_target(pp["Y_raw"], pp["kpi_scaler"], pp["cutoffs"])   # (1, kpi_dim)
    if mode == "class":
        pool = pp["kpi_pool"]
        if len(pool) == 0:
            raise SystemExit("[error] no positive-class models to sample KPI targets from.")
        rng = np.random.default_rng(seed)
        return pool[rng.integers(0, len(pool), size=n)].astype("float32")   # (n, kpi_dim)
    return kpi_target(pp["Y_raw"], pp["kpi_scaler"], pct)                   # (1, kpi_dim)


def generate(variant, model, cfg, y_tgt, n, temperature):
    fam, cont = family(variant), variant.endswith("_continuous")
    if fam == "cvae":
        cond = y_tgt if cont else np.array([[1.0]], dtype="float32")
        return generate_cvae(model[1], cond, n, cfg["latent_dim"], temperature)
    if fam == "gan":
        if cont:
            return model.generate(n_samples=n, context=y_tgt, temperature=temperature)
        return model.generate(n_samples=n, label=1, temperature=temperature)
    if cont:                                                     # nf
        return model.generate(n_samples=n, context=y_tgt, temperature=temperature)
    return model.generate(n_samples=n, label=1, temperature=temperature)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pgi-threshold", type=float, default=10.0,
                    help="Conditioning percentile for the single-PGI scenario only.")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS),
                    help="Comma-separated subset of: " + ",".join(ALL_VARIANTS))
    ap.add_argument("--out-dir", default=str(paths.SYNTH_DIR))
    ap.add_argument("--gen-dir", default=str(paths.GEN_DIR),
                    help="Root folder for trained-network checkpoints "
                         "(<gen-dir>/<variant>_<scenario>_seed<seed>/).")
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="Skip training: load each variant from --gen-dir and only generate.")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not write checkpoints after training.")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Fixed at 1.0 for a matched cross-architecture comparison.")
    ap.add_argument("--kpi-target-mode", choices=["class", "threshold", "percentile"],
                    default="class",
                    help="Continuous-KPI generation target. 'class': each sample is conditioned "
                         "on a KPI vector drawn from the training models in the positive class; "
                         "'threshold': each KPI at its labelling threshold; "
                         "'percentile': a single target at --kpi-target-pct of each KPI.")
    ap.add_argument("--kpi-target-pct", type=float, default=5.0,
                    help="Target percentile for --kpi-target-mode percentile.")
    ap.add_argument("--cvae-epochs", type=int, default=120)
    ap.add_argument("--gan-epochs", type=int, default=120)
    ap.add_argument("--nf-epochs", type=int, default=200)
    ap.add_argument("--nf-layers", type=int, default=6,
                    help="RealNVP coupling layers (matched-capacity default).")
    ap.add_argument("--nf-hidden", type=int, default=64,
                    help="RealNVP coupling width (matched-capacity default).")
    ap.add_argument("--latent-dim", type=int, default=80)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--nf-batch", type=int, default=512)
    ap.add_argument("--tag", default=None, help="Optional extra filename tag.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        raise SystemExit(f"[error] unknown variant(s): {bad}")
    verbose = 0 if args.quiet else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perts = SCENARIOS[args.scenario]["perts"]
    pgi_thr = args.pgi_threshold if args.scenario == "pgi" else None
    cutoffs = cfg_cutoffs(args.scenario, pgi_thr)
    scen_tag = scenario_tag(args.scenario, pgi_thr)
    if args.tag:
        scen_tag = f"{scen_tag}_{args.tag}"

    gen_root = Path(args.gen_dir)
    ckpt = lambda v: gen_root / f"{v}_{scen_tag}_seed{args.seed}"

    print(f"[scenario] {args.scenario} | perturbations={perts} | cutoffs={cutoffs} "
          f"| seed={args.seed} | device={device} "
          f"| mode={'from-checkpoint' if args.from_checkpoint else 'train'}")

    n, T = args.n_samples, args.temperature
    if args.kpi_target_mode == "class":
        target_tag = "cls"
    elif args.kpi_target_mode == "threshold":
        target_tag = ""
    else:
        target_tag = f"p{args.kpi_target_pct:g}"

    # ---- data (only needed for training; checkpoints carry their own preproc) ----
    if not args.from_checkpoint:
        reset_environment(args.seed); torch.manual_seed(args.seed)
        scn = build_scenario(args.scenario,
                             pgi_threshold=args.pgi_threshold if args.scenario == "pgi" else None)
        X = scn["X"]
        y_binary = scn["y_binary"]
        y_bin_col = y_binary.reshape(-1, 1).astype("float32")
        Y_kpi, Y_raw, kpi_scaler = scn["Y_kpi"], scn["Y_raw"], scn["kpi_scaler"]
        kpi_dim = Y_kpi.shape[1]
        preproc = {"x_scaler": scn["x_scaler"],
                   "kpi_scaler": kpi_scaler,
                   "column_names": scn["columns"],
                   "kpi_pool": Y_kpi[y_binary == 1].astype("float32"),
                   "Y_raw": Y_raw,
                   "perts": perts, "cutoffs": list(cutoffs)}

    if args.kpi_target_mode == "class":
        print("[target] class-matched: one positive-class KPI vector per sample")
    elif args.kpi_target_mode == "threshold":
        print(f"[target] each KPI at its labelling threshold (percentiles {cutoffs})")
    else:
        print(f"[target] fixed percentile p{args.kpi_target_pct:g} of each KPI")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    records = []

    def finalize(xg, train_s, gen_s, variant, pp):
        df, dropped = postprocess(xg, pp["x_scaler"], pp["column_names"])
        name = variant
        if variant.endswith("_continuous") and target_tag:
            name = f"{variant}_{target_tag}"        # e.g. cvae_continuous_cls
        path = out / f"benchmark_{name}_{scen_tag}_seed{args.seed}.csv"
        df.to_csv(path)
        print(f"[saved] {path}  ({len(df)} kept, {dropped} dropped) "
              f"| train {train_s:.1f}s, gen {gen_s:.2f}s")
        records.append({"variant": name,
                        "train_seconds": round(train_s, 2),
                        "train_minutes": round(train_s / 60.0, 3),
                        "gen_seconds": round(gen_s, 3),
                        "n_models_kept": len(df), "n_dropped": dropped})

    for variant in variants:                       # order follows --variants
        fam, cont = family(variant), variant.endswith("_continuous")
        print(f"\n===== {variant} =====")

        if args.from_checkpoint:
            model, cfg, pp = load_checkpoint(ckpt(variant), variant, device)
            train_s = float(cfg.get("train_seconds", 0.0))
            # fresh, seed-determined sampling noise
            reset_environment(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
        else:
            pp = preproc
            cond_dim = kpi_dim if cont else 1
            t0 = time.perf_counter()
            if fam == "cvae":
                model = train_cvae(X, Y_kpi if cont else y_bin_col, cond_dim,
                                   args.latent_dim, args.cvae_epochs, args.batch,
                                   args.seed, verbose)
                epochs, batch = args.cvae_epochs, args.batch
            elif fam == "gan":
                model = train_gan(X, Y_kpi if cont else y_binary, args.latent_dim,
                                  args.gan_epochs, args.batch, device, args.seed, verbose,
                                  kpi_dim=kpi_dim if cont else None)
                epochs, batch = args.gan_epochs, args.batch
            else:
                model = train_nf(X, Y_kpi if cont else y_bin_col, cond_dim,
                                 args.nf_layers, args.nf_hidden, args.nf_epochs,
                                 args.nf_batch, device, args.seed, verbose)
                epochs, batch = args.nf_epochs, args.nf_batch
            train_s = time.perf_counter() - t0
            cfg = {"variant": variant, "family": fam, "continuous": cont,
                   "scenario": args.scenario, "scen_tag": scen_tag, "seed": args.seed,
                   "perturbations": perts, "cutoffs": list(cutoffs),
                   "input_dim": int(X.shape[1]), "cond_dim": int(cond_dim),
                   "latent_dim": args.latent_dim,
                   "nf_layers": args.nf_layers, "nf_hidden": args.nf_hidden,
                   "epochs": epochs, "batch": batch,
                   "n_train": int(X.shape[0]), "n_positive": int(y_binary.sum()),
                   "train_seconds": round(train_s, 2)}
            if not args.no_save:
                save_checkpoint(ckpt(variant), variant, model, cfg, pp)

        y_tgt = make_kpi_target(pp, args.kpi_target_mode, args.kpi_target_pct,
                                n, args.seed) if cont else None
        t1 = time.perf_counter()
        xg = generate(variant, model, cfg, y_tgt, n, T)
        finalize(xg, train_s, time.perf_counter() - t1, variant, pp)

    # ---- timing summary table ----
    if records:
        tdf = pd.DataFrame(records).set_index("variant")
        tdf.insert(0, "scenario", scen_tag)
        tdf.insert(1, "seed", args.seed)
        ttag = f"_{target_tag}" if target_tag else ""
        ctag = "_fromckpt" if args.from_checkpoint else ""
        tpath = out / f"timing_{scen_tag}{ttag}{ctag}_seed{args.seed}.csv"
        tdf.to_csv(tpath)
        print("\n=== timing summary ===")
        print(tdf.to_string())
        print(f"[saved] {tpath}")

    print("[done]")


if __name__ == "__main__":
    main()
