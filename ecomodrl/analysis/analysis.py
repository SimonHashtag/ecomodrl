#   Testing     : evaluates each of the 100 policies on 100 held-out
#                            test seeds (10_000 cells per (model, algo)), plus a
#                            seed-independent Bellman value_grid.npz per policy
#                            writes artifacts/<model>/<algo>/seed_<S>/
#                                   {value_grid.npz,
#                                    test/values.json,
#                                    test/euler.npz,
#                                    test/visited.npz (stochastic only)}
#                            All per-test-seed results for one training seed
#                            are stored in those 2-3 consolidated files (keyed
#                            by str(test_seed) / "ts_<TEST_SEED>") instead of
#                            one directory per test seed, to keep the inode
#                            footprint bounded (~100x reduction).
#                            then summary.json aggregating across the 100 cells
#
#   Plots                  : distribution-aware figures over the 100 seed combinations
#                            per (model, algo)

import os
import json
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import analysis
from ecomodrl.analysis.plotting import (
    plot_policy_dist, 
    plot_cev_dist,
    plot_euler_errors_hist_dist,
    plot_convergence_dist,
    make_latex_tables,
    make_combined_latex_table
)
from ecomodrl.analysis.policy_interface import EGMPolicy, RLPolicy
from train import MODEL_CLASSES, EVAL_SEED_OFFSET, M_GRID_EVAL, T_GRID_EVAL
from search_spaces import (
    TRAIN_SEEDS, TEST_SEEDS,
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

MODELS = ["cass_koopmans", "brock_mirman", "buffer_stock"]
ALGOS  = ["ppo", "ddpg", "td3", "sac"]

WALLCLOCK = {
    "cass_koopmans": 600,
    "brock_mirman":  600,
    "buffer_stock":  600,
}

EVAL_INTERVAL_SECONDS = 5.0

# Stage-3 test pack sizes are inherited from train.py (M_GRID_EVAL, T_GRID_EVAL)
# so checkpoint-selection and Stage-3 V_grid use the exact same evaluation
# protocol. TEST_N_EULER is the per-cell Euler-error simulation length.
TEST_N_EULER = 100

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR     = "plots"

def _seed_dir(model, algo, train_seed):
    return os.path.join(ARTIFACTS_DIR, model, algo, f"seed_{train_seed}")


def _consolidated_test_paths(model, algo, train_seed):
    base = os.path.join(_seed_dir(model, algo, train_seed), "test")
    return {
        "base":    base,
        "values":  os.path.join(base, "values.json"),
        "euler":   os.path.join(base, "euler.npz"),
        "visited": os.path.join(base, "visited.npz"),
    }


def _load_consolidated_values(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)  # keys: str(test_seed) -> {v_rl, v_egm, cev_regret}


def _load_consolidated_npz(path):
    if not os.path.exists(path):
        return {}
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}  # keys: "ts_<TEST_SEED>"


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_write_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _test_packs_dir(model):
    return os.path.join(ARTIFACTS_DIR, model, "test_packs")


def _egm_dir(model):
    return os.path.join(ARTIFACTS_DIR, model, "egm")



def _solve_model(model_name):
    cls = MODEL_CLASSES[model_name]
    eco = cls()
    eco.solve(verbose=False)
    return eco


def _algo_class(algo):
    from stable_baselines3 import PPO, DDPG, TD3, SAC
    return {"ppo": PPO, "ddpg": DDPG, "td3": TD3, "sac": SAC}[algo]


def _build_test_packs(model_name):
    """Build (and persist) the test eval_packs for a given model.

    Runs in the main process so all workers can read identical bytes off
    disk -- guarantees the paired-comparison property across (algo,
    train_seed) cells sharing a test_seed. Also caches the EGM-policy V on
    each pack (needed for CEV regret) to test_packs/egm_values.json.
    """
    pack_dir = _test_packs_dir(model_name)
    os.makedirs(pack_dir, exist_ok=True)
    egm_v_path = os.path.join(pack_dir, "egm_values.json")

    need_packs = [ts for ts in TEST_SEEDS
                  if not os.path.exists(os.path.join(pack_dir, f"test_{ts}.pkl"))]
    need_egm_v = not os.path.exists(egm_v_path)

    eco = _solve_model(model_name)
    deterministic = (getattr(eco, "weights", None) is None) and not eco.HAS_EXO

    if deterministic:
        # CK / pure-deterministic: EGM Euler errors are policy-independent and
        # do not depend on a simulation seed. Cache once under egm/.
        egm_ee_path = os.path.join(_egm_dir(model_name), "euler.npy")
        need_egm_ee = not os.path.exists(egm_ee_path)
    else:
        need_egm_ee_seeds = [
            ts for ts in TEST_SEEDS
            if not os.path.exists(os.path.join(pack_dir, f"egm_euler_{ts}.npy"))
        ]
        need_egm_ee = bool(need_egm_ee_seeds)

    if not need_packs and not need_egm_v and not need_egm_ee:
        return

    egm_policy = EGMPolicy(eco)
    egm_values = {}
    if os.path.exists(egm_v_path):
        with open(egm_v_path) as f:
            egm_values = {int(k): float(v) for k, v in json.load(f).items()}

    for ts in TEST_SEEDS:
        pack_path = os.path.join(pack_dir, f"test_{ts}.pkl")
        if not os.path.exists(pack_path):
            pack = analysis.make_eval_pack(eco, ts, M_GRID_EVAL, T_GRID_EVAL)
            with open(pack_path, "wb") as f:
                pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
        if ts not in egm_values:
            with open(pack_path, "rb") as f:
                pack = pickle.load(f)
            egm_values[ts] = float(analysis.eval_return_grid(eco, egm_policy, pack))

    with open(egm_v_path, "w") as f:
        json.dump({str(k): v for k, v in egm_values.items()}, f, indent=2)

    if deterministic:
        if need_egm_ee:
            os.makedirs(_egm_dir(model_name), exist_ok=True)
            ee = analysis.euler_errors(eco, egm_policy, n_sim=TEST_N_EULER)
            np.save(egm_ee_path, np.asarray(ee, dtype=np.float64))
    else:
        for ts in TEST_SEEDS:
            p = os.path.join(pack_dir, f"egm_euler_{ts}.npy")
            if not os.path.exists(p):
                ee = analysis.euler_errors(eco, egm_policy,
                                           n_sim=TEST_N_EULER, seed=ts)
                np.save(p, np.asarray(ee, dtype=np.float64))

    print(f"[test] {model_name}: prepared {len(TEST_SEEDS)} test packs, "
          f"EGM reference values and EGM Euler errors")


def _cache_train_egm_values(model_name):
    """Cache V_EGM evaluated on each training-checkpoint eval_pack.

    Each training run uses an eval_pack seeded by `train_seed + EVAL_SEED_OFFSET`
    (see train.train -> analysis.make_eval_pack). The convergence plot needs
    the EGM value on those exact eval_packs to compute regret = V_EGM - V_RL.
    Writes artifacts/<model>/train_egm_values.json keyed by str(train_seed).
    Idempotent: only computes seeds missing from the on-disk JSON.
    """
    out_path = os.path.join(ARTIFACTS_DIR, model_name, "train_egm_values.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = {int(k): float(v) for k, v in json.load(f).items()}

    missing = [s for s in TRAIN_SEEDS if s not in existing]
    if not missing:
        return

    eco = _solve_model(model_name)
    egm_policy = EGMPolicy(eco)
    for s in missing:
        pack = analysis.make_eval_pack(eco, s + EVAL_SEED_OFFSET,
                                       M_GRID_EVAL, T_GRID_EVAL)
        existing[s] = float(analysis.eval_return_grid(eco, egm_policy, pack))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in existing.items()}, f, indent=2)
    print(f"[test] {model_name}: cached V_EGM on {len(missing)} train eval_packs")


def _test_worker(args):
    """Process all test work for one (model, train_seed):
       - for each algo: D.1 value_grid.npz (seed-indep) and
                        D.2 consolidated test/values.json + test/euler.npz
                            (+ test/visited.npz for stochastic models),
                            each holding all TEST_SEEDS results for this row.

    Migration: on entry, any leftover per-test-seed files written by an older
    layout (`test/<TS>/{value.json, euler.npy, visited.npy}`) are folded into
    the consolidated files, then the per-cell directories are removed. After
    that, missing test_seeds are computed and merged in. Each consolidated
    file is rewritten atomically (temp + os.replace).
    """
    import torch
    torch.set_num_threads(1)

    model_name, train_seed = args
    eco = _solve_model(model_name)
    deterministic = (getattr(eco, "weights", None) is None) and not eco.HAS_EXO

    # Load all test packs and EGM reference values once.
    pack_dir = _test_packs_dir(model_name)
    packs = {}
    for ts in TEST_SEEDS:
        with open(os.path.join(pack_dir, f"test_{ts}.pkl"), "rb") as f:
            packs[ts] = pickle.load(f)
    with open(os.path.join(pack_dir, "egm_values.json")) as f:
        v_egm_by_seed = {int(k): float(v) for k, v in json.load(f).items()}

    summary = []
    for algo in ALGOS:
        out_dir = _seed_dir(model_name, algo, train_seed)
        mz = os.path.join(out_dir, "model.zip")
        if not os.path.exists(mz):
            summary.append((algo, "missing_model"))
            continue

        algo_cls = _algo_class(algo)
        sb3 = algo_cls.load(mz)
        policy = RLPolicy(sb3, None, eco.HAS_EXO, model=eco)

        # D.1: seed-independent Bellman value-on-grid (for CEV maps)
        vg_path = os.path.join(out_dir, "value_grid.npz")
        if not os.path.exists(vg_path):
            grid_rl, V_rl_grid = analysis.policy_value_on_grid(eco, policy)
            if eco.HAS_EXO:
                np.savez(vg_path,
                         m_grid=grid_rl[0], z_grid=grid_rl[1], V=V_rl_grid)
            else:
                np.savez(vg_path, m_grid=grid_rl, V=V_rl_grid)

        # D.2: consolidated per-(algo, train_seed) test outputs
        cpaths = _consolidated_test_paths(model_name, algo, train_seed)
        os.makedirs(cpaths["base"], exist_ok=True)
        values   = _load_consolidated_values(cpaths["values"])
        eulers   = _load_consolidated_npz(cpaths["euler"])
        visiteds = (_load_consolidated_npz(cpaths["visited"])
                    if not deterministic else None)

        # ---- Migration: fold any legacy per-cell files into the dicts. ----
        migrated = False
        migrated_dirs = []
        if os.path.isdir(cpaths["base"]):
            for name in os.listdir(cpaths["base"]):
                sub = os.path.join(cpaths["base"], name)
                if not os.path.isdir(sub):
                    continue
                try:
                    ts_legacy = int(name)
                except ValueError:
                    continue
                k  = str(ts_legacy)
                ak = f"ts_{ts_legacy}"
                vp_old = os.path.join(sub, "value.json")
                ep_old = os.path.join(sub, "euler.npy")
                vis_old = os.path.join(sub, "visited.npy")
                if k not in values and os.path.exists(vp_old):
                    with open(vp_old) as f:
                        values[k] = json.load(f)
                    migrated = True
                if ak not in eulers and os.path.exists(ep_old):
                    eulers[ak] = np.load(ep_old)
                    migrated = True
                if (not deterministic) and ak not in visiteds \
                        and os.path.exists(vis_old):
                    visiteds[ak] = np.load(vis_old)
                    migrated = True
                migrated_dirs.append(sub)

        # Write consolidated *before* removing per-cell dirs so a crash here
        # cannot lose data.
        if migrated:
            _atomic_write_json(cpaths["values"], values)
            _atomic_write_npz(cpaths["euler"], eulers)
            if not deterministic:
                _atomic_write_npz(cpaths["visited"], visiteds)
        for d in migrated_dirs:
            shutil.rmtree(d, ignore_errors=True)

        # ---- Compute any missing test seeds. ----
        dirty_v = dirty_e = dirty_vi = False
        for ts in TEST_SEEDS:
            k  = str(ts)
            ak = f"ts_{ts}"

            if k not in values:
                v_rl = float(analysis.eval_return_grid(eco, policy, packs[ts]))
                v_egm = v_egm_by_seed[ts]
                if np.isfinite(v_rl) and v_rl != 0.0:
                    cev_r = float(analysis.cev(eco, v_rl, v_egm))
                else:
                    cev_r = float("nan")
                values[k] = {"v_rl": v_rl, "v_egm": v_egm,
                             "cev_regret": cev_r}
                dirty_v = True

            if ak not in eulers:
                ee = analysis.euler_errors(eco, policy,
                                           n_sim=TEST_N_EULER, seed=ts)
                eulers[ak] = np.asarray(ee, dtype=np.float64)
                dirty_e = True

            # Visited m-path used by Euler-error simulation (stochastic models
            # only). Same seed as the Euler call so the trajectory is
            # reproducibly identical.
            if (not deterministic) and ak not in visiteds:
                states = analysis.simulate_states(
                    eco, policy, n_sim=TEST_N_EULER, seed=ts
                )
                visiteds[ak] = np.asarray(states[..., 0], dtype=np.float64)
                dirty_vi = True

        if dirty_v:
            _atomic_write_json(cpaths["values"], values)
        if dirty_e:
            _atomic_write_npz(cpaths["euler"], eulers)
        if dirty_vi:
            _atomic_write_npz(cpaths["visited"], visiteds)

        summary.append((algo, "ok"))

    return model_name, int(train_seed), summary


def _summarize_test(model_name, algo):
    """Aggregate the test cells into a single summary.json.

    Per-metric we record both classic order stats (median/Q1/Q3/min/max)
    and the IQM with a 95% two-way crossed (pigeonhole) bootstrap CI, which
    resamples train-seed rows and test-seed columns with replacement so the
    interval reflects the crossed design instead of treating the n_train*n_test
    cells as independent. Deterministic models keep one representative test
    column (the others are exact duplicates), so the crossed bootstrap reduces
    to a one-way train-seed bootstrap there.
    """
    # Determine whether this model is deterministic (CK): EGM Euler errors
    # are then policy-independent and cached as a single file.
    eco = _solve_model(model_name)
    deterministic = (getattr(eco, "weights", None) is None) and not eco.HAS_EXO
    pack_dir = _test_packs_dir(model_name)
    egm_ee_single = None
    if deterministic:
        p = os.path.join(_egm_dir(model_name), "euler.npy")
        if os.path.exists(p):
            egm_ee_single = np.load(p)
    egm_ee_by_ts = {}
    if not deterministic:
        for ts in TEST_SEEDS:
            p = os.path.join(pack_dir, f"egm_euler_{ts}.npy")
            if os.path.exists(p):
                egm_ee_by_ts[ts] = np.load(p)

    active_test_seeds = TEST_SEEDS[:1] if deterministic else TEST_SEEDS
    shape = (len(TRAIN_SEEDS), len(active_test_seeds))
    metrics = {
        "v_rl":        np.full(shape, np.nan, dtype=np.float64),
        "cev_regret":  np.full(shape, np.nan, dtype=np.float64),
        "ee_mean":     np.full(shape, np.nan, dtype=np.float64),
        "ee_p95":      np.full(shape, np.nan, dtype=np.float64),
        "wasserstein": np.full(shape, np.nan, dtype=np.float64),
    }
    n_cells = 0
    for i, s in enumerate(TRAIN_SEEDS):
        cpaths = _consolidated_test_paths(model_name, algo, s)
        values = _load_consolidated_values(cpaths["values"])
        eulers = _load_consolidated_npz(cpaths["euler"])
        if not values or not eulers:
            continue
        for j, ts in enumerate(active_test_seeds):
            k  = str(ts)
            ak = f"ts_{ts}"
            if k not in values or ak not in eulers:
                continue
            d = values[k]
            metrics["v_rl"][i, j] = d["v_rl"]
            metrics["cev_regret"][i, j] = d["cev_regret"]
            ee = eulers[ak]
            ee_f = ee[np.isfinite(ee)]
            if ee_f.size > 0:
                metrics["ee_mean"][i, j] = float(ee_f.mean())
                metrics["ee_p95"][i, j] = float(np.percentile(ee_f, 95))
            ref = egm_ee_single if deterministic else egm_ee_by_ts.get(ts)
            if ref is not None:
                metrics["wasserstein"][i, j] = analysis.wasserstein_euler(ee, ref)
            n_cells += 1

    def _stats(matrix):
        M = np.asarray(matrix, dtype=np.float64)
        a = M[np.isfinite(M)]
        if a.size == 0:
            return {"n": 0}
        # Crossed (train x test) bootstrap: descriptive order stats stay on the
        # flat finite values; the IQM CI resamples rows and columns so the CI
        # reflects the crossed design rather than treating every cell as i.i.d.
        point, lo, hi = analysis.iqm_ci_crossed(M)
        return {
            "n":      int(a.size),
            "median": float(np.median(a)),
            "q1":     float(np.percentile(a, 25)),
            "q3":     float(np.percentile(a, 75)),
            "min":    float(a.min()),
            "max":    float(a.max()),
            "iqm":    float(point),
            "ci_lo":  float(lo),
            "ci_hi":  float(hi),
        }

    out = {
        "model": model_name, "algo": algo,
        "n_cells": int(n_cells),
        "v_rl":        _stats(metrics["v_rl"]),
        "cev_regret":  _stats(metrics["cev_regret"]),
        "ee_mean":     _stats(metrics["ee_mean"]),
        "ee_p95":      _stats(metrics["ee_p95"]),
        "wasserstein": _stats(metrics["wasserstein"]),
    }
    summ_dir = os.path.join(ARTIFACTS_DIR, model_name, algo, "test")
    os.makedirs(summ_dir, exist_ok=True)
    with open(os.path.join(summ_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)


def run_test():
    print("\n========== Stage 3 -- Multi-seed Testing ==========")
    # Gate
    missing = []
    for m in MODELS:
        for a in ALGOS:
            for s in TRAIN_SEEDS:
                mz = os.path.join(_seed_dir(m, a, s), "model.zip")
                if not os.path.exists(mz):
                    missing.append(mz)
    if missing:
        raise RuntimeError(
            f"RUN_TEST aborted: {len(missing)} trained policies missing. "
            f"First few:\n  " + "\n  ".join(missing[:5])
        )

    # Cache EGM value_grid (used later for CEV maps in plots)
    for m in MODELS:
        os.makedirs(_egm_dir(m), exist_ok=True)
        egp = os.path.join(_egm_dir(m), "value_grid.npz")
        if not os.path.exists(egp):
            eco = _solve_model(m)
            grid, V_egm = analysis.policy_value_on_grid(eco, EGMPolicy(eco))
            if eco.HAS_EXO:
                np.savez(egp, m_grid=grid[0], z_grid=grid[1], V=V_egm)
            else:
                np.savez(egp, m_grid=grid, V=V_egm)
            print(f"[test] cached EGM value grid for {m}")

    # Build/cache test packs + EGM reference values per model, plus
    # V_EGM on every training-checkpoint eval_pack (needed for the
    # regret-on-log-axis convergence plot).
    for m in MODELS:
        _build_test_packs(m)
        _cache_train_egm_values(m)

    units = [(m, s) for m in MODELS for s in TRAIN_SEEDS]
    workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"[test] {len(units)} (model, train_seed) units; "
          f"ProcessPoolExecutor(max_workers={workers})")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_test_worker, u): u for u in units}
        done = 0
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                m, s, summary = fut.result()
                done += 1
                tags = " ".join(f"{a}:{st}" for a, st in summary)
                print(f"[test] {m} seed={s} OK [{tags}] "
                      f"({done}/{len(units)})", flush=True)
            except Exception as e:
                done += 1
                print(f"[test] FAILED {u[0]} seed={u[1]}: {e!r} "
                      f"({done}/{len(units)})", flush=True)

    # Summarize
    for m in MODELS:
        for a in ALGOS:
            _summarize_test(m, a)
            print(f"[test] {m}/{a} summary.json written")


# =============================================================================
# Plots
# =============================================================================

def _load_history(path):
    d = np.load(path)
    return np.asarray(d["t"]), np.asarray(d["V_grid"])


def run_plots():
    print("\n========== Plots ==========")
    combined_summaries = []  # list of (model_name, {algo: summary}) for the
                             # cross-model comparison table
    for model_name in MODELS:
        eco = _solve_model(model_name)
        egm_policy = EGMPolicy(eco)

        # --- Per-algo, per-train_seed data ---
        rl_policies   = {a: [] for a in ALGOS}    # list of <=N_TRAIN_SEEDS RLPolicy
        rl_V_grids    = {a: [] for a in ALGOS}    # list of <=N V grids on display grid
        histories     = {a: [] for a in ALGOS}    # list of <=N (t, V) pairs
        history_seeds = {a: [] for a in ALGOS}    # train_seed aligned with histories
        ee_pools      = {a: [] for a in ALGOS}    # list of arrays (stochastic)
        ee_seedwise   = {a: [] for a in ALGOS}    # CK: <=N arrays (one per policy)
        visited_pools = {a: [] for a in ALGOS}    # stochastic: pooled visited m

        # EGM value grid (for CEV map computation in plots)
        egp = np.load(os.path.join(_egm_dir(model_name), "value_grid.npz"))
        if eco.HAS_EXO:
            grid_disp = (egp["m_grid"], egp["z_grid"])
        else:
            grid_disp = (egp["m_grid"],)
        V_egm_grid = egp["V"]

        deterministic = (getattr(eco, "weights", None) is None) and not eco.HAS_EXO

        for algo in ALGOS:
            for s in TRAIN_SEEDS:
                seed_dir = _seed_dir(model_name, algo, s)
                mz = os.path.join(seed_dir, "model.zip")
                if not os.path.exists(mz):
                    continue
                sb3 = _algo_class(algo).load(mz)
                rl_policies[algo].append(RLPolicy(sb3, None, eco.HAS_EXO, model=eco))

                vg_path = os.path.join(seed_dir, "value_grid.npz")
                if os.path.exists(vg_path):
                    vd = np.load(vg_path)
                    rl_V_grids[algo].append(vd["V"])

                hp = os.path.join(seed_dir, "history.npz")
                if os.path.exists(hp):
                    histories[algo].append(_load_history(hp))
                    history_seeds[algo].append(s)

                # cells (consolidated)
                cpaths = _consolidated_test_paths(model_name, algo, s)
                eulers = _load_consolidated_npz(cpaths["euler"])
                visiteds = (_load_consolidated_npz(cpaths["visited"])
                            if not deterministic else {})
                got_ck_ee = False
                for ts in TEST_SEEDS:
                    ak = f"ts_{ts}"
                    if ak in eulers:
                        ee_pools[algo].append(eulers[ak])
                        if deterministic and not got_ck_ee:
                            ee_seedwise[algo].append(eulers[ak])
                            got_ck_ee = True
                    if (not deterministic) and ak in visiteds:
                        visited_pools[algo].append(visiteds[ak])

        # V_EGM aligned with each retained training history (for regret plot)
        train_egm_path = os.path.join(ARTIFACTS_DIR, model_name,
                                      "train_egm_values.json")
        v_egm_by_train_seed = {}
        if os.path.exists(train_egm_path):
            with open(train_egm_path) as f:
                v_egm_by_train_seed = {int(k): float(v) for k, v in json.load(f).items()}
        v_egms = {a: [v_egm_by_train_seed.get(s, float("nan"))
                      for s in history_seeds[a]] for a in ALGOS}

        # Pool visited m per algo (stochastic only)
        visited_pooled = {}
        for a in ALGOS:
            arrs = visited_pools.get(a, [])
            if arrs:
                visited_pooled[a] = np.concatenate(
                    [np.asarray(x).ravel() for x in arrs]
                )

        base = os.path.join(PLOTS_DIR, model_name)
        plot_policy_dist(
            eco, egm_policy, rl_policies, os.path.join(base, "policy.png")
        )
        plot_cev_dist(
            eco, grid_disp, V_egm_grid, rl_V_grids,
            os.path.join(base, "cev.png"),
            visited_pools=visited_pooled,
        )
        plot_euler_errors_hist_dist(
            eco, ee_pools, ee_seedwise,
            os.path.join(base, "euler_hist.png")
        )
        plot_convergence_dist(
            histories, v_egms, WALLCLOCK[model_name],
            os.path.join(base, "convergence.png"),
            grid_step=EVAL_INTERVAL_SECONDS,
        )

        # LaTeX comparison table per model
        summaries = {}
        for a in ALGOS:
            sp = os.path.join(ARTIFACTS_DIR, model_name, a, "test", "summary.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    summaries[a] = json.load(f)
        make_latex_tables(eco, summaries, os.path.join(base, "table.tex"))
        combined_summaries.append((model_name, summaries))

        print(f"[plots] saved -> {base}/")

    # Combined point-estimate comparison table across all models
    make_combined_latex_table(
        combined_summaries, os.path.join(PLOTS_DIR, "comparison_table.tex")
    )
    print(f"[plots] saved -> {PLOTS_DIR}/comparison_table.tex")