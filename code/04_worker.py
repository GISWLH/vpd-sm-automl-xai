# Per-grid-cell AutoML (FLAML rf/xgboost) + TreeSHAP, paper Methods. One process handles a slice of cells.
# usage: 04_worker.py <proc_dir> <out_dir> <rank> <nrank> <time_budget_s> [max_cells]
import os, sys, time, warnings, logging
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from flaml import AutoML
import shap
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind

logging.getLogger("flaml").setLevel(logging.ERROR)
proc, outd, rank, nrank, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
max_cells = int(sys.argv[6]) if len(sys.argv) > 6 else None
os.makedirs(outd, exist_ok=True)

z = np.load(f"{proc}/anomalies.npz")           # float32 (time, lat, lon) residuals; bool gs; cell list
res = {k: np.load(f"{proc}/res_{k}.npy", mmap_mode="r") for k in ["ndvi", "sm", "vpd", "tas", "swr"]}
gs = z["gs"]                                   # (12, lat, lon) growing-season months
cells = z["cells"]                             # (n, 2) target cell indices (land-cover filtered)
year = z["year"]; month = z["month"]
PRED = ["sm", "vpd", "tas", "swr"]
ny, nx = gs.shape[1:]
cells = cells[rank::nrank]
if max_cells:
    cells = cells[:max_cells]


def pick(i, j):
    y = np.asarray(res["ndvi"][:, i, j], "f4")
    X = np.stack([np.asarray(res[k][:, i, j], "f4") for k in PRED], 1)
    t = gs[month - 1, i, j] & np.isfinite(y) & np.isfinite(X).all(1)
    return X[t], y[t], year[t]


rows = []
fp = f"{outd}/part_{rank:04d}.npy"
if os.path.exists(fp):                         # resume after a crash
    rows = np.load(fp).tolist()
start = len(rows)
t0 = time.time()
for n, (i, j) in enumerate(cells):
    if n < start:
        continue
    Xs, ys = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            a, b = i + di, (j + dj) % nx      # 3x3 pooling, wrap in longitude
            if 0 <= a < ny:
                X, y, _ = pick(a, b)
                Xs.append(X); ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    Xc, yc, yrc = pick(i, j)
    row = [i, j, len(y), len(yc)] + [np.nan] * 20
    if len(yc) >= 40 and len(y) >= 80:
        rng = np.random.RandomState(int(i) * 1000 + int(j))
        p = rng.permutation(len(y)); nt = int(0.8 * len(y))
        tr, te = p[:nt], p[nt:]
        am = AutoML()
        am.fit(X[tr], y[tr], task="regression", metric="r2", estimator_list=["rf", "xgboost"],
               time_budget=budget, n_jobs=1, eval_method="holdout", split_ratio=0.2, seed=int(rng.randint(1 << 30)),
               verbose=0, log_file_name="", retrain_full=True)
        r2 = r2_score(y[te], am.predict(X[te]))
        phi = np.abs(shap.TreeExplainer(am.model.estimator, feature_perturbation="tree_path_dependent").shap_values(Xc))
        pre, post = yrc <= 2001, yrc >= 2002
        tot = phi.mean(0)
        ri = lambda m: m / m.sum() * 100
        ra, r1, r2_ = ri(tot), ri(phi[pre].mean(0)), ri(phi[post].mean(0))
        p_dom = ttest_ind(phi[:, 1], phi[:, 0], equal_var=False).pvalue
        p_sm = ttest_ind(phi[post, 0], phi[pre, 0], equal_var=False).pvalue
        p_vpd = ttest_ind(phi[post, 1], phi[pre, 1], equal_var=False).pvalue
        est = 0 if am.best_estimator == "rf" else 1
        row = [i, j, len(y), len(yc), r2, est, *ra, *r1, *r2_, p_dom, p_sm, p_vpd, r2_score(yc, am.predict(Xc)), pre.sum(), post.sum()]
    rows.append(row)
    if (n + 1) % 25 == 0 or n + 1 == len(cells):
        np.save(f"{outd}/part_{rank:04d}.tmp.npy", np.array(rows, "f8"))
        os.replace(f"{outd}/part_{rank:04d}.tmp.npy", f"{outd}/part_{rank:04d}.npy")
        print(rank, n + 1, len(cells), round((time.time() - t0) / (n + 1 - start), 2), "s/cell", flush=True)
print("DONE", rank, flush=True)
