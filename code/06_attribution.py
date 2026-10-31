# Spatial attribution of one target across cells (FLAML+TreeSHAP); SHAP is split over processes, it is single-threaded for rf
# usage: 06_attribution.py <data_dir> <tag> <budget_s> <nproc>
import json, multiprocessing as mp, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from flaml import AutoML
import shap
from sklearn.metrics import r2_score

out, tag, budget, nproc = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
z = np.load(f"{out}/attr_inputs_{tag}.npz")
X_, y_, names = z["X"], z["y"], [str(n) for n in z["names"]]
p = np.random.RandomState(0).permutation(len(y_)); nt = int(.8 * len(y_))
t0 = time.time()
am = AutoML()
am.fit(X_[p[:nt]], y_[p[:nt]], task="regression", metric="r2", estimator_list=["rf", "xgboost"], time_budget=budget,
       n_jobs=nproc, eval_method="holdout", split_ratio=0.2, seed=0, verbose=0, log_file_name="")
r2 = r2_score(y_[p[nt:]], am.predict(X_[p[nt:]]))
est = am.model.estimator
pickle.dump(est, open(f"{out}/attr_model_{tag}.pkl", "wb"))
print(tag, "fit", am.best_estimator, am.best_config, "test r2", r2, "s", time.time() - t0, flush=True)

exp = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
t0 = time.time()
exp.shap_values(X_[:50])                       # includes the additivity check
rate = (time.time() - t0) / 50
print(tag, "shap s/row", rate, "projected h", rate * len(X_) / nproc / 3600, flush=True)


def f(c):
    return exp.shap_values(c, check_additivity=False)


t0 = time.time()
if am.best_estimator == "rf":
    est.n_jobs = 1
    chunks = np.array_split(X_, max(1, len(X_) // 100))
    sv = []
    with mp.Pool(nproc) as pool:
        for n, r in enumerate(pool.imap(f, chunks), 1):
            sv.append(r)
            if n % 20 == 0:
                print(tag, n, len(chunks), "elapsed h", (time.time() - t0) / 3600, flush=True)
    sv = np.concatenate(sv)
else:
    sv = f(X_)
pd.DataFrame(X_, columns=names).to_csv(f"{out}/shap_scatter_{tag}_xx.csv.gz", index=False)
pd.DataFrame(sv, columns=names).to_csv(f"{out}/shap_scatter_{tag}_yy.csv.gz", index=False)
imp = np.abs(sv).mean(0); imp = imp / imp.sum() * 100
s = {"estimator": am.best_estimator, "config": am.best_config, "r2": float(r2), "n": len(X_), **{n: round(float(v), 1) for n, v in zip(names, imp)}}
json.dump(s, open(f"{out}/attr_{tag}.json", "w"), indent=1)
print(tag, s, "DONE", flush=True)
