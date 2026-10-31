# Area-weighted 5x5 block means of the attribution targets and predictors (0.5 to 2.5 deg), inputs of 06_attribution.py
# usage: 05b_coarsen.py <proc_dir> <data_dir> <out_dir>
import os, sys
import numpy as np, xarray as xr

proc, data, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
m = np.load(f"{proc}/metrics.npz")
A = xr.open_dataset(f"{proc}/grid_area_0.5deg.nc")["area"].sel(lat=slice(-50, 75)).values
D = xr.open_dataset(f"{data}/relative_importance_fields.nc")["diff"].values
dif = xr.open_dataset(f"{data}/relative_importance_change_fields.nc")["dif"].values
dD = xr.open_dataset(f"{data}/dominance_change_fields.nc")["diff_change"].values
ny, nx, B, MIN = D.shape[0] // 5, D.shape[1] // 5, 5, 5


def block(x):
    return x.reshape(ny, B, nx, B).sum((1, 3))


base = ["vpd_mean", "sm_mean", "vpd_nor_cv", "sm_nor_cv"]
chg = ["vpd_mean", "sm_mean", "vpd_mean_change", "sm_mean_change", "vpd_nor_cv", "sm_nor_cv", "vpd_nor_cv_change", "sm_nor_cv_change"]
for tag, y, names in [("importance", D, base), ("importance_change_SM", dif[0], chg),
                      ("importance_change_VPD", dif[1], chg), ("dominance_change", dD, chg)]:
    X = np.stack([m[n] for n in names], -1)
    k = np.isfinite(y) & np.isfinite(X).all(-1)
    w = np.where(k, A, 0.)
    W = block(w); ok = block(k.astype(float)) >= MIN
    yc = block(np.where(k, y, 0.) * w)[ok] / W[ok]
    Xc = np.stack([block(np.where(k, X[..., n], 0.) * w)[ok] / W[ok] for n in range(len(names))], -1)
    np.savez(f"{out}/attr_inputs_{tag}.npz", X=Xc, y=yc, names=np.array(names))
    print(tag, Xc.shape, "from", int(k.sum()), flush=True)
