# Merge per-cell results, apply R2>=0.2, write the authors' map inputs and the inputs of 06_attribution.py
# usage: 05_merge_attribution.py <proc_dir> <part_dir> <data_out_dir>
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr

proc, parts, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
z = np.load(f"{proc}/anomalies.npz"); lat, lon = z["lat"], z["lon"]
m = np.load(f"{proc}/metrics.npz")
ga = xr.open_dataset(f"{proc}/grid_area_0.5deg.nc")["area"].sel(lat=slice(-50, 75))
A = ga.values
a = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{parts}/part_[0-9][0-9][0-9][0-9].npy"))])
cols = ["i", "j", "n", "nc", "r2", "est", "ra_sm", "ra_vpd", "ra_tas", "ra_swr", "r1_sm", "r1_vpd", "r1_tas", "r1_swr",
        "r2_sm", "r2_vpd", "r2_tas", "r2_swr", "p_dom", "p_sm", "p_vpd", "r2c", "npre", "npost"]
d = pd.DataFrame(a, columns=cols)
print("cells", len(d), "fitted", d.r2.notna().sum(), "mean test R2 (all fitted)", d.r2.mean())
print("xgboost share", d.est.mean())
print("cells R2>=0.2", (d.r2 >= 0.2).sum(), "mean R2 kept", d.r2[d.r2 >= 0.2].mean(), "centre-cell R2 kept", d.r2c[d.r2 >= 0.2].mean())
d = d[d.r2 >= 0.2]
i = d.i.astype(int).values; j = d.j.astype(int).values


def g(v):
    x = np.full((len(lat), len(lon)), np.nan); x[i, j] = v; return x


def da(x):
    return xr.DataArray(x, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


ri_sm, ri_vpd = g(d.ra_sm), g(d.ra_vpd)
D = ri_vpd - ri_sm
dri_sm = g((d.r2_sm - d.r1_sm) / d.r1_sm * 100)
dri_vpd = g((d.r2_vpd - d.r1_vpd) / d.r1_vpd * 100)
dD = g((d.r2_vpd - d.r2_sm) - (d.r1_vpd - d.r1_sm))

xr.Dataset({"ri": (("v", "lat", "lon"), np.stack([ri_sm, ri_vpd])), "diff": (("lat", "lon"), D), "pval": (("lat", "lon"), g(d.p_dom))},
           coords={"lat": lat, "lon": lon}).to_netcdf(f"{out}/relative_importance_fields.nc")
xr.Dataset({"dif": (("v", "lat", "lon"), np.stack([dri_sm, dri_vpd])), "pval": (("v", "lat", "lon"), np.stack([g(d.p_sm), g(d.p_vpd)]))},
           coords={"lat": lat, "lon": lon}).to_netcdf(f"{out}/relative_importance_change_fields.nc")
xr.Dataset({"diff_overall": (("lat", "lon"), D), "diff_change": (("lat", "lon"), dD)},
           coords={"lat": lat, "lon": lon}).to_netcdf(f"{out}/dominance_change_fields.nc")


def amean(x):
    k = np.isfinite(x); return float((x[k] * A[k]).sum() / A[k].sum())


def apct(k, x):
    v = np.isfinite(x); return float(A[k & v].sum() / A[v].sum() * 100)


s = {"n_cells": int(len(d)), "mean_test_r2": float(d.r2.mean()), "RI_SM": amean(ri_sm), "RI_VPD": amean(ri_vpd),
     "SM_dominated_pct": apct(D < 0, D), "dRI_VPD_up_pct": apct(dri_vpd > 0, dri_vpd), "dRI_SM_down_pct": apct(dri_sm < 0, dri_sm),
     "dD_pos_pct": apct(dD > 0, dD), "SM_dom_to_VPD_pct": apct((D < 0) & (dD > 0), D), "SM_dom_to_SM_pct": apct((D < 0) & (dD < 0), D),
     "VPD_dom_to_VPD_pct": apct((D > 0) & (dD > 0), D), "VPD_dom_to_SM_pct": apct((D > 0) & (dD < 0), D)}

# sensor-switch diagnostic: NDVI residual variability AVHRR era vs MODIS era against dD
r = np.load(f"{proc}/res_ndvi.npy", mmap_mode="r"); yr = z["year"]; gs = z["gs"][z["month"] - 1]
v1 = np.nanstd(np.where(gs[yr <= 2002], r[yr <= 2002], np.nan), 0)
v2 = np.nanstd(np.where(gs[yr >= 2003], r[yr >= 2003], np.nan), 0)
ratio = v2 / v1
k = np.isfinite(ratio) & np.isfinite(dD)
s["ndvi_resid_std_ratio_modis_over_avhrr_median"] = float(np.median(ratio[k]))
s["corr_logratio_vs_dD_spearman"] = float(pd.Series(np.log(ratio[k])).corr(pd.Series(dD[k]), method="spearman"))
xr.Dataset({"ndvi_std_ratio": (("lat", "lon"), ratio)}, coords={"lat": lat, "lon": lon}).to_netcdf(f"{out}/diag_sensor_ratio.nc")

base = ["vpd_mean", "sm_mean", "vpd_nor_cv", "sm_nor_cv"]
chg = ["vpd_mean", "sm_mean", "vpd_mean_change", "sm_mean_change", "vpd_nor_cv", "sm_nor_cv", "vpd_nor_cv_change", "sm_nor_cv_change"]
for tag, y, names in [("importance", D, base), ("importance_change_SM", dri_sm, chg),
                      ("importance_change_VPD", dri_vpd, chg), ("dominance_change", dD, chg)]:
    X = np.stack([m[n] for n in names], -1)
    k = np.isfinite(y) & np.isfinite(X).all(-1)
    np.savez(f"{out}/attr_inputs_{tag}.npz", X=X[k], y=y[k], names=np.array(names))
json.dump(s, open(f"{out}/summary_merge.json", "w"), indent=1)
print(json.dumps(s, indent=1))
