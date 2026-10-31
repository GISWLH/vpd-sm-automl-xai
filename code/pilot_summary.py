import glob, numpy as np, pandas as pd
c = ["i","j","n","nc","r2","est","ra_sm","ra_vpd","ra_tas","ra_swr","r1_sm","r1_vpd","r1_tas","r1_swr","r2_sm","r2_vpd","r2_tas","r2_swr","p_dom","p_sm","p_vpd","r2c","npre","npost"]
d = pd.DataFrame(np.concatenate([np.load(f) for f in glob.glob("/data/lhwang/vpd_sm_repro/out/pilot/part_*.npy")]), columns=c)
lat = -49.75 + d.i * .5
print(len(d), "fitted", d.r2.notna().sum(), "xgb share", round(d.est.mean(), 2))
print("test R2 pct", d.r2.describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_dict())
print("centre R2 median", round(d.r2c.median(), 3), " kept>=0.2:", round((d.r2 >= .2).mean(), 3))
k = d[d.r2 >= .2]; w = np.cos(np.deg2rad(lat[k.index]))
print("n kept", len(k))
for v in ["sm","vpd","tas","swr"]: print(v, "RI", round(np.average(k["ra_"+v], weights=w), 1))
D = k.ra_vpd - k.ra_sm; print("SM dominated", round(np.average(D < 0, weights=w) * 100, 1), "%")
dv = (k.r2_vpd - k.r1_vpd) / k.r1_vpd; ds = (k.r2_sm - k.r1_sm) / k.r1_sm
print("VPD up", round(np.average(dv > 0, weights=w) * 100, 1), "SM down", round(np.average(ds < 0, weights=w) * 100, 1))
dD = (k.r2_vpd - k.r2_sm) - (k.r1_vpd - k.r1_sm); print("dD>0", round(np.average(dD > 0, weights=w) * 100, 1))
b = pd.cut(lat[k.index], [-50,-35,-23.5,23.5,35,50,75]); print(k.groupby(b)[["r2","ra_sm","ra_vpd"]].mean().round(2)); print(k.groupby(b).size().to_dict())
print("frac n<expected", (d.nc < 40).sum())
