# MCD12C1 2001 IGBP 0.05 deg -> 0.5 deg mode, merged into the paper's 9 classes + grid area (run on xuanze)
import numpy as np, xarray as xr
from osgeo import gdal
f = "/data/data_research_backup/师姐/data/land_use/MCD12C1.A2001001.006.2018053185512.hdf"
a = gdal.Open(f'HDF4_EOS:EOS_GRID:"{f}":MOD12C1:Majority_Land_Cover_Type_1').ReadAsArray()  # 3600x7200, 90N->90S
a = a[::-1]                                  # ascending lat
b = a.reshape(360, 10, 720, 10).transpose(0, 2, 1, 3).reshape(360, 720, 100)
cnt = np.stack([(b == k).sum(-1) for k in range(18)], -1)
mode = cnt.argmax(-1)
lat = np.arange(-89.75, 90, 0.5); lon = np.arange(-179.75, 180, 0.5)
merge = {1: 1, 3: 1, 2: 2, 4: 5, 5: 5, 6: 6, 7: 6, 8: 8, 9: 8, 10: 10, 12: 12, 14: 12}
lc = np.full(mode.shape, np.nan)
for k, v in merge.items():
    lc[mode == k] = v
la = np.broadcast_to(lat[:, None], lc.shape)
lc[(lc == 6) & (la > 50)] = 7                # boreal shrubland
lc[(lc == 8) & (la > 50)] = 9                # boreal savanna
R = 6371007.2
area = (np.sin(np.deg2rad(lat + .25)) - np.sin(np.deg2rad(lat - .25))) * np.deg2rad(.5) * R ** 2
area = np.broadcast_to(area[:, None], lc.shape).copy()
out = "/data/lhwang/vpd_sm_repro/proc/"
xr.Dataset({"Band1": (("lat", "lon"), lc)}, coords={"lat": lat, "lon": lon}).to_netcdf(out + "LC_2001_MCD12C1_IGBP_Type1_mode_05deg.nc")
xr.Dataset({"area": (("lat", "lon"), area)}, coords={"lat": lat, "lon": lon}).to_netcdf(out + "grid_area_0.5deg.nc")
s = (la >= -50) & (la <= 75) & ~np.isnan(lc)
tot = area[s].sum()
for k in [1, 2, 5, 6, 7, 8, 9, 10, 12]:
    print(k, round(area[s & (lc == k)].sum() / tot * 100, 1), "%")
print("cells", s.sum())
