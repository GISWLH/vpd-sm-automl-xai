# ERA5 daily 0.25 deg -> monthly 0.5 deg VPD / SM / TAS / SWR, 1982-2022 (run on xuanze)
import os, sys
import numpy as np
import xarray as xr
from multiprocessing import Pool

SRC = "/data/lhwang/ERA5_raw/ERA5_1940_2023/surface"
OUT = "/data/lhwang/vpd_sm_repro/proc/era5_monthly"
os.makedirs(OUT, exist_ok=True)
YM = [(y, m) for y in range(1982, 2023) for m in range(1, 13)]
LAT = np.arange(-49.75, 75, 0.5)
LON = np.arange(-179.75, 180, 0.5)


def read(var, name, y, m):
    with xr.open_dataset(f"{SRC}/{var}/ERA5_{y}{m:02d}.nc") as ds:
        return ds[name].values.astype("f8")


def to_ascending(a):
    # (..., 721, 1440), lat 90..-90, lon 0..359.75 -> lat -90..90, lon -180..179.75
    return np.roll(a[..., ::-1, :], 720, axis=-1)


def coarsen(a, w=None):
    # 0.5 deg cell centred on a 0.25 deg node: [1,2,1]/4 in each direction, normalised by valid weight
    w = np.ones_like(a) if w is None else w.astype("f8")
    x = a * w
    def k(z):
        z = (np.roll(z, 1, 1) + 2 * z + np.roll(z, -1, 1)) / 4
        z[1:-1] = (z[:-2] + 2 * z[1:-1] + z[2:]) / 4
        return z
    c = k(x) / np.where(k(w) > 0, k(w), np.nan)
    return c[161:660:2, 1::2]


def month(ym):
    y, m = ym
    f = f"{OUT}/{y}{m:02d}.npz"
    if os.path.exists(f):
        return f
    t = read("2m_temperature", "t2m", y, m) - 273.15
    d = read("2m_dewpoint_temperature", "d2m", y, m) - 273.15
    es = 6.108 * np.exp(17.27 * t / (t + 237.3))
    ea = 6.108 * np.exp(17.27 * d / (d + 237.3))
    vpd = to_ascending(np.clip(es - ea, 0, None).mean(0))
    tas = to_ascending(t.mean(0))
    sm = to_ascending(read("volumetric_soil_water_layer", "swvl", y, m).mean(0))
    swr = to_ascending(read("mean_surface_net_short_wave_radiation_flux", "avg_snswrf", y, m).mean(0))
    land = sm > 1e-4  # swvl is ~0 over sea; keep coastal 0.5 deg cells land-only
    np.savez(f, vpd=coarsen(vpd).astype("f4"), tas=coarsen(tas).astype("f4"),
             sm=coarsen(sm, land).astype("f4"), swr=coarsen(swr).astype("f4"))
    return f


if __name__ == "__main__":
    ym = YM[:1] if len(sys.argv) > 1 and sys.argv[1] == "test" else YM
    with Pool(8) as p:
        for i, f in enumerate(p.imap_unordered(month, ym), 1):
            if i % 12 == 0 or i == len(ym):
                print(i, "/", len(ym), f, flush=True)
    names = ["vpd", "tas", "sm", "swr"]
    if len(ym) == len(YM):
        s = {k: [] for k in names}
        for y, m in YM:
            z = np.load(f"{OUT}/{y}{m:02d}.npz")
            for k in names:
                s[k].append(z[k])
        np.savez("/data/lhwang/vpd_sm_repro/proc/era5_all.npz", lat=LAT, lon=LON,
                 **{k: np.stack(v) for k, v in s.items()})
        print("done")
