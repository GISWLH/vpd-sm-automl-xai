# NDVI -> monthly 0.5 deg, STL residuals for all series, growing-season mask, climatology/CV metrics (run on xuanze)
import glob, os, zipfile, io, sys
import numpy as np
import xarray as xr
import rasterio
from multiprocessing import Pool
from statsmodels.tsa.seasonal import STL

P = "/data/lhwang/vpd_sm_repro/proc/"
RAW = "/data/lhwang/vpd_sm_repro/raw/pku_ndvi/"
LAT = np.arange(-49.75, 75, 0.5); LON = np.arange(-179.75, 180, 0.5)
year = np.repeat(np.arange(1982, 2023), 12); month = np.tile(np.arange(1, 13), 41)


def read_half(a):
    zf, name = a
    with zipfile.ZipFile(zf) as z:
        with rasterio.MemoryFile(z.read(name)) as m, m.open() as d:
            x = d.read(1).astype("f4")
            q = d.read(2).astype("i4")
    x[(x > 1000) | (x < 0)] = np.nan
    # QC digits: 2nd = GIMMS (2 snow/cloud), 3rd = MODIS (2 snow/ice, 3 cloudy); both eras drop snow/cloud only
    x[((q // 10 % 10) == 2) | np.isin(q % 10, (2, 3))] = np.nan
    n = np.isfinite(x)
    v = np.nan_to_num(x).reshape(360, 6, 720, 6).sum((1, 3), dtype="f8")
    c = n.reshape(360, 6, 720, 6).sum((1, 3), dtype="f8")
    return name, v.astype("f4"), c.astype("f4")   # sums on a 0.5 deg global grid, 90N first


def ndvi_monthly():
    f = P + "ndvi_monthly.npy"
    if os.path.exists(f):
        return np.load(f)
    items = []
    for zf in sorted(glob.glob(RAW + "*consolidated*.zip")):
        with zipfile.ZipFile(zf) as z:
            items += [(zf, n) for n in z.namelist() if n.lower().endswith(".tif")]
    print("half-month tifs", len(items), flush=True)
    assert len(items) == 984, len(items)
    sv = np.zeros((492, 360, 720), "f8"); sc = np.zeros((492, 360, 720), "f8")
    with Pool(8) as p:
        for k, (name, v, c) in enumerate(p.imap_unordered(read_half, items)):
            s = os.path.basename(name).split("_")[-1].split(".")[0]    # YYYYMM + half tag
            y, m = int(s[:4]), int(s[4:6])
            t = (y - 1982) * 12 + m - 1
            sv[t] += v; sc[t] += c
            if k % 100 == 0:
                print(k, flush=True)
    a = sv / np.where(sc > 0, sc, np.nan) / 1000.0
    a = a[:, ::-1][:, 80:330].astype("f4")              # ascending; lat -50..75
    np.save(f, a)
    return a


def stl_cell(x):
    ok = np.isfinite(x)
    if ok.sum() < 120:
        return np.full(x.shape, np.nan, "f4")
    y = x.astype("f8").copy()
    clim = np.array([np.nanmean(y[m::12]) if np.isfinite(y[m::12]).any() else np.nanmean(y) for m in range(12)])
    bad = ~ok
    y[bad] = clim[(month - 1)[bad]]
    r = STL(y, period=12).fit().resid
    r[bad] = np.nan
    return r.astype("f4")


def stl_block(a):
    b = np.empty(a.shape, "f4")
    for c in range(a.shape[1]):
        b[:, c] = stl_cell(a[:, c])
    return b


if __name__ == "__main__":
    ndvi = ndvi_monthly()
    print("ndvi", ndvi.shape, np.nanmean(ndvi), flush=True)
    e = np.load(P + "era5_all.npz")
    lc = xr.open_dataset(P + "LC_2001_MCD12C1_IGBP_Type1_mode_05deg.nc").Band1.sel(lat=slice(-50, 75)).values
    clim_t = np.stack([e["tas"][m::12].mean(0) for m in range(12)])
    gs = clim_t > 5
    has_gs = gs.any(0)

    # pad 1 cell around target cells so that 3x3 neighbours are available
    tgt = np.isfinite(lc) & has_gs & np.isfinite(e["sm"]).all(0) & (np.isfinite(ndvi).sum(0) >= 120)
    near = tgt.copy()
    near[1:] |= tgt[:-1]; near[:-1] |= tgt[1:]
    near |= np.roll(near, 1, 1) | np.roll(near, -1, 1)
    near[1:] |= near[:-1].copy(); near[:-1] |= near[1:].copy()
    idx = np.argwhere(near)
    print("targets", tgt.sum(), "with neighbours", near.sum(), flush=True)

    fields = {"ndvi": ndvi, "sm": e["sm"], "vpd": e["vpd"], "tas": e["tas"], "swr": e["swr"]}
    for k, a in fields.items():
        out = np.lib.format.open_memmap(P + f"res_{k}.npy", "w+", "f4", a.shape)
        out[:] = np.nan
        v = a[:, idx[:, 0], idx[:, 1]]
        blocks = [v[:, s] for s in np.array_split(np.arange(v.shape[1]), 64)]
        with Pool(8) as p:
            r = np.concatenate(p.map(stl_block, blocks), 1)
        out[:, idx[:, 0], idx[:, 1]] = r
        out.flush()
        print("stl", k, flush=True)

    np.savez(P + "anomalies.npz", gs=gs, cells=np.argwhere(tgt), year=year, month=month, lat=LAT, lon=LON)

    # climatological means and CV of normalised (STL residual) values, growing season only
    def metric(name, tmask):
        g = gs[month - 1] & tmask[:, None, None]
        out = {}
        for v in ["sm", "vpd"]:
            raw = np.where(g, e[v], np.nan)
            rs = np.where(g, np.load(P + f"res_{v}.npy", mmap_mode="r"), np.nan)
            out[v + "_mean"] = np.nanmean(raw, 0)
            out[v + "_nor_cv"] = np.nanstd(rs, 0) / out[v + "_mean"]
        return out
    A = metric("all", np.ones(492, bool)); A1 = metric("pre", year <= 2001); A2 = metric("post", year >= 2002)
    for v in ["sm", "vpd"]:
        A[v + "_mean_change"] = A2[v + "_mean"] - A1[v + "_mean"]
        A[v + "_nor_cv_change"] = A2[v + "_nor_cv"] - A1[v + "_nor_cv"]
    np.savez(P + "metrics.npz", **A)
    print("done", flush=True)
