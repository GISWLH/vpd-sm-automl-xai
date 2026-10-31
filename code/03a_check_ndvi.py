import zipfile, glob, re, collections
for zf in sorted(glob.glob("/data/lhwang/vpd_sm_repro/raw/pku_ndvi/*consolidated*.zip")):
    n = zipfile.ZipFile(zf).namelist()
    t = [x for x in n if x.lower().endswith(".tif")]
    print(zf[-13:], len(n), len(t), t[:3], t[-1])
    c = collections.Counter(re.search(r"_(\d{6})", x.split("/")[-1].replace("V1.2_", "")).group(1) for x in t)
    print(" months", len(c), "counts", set(c.values()))
