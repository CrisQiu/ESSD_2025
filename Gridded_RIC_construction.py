import os
import math
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from joblib import Parallel, delayed
from contextlib import contextmanager

# --------------------
# Basic parameters
# --------------------
input_dir = r"E:/research data/Arctic_rivers/YeniseyRiver/YeniseyRiverIce_MODIS_CloudRemoved"   ####Path to the daily mapped river ice (xxx.tif)
output_dir_ric = r"E:/research data/Arctic_rivers/YeniseyRiver/Gridded_RIC_3km_whole_year2"         ####Path to the daily gridded 3km RIC data
output_dir_wc  = r"E:/research data/Arctic_rivers/YeniseyRiver/Gridded_WC_3km_whole_year2"
os.makedirs(output_dir_ric, exist_ok=True)
os.makedirs(output_dir_wc, exist_ok=True)

pixel_size = 500    
large_grid_size = 3000  
factor = large_grid_size // pixel_size 

# --------------------
# Environment variables: enable GDAL multithreading
# --------------------
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("NUM_THREADS", "ALL_CPUS")

# --------------------
# Utility functions
# --------------------
def pad_to_factor(arr, factor):
    """Pad the array edges to multiples of 'factor' with NaN for convenient reshaping during aggregation."""
    h, w = arr.shape
    rh = (math.ceil(h / factor) * factor) - h
    rw = (math.ceil(w / factor) * factor) - w
    if rh == 0 and rw == 0:
        return arr
    out = np.full((h + rh, w + rw), np.nan, dtype=arr.dtype)
    out[:h, :w] = arr
    return out

def aggregate_proportions(data, factor=6):
    """
    data: 2D array, pixel values are {0 (water), 10 (cloud), 255 (ice)} or NaN (invalid)
    Returns: proportions of RIC, WC, and CC (normalized by the number of valid pixels),
             with shape (H//factor, W//factor)
    """
    # Valid pixels (not NaN)
    valid = ~np.isnan(data)

    # Construct boolean masks and convert to float32 (for summation)
    m_ice   = (data == 255).astype(np.float32)
    m_water = (data == 0).astype(np.float32)
    m_cloud = (data == 10).astype(np.float32)
    m_valid = valid.astype(np.float32)

    # Pad edges to multiples of 'factor' to avoid looping over small windows
    m_ice   = pad_to_factor(m_ice,   factor)
    m_water = pad_to_factor(m_water, factor)
    m_cloud = pad_to_factor(m_cloud, factor)
    m_valid = pad_to_factor(m_valid, factor)

    H, W = m_ice.shape
    fh, fw = factor, factor

    # Reshape to 4D: (H/f, f, W/f, f), and sum within each block
    def block_sum(x):
        x4 = x.reshape(H // fh, fh, W // fw, fw)
        return x4.sum(axis=(1, 3))

    ice_cnt   = block_sum(m_ice)
    water_cnt = block_sum(m_water)
    cloud_cnt = block_sum(m_cloud)
    valid_cnt = block_sum(m_valid)

    # Normalize by valid pixel count to obtain proportions
    with np.errstate(invalid="ignore", divide="ignore"):
        ric = ice_cnt   / valid_cnt
        wc  = water_cnt / valid_cnt
        cc  = cloud_cnt / valid_cnt

    # Set NaN where valid pixel count equals zero
    ric[valid_cnt == 0] = np.nan
    wc[valid_cnt == 0]  = np.nan
    cc[valid_cnt == 0]  = np.nan

    return ric.astype(np.float32), wc.astype(np.float32), cc.astype(np.float32)

def reclassify_cloud_vectorized(ric, wc, cc):
    """
    Fully vectorized cloud post-classification:
      - If RIC > 0.8: add CC to RIC
      - elif WC > 0.8: add CC to WC
      - else: distribute CC proportionally according to ric/(ric+wc)
    """
    # Valid mask: all three are not NaN
    valid = ~np.isnan(ric) & ~np.isnan(wc) & ~np.isnan(cc)
    if not np.any(valid):
        return ric, wc, cc

    cond1 = valid & (ric > 0.8)
    cond2 = valid & (~cond1) & (wc > 0.8)
    cond3 = valid & (~cond1) & (~cond2)

    # Condition 1
    ric[cond1] = ric[cond1] + cc[cond1]
    cc[cond1] = 0.0

    # Condition 2
    wc[cond2] = wc[cond2] + cc[cond2]
    cc[cond2] = 0.0

    # Condition 3: proportional distribution
    total = ric + wc
    total[~cond3] = 1.0  # avoid division by zero in irrelevant positions
    ratio = np.zeros_like(ric, dtype=np.float32)
    ratio[cond3] = ric[cond3] / total[cond3]
    ratio = np.clip(ratio, 0.0, 1.0)

    ric[cond3] = ric[cond3] + cc[cond3] * ratio[cond3]
    wc[cond3]  = wc[cond3]  + cc[cond3] * (1.0 - ratio[cond3])
    cc[cond3]  = 0.0

    return ric, wc, cc

def rolling_nanmean_time(series, window=10):
    """
    Compute a NaN-safe rolling mean along the time dimension (axis=0).
    series: (T, H, W) float32
    """
    T = series.shape[0]
    half = window // 2

    # Cumulative sums of values and valid counts (treat NaN as 0)
    data = np.nan_to_num(series, nan=0.0).astype(np.float32)
    valid = (~np.isnan(series)).astype(np.float32)

    # Prefix sums (shape (T+1, H, W) for convenient interval summation)
    csum = np.concatenate([np.zeros((1, *data.shape[1:]), dtype=np.float32), np.cumsum(data, axis=0)], axis=0)
    vsum = np.concatenate([np.zeros((1, *valid.shape[1:]), dtype=np.float32), np.cumsum(valid, axis=0)], axis=0)

    out = np.empty_like(series, dtype=np.float32)
    for t in range(T):
        s = max(0, t - half)
        e = min(T - 1, t + half)
        # Since csum has one prepended row, interval [s, e] sum = csum[e+1] - csum[s]
        num = csum[e + 1] - csum[s]
        den = vsum[e + 1] - vsum[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = num / den
        out[t][den == 0] = np.nan
    return out

@contextmanager
def rasterio_profile_like(src, grid_h, grid_w, scale_factor):
    meta = src.meta.copy()
    # Update resolution and affine transformation
    transform = src.transform
    new_transform = Affine(
        transform.a * scale_factor, transform.b, transform.c,
        transform.d, transform.e * scale_factor, transform.f
    )
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "compress": "lzw",  # alternatively "DEFLATE" or "ZSTD"
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "height": grid_h,
        "width": grid_w,
        "transform": new_transform,
        "BIGTIFF": "IF_SAFER"
    })
    yield meta

def process_one_file(tif_path):
    """
    Read one 500 m tif, aggregate to 3 km in memory using vectorized operations,
    and return (ric, wc, cc, meta_like_info)
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        ric, wc, cc = aggregate_proportions(data, factor=factor)

        # Generate output metadata (based on input transform and 3 km scale)
        grid_h, grid_w = ric.shape
        with rasterio_profile_like(src, grid_h, grid_w, scale_factor=factor) as meta:
            return ric, wc, cc, meta

def write_one_day(ric, wc, base_name, meta):
    """Write RIC and WC (Float32) outputs."""
    out_ric = os.path.join(output_dir_ric, f"{base_name}_RIC.tif")
    out_wc  = os.path.join(output_dir_wc,  f"{base_name}_WC.tif")
    with rasterio.open(out_ric, "w", **meta) as dst:
        dst.write(ric, 1)
    with rasterio.open(out_wc, "w", **meta) as dst:
        dst.write(wc, 1)

# --------------------
# Main workflow
# --------------------
file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".tif")])
if len(file_list) == 0:
    raise RuntimeError("No .tif files found in the input directory.")

# 1) Parallel reading and aggregation (each file is independent, CPU-friendly)
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(process_one_file)(fp) for fp in file_list
)

# Unpack results
ric_list, wc_list, cc_list, meta_list = zip(*results)
ric_series = np.stack(ric_list, axis=0)  # (T, H, W)
wc_series  = np.stack(wc_list,  axis=0)
cc_series  = np.stack(cc_list,  axis=0)

# 2) Rolling mean along the time dimension (O(T))
ric_sm = rolling_nanmean_time(ric_series, window=10)
wc_sm  = rolling_nanmean_time(wc_series,  window=10)
cc_sm  = rolling_nanmean_time(cc_series,  window=10)

# 3) Cloud post-classification (vectorized)
#    Note: processed day by day to avoid excessive memory usage
for i, tif_path in enumerate(file_list):
    base = os.path.splitext(os.path.basename(tif_path))[0]
    ric = ric_sm[i].copy()
    wc  = wc_sm[i].copy()
    cc  = cc_sm[i].copy()

    ric, wc, _ = reclassify_cloud_vectorized(ric, wc, cc)
    meta = meta_list[i]
    write_one_day(ric, wc, base, meta)

print("Processing completed, results saved!")
