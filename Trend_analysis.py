"""
This code computes and visualizes the multi-year trend of river ice duration. To estimate trends for the freeze-up date and breakup date, simply change the input and output paths.
"""
import os
import re
import math
from glob import glob
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from scipy.stats import theilslopes, kendalltau
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from tqdm import tqdm
import geopandas as gpd
from PIL import Image

# —— Vector-friendly settings ——
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 15,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "image.composite_image": False,
})

# -----------------------------
# User settings
# -----------------------------
INPUT_DIR  = r"xxx/Kolyma_ice_duration_DAY"    ###Path to the annual river ice duration; the corresponding data have been shared.
FNAME_RE   = re.compile(r"ice_duration_DAY_(\d{4})\.tif$")
BASIN_SHP  = r"E:/research data/Arctic_rivers/All_six_rivers/Basins_3571.shp"    ### If needed, I can share this shapefile, or you can directly access and download the river basin shapefiles from https://zenodo.org/records/1297434.
OUTPUT_DIR = os.path.join(INPUT_DIR, "trend_outputs2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGEND_BOX = [0.46, 0.41, 0.20, 0.18]    ###Absolute position attribute, which can be adjusted according to your specific situation.

# Colors (RGB)
COLOR_LIGHT_BLUE = "#91C8E4"  # Increase (light blue)
COLOR_DARK_BLUE  = "#3A59D1"  # Significant increase
COLOR_LIGHT_RED  = "#ff8f8f"  # Decrease (light red)
COLOR_DARK_RED   = "#ff0000"  # Significant decrease
COLOR_BASIN_FILL = "#EEEEEE"
COLOR_BASIN_EDGE = "#DDDDDD"

ALPHA = 0.05
MIN_VALID_YEARS_ABS  = 8
MIN_VALID_YEARS_FRAC = 0.6

# -----------------------------
# Helpers
# -----------------------------
def list_yearly_files(folder: str) -> List[Tuple[int, str]]:
    files = []
    for fp in glob(os.path.join(folder, "*.tif")):
        m = FNAME_RE.search(os.path.basename(fp))
        if m:
            files.append((int(m.group(1)), fp))
    if not files:
        raise FileNotFoundError("No matching TIFF files found in the folder.")
    files.sort(key=lambda x: x[0])
    return files

def read_stack(files: List[Tuple[int, str]]):
    years = [y for y, _ in files]
    with rasterio.open(files[0][1]) as src0:
        profile = src0.profile.copy()
        H, W = src0.height, src0.width
    stack = np.full((len(files), H, W), np.nan, dtype=np.float32)
    for _, fp in tqdm(files, desc="Reading rasters"):
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            stack[len(years) - len(files)] = arr
    stack = np.stack([rasterio.open(fp).read(1).astype(np.float32) for _, fp in files])
    with rasterio.open(files[0][1]) as src0:
        prof = src0.profile.copy()
    return years, stack, prof

def compute_trend_maps(years: List[int], stack: np.ndarray, alpha: float):
    n, H, W = stack.shape
    years_arr = np.asarray(years, dtype=np.float32)
    min_valid = max(MIN_VALID_YEARS_ABS, int(math.ceil(MIN_VALID_YEARS_FRAC * n)))

    sen_slope = np.full((H, W), np.nan, dtype=np.float32)
    p_value   = np.full((H, W), np.nan, dtype=np.float32)
    trend_cls = np.zeros((H, W), dtype=np.int16)

    flat_stack = stack.reshape(n, -1)
    for idx in tqdm(range(flat_stack.shape[1]), desc="Computing trends"):
        y = flat_stack[:, idx]
        msk = ~np.isnan(y)
        if msk.sum() < min_valid:
            continue
        yv = y[msk].astype(np.float64)
        xv = years_arr[msk].astype(np.float64)
        try:
            slope, _, _, _ = theilslopes(yv, xv)
        except Exception:
            x0 = xv - xv.mean()
            denom = (x0 * x0).sum()
            slope = ((x0 * yv).sum() / denom) if denom > 0 else np.nan
        try:
            _, p = kendalltau(xv, yv, nan_policy='omit')
        except Exception:
            p = np.nan

        sen_slope.ravel()[idx] = np.float32(slope)
        p_value.ravel()[idx]   = np.float32(p)
        if not np.isnan(slope) and not np.isnan(p):
            if slope > 0:
                trend_cls.ravel()[idx] = 2 if p < alpha else 1
            elif slope < 0:
                trend_cls.ravel()[idx] = -2 if p < alpha else -1
            else:
                trend_cls.ravel()[idx] = 0
    return sen_slope, p_value, trend_cls

def write_raster(path: str, arr: np.ndarray, ref_profile: dict, dtype, nodata_val=None):
    profile = ref_profile.copy()
    profile.update({"count": 1, "dtype": np.dtype(dtype).name})
    if nodata_val is not None:
        profile["nodata"] = nodata_val
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)

def make_extent_and_crs(example_tif: str):
    with rasterio.open(example_tif) as src:
        extent = plotting_extent(src)
        crs    = src.crs
    return extent, crs

def load_basins_to_raster_crs(shp_path: str, raster_crs):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(raster_crs)
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf["geometry"] = gdf.buffer(0)
    except Exception:
        pass
    return gdf

# clip path
def _poly_to_path(poly):
    ext_x, ext_y = poly.exterior.coords.xy
    verts = list(zip(ext_x, ext_y))
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    for ring in poly.interiors:
        x, y = ring.coords.xy
        v = list(zip(x, y))
        verts += v
        codes += [Path.MOVETO] + [Path.LINETO]*(len(v)-2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

def make_clip_patch_from_gdf(ax, gdf):
    geom = gdf.geometry.unary_union
    if geom.geom_type == "Polygon":
        path = _poly_to_path(geom)
    else:
        verts, codes = [], []
        for poly in geom.geoms:
            p = _poly_to_path(poly)
            verts.extend(p.vertices); codes.extend(p.codes)
        path = Path(verts, codes)
    return PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')

# -----------------------------
# Plotting
# -----------------------------
def build_trend_figure(trend_cls: np.ndarray, example_tif: str, legend_box=LEGEND_BOX):
    extent, raster_crs = make_extent_and_crs(example_tif)
    basins = load_basins_to_raster_crs(BASIN_SHP, raster_crs)

    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    cmap = ListedColormap([
        COLOR_DARK_RED,     # -2  Significant decrease
        COLOR_LIGHT_RED,    # -1  Decrease
        COLOR_BASIN_FILL,   #  0  Same as background color
        COLOR_LIGHT_BLUE,   # +1  Increase
        COLOR_DARK_BLUE,    # +2  Significant increase
    ])
    norm = BoundaryNorm(bounds, cmap.N)

    draw_arr = trend_cls.astype(float)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("white")

    basins.plot(ax=ax, facecolor=COLOR_BASIN_FILL, edgecolor='none', linewidth=0, zorder=1)

    im = ax.imshow(draw_arr, origin="upper", extent=extent,
                   cmap=cmap, norm=norm, interpolation="nearest", zorder=2)
    im.set_alpha(None)
    im.set_rasterized(True)
    im.set_clip_path(make_clip_patch_from_gdf(ax, basins))  # Clip to basin boundaries

    basins.boundary.plot(ax=ax, edgecolor=COLOR_BASIN_EDGE, linewidth=1.0, zorder=3)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    ax_leg = fig.add_axes(legend_box); ax_leg.axis("off")
    entries = [
        ("Increase",                      COLOR_LIGHT_BLUE),
        ("Decrease",                      COLOR_LIGHT_RED),
        ("Significant increase (p<0.05)", COLOR_DARK_BLUE),
        ("Significant decrease (p<0.05)", COLOR_DARK_RED),
    ]
    y0, dy, size = 0.8, 0.22, 0.12
    for i, (lab, col) in enumerate(entries):
        y = y0 - i*dy
        ax_leg.add_patch(plt.Rectangle((0.0, y - size/2), 0.18, size,
                                       transform=ax_leg.transAxes,
                                       facecolor=col, edgecolor="none"))
        ax_leg.text(0.22, y, lab, transform=ax_leg.transAxes,
                    va="center", ha="left")
    return fig

# -----------------------------
# Export
# -----------------------------
def export_vector_pdf(fig, out_pdf_vector: str, raster_dpi=600):
    fig.savefig(out_pdf_vector, dpi=raster_dpi, bbox_inches='tight', pad_inches=0,
                transparent=False, facecolor='white')

def export_safe_bitmap_pdf(fig, out_pdf_safe: str, dpi=600):
    tmp = out_pdf_safe.replace(".pdf", ".__tmp_rgb.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False, facecolor='white')
    with Image.open(tmp) as im:
        im.convert("RGB").save(out_pdf_safe, "PDF", resolution=dpi)
    try: os.remove(tmp)
    except Exception: pass

def export_png(fig, out_png: str, dpi=300):
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False, facecolor='white')

# -----------------------------
# Main
# -----------------------------
def main():
    files = list_yearly_files(INPUT_DIR)
    print(f"Found {len(files)} yearly rasters: {files[0][0]}–{files[-1][0]}")
    years, stack, profile = read_stack(files)
    print(f"Stack: {stack.shape} (years, rows, cols)")

    sen_slope, p_value, trend_cls = compute_trend_maps(years, stack, ALPHA)

    write_raster(os.path.join(OUTPUT_DIR, "sen_slope_days_per_year.tif"), sen_slope, profile, np.float32, np.nan)
    write_raster(os.path.join(OUTPUT_DIR, "mk_pvalue.tif"),               p_value,   profile, np.float32, np.nan)
    write_raster(os.path.join(OUTPUT_DIR, "trend_class.tif"),             trend_cls, profile, np.int16,   0)

    fig = build_trend_figure(trend_cls, files[0][1], legend_box=LEGEND_BOX)

    out_png        = os.path.join(OUTPUT_DIR, "trend_map.png")
    out_pdf_vector = os.path.join(OUTPUT_DIR, "trend_map_vector.pdf")
    out_pdf_safe   = os.path.join(OUTPUT_DIR, "trend_map_safe.pdf")

    export_png(fig, out_png, dpi=300)
    export_vector_pdf(fig, out_pdf_vector, raster_dpi=600)
    export_safe_bitmap_pdf(fig, out_pdf_safe, dpi=600)

    plt.close(fig)
    print("Saved:", out_png)
    print("Saved:", out_pdf_vector)
    print("Saved:", out_pdf_safe)

if __name__ == "__main__":
    main()
