#!/usr/bin/env python
"""
Download Sentinel-2 L2A COGs via Earth-Search STAC API and build
before/after 2-band (B4, B8) composites over one or more AOIs, then
compute NDVI and generate patches consistent with the Hansen script.

Raw outputs per AOI:
  data/raw/sentinel2/<AOI>/s2_2018_dry_aoi.tif   (bands: B4, B8)
  data/raw/sentinel2/<AOI>/s2_2022_dry_aoi.tif   (bands: B4, B8)

Processed outputs per AOI:
  data/processed/sentinel2/<AOI>/s2_ndvi_2018_aoi.tif
  data/processed/sentinel2/<AOI>/s2_ndvi_2022_aoi.tif
  data/processed/sentinel2/<AOI>/patches_ndvi/patch_000000.npz  (NDVI 2018+2022, 32x32)
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import rasterio
import xarray as xr
import rioxarray  # noqa: F401  # needed for .rio accessor
import pystac_client
import stackstac

from config_aoi import AOIS, PATCH_SIZE, STRIDE, TARGET_PATCHES_MIN, TARGET_PATCHES_MAX

# --- Config ---

DATA_RAW = Path("data/raw/sentinel2")
DATA_PROCESSED = Path("data/processed/sentinel2")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Earth-Search STAC API (Sentinel-2 L2A COGs)
STAC_URL = "https://earth-search.aws.element84.com/v1"  # v1 API

STAC_COLLECTION = "sentinel-2-l2a"

# Use a fixed UTM zone for this region of Peru (around lon -70, southern hemisphere)
TARGET_EPSG = 32719  # UTM zone 19S

# Limit how many S2 scenes we stack to keep Dask / memory reasonable
MAX_ITEMS_PER_COMPOSITE = 40

from typing import Optional

def get_s2_median_composite_for_aoi(
    aoi_bounds,
    start_date: str,
    end_date: str,
    out_path: Path,
) -> Optional[Path]:
    """
    Build a median Sentinel-2 composite for given AOI bounds and date range,
    using red (≈B04) and nir (≈B08), and save as 2-band GeoTIFF.

    Returns:
      out_path if successful, or None if no items found.
    """
    print(f"[INFO] Building Sentinel-2 composite {start_date} to {end_date} for {out_path}")

    catalog = pystac_client.Client.open(STAC_URL)

    def search_with_cloud(max_cloud: Optional[int]):
        params = dict(
            collections=[STAC_COLLECTION],  # "sentinel-2-l2a"
            bbox=aoi_bounds,
            datetime=f"{start_date}/{end_date}",
        )
        if max_cloud is not None:
            params["query"] = {"eo:cloud_cover": {"lt": max_cloud}}

        s = catalog.search(**params)
        items_list = list(s.get_items())
        if items_list:
            if max_cloud is not None:
                print(
                    f"[INFO] Found {len(items_list)} items with cloud_cover < {max_cloud}"
                )
            else:
                print(f"[INFO] Found {len(items_list)} items without cloud filter")
        return items_list

    # 1) Try strict cloud filter first
    items = search_with_cloud(20)

    # 2) Relax cloud threshold if needed
    if not items:
        print("[WARN] No items with cloud_cover < 20, trying < 60...")
        items = search_with_cloud(60)

    # 3) Drop cloud filter completely if still none
    if not items:
        print("[WARN] Still no items, trying without cloud filter...")
        items = search_with_cloud(None)

    if not items:
        print(
            "[ERROR] No Sentinel-2 items found for AOI and date range "
            f"{start_date} to {end_date}."
        )
        return None

    # ---- NEW: sort by cloud_cover and keep only the best N items ----
    items_sorted = sorted(
        items,
        key=lambda it: it.properties.get("eo:cloud_cover", 100.0)
    )
    items_used = items_sorted[:MAX_ITEMS_PER_COMPOSITE]
    print(
        f"[INFO] Using top {len(items_used)} items (lowest cloud_cover) "
        f"out of {len(items)} total"
    )

    # Optional debug: check assets on the first item
    first = items_used[0]
    print("[DEBUG] First item id:", first.id)
    print("[DEBUG] First item assets (subset):", list(first.assets.keys())[:10])

    # Stack as xarray DataArray: dims = (time, band, y, x)
    da = stackstac.stack(
        items_used,
        # On sentinel-2-l2a the COG assets are named by common name, not B04/B08
        assets=["red", "nir"],          # red ≈ B04, nir ≈ B08
        bounds_latlon=aoi_bounds,       # (lon_min, lat_min, lon_max, lat_max) in WGS84
        epsg=TARGET_EPSG,               # reproject to UTM zone 19S
        resolution=10,                  # 10 m
        chunksize=2048,                 # keep Dask chunks reasonable
    )

    median = da.median(dim="time", keep_attrs=True).compute()
    median = median.assign_coords(band=["B4", "B8"]).transpose("band", "y", "x")
    median.rio.write_crs(f"EPSG:{TARGET_EPSG}", inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    median.rio.to_raster(out_path)
    print(f"[INFO] Saved composite to {out_path}")

    return out_path


def compute_ndvi_from_two_band_tif(src_path: Path, dest_path: Path) -> Path:
    """
    Read a 2-band GeoTIFF (B4, B8), compute NDVI, and save as single-band GeoTIFF.
    """
    with rasterio.open(src_path) as src:
        red = src.read(1).astype("float32")  # B4
        nir = src.read(2).astype("float32")  # B8
        profile = src.profile

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0).astype("float32")

    profile.update(count=1, dtype="float32")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    print(f"[INFO] Saved NDVI raster to {dest_path}")
    return dest_path


def generate_patches_from_ndvi_pair(
    ndvi_2018_path: Path,
    ndvi_2022_path: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    target_min: int = TARGET_PATCHES_MIN,
    target_max: int = TARGET_PATCHES_MAX,
) -> int:
    """
    Generate patches from stacked NDVI(2018, 2022) rasters, consistent
    with Hansen patching (32x32, overlapping, shuffled).

    Saves .npz files with:
      - 'ndvi': array of shape (2, H, W) -> [2018, 2022]
      - metadata: row, col, patch_size, years=[2018, 2022]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ndvi_2018_path) as src18, rasterio.open(ndvi_2022_path) as src22:
        ndvi18 = src18.read(1)
        ndvi22 = src22.read(1)

        if ndvi18.shape != ndvi22.shape:
            raise ValueError(
                f"Shape mismatch between NDVI 2018 {ndvi18.shape} and NDVI 2022 {ndvi22.shape}"
            )

        stacked = np.stack([ndvi18, ndvi22], axis=0)  # (2, H, W)
        _, height, width = stacked.shape

        coords = []
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                coords.append((row, col))

        coords = np.array(coords)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(coords)

        total_patches = 0

        for row, col in coords:
            if total_patches >= target_max:
                break

            row = int(row)
            col = int(col)

            patch = stacked[:, row:row + patch_size, col:col + patch_size]

            # Optional: skip if both years are essentially nodata
            # if np.isnan(patch).mean() > 0.5:
            #     continue

            patch_id = f"patch_{total_patches:06d}"
            out_path = out_dir / f"{patch_id}.npz"

            np.savez_compressed(
                out_path,
                ndvi=patch,
                row=row,
                col=col,
                patch_size=patch_size,
                years=np.array([2018, 2022]),
            )

            total_patches += 1

    print(f"[INFO] Generated {total_patches} Sentinel-2 NDVI patches at {out_dir}")

    if total_patches < target_min:
        print(
            f"[WARN] Only {total_patches} patches generated "
            f"(TARGET_PATCHES_MIN={target_min}). "
            f"Consider decreasing STRIDE or PATCH_SIZE or expanding AOI."
        )

    return total_patches


def run_for_aoi(aoi_key: str):
    """
    Run the full Sentinel-2 pipeline for a single AOI:
      - 2018 composite (B4/B8)
      - 2022 composite (B4/B8)
      - NDVI 2018 + 2022
      - NDVI patches (2018+2022 stacked)
    """
    if aoi_key not in AOIS:
        raise ValueError(f"Unknown AOI '{aoi_key}'. Available: {list(AOIS.keys())}")

    aoi = AOIS[aoi_key]
    bounds = aoi.bounds
    print(f"[INFO] Processing Sentinel-2 for AOI '{aoi.name}' with bounds {bounds}")

    # Directories per AOI
    raw_dir = DATA_RAW / aoi.name
    proc_dir = DATA_PROCESSED / aoi.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1) 2018 composite
    s2_2018_path = raw_dir / "s2_2018_dry_aoi.tif"
    if not s2_2018_path.exists():
        # primary window: 2018
        ok_path = get_s2_median_composite_for_aoi(
            bounds, "2018-01-01", "2018-12-31", s2_2018_path
        )
        if ok_path is None:
            print("[WARN] Falling back to wider date range 2016–2020 for '2018' composite")
            ok_path = get_s2_median_composite_for_aoi(
                bounds, "2016-01-01", "2020-12-31", s2_2018_path
            )
        if ok_path is None:
            print(f"[ERROR] Skipping AOI '{aoi.name}' – no data for 2018 even after fallback.")
            return  # abort this AOI cleanly
    else:
        print(f"[INFO] Found existing {s2_2018_path}")

    # 2) 2022 composite
    s2_2022_path = raw_dir / "s2_2022_dry_aoi.tif"
    s2_2022_path = raw_dir / "s2_2022_dry_aoi.tif"
    if not s2_2022_path.exists():
        ok_path = get_s2_median_composite_for_aoi(
            bounds, "2022-01-01", "2022-12-31", s2_2022_path
        )
        if ok_path is None:
            print("[WARN] Falling back to wider date range 2020–2024 for '2022' composite")
            ok_path = get_s2_median_composite_for_aoi(
                bounds, "2020-01-01", "2024-12-31", s2_2022_path
            )
        if ok_path is None:
            print(f"[ERROR] Skipping AOI '{aoi.name}' – no data for 2022 even after fallback.")
            return
    else:
        print(f"[INFO] Found existing {s2_2022_path}")

    # 3) NDVI rasters
    ndvi_2018_path = proc_dir / "s2_ndvi_2018_aoi.tif"
    ndvi_2022_path = proc_dir / "s2_ndvi_2022_aoi.tif"

    if not ndvi_2018_path.exists():
        compute_ndvi_from_two_band_tif(s2_2018_path, ndvi_2018_path)
    else:
        print(f"[INFO] Found existing {ndvi_2018_path}")

    if not ndvi_2022_path.exists():
        compute_ndvi_from_two_band_tif(s2_2022_path, ndvi_2022_path)
    else:
        print(f"[INFO] Found existing {ndvi_2022_path}")

    # 4) Generate NDVI patches (2018 + 2022 stacked)
    patches_dir = proc_dir / "patches_ndvi"
    total_patches = generate_patches_from_ndvi_pair(
        ndvi_2018_path=ndvi_2018_path,
        ndvi_2022_path=ndvi_2022_path,
        out_dir=patches_dir,
    )

    print(f"[DONE] AOI '{aoi.name}' – total Sentinel-2 NDVI patches: {total_patches}")


def main(aoi: str = "all"):
    if aoi == "all":
        for aoi_key in AOIS.keys():
            run_for_aoi(aoi_key)
    else:
        run_for_aoi(aoi)


if __name__ == "__main__":
    # Minimal CLI; consistent with Hansen’s `--aoi all|la_pampa|tambopata|...`
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Sentinel-2 composites, NDVI, and patches for one or more AOIs."
    )
    parser.add_argument(
        "--aoi",
        type=str,
        default="all",
        help="AOI key from config_aoi.AOIS or 'all'",
    )
    args = parser.parse_args()
    main(aoi=args.aoi)
