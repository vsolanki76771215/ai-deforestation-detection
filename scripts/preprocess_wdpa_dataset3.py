#!/usr/bin/env python
"""
Dataset 3 preprocessing: WDPA protected areas for Madre de Dios AOI.

- Reads WDPA shapefile exported from Google Earth Engine
- Fixes text encoding (latin1 / windows-1252)
- Clips to AOI (La Pampa / Tambopata buffer zone)
- Saves a standardized GeoPackage (UTF-8) and a sample CSV

Inputs (expected):
    data/raw/wdpa_aoi.shp  (+ .dbf, .shx, .prj)

Outputs:
    data/processed/wdpa_aoi_clean.gpkg   (layer: wdpa_aoi)
    data/processed/wdpa_aoi_sample.csv
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

# --- Config ---

# AOI: La Pampa / Tambopata buffer zone (Madre de Dios, Peru)
# (lon_min, lat_min, lon_max, lat_max)
AOI_BOUNDS = (-70.6, -13.3, -69.9, -12.7)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

WDPA_SHP = DATA_RAW / "wdpa_aoi.shp"
GPKG_OUT = DATA_PROCESSED / "wdpa_aoi_clean.gpkg"
CSV_SAMPLE = DATA_PROCESSED / "wdpa_aoi_sample.csv"


def load_wdpa_with_encoding(path: Path) -> gpd.GeoDataFrame:
    """
    Try loading WDPA shapefile with several common encodings until one works.
    This fixes UnicodeDecodeError issues from non-UTF-8 DBF files.
    """
    encodings_to_try = ["utf-8", "latin1", "iso-8859-1", "windows-1252"]
    last_error = None

    for enc in encodings_to_try:
        try:
            print(f"[INFO] Trying encoding: {enc}")
            gdf = gpd.read_file(path, encoding=enc)
            print(f"[INFO] Successfully loaded with encoding: {enc}")
            return gdf
        except Exception as e:
            print(f"[WARN] Failed with encoding {enc}: {e}")
            last_error = e

    raise RuntimeError(
        f"Could not read {path} with any tried encoding. Last error: {last_error}"
    )


def clip_to_aoi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip WDPA polygons to AOI rectangle and return the subset."""
    lon_min, lat_min, lon_max, lat_max = AOI_BOUNDS
    aoi_geom = box(lon_min, lat_min, lon_max, lat_max)

    # Ensure CRS is WGS84
    if gdf.crs is None:
        print("[INFO] No CRS detected; assuming EPSG:4326 (WGS84).")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        print(f"[INFO] Reprojecting from {gdf.crs} to EPSG:4326.")
        gdf = gdf.to_crs("EPSG:4326")

    # Clip using intersection
    print("[INFO] Clipping WDPA to AOI bounds...")
    # Using intersects & filter for simplicity
    gdf_aoi = gdf[gdf.intersects(aoi_geom)].copy()
    print(f"[INFO] {len(gdf_aoi)} features intersect AOI.")

    return gdf_aoi


def save_outputs(gdf_aoi: gpd.GeoDataFrame):
    """Save cleaned AOI WDPA as GeoPackage + small sample CSV."""
    if gdf_aoi.empty:
        print("[WARN] No WDPA features within AOI. Saving empty outputs anyway.")
    else:
        print("[INFO] Saving cleaned WDPA AOI to GeoPackage...")

    # 1) Save as GeoPackage (UTF-8, robust for future reads)
    GPKG_OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf_aoi.to_file(GPKG_OUT, driver="GPKG", layer="wdpa_aoi")
    print(f"[INFO] Saved GeoPackage: {GPKG_OUT}")

    # 2) Save a small CSV sample of attributes
    # Common WDPA fields (if present)
    candidate_cols = [
        "WDPAID", "WDPA_PID", "NAME", "DESIG", "DESIG_ENG",
        "ISO3", "STATUS", "STATUS_YR", "REP_AREA", "MARINE"
    ]
    cols = [c for c in candidate_cols if c in gdf_aoi.columns]

    if cols:
        sample = gdf_aoi[cols].head(10)
    else:
        # Fallback: just dump first 10 rows of whatever columns exist
        sample = gdf_aoi.head(10)

    sample.to_csv(CSV_SAMPLE, index=False)
    print(f"[INFO] Saved sample CSV: {CSV_SAMPLE}")


def main():
    if not WDPA_SHP.exists():
        raise FileNotFoundError(
            f"Expected WDPA shapefile not found at {WDPA_SHP}. "
            "Make sure you downloaded wdpa_aoi.shp from Earth Engine "
            "and placed it in data/raw/."
        )

    print(f"[INFO] Loading WDPA shapefile from {WDPA_SHP}")
    gdf = load_wdpa_with_encoding(WDPA_SHP)

    print(f"[INFO] Loaded {len(gdf)} features in WDPA dataset.")
    gdf_aoi = clip_to_aoi(gdf)
    save_outputs(gdf_aoi)

    print("[DONE] WDPA preprocessing complete.")


if __name__ == "__main__":
    main()
