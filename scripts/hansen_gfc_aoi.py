# hansen_gfc_aoi.py

"""
Download Hansen Global Forest Change (GFC) v1.12 tiles for a given tile ID
and clip them to multiple AOIs in Madre de Dios, Peru (La Pampa, Tambopata,
Madre de Dios river corridor). Optionally generate patches.

Outputs per AOI:
- data/raw/Hansen_GFC-2024-v1.12_treecover2000_<TILE>.tif
- data/raw/Hansen_GFC-2024-v1.12_lossyear_<TILE>.tif
- data/processed/hansen/<AOI>/gfc_treecover2000_<TILE>_aoi.tif
- data/processed/hansen/<AOI>/gfc_lossyear_<TILE>_aoi.tif
- data/processed/hansen/<AOI>/gfc_loss_2018_2022_aoi.tif  (binary loss map)
- data/processed/hansen/<AOI>/patches/patch_000000.npz (32×32 patches)
"""

import argparse
from pathlib import Path

import requests
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping

from config_aoi import AOIS, PATCH_SIZE, STRIDE, TARGET_PATCHES_MIN, TARGET_PATCHES_MAX


# --- Config ---

# Loss window for binary map (Hansen encoding: 1 = 2001, ..., 24 = 2024)
LOSS_START = 18  # 2018
LOSS_END = 22    # 2022

BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"


# --- Helper functions ---

def download_hansen_tile(band: str, tile: str, dest_folder: Path) -> Path:
    """
    Download a Hansen GFC v1.12 GeoTIFF tile for a specific band and tile ID.

    band: 'treecover2000' or 'lossyear'
    tile: e.g. '10S_070W'
    """
    dest_folder.mkdir(parents=True, exist_ok=True)
    filename = f"Hansen_GFC-2024-v1.12_{band}_{tile}.tif"
    url = f"{BASE_URL}/{filename}"
    dest_path = dest_folder / filename

    if dest_path.exists():
        print(f"[INFO] Already exists: {dest_path}")
        return dest_path

    print(f"[INFO] Downloading {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[INFO] Saved to {dest_path}")
    return dest_path


def clip_raster_to_aoi(src_path: Path, aoi_bounds, dest_path: Path) -> Path:
    """
    Clip a raster to the rectangular AOI bounds.

    aoi_bounds: (lon_min, lat_min, lon_max, lat_max)
    """
    lon_min, lat_min, lon_max, lat_max = aoi_bounds
    aoi_geom = box(lon_min, lat_min, lon_max, lat_max)
    shapes = [mapping(aoi_geom)]

    with rasterio.open(src_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"[INFO] Clipped raster saved to {dest_path}")
    return dest_path


def make_binary_loss_map(lossyear_aoi_path: Path, dest_path: Path,
                         loss_start: int, loss_end: int) -> Path:
    """
    Create a binary loss map for a given year range [loss_start, loss_end].

    Output:
      uint8 GeoTIFF with 1 = forest loss in window, 0 = no loss / outside.
    """
    with rasterio.open(lossyear_aoi_path) as src:
        lossyear = src.read(1)
        profile = src.profile

    # Mask for pixels that lost forest between loss_start and loss_end
    loss_mask = (lossyear >= loss_start) & (lossyear <= loss_end)
    loss_binary = loss_mask.astype("uint8")

    profile.update(dtype="uint8", count=1)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest_path, "w", **profile) as dst:
        dst.write(loss_binary, 1)

    print(f"[INFO] Binary loss map saved to {dest_path}")
    return dest_path


def generate_patches_from_hansen(
    treecover_aoi_path: Path,
    loss_binary_path: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    target_min: int = TARGET_PATCHES_MIN,
    target_max: int = TARGET_PATCHES_MAX,
) -> int:
    """
    Tile the AOI into patches using treecover2000 and binary loss rasters.

    Each patch is saved as a .npz file with:
      - 'treecover': treecover2000 patch (H x W)
      - 'loss': binary loss patch (H x W)
      - metadata: row, col, patch_size
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(treecover_aoi_path) as src_tree, rasterio.open(loss_binary_path) as src_loss:
        tree = src_tree.read(1)
        loss = src_loss.read(1)

        if tree.shape != loss.shape:
            raise ValueError(
                f"Shape mismatch between treecover {tree.shape} and loss {loss.shape}"
            )

        height, width = tree.shape

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

            patch_tree = tree[row:row + patch_size, col:col + patch_size]
            patch_loss = loss[row:row + patch_size, col:col + patch_size]

            patch_id = f"patch_{total_patches:06d}"
            out_path = out_dir / f"{patch_id}.npz"

            np.savez_compressed(
                out_path,
                treecover=patch_tree,
                loss=patch_loss,
                row=row,
                col=col,
                patch_size=patch_size,
            )

            total_patches += 1

    print(f"[INFO] Generated {total_patches} Hansen patches at {out_dir}")

    if total_patches < target_min:
        print(
            f"[WARN] Only {total_patches} patches generated "
            f"(TARGET_PATCHES_MIN={target_min}). "
            f"Consider decreasing STRIDE or PATCH_SIZE or expanding AOI."
        )

    return total_patches


# --- Main CLI ---

def run_for_aoi(tile: str, aoi_name: str):
    if aoi_name not in AOIS:
        raise ValueError(f"Unknown AOI '{aoi_name}'. Available: {list(AOIS.keys())}")

    aoi = AOIS[aoi_name]
    print(f"[INFO] Processing AOI '{aoi.name}' with bounds {aoi.bounds}")

    data_raw = Path("data/raw/hansen")
    data_processed = Path("data/processed/hansen") / aoi.name

    # 1) Download full tiles
    treecover_path = download_hansen_tile("treecover2000", tile, data_raw)
    lossyear_path = download_hansen_tile("lossyear", tile, data_raw)

    # 2) Clip to AOI
    treecover_aoi_path = data_processed / f"gfc_treecover2000_{tile}_aoi.tif"
    lossyear_aoi_path = data_processed / f"gfc_lossyear_{tile}_aoi.tif"

    clip_raster_to_aoi(treecover_path, aoi.bounds, treecover_aoi_path)
    clip_raster_to_aoi(lossyear_path, aoi.bounds, lossyear_aoi_path)

    # 3) Make binary loss map (2018–2022)
    loss_binary_path = data_processed / "gfc_loss_2018_2022_aoi.tif"
    make_binary_loss_map(
        lossyear_aoi_path,
        loss_binary_path,
        loss_start=LOSS_START,
        loss_end=LOSS_END,
    )

    # 4) Generate 32×32 patches
    patches_dir = data_processed / "patches"
    total_patches = generate_patches_from_hansen(
        treecover_aoi_path=treecover_aoi_path,
        loss_binary_path=loss_binary_path,
        out_dir=patches_dir,
    )

    print(f"[DONE] AOI '{aoi.name}' – total patches: {total_patches}")


def main(tile: str, aoi: str):
    if aoi == "all":
        for aoi_name in AOIS.keys():
            run_for_aoi(tile, aoi_name)
    else:
        run_for_aoi(tile, aoi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Hansen GFC v1.12 tile, clip to AOIs, and generate patches."
    )
    parser.add_argument(
        "--tile",
        type=str,
        default="10S_070W",
        help="Tile ID, e.g. 10S_070W (default: 10S_070W)",
    )
    parser.add_argument(
        "--aoi",
        type=str,
        default="all",
        help="AOI key from config_aoi.AOIS or 'all'",
    )
    args = parser.parse_args()
    main(tile=args.tile, aoi=args.aoi)
