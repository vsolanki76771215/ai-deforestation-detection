#!/usr/bin/env python
"""
Build tabular ML datasets (CSV) from NPZ patch tensors.

Reads:
  data/processed/patches/<aoi>/features/patch_*.npz
  data/processed/patches/<aoi>/labels/patch_*.npz

and writes:
  data/processed/dataset_ml/<aoi>_patches.csv
  data/processed/dataset_ml/all_patches_combined.csv
  data/processed/dataset_ml/all_patches_features_labels_s2_ndvi.csv  # alias

Each row in the CSV corresponds to one patch with:
  - aoi
  - patch_filename
  - ndvi_2018_mean
  - ndvi_2022_mean
  - ndvi_diff
  - loss_fraction   (fraction of pixels marked as loss)
  - loss_binary     (1 if loss_fraction >= 0.5 else 0)
"""

from pathlib import Path
import numpy as np
import pandas as pd


AOIS = ["la_pampa", "tambopata", "madre_de_dios_corridor"]

PATCH_ROOT = Path("data/processed/patches")
OUT_DIR = Path("data/processed/dataset_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_ndvi_means(feat_npz: np.lib.npyio.NpzFile):
    """
    Try to robustly get ndvi_2018 and ndvi_2022 arrays from a feature NPZ.

    Supported patterns:
      - keys 'ndvi_2018' and 'ndvi_2022'
      - single key, 3D array (H, W, 2) -> [:,:,0] and [:,:,1]
      - single key, 3D array (2, H, W) -> [0] and [1]
    """
    keys = feat_npz.files

    # Case 1: explicit ndvi_2018 / ndvi_2022 keys
    if "ndvi_2018" in keys and "ndvi_2022" in keys:
        ndvi18 = feat_npz["ndvi_2018"]
        ndvi22 = feat_npz["ndvi_2022"]
        return ndvi18, ndvi22

    # Fallback: use first key
    arr = feat_npz[keys[0]]

    if arr.ndim == 3:
        # (H, W, bands)
        if arr.shape[2] >= 2:
            return arr[:, :, 0], arr[:, :, 1]
        # (bands, H, W)
        if arr.shape[0] >= 2:
            return arr[0, :, :], arr[1, :, :]

    # Last resort: treat same array as both (no temporal diff info)
    return arr, arr


def extract_label_array(label_npz: np.lib.npyio.NpzFile):
    """
    Extract label array from NPZ.

    Supported patterns:
      - key 'label'
      - first key in the file
    """
    keys = label_npz.files
    if "label" in keys:
        return label_npz["label"]
    return label_npz[keys[0]]


def build_aoi_csv(aoi: str) -> Path:
    """
    Build a CSV for a single AOI by combining feature + label NPZ patches.
    """
    feat_dir = PATCH_ROOT / aoi / "features"
    lab_dir = PATCH_ROOT / aoi / "labels"

    if not feat_dir.exists() or not lab_dir.exists():
        print(f"[WARN] Missing features or labels directory for AOI '{aoi}', skipping.")
        return None

    print(f"[INFO] Building CSV for AOI: {aoi}")
    rows = []

    # Iterate over feature patches; assume matching filenames in labels
    for feat_path in sorted(feat_dir.glob("patch_*.npz")):
        patch_name = feat_path.name
        lab_path = lab_dir / patch_name

        if not lab_path.exists():
            print(f"[WARN] No matching label NPZ for {feat_path}, skipping.")
            continue

        # Load data
        feat_npz = np.load(feat_path)
        lab_npz = np.load(lab_path)

        ndvi18, ndvi22 = extract_ndvi_means(feat_npz)
        label_arr = extract_label_array(lab_npz)

        # Compute summary statistics
        ndvi18_mean = float(np.nanmean(ndvi18))
        ndvi22_mean = float(np.nanmean(ndvi22))
        ndvi_diff = ndvi22_mean - ndvi18_mean

        # labels are usually 0/1
        loss_fraction = float(np.nanmean(label_arr))
        loss_binary = int(loss_fraction >= 0.5)

        rows.append(
            dict(
                aoi=aoi,
                patch_file=patch_name,
                ndvi_2018_mean=ndvi18_mean,
                ndvi_2022_mean=ndvi22_mean,
                ndvi_diff=ndvi_diff,
                loss_fraction=loss_fraction,
                loss_binary=loss_binary,
            )
        )

    if not rows:
        print(f"[WARN] No rows created for AOI {aoi}.")
        return None

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{aoi}_patches.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(df)} rows to {out_path}")
    return out_path


def main():
    all_dfs = []

    for aoi in AOIS:
        csv_path = build_aoi_csv(aoi)
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No AOI CSVs created, aborting combined dataset.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_out = OUT_DIR / "all_patches_combined.csv"
    combined.to_csv(combined_out, index=False)
    print(f"[INFO] Saved combined dataset with {len(combined)} rows to {combined_out}")

    # Also save with the more descriptive name requested
    alias_out = OUT_DIR / "all_patches_features_labels_s2_ndvi.csv"
    combined.to_csv(alias_out, index=False)
    print(f"[INFO] Saved alias dataset to {alias_out}")


if __name__ == "__main__":
    main()
