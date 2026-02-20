from pathlib import Path

feat_dir = Path("data/processed/patches/la_pampa/features")
lab_dir  = Path("data/processed/patches/la_pampa/labels")

# n_feat = len(list(feat_dir.glob("patch_*.npz")))
# n_lab  = len(list(lab_dir.glob("patch_*.npz")))

# print("Features:", n_feat)
# print("Labels:", n_lab)
# print("Missing labels (features - labels):", n_feat - n_lab)

# feat_names = {p.name for p in feat_dir.glob("patch_*.npz")}
# lab_names  = {p.name for p in lab_dir.glob("patch_*.npz")}

# missing = sorted(list(feat_names - lab_names))[:20]
# print("Missing patches:", missing)


lab_dir = Path("data/processed/patches/la_pampa/labels")
lab_ids = sorted(int(p.stem.split("_")[1]) for p in lab_dir.glob("patch_*.npz"))
print("Label patch min/max:", lab_ids[0], lab_ids[-1], "count:", len(lab_ids))