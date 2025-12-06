# 🌍 AI-Driven Deforestation & Illegal Mining Detection  
**Madre de Dios, Peru — La Pampa • Tambopata • Madre de Dios Corridor**

---

## 📘 Overview

Illegal gold mining and unregulated land clearing are rapidly transforming the Amazon rainforest, especially in the **Madre de Dios** region of Peru.  
This project builds an **AI/ML pipeline** that integrates **remote sensing, geospatial processing, and machine learning** to detect forest loss between **2018 and 2022**.

The pipeline uses:
- **Hansen Global Forest Change v1.12** (deforestation labels)  
- **Sentinel-2 NDVI** (vegetation features)  
- **WDPA protected areas** (context & enforcement insights)  

Final output:  
✔ Patch-level ML dataset (43,576 samples)  
✔ Fully reproducible data pipeline  

---

# 🗺️ Areas of Interest (AOIs)

Defined in `config_aoi.py`:

| AOI | Description |
|-----|-------------|
| **La Pampa** | Illegal gold mining hotspot |
| **Tambopata** | Buffer of Tambopata National Reserve |
| **Madre de Dios Corridor** | Agricultural, mining, and transport corridor |

Each dataset is **clipped and tiled per AOI**.

---

# 📦 Data Collection

---

## 1️⃣ Hansen Global Forest Change v1.12 (GFC)

**Source:** https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12  

**Bands used:**
- `treecover2000`
- `lossyear`

### 🔧 Processing Steps
1. Download full tile (`10S_070W`)  
2. Clip tile to AOI  
3. Create binary forest loss map (2018–2022)  
4. Generate 32×32 NPZ label patches  

### 📁 Raw Files

```
data/raw/hansen/
    Hansen_GFC-2024-v1.12_treecover2000_10S_070W.tif
    Hansen_GFC-2024-v1.12_lossyear_10S_070W.tif
```

### 📁 Processed Outputs

```
data/processed/hansen/<AOI>/
    gfc_treecover2000_10S_070W_aoi.tif
    gfc_lossyear_10S_070W_aoi.tif
    gfc_loss_2018_2022_aoi.tif
```

**Purpose:** Provides *supervised labels* for deforestation detection.

---

## 2️⃣ Sentinel-2 NDVI (2018 & 2022)

**Source:** AWS Earth Search STAC API  
https://earth-search.aws.element84.com/v1

**Bands used:**  
- B04 (Red)  
- B08 (NIR)

### 🔧 Processing Steps
1. Query cloud-filtered Sentinel-2 L2A scenes  
2. Build dry-season composites for **2018** & **2022**  
3. Clip to AOI  
4. Compute NDVI  
5. Generate 32×32 NPZ feature patches
   
### 📁 Raw Files

```
data/raw/sentinel2/<AOI>
    s2_2018_dry_aoi.tif
    s2_2022_dry_aoi.tif
```

### 📁 Processed Outputs

```
data/processed/sentinel2/<AOI>/
    s2_ndvi_2018_aoi.tif
    s2_ndvi_2022_aoi.tif
```

**Purpose:** Provides NDVI-based vegetation change features.

---

## 3️⃣ WDPA Protected Areas

**Source:** UNEP-WCMC via Google Earth Engine  
Dataset ID: `WCMC/WDPA/current/polygons`

### 📁 Processed Outputs

```
data/processed/wdpa/
    wdpa_aoi_clean.gpkg
    la_pampa/wdpa_la_pampa.gpkg
    tambopata/wdpa_tambopata.gpkg
    madre_de_dios_corridor/wdpa_mdd_corridor.gpkg
```

**Purpose:** Identify patches inside protected areas & monitor encroachment.

---

# 🧩 Patch Extraction (Raster → NPZ → CSV)

Repository uses this **final patch directory layout**:

```
data/processed/patches/
    la_pampa/
        features/
            patch_000000.npz
            patch_000001.npz
            ...
        labels/
            patch_000000.npz
            patch_000001.npz
            ...
    tambopata/
        features/
        labels/
    madre_de_dios_corridor/
        features/
        labels/
```

### 📦 Patch Contents

**Feature patch (`features/*.npz`)** contains:
```
ndvi_2018      → 32×32 array  
ndvi_2022      → 32×32 array  
metadata       → row, col, patch_size, aoi
```

**Label patch (`labels/*.npz`)** contains:
```
loss_mask      → 32×32 binary array (1 = loss)
treecover      → optional treecover2000
metadata       → row, col, patch_size, aoi
```

### 🔗 Patch Pairing
Matched by identical filenames:

```
features/patch_012345.npz
labels/patch_012345.npz
```

If no pair exists → skipped.

---

# 📊 Final Machine Learning Dataset

Converted into CSVs:

```
data/processed/dataset_ml/
    la_pampa_patches.csv
    tambopata_patches.csv
    madre_de_dios_corridor_patches.csv
    all_patches_combined.csv
    all_patches_features_labels_s2_ndvi.csv
```

### Dataset Size
| AOI | Samples |
|-----|---------|
| La Pampa | 3,576 |
| Tambopata | 20,000 |
| Madre de Dios Corridor | 20,000 |
| **Total** | **43,576** |

### Features Included
- NDVI mean/std/min/max (2018 & 2022)  
- NDVI Δ (2022 − 2018)  
- Loss fraction  
- Binary loss label  

---

# 🔁 Reproducibility

Regenerate the entire dataset:

```bash
python hansen_gfc_aoi.py --aoi all
python sentinel2_ndvi_aoi.py --aoi all
python preprocess_wdpa_dataset3.py
python build_patch_csv_from_npz.py
```

---
### 📁 Project layout

```
project-root/
│
├── data/
│   ├── raw/
│   │   ├── hansen/
│   │   │   ├── Hansen_GFC-2024-v1.12_treecover2000_10S_070W.tif
│   │   │   └── Hansen_GFC-2024-v1.12_lossyear_10S_070W.tif
│   │   │
│   │   ├── sentinel2/
│   │   │   ├── la_pampa/
│   │   │   ├── tambopata/
│   │   │   └── madre_de_dios_corridor/
│   │   │
│   │   └── wdpa/
│   │       ├── osm_protected_areas.geojson
│   │       ├── wdpa_aoi.cpg
│   │       ├── wdpa_aoi.dbf
│   │       ├── wdpa_aoi.fix
│   │       ├── wdpa_aoi.prj
│   │       ├── wdpa_aoi.shp
│   │       └── wdpa_aoi.shx
│   │
│   ├── processed/
│   │   ├── hansen/
│   │   │   ├── la_pampa/
│   │   │   │   ├── gfc_treecover2000_10S_070W_aoi.tif
│   │   │   │   ├── gfc_lossyear_10S_070W_aoi.tif
│   │   │   │   ├── gfc_loss_2018_2022_aoi.tif
│   │   │   │   └── patches/
│   │   │   │       ├── patch_000000.npz
│   │   │   │       └── ...
│   │   │   ├── tambopata/
│   │   │   └── madre_de_dios_corridor/
│   │   │
│   │   ├── sentinel2/
│   │   │   ├── la_pampa/
│   │   │   │   ├── s2_ndvi_2018_aoi.tif
│   │   │   │   ├── s2_ndvi_2022_aoi.tif
│   │   │   │   └── patches/
│   │   │   │       ├── patch_000000.npz
│   │   │   │       └── ...
│   │   │   ├── tambopata/
│   │   │   └── madre_de_dios_corridor/
│   │   │
│   │   ├── patches/
│   │   │   ├── la_pampa/
│   │   │   │   ├── features/
│   │   │   │   │   ├── patch_000000.npz
│   │   │   │   │   └── ...
│   │   │   │   └── labels/
│   │   │   │       ├── patch_000000.npz
│   │   │   │       └── ...
│   │   │   ├── tambopata/
│   │   │   └── madre_de_dios_corridor/
│   │   │
│   │   ├── wdpa/
│   │   │   ├── wdpa_aoi_clean.gpkg
│   │   │   ├── la_pampa/
│   │   │   ├── tambopata/
│   │   │   └── madre_de_dios_corridor/
│   │   │
│   │   └── dataset_ml/
│   │       ├── la_pampa_patches.csv
│   │       ├── tambopata_patches.csv
│   │       ├── madre_de_dios_corridor_patches.csv
│   │       ├── all_patches_combined.csv
│   │       └── all_patches_features_labels_s2_ndvi.csv
│
├── scripts/
│   ├── hansen_gfc_aoi.py
│   ├── sentinel2_ndvi_aoi.py
│   ├── preprocess_wdpa_dataset3.py
│   └── build_patch_csv_from_npz.py
│
└── README.md
```
---

### 📁 Dataset access
Because the full geospatial dataset exceeds GitHub’s file-size limits, a curated data subset is provided through Google Drive to support reproducibility, evaluation, and demonstration of the full data pipeline.  

🔗 **Google Drive Dataset Link:**  
https://drive.google.com/drive/folders/17pf9_Pf-NWv1betSzZdS5weRk9DrL5P9?usp=sharing  

Folder structure mirrors the repository:

```
data-subset/
│
├── raw/
│   ├── hansen/
│   │   └── 10S_070W/
│   │
│   ├── sentinel2/
│   │   └── la_pampa/
│   │
│   └── wdpa/
│
├── processed/
│   ├── hansen/
│   │   └── la_pampa/
│   │
│   ├── sentinel2/
│   │   └── la_pampa/
│   │
│   ├── wdpa/
│   │   └── la_pampa/
│   │
│   ├── patches/
│   │   └── la_pampa/
│   │
│   └── dataset_ml/


```

---

# 📚 Acknowledgements
- Hansen GFC: © University of Maryland, Google, USGS, NASA  
- Sentinel-2: © ESA Copernicus Programme  
- WDPA: © UNEP-WCMC  

---

