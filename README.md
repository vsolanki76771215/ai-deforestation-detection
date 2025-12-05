# AI-Driven Deforestation & Illegal Mining Detection

Monitoring the La Pampa – Tambopata – Madre de Dios Corridor, Peru
Capstone Project — UMass Global AI/ML Bootcamp

📌 Overview

This project builds an automated system to detect deforestation and illegal mining activity in the Amazon using satellite imagery.
The system integrates multi-source geospatial datasets, generates patch-level training labels, extracts remote-sensing features, and prepares the data for downstream ML modeling.


📦 Data Collection

This project integrates multi-source geospatial datasets to monitor deforestation and illegal mining across three AOIs (La Pampa, Tambopata, Madre de Dios Corridor) in Madre de Dios, Peru. All datasets were collected from public, authoritative sources, processed using reproducible scripts, and aligned to a unified patch-based ML pipeline.

🗺️ Areas of Interest (AOIs)

Defined in config_aoi.py:

AOI	Description	Purpose
La Pampa	Illegal gold-mining hotspot	High-intensity deforestation
Tambopata	Buffer zone of Tambopata National Reserve	Monitoring protected area pressure
Madre de Dios Corridor	Agricultural & river-mining corridor	Mixed drivers of deforestation

All datasets are clipped and tiled per AOI to ensure consistent spatial alignment.

1️⃣ Hansen Global Forest Change v1.12 (GFC)

Source:
https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12

Bands used:

treecover2000

lossyear

Download Method: Direct HTTP from Google Cloud Bucket.

Raw files stored under:

data/raw/hansen/
    Hansen_GFC-2024-v1.12_treecover2000_10S_070W.tif
    Hansen_GFC-2024-v1.12_lossyear_10S_070W.tif


Processing Steps:

Clip each tile to AOI bounding boxes

Convert lossyear → binary forest-loss mask (2018–2022)

Tile AOI rasters into 32×32 patches (labels)

Processed outputs:

data/processed/hansen/<AOI>/
    gfc_treecover2000_10S_070W_aoi.tif
    gfc_lossyear_10S_070W_aoi.tif
    gfc_loss_2018_2022_aoi.tif
    patches/*.npz               ← label patches


Purpose:
Provides supervised labels for detecting deforestation.

2️⃣ Sentinel-2 Optical Imagery (NDVI 2018 & 2022)

Source:
AWS Earth Search STAC API
https://earth-search.aws.element84.com/v1

Bands used:

B04 (Red)

B08 (NIR)

Collection Method:

Query STAC for cloud-filtered L2A scenes

Build median composites for 2018 dry season & 2022 dry season

Clip to AOI

Compute NDVI

Generate 32×32 feature patches

Processed outputs:

data/processed/sentinel2/<AOI>/
    s2_ndvi_2018_aoi.tif
    s2_ndvi_2022_aoi.tif
    patches/*.npz               ← feature patches


Purpose:
Provides temporal vegetation change features for supervised ML.

3️⃣ Protected Areas — WDPA

Source:
UNEP-WCMC / Google Earth Engine
Dataset ID: WCMC/WDPA/current/polygons

Processing Steps:

Export WDPA polygons for Madre de Dios region

Clean attribute encoding

Convert to GeoPackage

Clip per AOI

Processed outputs:

data/processed/wdpa/
    wdpa_aoi_clean.gpkg                 ← full AOI WDPA layer
    la_pampa/wdpa_la_pampa.gpkg
    tambopata/wdpa_tambopata.gpkg
    madre_de_dios_corridor/wdpa_mdd_corridor.gpkg


Purpose:
Adds spatial context such as:

Whether a patch lies inside a protected area

Distance to reserve boundary

🧩 Patch Extraction Pipeline (Raster → NPZ → CSV)

Each AOI produces two patch sets:

Source	Output	Description
Sentinel-2	features/*.npz	NDVI patch tensor
Hansen GFC	labels/*.npz	Binary forest-loss patch

Each .npz → 32×32 array (or 32×32×N for multi-band).

Example structure:

data/processed/patches/<AOI>/
    features/patch_000123.npz
    labels/patch_000123.npz

📊 Final Machine Learning Dataset (CSV)

NPZ patches are merged into per-AOI and global tabular datasets:

data/processed/dataset_ml/
    la_pampa_patches.csv
    tambopata_patches.csv
    madre_de_dios_corridor_patches.csv
    all_patches_combined.csv
    all_patches_features_labels_s2_ndvi.csv

Columns include:

aoi

patch_file

ndvi_2018_mean

ndvi_2022_mean

ndvi_diff

loss_fraction

loss_binary

Total samples: 43,576 patches

✔ Exceeds rubric requirement of ≥15,000 samples.

🗂️ Data Storage & Versioning

Large rasters (>100 MB) kept in /data/raw/ and excluded via .gitignore.

Project is compatible with Git LFS or cloud storage.

All processed outputs are reproducible from scripts in /scripts/.

🔁 Reproducibility

The complete dataset can be regenerated using:

python hansen_gfc_aoi.py --aoi all
python sentinel2_ndvi_aoi.py --aoi all
python preprocess_wdpa_dataset3.py
python build_patch_csv_from_npz.py
