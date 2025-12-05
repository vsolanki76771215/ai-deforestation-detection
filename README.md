# AI-Driven Deforestation & Illegal Mining Detection

Monitoring the La Pampa – Tambopata – Madre de Dios Corridor, Peru
Capstone Project — UMass Global AI/ML Bootcamp

📌 Overview

This project builds an automated system to detect deforestation and illegal mining activity in the Amazon using satellite imagery.
The system integrates multi-source geospatial datasets, generates patch-level training labels, extracts remote-sensing features, and prepares the data for downstream ML modeling.

📦 Data Collection
This project integrates multi-source geospatial datasets to detect deforestation and illegal mining across three AOIs in Madre de Dios, Peru: La Pampa, Tambopata, and the Madre de Dios Corridor.
Data was collected from public, authoritative sources, processed using reproducible Python scripts, and structured following remote-sensing best practices.
All datasets are used under their respective open licenses.

🌍 Areas of Interest (AOIs)
Each dataset is clipped to the following AOIs (defined in config_aoi.py):

La Pampa                Illegal gold-mining hotspot	High-intensity deforestation
Tambopata               Tambopata National Reserve buffer	Protected area monitoring
Madre de Dios Corridor  Agricultural & river-mining corridor	Mixed land-use change

All downstream datasets (rasters → patches → features → labels) are generated per AOI, ensuring consistency and scalability.


1️⃣ Dataset 1 — Hansen Global Forest Change v1.12 (GFC)

Source:
https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12

Bands used:
  treecover2000 — % tree cover in year 2000
  lossyear — year of forest loss (1=2001 … 24=2024)

Collection method:
Full tile download via HTTPS (no API key required), then clipped to each AOI using Rasterio.

Raw files (per tile):
  Hansen_GFC-2024-v1.12_treecover2000_10S_070W.tif
  Hansen_GFC-2024-v1.12_lossyear_10S_070W.tif

Processed outputs (per AOI):
  gfc_treecover2000_10S_070W_aoi.tif
  gfc_lossyear_10S_070W_aoi.tif
  gfc_loss_2018_2022_aoi.tif
  patches/*.npz

Purpose:
Provides supervised labels for patch-level forest loss (2018–2022).


2️⃣ Dataset 2 — Sentinel-2 Optical Imagery (Before/After NDVI)

Source:
AWS STAC API (Earth Search)
https://earth-search.aws.element84.com/v1

Bands used:
  B04 (Red)
  B08 (NIR)

Collection method:
For each AOI, the pipeline:
  Queries STAC for cloud-filtered Sentinel-2 L2A imagery
  Builds median composites for 2018 dry season & 2022 dry season
  Clips composites to AOI
  Computes NDVI rasters
  Generates NPZ feature patches

Raw files (per AOI):
  s2_2018_raw.tif
  s2_2022_raw.tif

Processed outputs (per AOI):
  s2_ndvi_2018_aoi.tif
  s2_ndvi_2022_aoi.tif
  patches/*.npz

Purpose:
Provides temporal vegetation change features for ML models.


3️⃣ Dataset 3 — Protected Areas (WDPA)

Source:
UNEP-WCMC WDPA via Google Earth Engine
Dataset: WCMC/WDPA/current/polygons

Collection method:
  Exported from GEE as shapefile
  Cleaned encoding (Latin1 → UTF-8)
  Converted to GeoPackage
  Clipped per AOI

Raw files:
  osm_protected_areas.geojson
  wdpa_aoi.* .cpg, .dbf, .fix, .prj, shp, shx

Processed outputs:
  wdpa_aoi_clean.gpkg
  wdpa/<AOI>/wdpa_<AOI>.gpkg

Purpose:
Adds spatial context:
  inside protected area
  distance to boundary
  encroachment detection

