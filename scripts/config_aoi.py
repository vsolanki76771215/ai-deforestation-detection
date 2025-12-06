# config_aoi.py
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class AOI:
    name: str
    bounds: Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)

# Placeholder bounding boxes (replace with your real ones)
AOIS: Dict[str, AOI] = {
    "la_pampa": AOI(
        name="la_pampa",
        bounds=(-70.6, -13.3, -69.9, -12.7),
    ),
    "tambopata": AOI(
        name="tambopata",
        bounds=(-69.9, -13.4, -69.1, -12.4),
    ),
    "madre_de_dios_corridor": AOI(
        name="madre_de_dios_corridor",
        bounds=(-70.8, -13.5, -69.5, -12.3),
    ),
}

# Patch configuration
PATCH_SIZE = 32
STRIDE = 16
TARGET_PATCHES_MIN = 15000
TARGET_PATCHES_MAX = 20000