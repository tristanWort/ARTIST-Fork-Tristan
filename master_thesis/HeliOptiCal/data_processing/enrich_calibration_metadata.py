import json
import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pathlib
from typing import Tuple, List


"""
This script serves the purpose of completing the missing columns `Azimuth` and `Elevation` in `PAINT` calibration metadata.

Requires 
    - An existing CSV-type metadata file with column `Id` (calibration ID) and `HeliostatId`.
    - Calibration data as downloaded from `PAINT` for each Heliostat and calibration ID (missing IDs will be removed from CSV).
"""


def load_sun_position_from_json(json_path: pathlib.Path):
    """
    Stub function to load sun azimuth and elevation from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        sun_azimuth = data.get("sun_azimuth")
        sun_elevation = data.get("sun_elevation")
        return sun_azimuth, sun_elevation


def enrich_metadata_with_sun_positions(metadata_csv_path: str, 
                                       base_data_dir: str, 
                                       output_csv_path: str, 
                                       json_suffix: str = "-calibration-properties.json",
                                       flux_suffix: str = "-flux.png"):
    """
    Augments calibration metadata CSV with sun azimuth and elevation from JSON files.

    Parameters
    ----------
    metadata_csv_path : str
        Path to the CSV metadata file (must contain 'Id' and 'HeliostatId').
    base_data_dir : str
        Base directory where each HeliostatId has a 'Calibration' folder.
    output_csv_path : str
        Path to save the updated metadata CSV.
    json_suffix : str
        Suffix for the calibration JSON file (default: '-calibration-properties.json').
    flux_suffix : str
        Suffix for the flux image file (default: '-flux.png').
    """
    # Step 1–2: Load and clean metadata
    df = pd.read_csv(metadata_csv_path)
    df = df.dropna(subset=["Id", "HeliostatId"])
    df = df[df["Id"].duplicated(keep=False) == False]  # Keep only unique Ids

    # Columns to be added
    valid_rows = []
    azimuth_list = []
    elevation_list = []

    for idx, row in df.iterrows():
        heliostat_id = str(row["HeliostatId"])
        calibration_id = str(row["Id"])

        calibration_dir = pathlib.Path(base_data_dir) / heliostat_id / "Calibration"
        json_path = calibration_dir / f"{calibration_id}{json_suffix}"
        png_path = calibration_dir / f"{calibration_id}{flux_suffix}"
        
        if not json_path.exists():
            print(f"[SKIP] JSON not found: {json_path}")
            continue
        if not png_path.exists():
            print(f"[SKIP] PNG not found: {png_path}")
            continue
        
        try:
            az, el = load_sun_position_from_json(json_path)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON {json_path}: {e}")
            continue

        valid_rows.append(row)
        azimuth_list.append(az)
        elevation_list.append(el)

    # Step 5: Rebuild Datframe from valid rows
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    valid_df["Azimuth"] = azimuth_list
    valid_df["Elevation"] = elevation_list

    # Step 6: Save output
    valid_df.to_csv(output_csv_path, index=False)
    print(f"Completed metadata saved to: {output_csv_path}")
    

if __name__ == '__main__':

    metadata_path = '/dss/dsshome1/05/di38kid/data/paint/selected_20/metadata/old_calibration_metadata_selected_heliostats_20250525_161028.csv'
    paint_directory = '/dss/dsshome1/05/di38kid/data/paint/selected_20'
    save_as = '/dss/dsshome1/05/di38kid/data/paint/selected_20/metadata/calibration_metadata_selected_heliostats_20250525_161028__1.csv'
    enrich_metadata_with_sun_positions(metadata_path, paint_directory, save_as)
    
    
    