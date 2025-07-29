import json
import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# import tikzplotlib
from pathlib import Path
from typing import Tuple, List

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util import utils

"""
This script has the purpose of selecting stratisfied samples of Helistats considering their positional features. 
As a method of data splitting and sample selection ``kmeans`` is used. Only Heliostats with available deflectometry are used.
"""
# === Load and preprocess the data ===
# Load power plant position and use as reference point for ENU-conversion
device = torch.device("cpu")
print("Loading tower config...")
tower_measurements = '/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json'
tower_dict = json.load(open(tower_measurements, 'r')) 
power_plant_position = power_plant_position = torch.tensor(tower_dict['power_plant_properties']['coordinates'], dtype=torch.float64, device=device)

# Load Target Tower data
receiver_towers = []
solar_tower_juelich_position =  torch.tensor(tower_dict['solar_tower_juelich_upper']['coordinates']['center'], dtype=torch.float64, device=device)
mutlifocus_tower_position = torch.tensor(tower_dict['multi_focus_tower']['coordinates']['center'], dtype=torch.float64, device=device)
receiver_towers.append({'id': 'Solar Tower Juelich', 'wgs84': solar_tower_juelich_position, 'east': 0, 'north': 0})
receiver_towers.append({'id': 'Multifocus Tower', 'wgs84': mutlifocus_tower_position, 'east': 0, 'north': 0})

# Load the Heliostat properties metadta
print("Loading properties metadata csv...")
heliostats_properties_metadata = '/dss/dsshome1/05/di38kid/data/paint/metadata/properties_metadata_all_heliostats.csv'
df_properties = pd.read_csv(heliostats_properties_metadata)
df_properties = df_properties[['HeliostatId', 'latitude', 'longitude']].dropna()
df_properties = df_properties.drop_duplicates(subset='HeliostatId', keep='first').reset_index(drop=True)
print(f"Properties metadata for {len(df_properties)} heliostats.")

# Load the Heliostat deflectometry data
print("Loading deflectometry metadata csv...")
heliostats_deflectometry_metadata = '/dss/dsshome1/05/di38kid/data/paint/metadata/deflectometry_metadata_all_heliostats.csv'
df_deflect = pd.read_csv(heliostats_deflectometry_metadata)
df_deflect = df_deflect[['HeliostatId', 'latitude', 'longitude']].dropna()
df_deflect = df_deflect.drop_duplicates(subset='HeliostatId', keep='first').reset_index(drop=True)
print(f"Deflectometry metadata for {len(df_deflect)} heliostats.")

# === Transform positions to ENU ===
print("Transform coordinates to local ENU...")

# Tower coordinates
for tower in receiver_towers:
    tower_wgs84_pos = tower['wgs84']
    tower_pos_enu = utils.convert_wgs84_coordinates_to_local_enu(tower_wgs84_pos, power_plant_position, device=device)
    tower['east'] = tower_pos_enu[0].item()
    tower['north'] = tower_pos_enu[1].item()

# === Coordinate conversion stub ===
def to_enu(df, receiver_latlon):
    enu_list = []
    for _, row in df.iterrows():
        coords = torch.tensor([row["latitude"], row["longitude"], 0.0], dtype=torch.float64)
        enu = utils.convert_wgs84_coordinates_to_local_enu(coords, receiver_latlon, device)
        enu_list.append(enu[:2].tolist())  # east, north
    df[["east", "north"]] = pd.DataFrame(enu_list, index=df.index)
    return df
    

# Convert coordinates to EN
df_properties = to_enu(df_properties, power_plant_position)
df_deflect = to_enu(df_deflect, power_plant_position)


# Load the Heliostat calibration data and remove Heliostats with insufficient calibration data
print("Loading calibration metadata")
calibration_csv_path = "/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_all_heliostats.csv" 
calib_df = pd.read_csv(calibration_csv_path)
required_cols = {"Id", "HeliostatId"}
if not required_cols.issubset(calib_df.columns):
    raise ValueError(f"CSV is missing one or more required columns: {required_cols}")
calib_counts = calib_df.groupby("HeliostatId")["Id"].count().reset_index()
calib_counts.columns = ["HeliostatId", "num_measurements"]

# Keep only Heliostats with at least 90 measurements
print("Filtering for Heliostats with 90 measurements or more...")
sufficient_heliostats = calib_counts[calib_counts["num_measurements"] >= 90]["HeliostatId"]
df_calib_filtered = df_deflect[df_deflect["HeliostatId"].isin(sufficient_heliostats)]
print(f"Filtered deflectometry data for {len(df_calib_filtered)} heliostats.")

# === Perform K-means clustering on ENU positions ===
print("Preform K-means split..")
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
df_calib_filtered['cluster'] = kmeans.fit_predict(df_calib_filtered[['east', 'north']])

# === Select one representative heliostat per cluster ===
selected_ids = df_calib_filtered.groupby('cluster').apply(lambda g: g.sample(1, random_state=42)).reset_index(drop=True)
selected_ids_list = selected_ids['HeliostatId'].tolist()

selected_df = df_calib_filtered[df_calib_filtered['HeliostatId'].isin(selected_ids_list)].copy()
# full_df = df.copy()


# === Enforce certain Heliostats to be in the final data batch by exchanging them with their closest neighbor in k-means set
def enforce_heliostats_in_selection(df: pd.DataFrame, selected_df: pd.DataFrame, target_heliostat_ids: List[str]):
    """
    Ensure all heliostats in target_heliostat_ids are present in selected_df.
    Replaces the closest heliostats in selected_df with those targets (if not already present).

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with 'HeliostatId', 'east', 'north'.
    selected_df : pd.DataFrame
        Initial selection DataFrame with shape (k, ...) and the same columns.
    target_heliostat_ids : List[str]
        List of Heliostat IDs to enforce in the final selection.

    Returns
    -------
    Tuple[pd.DataFrame, List[Tuple[str, str]]]
        Updated selected_df and a list of (replaced_id, inserted_id) tuples.
    """
    selected_df = selected_df.copy()
    replacements = []

    for target_id in target_heliostat_ids:
        if target_id in selected_df['HeliostatId'].values:
            continue  # Already included

        target_row = df[df['HeliostatId'] == target_id]
        if target_row.empty:
            raise ValueError(f"Target HeliostatId '{target_id}' not found in dataset.")

        target_pos = target_row[['east', 'north']].values.reshape(1, -1)
        selected_positions = selected_df[['east', 'north']].values

        # Compute distances and find nearest
        distances = cdist(selected_positions, target_pos)
        closest_idx = distances.argmin()
        replaced_id = selected_df.iloc[closest_idx]['HeliostatId']

        # Replace the nearest one
        selected_df = selected_df.drop(index=selected_df.index[closest_idx])
        selected_df = pd.concat([selected_df, target_row], ignore_index=True)

        replacements.append((replaced_id, target_id))

    return selected_df.reset_index(drop=True), replacements


def remove_heliostats_from_selection(df: pd.DataFrame, selected_df: pd.DataFrame, heliostat_ids_to_remove: List[str]):
    """
    Remove specific heliostats from the selected set and replace each with the closest non-selected heliostat.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame containing all available heliostats ('HeliostatId', 'east', 'north').
    selected_df : pd.DataFrame
        The currently selected subset of df.
    heliostat_ids_to_remove : List[str]
        List of Heliostat IDs to be removed from the selection.

    Returns
    -------
    Tuple[pd.DataFrame, List[Tuple[str, str]]]
        Updated selected_df and a list of (removed_id, inserted_id) tuples.
    """
    selected_df = selected_df.copy()
    replacements = []

    # Build a set for fast membership check
    selected_ids_set = set(selected_df['HeliostatId'])

    for remove_id in heliostat_ids_to_remove:
        if remove_id not in selected_ids_set:
            continue  # Already not in selection

        remove_row = selected_df[selected_df['HeliostatId'] == remove_id]
        if remove_row.empty:
            continue

        # Candidate pool: heliostats not in the selection
        candidate_pool = df[~df['HeliostatId'].isin(selected_ids_set)]
        if candidate_pool.empty:
            raise ValueError("No available heliostats to use as replacement.")

        remove_pos = remove_row[['east', 'north']].values.reshape(1, -1)
        candidate_positions = candidate_pool[['east', 'north']].values

        # Find closest candidate
        distances = cdist(candidate_positions, remove_pos)
        closest_idx = distances.argmin()
        replacement_row = candidate_pool.iloc[[closest_idx]]  # keep as DataFrame
        inserted_id = replacement_row.iloc[0]['HeliostatId']

        # Replace in selection
        selected_df = selected_df[selected_df['HeliostatId'] != remove_id]
        selected_df = pd.concat([selected_df, replacement_row], ignore_index=True)

        # Update selected IDs for subsequent iterations
        selected_ids_set.remove(remove_id)
        selected_ids_set.add(inserted_id)

        replacements.append((remove_id, inserted_id))

    return selected_df.reset_index(drop=True), replacements



# Enforce target Heliostats to be in selection
target_ids = ['AA39', 'AM35']
selected_df, replaced_pairs = enforce_heliostats_in_selection(df_calib_filtered, selected_df, target_ids)

for old_id, new_id in replaced_pairs:
    print(f"Replaced heliostat '{old_id}' with '{new_id}'")

# Remove manually checked Heliostats that have too few Calibration data
# to_remove = ['AW19', 'AU55', 'AU50', "AU54", "AU53"]
# selected_df, replacements = remove_heliostats_from_selection(df, selected_df, to_remove)

# for removed, added in replacements:
#     print(f"Replaced heliostat '{removed}' with '{added}'")

# === Plotting ===
print("Generate Plots...")
ref_point = (0.0, 0.0)  # ENU origin is receiver center
output_dir = Path('/dss/dsshome1/05/di38kid/data/paint/selected_20/plots')

# --- Plot 0: Plot all heliostats, and steps of selection ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(df_properties["east"], df_properties["north"], color="lightgray", s=25, label="Missing Deflectometry")
ax.scatter(df_deflect["east"], df_deflect["north"], color="lightblue", s=25, label="< 90 Calibration Samples")
ax.scatter(selected_df["east"], selected_df["north"], color="black", s=30, label="Selection of 20 (K-Means)")
ax.scatter(df_calib_filtered["east"], df_calib_filtered["north"], color="steelblue", s=25, label="Remaining Heliostats")
ax.scatter(selected_df["east"], selected_df["north"], color="black", s=30)

# Annotate selected heliostats
for _, row in selected_df.iterrows():
    ax.text(row["east"] + 1.5, row["north"] + 1.5, row["HeliostatId"], fontsize=9, color="black")

# Plot Receiver Towers
tower_colors = ['darkred', 'darkorange']
east_diff = [7, -40]
nort_diff = [-3, -12]
# Plot Receiver Towers with distinct colors and legend
for idx, tower in enumerate(receiver_towers):
    ax.scatter(tower['east'], tower['north'], marker='P', s=80,
                color=tower_colors[idx % len(tower_colors)])
    ax.text(tower["east"]+east_diff[idx], tower["north"]+nort_diff[idx], tower['id'], fontsize=11, color=tower_colors[idx])

ax.set_xlabel("East [m]", fontsize=11)
ax.set_ylabel("North [m]", fontsize=11)
# ax.set_title("Heliostat Field Stratified by Metadata", fontsize=14)
ax.legend()
# ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect("equal", adjustable="box")

# Save both PNG and PGF
fig.tight_layout()
fig.savefig(output_dir / "heliostat_stratified_overview.png", dpi=300)
# fig.savefig(output_dir / "heliostat_stratified_overview.pgf")  # requires LaTeX-compatible backend
# tikzplotlib.save(output_dir / "heliostat_stratified_overview.tex")
plt.close(fig)

print(f"Saved stratified overview to {output_dir}")

# --- Plot 1: Only selected heliostats ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(selected_df['east'], selected_df['north'], color='blue', s=50, label='Selected Heliostats')
ax1.scatter(0.0, 0.0, color='red', marker='*', s=200, label='Power Plant Receiver')
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("Stratified K-Means Sample of 20 Heliostats")
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
fig1.savefig(output_dir / f"{k}_selected_heliostats_only.png", dpi=300)
# tikzplotlib.save(output_dir / f"{k}_selected_heliostats_only.tex")

#  --- Plot 2: All vs selected  ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(df_deflect['east'], df_deflect['north'], color='lightgray', s=20, label='All Heliostats')
ax2.scatter(selected_df['east'], selected_df['north'], color='blue', s=50, label='Selected Heliostats')
ax2.scatter(0.0, 0.0, color='red', marker='*', s=200, label='Power Plant Receiver')
ax2.set_xlabel("East [m]")
ax2.set_ylabel("North [m]")
ax2.set_title("Heliostat Field with K-Means Sample Highlighted")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()
fig2.savefig(output_dir / f"{k}_full_vs_selected_heliostats.png", dpi=300)

# --- Plot 3: Selected Heliostats with ID labels ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(selected_df['east'], selected_df['north'], color='blue', s=50, label='Selected Heliostats')
ax3.scatter(0.0, 0.0, color='red', marker='*', s=200, label='Power Plant Receiver')
# Add HeliostatId annotations
for _, row in selected_df.iterrows():
    ax3.text(row['east'] + 1.5, row['north'] + 1.5, row['HeliostatId'],
             fontsize=9, color='black')
ax3.set_xlabel("East [m]")
ax3.set_ylabel("North [m]")
ax3.set_title("Selected Heliostats (IDs Annotated)")
ax3.legend()
ax3.grid(True)
fig3.tight_layout()
fig3.savefig(output_dir / f"{k}_selected_heliostats_annotated.png", dpi=300)

# --- Plot 4: All vs selected with Receiver Towers ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(og_df['east'], og_df['north'], color='lightgray', s=20, label='All Heliostats')
ax4.scatter(selected_df['east'], selected_df['north'], color='blue', s=50, label='Selected Heliostats')
# Plot Receiver Towers
tower_colors = ['darkred', 'darkorange']
# Plot Receiver Towers with distinct colors and legend
for idx, tower in enumerate(receiver_towers):
    ax4.scatter(tower['east'], tower['north'], marker='P', s=120,
                color=tower_colors[idx % len(tower_colors)], label=tower['id'])
ax4.set_xlabel("East [m]")
ax4.set_ylabel("North [m]")
ax4.set_title("Field with Selected Heliostats and Receiver Towers")
ax4.legend()
ax4.grid(True)
fig4.tight_layout()
fig4.savefig(output_dir / f"{k}_full_vs_selected_with_receivers.png", dpi=300)

# --- Plot 5: Selected Heliostats with ID labels and Receiver Towers ---
fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(selected_df['east'], selected_df['north'], color='blue', s=50, label='Selected Heliostats')
# Annotate Heliostat IDs
for _, row in selected_df.iterrows():
    ax5.text(row['east'] + 1.5, row['north'] + 1.5, row['HeliostatId'], fontsize=9, color='black')
# Plot Receiver Towers
for idx, tower in enumerate(receiver_towers):
    ax5.scatter(tower['east'], tower['north'], marker='P', s=120,
                color=tower_colors[idx % len(tower_colors)], label=tower['id'])
ax5.set_xlabel("East [m]")
ax5.set_ylabel("North [m]")
ax5.set_title("Selected Heliostats with Receiver Towers (Annotated)")
ax5.legend()
ax5.grid(True)
fig5.tight_layout()
fig5.savefig(output_dir / f"{k}_selected_heliostats_annotated_with_receivers.png", dpi=300)

print("Done!")
