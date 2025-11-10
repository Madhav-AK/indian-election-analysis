import json
import re
import warnings
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

# Suppress noisy geopandas/future warnings
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
warnings.filterwarnings("ignore", category=FutureWarning)

VERBOSE = False

__all__ = [
    "clean_pc_name",
    "clean_state_name",
    "report_merge_diagnostics",
    "PC_NAME_MAP",
    "PC_NAME_MAP_TURNOUT",
    "PC_NAME_MAP_CANDIDATES",
    "STATE_NAME_MAP",
    "load_geojson",
]

# -----------------------------
# Name cleaning utilities
# -----------------------------
def clean_pc_name(name: Optional[str]) -> str:
    """Clean PC (constituency) names: drop (SC)/(ST) suffixes, uppercase, trim spaces."""
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s*\((SC|ST)\)$", "", name, flags=re.IGNORECASE)
    name = name.upper().strip()
    name = re.sub(r"\s+", " ", name)
    return name


def clean_state_name(name: Optional[str]) -> str:
    """Clean state names: uppercase, normalize '&' to 'AND', collapse whitespace."""
    if not isinstance(name, str):
        return ""
    name = name.upper().strip()
    name = name.replace("&", "AND")
    name = re.sub(r"\s+", " ", name)
    return name


# -----------------------------
# Merge diagnostics
# -----------------------------
def report_merge_diagnostics(
    map_df: pd.DataFrame,
    data_df: pd.DataFrame,
    merge_col: str,
    map_name: str = "GeoJSON",
    data_name: str = "CSV",
) -> None:
    """
    Compare unique keys between map and data frames on `merge_col`
    and print quick match/miss summaries.
    """
    if VERBOSE: print(f"\n--- Merge Diagnostics (on '{merge_col}') ---")

    map_items = set(map_df.get(merge_col, pd.Series(dtype=object)).dropna().unique())
    data_items = set(data_df.get(merge_col, pd.Series(dtype=object)).dropna().unique())

    # Filter out empty strings
    map_items = {x for x in map_items if isinstance(x, str) and x.strip()}
    data_items = {x for x in data_items if isinstance(x, str) and x.strip()}

    matched_items = map_items.intersection(data_items)
    missing_from_data = sorted(map_items - data_items)  # On map, not in data
    missing_from_map = sorted(data_items - map_items)   # In data, not on map

    if VERBOSE: print(f"Total items in {map_name}: {len(map_items)}")
    if VERBOSE: print(f"Total items in {data_name}: {len(data_items)}")
    if VERBOSE: print(f"Matched items: {len(matched_items)}")
    if VERBOSE: print(f"Items on {map_name} missing {data_name} data (grey spots): {len(missing_from_data)}")
    if VERBOSE: print(f"Items in {data_name} missing {map_name} match: {len(missing_from_map)}")

    if missing_from_data:
        if VERBOSE: print(f"\n(1) Items on {map_name} missing {data_name} data (Grey Spots):")
        for i, nm in enumerate(missing_from_data, 1):
            if VERBOSE: print(f"{i:3}. {nm}")

    if missing_from_map:
        if VERBOSE: print(f"\n(2) Items in {data_name} missing {map_name} match:")
        for i, nm in enumerate(missing_from_map, 1):
            if VERBOSE: print(f"{i:3}. {nm}")

    if missing_from_data or missing_from_map:
        if VERBOSE: print("\nTo fix, compare lists (1) and (2) and update NAME_MAP dictionaries.")
    if VERBOSE: print("-" * 50)


# -----------------------------
# Name maps (PC / State)
# -----------------------------
PC_NAME_MAP: Dict[str, str] = {
    "AHMADNAGAR": "AHMEDNAGAR",
    "ANAKAPALLI": "ANAKAPALLE",
    "ANANTAPUR": "ANANTHAPUR",
    "ANANTNAG - RAJOURI": "ANANTNAG-RAJOURI",
    "ARUKU": "ARAKU",
    "AUTONOMOUS DISTRICT": "DIPHU",
    "BAHRAICH": "BAHARAICH",
    "BARDHAMAN DURGAPUR": "BARDHAMAN-DURGAPUR",
    "BARRACKPORE": "BARRACKPUR",
    "BHANDARA - GONDIYA": "BHANDARA GONDIYA",
    "COOCH BEHAR": "COOCHBEHAR",
    "DADRA & NAGAR HAVELI": "DADAR & NAGAR HAVELI",
    "GADCHIROLI-CHIMUR": "GADCHIROLI - CHIMUR",
    "GAUHATI": "GUWAHATI",
    "HARDWAR": "HARIDWAR",
    "HATKANANGLE": "HATKANANGALE",
    "KURNOOL": "KURNOOLU",
    "MANGALDOI": "DARRANG-UDALGURI",
    "NARASARAOPET": "NARSARAOPET",
    "NORTH EAST DELHI": "NORTH-EAST DELHI",
    "NORTH WEST DELHI": "NORTH-WEST DELHI",
    "NOWGONG": "NAGAON",
    "PATALIPUTRA": "PATLIPUTRA",
    "RANN OF KUTCH": "KACHCHH",
    "SARGUJA": "SURGUJA",
    "SECUNDRABAD": "SECUNDERABAD",
    "TEZPUR": "SONITPUR",
    "THIRUVALLUR": "TIRUVALLUR",
    "TIRUPATI": "THIRUPATHI",
    "YAVATMAL-WASHIM": "YAVATMAL- WASHIM",
}

# Turnout maps (extends base)
PC_NAME_MAP_TURNOUT: Dict[str, str] = {
    **PC_NAME_MAP,
    "ARAMBAGH": "ARAMBAG",
    "JOYNAGAR": "JAYNAGAR",
    "PALAMAU": "PALAMU",
    "RATNAGIRI - SINDHUDURG": "RATNAGIRI- SINDHUDURG",
    "SRERAMPUR": "SREERAMPUR",
}

# Candidate maps (reverse of some turnout ones)
PC_NAME_MAP_CANDIDATES: Dict[str, str] = {
    **PC_NAME_MAP,
    "ARAMBAG": "ARAMBAGH",
    "JAYNAGAR": "JOYNAGAR",
    "PALAMU": "PALAMAU",
    "RATNAGIRI - SINDHUDURG": "RATNAGIRI-SINDHUDURG",
    "SREERAMPUR": "SRERAMPUR",
}

# State name mappings (GeoJSON -> CSV)
STATE_NAME_MAP: Dict[str, str] = {
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "ANDAMAN AND NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "DELHI": "NCT OF DELHI",
    "DADRANAGARHAVELI DAMANDIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
}


# -----------------------------
# GeoJSON loader + normalizer
# -----------------------------
def _normalize_feature_properties(props: Optional[dict]) -> dict:
    """
    Normalize property keys into a standard schema:
      - pc_name (from PC_NAME or pc_name; default '')
      - STATE_NAME (from a variety of keys; default '')
    """
    if props is None:
        props = {}

    # pc_name
    if "PC_NAME" in props:      # uppercase key
        props["pc_name"] = props.pop("PC_NAME")
    else:
        props.setdefault("pc_name", "")

    # STATE_NAME from several possible keys
    state_val = ""
    for key in ("st_name", "STATE_UT", "state_ut", "STATE_NAME", "state_name"):
        if key in props and props.get(key) is not None:
            state_val = props.get(key)
            break
    props["STATE_NAME"] = state_val if state_val is not None else ""
    return props


def load_geojson(
    json_file: str = "geo_utils/india_pc_2024_simplified.geojson",
    feature_to_add_file: Optional[str] = "geo_utils/kaziranga.txt",
):
    """
    Load and clean a PC-level India GeoJSON.
    - Accepts either a FeatureCollection or a raw list of Feature dicts.
    - Optionally appends a single extra Feature (e.g., Kaziranga).
    - Normalizes properties to include 'pc_name' and 'STATE_NAME'.
    - Cleans names and applies STATE_NAME_MAP.

    Returns
    -------
    gpd.GeoDataFrame  (columns include: ['PC_NAME', 'STATE_NAME', 'geometry', ...])
    or None on failure.
    """
    if VERBOSE: print(f"Loading GeoJSON from '{json_file}'...")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: GeoJSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        if VERBOSE: print(f"Error: The file '{json_file}' is not valid JSON.")
        return None

    # Extract features
    if isinstance(geojson_data, dict) and "features" in geojson_data:
        if VERBOSE: print("Detected FeatureCollection. Extracting 'features' list.")
        feature_list = geojson_data["features"]
    elif isinstance(geojson_data, list):
        if VERBOSE: print("Detected list of features.")
        feature_list = geojson_data
    else:
        if VERBOSE: print("Error: GeoJSON data is not in the expected format.")
        return None

    # Optional: append extra Feature (e.g., Kaziranga)
    if feature_to_add_file:
        try:
            if VERBOSE: print(f"Attempting to load and append feature from '{feature_to_add_file}'...")
            with open(feature_to_add_file, "r", encoding="utf-8") as f:
                new_feature = json.load(f)
            if isinstance(new_feature, dict) and new_feature.get("type") == "Feature":
                feature_list.append(new_feature)
                if VERBOSE: print(f"Successfully appended feature: {new_feature.get('properties', {}).get('PC_NAME')}")
            else:
                if VERBOSE: print(f"Warning: '{feature_to_add_file}' is not a single GeoJSON Feature.")
        except FileNotFoundError:
            if VERBOSE: print(f"Warning: Feature file '{feature_to_add_file}' not found. Continuing without it.")
        except Exception as e:
            if VERBOSE: print(f"Warning: Could not append feature. Error: {e}")

    # Normalize properties
    if VERBOSE: print("Normalizing properties (pc_name, STATE_NAME)...")
    normalized_features = []
    for feat in feature_list:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {})
        feat["properties"] = _normalize_feature_properties(props)
        normalized_features.append(feat)

    if VERBOSE: print(f"Original feature count (after appends): {len(normalized_features)}")
    valid_features = [
        f for f in normalized_features
        if isinstance(f, dict) and f.get("geometry") is not None
    ]
    if VERBOSE: print(f"Valid features (with geometry): {len(valid_features)}")
    if not valid_features:
        if VERBOSE: print("Error: No valid features with geometry found.")
        return None

    if VERBOSE: print("Creating GeoDataFrame from valid features...")
    gdf = gpd.GeoDataFrame.from_features(valid_features)

    if "pc_name" not in gdf.columns:
        if VERBOSE: print("Error: missing 'pc_name' in properties after normalization.")
        if VERBOSE: print(f"Found columns: {list(gdf.columns)}")
        return None

    # Clean & standardize names
    gdf["pc_name"] = gdf["pc_name"].apply(clean_pc_name)
    gdf = gdf.rename(columns={"pc_name": "PC_NAME"})
    gdf["STATE_NAME"] = gdf["STATE_NAME"].apply(clean_state_name).replace(STATE_NAME_MAP)

    if VERBOSE: print(f"Successfully loaded and cleaned {len(gdf)} shapes from GeoJSON.")
    return gdf


file_7 =  "data/ConstituencyWiseSummary2.csv"
file_14 = "data/VoteDistribution.csv"
file_23 = "data/WomenElectorParticipation.csv"
file_24 = "data/WomenCandidateParticipation.csv"
file_33 = "data/ConstituencyWiseResult.csv"
candidate_data_file = "data/Candidate_Data.csv"


# -----------------------------
# Plot 1: Voter Turnout (PC-level)
# -----------------------------
def plot_voter_turnout(
    csv_file: str = file_7,
    gdf: Optional[gpd.GeoDataFrame] = None,
    feature_to_add_file: Optional[str] = "geo_utils/kaziranga.txt",
):
    """Plot voter turnout by constituency as a choropleth."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING VOTER TURNOUT MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson(feature_to_add_file=feature_to_add_file)
        if gdf is None:
            return None

    # Apply turnout-specific name mappings
    gdf_turnout = gdf.copy()
    gdf_turnout["PC_NAME"] = gdf_turnout["PC_NAME"].replace(PC_NAME_MAP_TURNOUT)

    # Load CSV
    try:
        df = pd.read_csv(csv_file, header=2)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    # Required columns
    required_csv_cols = ["PC No", "PC Name", "VTR (%)"]
    if not all(col in df.columns for col in required_csv_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_csv_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    # Keep only proper PC rows
    df = df[pd.to_numeric(df["PC No"], errors="coerce").notna()]

    # Select/clean
    df = df[["PC Name", "VTR (%)"]].copy()
    df.rename(columns={"PC Name": "PC_NAME", "VTR (%)": "VTR"}, inplace=True)
    df["PC_NAME"] = df["PC_NAME"].apply(clean_pc_name)
    df["VTR"] = pd.to_numeric(df["VTR"], errors="coerce")
    df.dropna(subset=["PC_NAME", "VTR"], inplace=True)

    # Average VTR if duplicates
    df = df.groupby("PC_NAME", as_index=False).agg(VTR=("VTR", "mean"))
    if VERBOSE: print(f"Successfully loaded and cleaned {len(df)} data rows from CSV.")

    # Merge
    if VERBOSE: print("Merging CSV data with GeoJSON shapes...")
    merged_gdf = gdf_turnout.merge(df, on="PC_NAME", how="left")

    # Diagnostics
    report_merge_diagnostics(
        gdf_turnout,
        df,
        "PC_NAME",
        map_name="GeoJSON (Turnout Mapped)",
        data_name="Voter Turnout CSV",
    )

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    merged_gdf.plot(
        column="VTR",
        ax=ax,
        legend=True,
        cmap="viridis",
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "Voter Turnout (%)",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )
    ax.set_title(
        "Constituency-wise Voter Turnout (VTR %)",
        fontdict={"fontsize": 20, "fontweight": "bold"},
    )
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Plot 2: Candidate Heatmap (count by PC)
# -----------------------------
def plot_candidate_heatmap(
    csv_file: str = candidate_data_file,
    gdf: Optional[gpd.GeoDataFrame] = None,
    feature_to_add_file: Optional[str] = "geo_utils/kaziranga.txt",
):
    """Plot number of candidates by constituency as a choropleth."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING CANDIDATE HEATMAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson(feature_to_add_file=feature_to_add_file)
        if gdf is None:
            return None

    # Apply candidate-specific name mappings
    gdf_candidates = gdf.copy()
    gdf_candidates["PC_NAME"] = gdf_candidates["PC_NAME"].replace(PC_NAME_MAP_CANDIDATES)

    # Load CSV
    try:
        df = pd.read_csv(csv_file, header=0)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    # Required columns
    required_csv_cols = ["Constituency", "Party"]
    if not all(col in df.columns for col in required_csv_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_csv_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    # Clean
    df = df[["Constituency", "Party"]].copy()
    df.rename(columns={"Constituency": "PC_NAME"}, inplace=True)
    df.dropna(subset=["PC_NAME"], inplace=True)
    df["PC_NAME"] = df["PC_NAME"].apply(clean_pc_name)

    # Count candidates per PC
    df_agg = df.groupby("PC_NAME", as_index=False).size().rename(columns={"size": "Num_Candidates"})
    if VERBOSE: print(f"Successfully loaded and aggregated {len(df_agg)} data rows from CSV.")

    # Merge
    if VERBOSE: print("Merging CSV data with GeoJSON shapes...")
    merged_gdf = gdf_candidates.merge(df_agg, on="PC_NAME", how="left")

    # Diagnostics
    report_merge_diagnostics(
        gdf_candidates, df_agg, "PC_NAME", map_name="GeoJSON (Candidate Mapped)", data_name="Candidate CSV"
    )

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    merged_gdf.plot(
        column="Num_Candidates",
        ax=ax,
        legend=True,
        cmap="Greens",
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "Number of Candidates",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )
    ax.set_title("Constituency-wise Number of Candidates", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Plot 3: Voter Deviation from State Mean
# -----------------------------
def _prepare_voter_csv_for_deviation(csv_file: str) -> Optional[pd.DataFrame]:
    """
    Reads the turnout CSV and returns a cleaned dataframe with:
      ['STATE_NAME', 'PC_NAME', 'Total Voters']
    """
    try:
        df_voter_csv = pd.read_csv(csv_file, header=2)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    # Build 'State' by forward-filling state header rows
    df_voter_csv["State"] = df_voter_csv["PC No"]
    df_voter_csv.loc[pd.to_numeric(df_voter_csv["State"], errors="coerce").notna(), "State"] = np.nan
    df_voter_csv["State"] = df_voter_csv["State"].ffill()

    # Keep only PC rows
    df_voter_csv = df_voter_csv[pd.to_numeric(df_voter_csv["PC No"], errors="coerce").notna()]

    # Select / rename
    needed = ["State", "PC Name", "Total Voters"]
    missing = [c for c in needed if c not in df_voter_csv.columns]
    if missing:
        if VERBOSE: print(f"Error: Missing expected columns in CSV: {missing}")
        return None

    df_constituencies = df_voter_csv[["State", "PC Name", "Total Voters"]].copy()
    df_constituencies.rename(columns={"State": "STATE_NAME", "PC Name": "PC_NAME"}, inplace=True)
    df_constituencies["STATE_NAME"] = df_constituencies["STATE_NAME"].apply(clean_state_name)
    df_constituencies["PC_NAME"] = df_constituencies["PC_NAME"].apply(clean_pc_name)
    df_constituencies.dropna(subset=["STATE_NAME", "PC_NAME"], inplace=True)
    return df_constituencies


def plot_voter_deviation_map(
    gdf: gpd.GeoDataFrame,
    df_const: pd.DataFrame,
):
    """
    Calculates and plots the percentage deviation of voters for each constituency
    from its state's average.
    """
    # Copies to avoid warnings
    df_const_copy = df_const.copy()
    gdf_copy = gdf.copy()

    # Apply turnout-specific name mappings to the GeoDataFrame
    gdf_copy["PC_NAME"] = gdf_copy["PC_NAME"].replace(PC_NAME_MAP_TURNOUT)

    # Ensure numeric
    df_const_copy["Total Voters"] = pd.to_numeric(df_const_copy["Total Voters"], errors="coerce")
    df_const_copy = df_const_copy.dropna(subset=["Total Voters", "STATE_NAME", "PC_NAME"])

    # State mean
    df_state_mean = df_const_copy.groupby("STATE_NAME", as_index=False)["Total Voters"].mean()
    df_state_mean = df_state_mean.rename(columns={"Total Voters": "State Mean Voters"})

    # Merge mean into PC data
    df_analysis = df_const_copy.merge(df_state_mean, on="STATE_NAME", how="left")

    # % deviation
    df_analysis["Voter Deviation %"] = 100.0 * (
        (df_analysis["Total Voters"] - df_analysis["State Mean Voters"]) / df_analysis["State Mean Voters"]
    )
    df_analysis["Voter Deviation %"] = df_analysis["Voter Deviation %"].replace([np.inf, -np.inf], np.nan)

    # Merge to shapes
    gdf_plot = gdf_copy.merge(df_analysis[["PC_NAME", "Voter Deviation %"]], on="PC_NAME", how="left")

    # Diagnostics
    report_merge_diagnostics(
        gdf_copy, df_analysis, "PC_NAME", map_name="GeoJSON (Turnout Mapped)", data_name="Voter Data CSV"
    )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    max_abs_dev = gdf_plot["Voter Deviation %"].abs().max()
    if pd.isna(max_abs_dev) or max_abs_dev == 0:
        max_abs_dev = 100.0

    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_dev, vcenter=0, vmax=max_abs_dev)
    gdf_plot.plot(
        column="Voter Deviation %",
        ax=ax,
        legend=True,
        cmap="RdBu_r",
        norm=norm,
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "% Deviation from State Average Voters",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )

    ax.set_title("Voter Number Deviation from State Mean", fontdict={"fontsize": 18, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# One-time loader & runners
# -----------------------------
def _maybe_save(fig, path: str):
    if fig is not None:
        plt.savefig(path, bbox_inches="tight", dpi=150)
        if VERBOSE: print(f"\nSuccess! Map saved as '{path}'")
        plt.show()


if __name__ == "__main__":
    # Load shared GeoJSON once
    gdf_constituencies = load_geojson(feature_to_add_file="geo_utils/kaziranga.txt")

    if gdf_constituencies is not None:
        # 1) Voter Turnout map
        fig_turnout = plot_voter_turnout(gdf=gdf_constituencies)
        _maybe_save(fig_turnout, "voter_turnout_map.png")

        # 2) Candidate Heatmap
        fig_candidates = plot_candidate_heatmap(gdf=gdf_constituencies)
        _maybe_save(fig_candidates, "candidate_heatmap.png")

        # 3) Voter Deviation map
        if VERBOSE: print("Generating voter deviation heatmap...")
        df_constituencies = _prepare_voter_csv_for_deviation(file_7)
        if df_constituencies is not None:
            fig_voter_dev = plot_voter_deviation_map(gdf_constituencies, df_constituencies)
            _maybe_save(fig_voter_dev, "voter_deviation_map.png")
        else:
            if VERBOSE: print("Skipped voter deviation map due to CSV load error.")
    else:
        if VERBOSE: print("Error: Failed to load base GeoJSON; skipping all plots.")


# -----------------------------
# Plot 4: % NOTA votes by PC
# -----------------------------
def plot_nota_map(
    csv_file: str = file_14,
    gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Plot percentage of NOTA votes by constituency."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING NOTA VOTES MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson()
        if gdf is None:
            return None

    # Name mapping (turnout mapping works for this file format)
    gdf_nota = gdf.copy()
    gdf_nota["PC_NAME"] = gdf_nota["PC_NAME"].replace(PC_NAME_MAP_TURNOUT)

    # Load CSV (header on 2nd row, skip the 3rd row)
    try:
        df = pd.read_csv(csv_file, header=1, skiprows=[2])
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    required_csv_cols = ["PC NAME", "% Votes to NOTA Out of Total Votes Polled"]
    if not all(col in df.columns for col in required_csv_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_csv_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    df = df[required_csv_cols].copy()
    df.rename(
        columns={
            "PC NAME": "PC_NAME",
            "% Votes to NOTA Out of Total Votes Polled": "NOTA_PCT",
        },
        inplace=True,
    )
    df["PC_NAME"] = df["PC_NAME"].apply(clean_pc_name)
    df["NOTA_PCT"] = pd.to_numeric(df["NOTA_PCT"], errors="coerce")
    df.dropna(subset=["PC_NAME", "NOTA_PCT"], inplace=True)

    # Aggregate duplicates if any
    df = df.groupby("PC_NAME", as_index=False).agg(NOTA_PCT=("NOTA_PCT", "mean"))
    if VERBOSE: print(f"Successfully loaded and cleaned {len(df)} data rows from CSV.")

    # Merge + diagnostics
    if VERBOSE: print("Merging CSV data with GeoJSON shapes...")
    merged_gdf = gdf_nota.merge(df, on="PC_NAME", how="left")
    report_merge_diagnostics(
        gdf_nota, df, "PC_NAME", map_name="GeoJSON (Turnout Mapped)", data_name="NOTA Votes CSV"
    )

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    merged_gdf.plot(
        column="NOTA_PCT",
        ax=ax,
        legend=True,
        cmap="Reds",
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "% NOTA Votes",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )
    ax.set_title("% of Votes for NOTA per Constituency", fontdict={"fontsize": 18, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Plot 5: Winning Alliance by PC (2024)
# -----------------------------
def plot_alliance_map(
    csv_file: str = file_33,
    gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Plot winning alliance by constituency (2024)."""
    import matplotlib.patches as mpatches

    if VERBOSE: print(f"\n{'='*60}\nGENERATING ALLIANCE MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson()
        if gdf is None:
            return None

    gdf_alliance = gdf.copy()
    gdf_alliance["PC_NAME"] = gdf_alliance["PC_NAME"].replace(PC_NAME_MAP_TURNOUT)

    # Load CSV (real header at line index=2)
    try:
        df = pd.read_csv(csv_file, header=2)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    required_cols = ["PC Name", "Party Name", "Total"]
    if not all(col in df.columns for col in required_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    df_winners = df[required_cols].copy()
    df_winners.rename(
        columns={"PC Name": "PC_NAME", "Party Name": "PARTY_NAME", "Total": "VOTES"}, inplace=True
    )
    df_winners["PC_NAME"] = df_winners["PC_NAME"].apply(clean_pc_name)
    df_winners["PARTY_NAME"] = df_winners["PARTY_NAME"].astype(str).str.strip()
    df_winners = df_winners[df_winners["PARTY_NAME"] != "NOTA"].copy()

    df_winners["VOTES"] = pd.to_numeric(df_winners["VOTES"], errors="coerce")
    df_winners.sort_values(by=["PC_NAME", "VOTES"], ascending=[True, False], inplace=True)
    df_winners.drop_duplicates(subset="PC_NAME", keep="first", inplace=True)

    if VERBOSE: print(f"Successfully loaded and found {len(df_winners)} winners from CSV.")

    # Alliance lists (2024)
    nda_parties = [
        "BJP",
        "TDP",
        "JD(U)",
        "SHS",
        "LJPRV",
        "JD(S)",
        "PMK",
        "RLD",
        "NCP",
        "JnP",
        "AGP",
        "AJSUP",
        "ADAL",
    ]
    india_parties = [
        "INC",
        "DMK",
        "AITC",
        "SP",
        "AAP",
        "SHSUBT",
        "NCPSP",
        "RJD",
        "CPI(M)",
        "IUML",
        "JKN",
        "JMM",
        "CPI",
        "CPI(ML)(L)",
        "VCK",
        "RSP",
    ]

    def get_alliance(party: str) -> str:
        if party == "BJP":
            return "BJP"
        if party == "INC":
            return "INC"
        if party in nda_parties:
            return "NDA_Other"
        if party in india_parties:
            return "INDIA_Other"
        return "Other"

    df_winners["Alliance"] = df_winners["PARTY_NAME"].apply(get_alliance)

    # Colors + legend
    color_map = {
        "BJP": "#E67E22",  # dark saffron/orange
        "NDA_Other": "#FAA94A",
        "INC": "#2E4172",  # deep navy
        "INDIA_Other": "#465A8C",
        "Other": "#B0B0B0",
    }
    legend_patches = [
        mpatches.Patch(color=color_map["BJP"], label="BJP"),
        mpatches.Patch(color=color_map["NDA_Other"], label="NDA (Others)"),
        mpatches.Patch(color=color_map["INC"], label="INC"),
        mpatches.Patch(color=color_map["INDIA_Other"], label="INDIA (Others)"),
        mpatches.Patch(color=color_map["Other"], label="Other"),
        mpatches.Patch(color="#cccccc", edgecolor="#777777", hatch="///", label="Data Not Available"),
    ]

    # Merge + diagnostics
    if VERBOSE: print("Merging CSV data with GeoJSON shapes...")
    merged_gdf = gdf_alliance.merge(df_winners[["PC_NAME", "Alliance"]], on="PC_NAME", how="left")
    report_merge_diagnostics(
        gdf_alliance, df_winners, "PC_NAME", map_name="GeoJSON (Turnout Mapped)", data_name="Winning Party CSV"
    )

    merged_gdf["color"] = merged_gdf["Alliance"].map(color_map).fillna("#cccccc")

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    merged_gdf.plot(
        color=merged_gdf["color"],
        ax=ax,
        categorical=True,
        legend=False,
        missing_kwds={"color": "#cccccc", "edgecolor": "#777777", "hatch": "///"},
    )
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    ax.set_title("Winning Alliance per Constituency (2024)", fontdict={"fontsize": 18, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Plot 6: Victory Margin (% of total polled) by PC
# -----------------------------
def plot_victory_margin_map(
    csv_file: str = file_33,
    gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Plot margin of victory by constituency (% of total votes polled)."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING VICTORY MARGIN MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson()
        if gdf is None:
            return None

    gdf_margin = gdf.copy()
    gdf_margin["PC_NAME"] = gdf_margin["PC_NAME"].replace(PC_NAME_MAP_TURNOUT)

    # Load CSV (header at index=2)
    try:
        df = pd.read_csv(csv_file, header=2)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    # Required (note the newline in TOTAL_POLLED col name)
    required_cols = ["PC Name", "Party Name", "Total", "Total Votes Polled In\nThe Constituency"]
    if not all(col in df.columns for col in required_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    df_data = df[required_cols].copy()
    df_data.rename(
        columns={
            "PC Name": "PC_NAME",
            "Party Name": "PARTY_NAME",
            "Total": "VOTES",
            "Total Votes Polled In\nThe Constituency": "TOTAL_POLLED",
        },
        inplace=True,
    )

    # Clean
    df_data["PC_NAME"] = df_data["PC_NAME"].apply(clean_pc_name)
    df_data["PARTY_NAME"] = df_data["PARTY_NAME"].astype(str).str.strip()
    df_data["VOTES"] = pd.to_numeric(df_data["VOTES"], errors="coerce")
    df_data["TOTAL_POLLED"] = pd.to_numeric(df_data["TOTAL_POLLED"].astype(str).str.replace(",", ""), errors="coerce")
    df_data = df_data[df_data["PARTY_NAME"] != "NOTA"].copy()

    # Rank per PC
    df_data.sort_values(by=["PC_NAME", "VOTES"], ascending=[True, False], inplace=True)
    winners = df_data.groupby("PC_NAME").nth(0).reset_index()
    runners = df_data.groupby("PC_NAME").nth(1).reset_index()

    # Merge W vs R
    df_analysis = winners[["PC_NAME", "VOTES", "TOTAL_POLLED"]].merge(
        runners[["PC_NAME", "VOTES"]], on="PC_NAME", how="left", suffixes=("_win", "_run")
    )
    df_analysis["VOTES_run"] = df_analysis["VOTES_run"].fillna(0)

    # Margin % of total polled
    df_analysis["MARGIN_VOTES"] = df_analysis["VOTES_win"] - df_analysis["VOTES_run"]
    df_analysis["MARGIN_PCT"] = (df_analysis["MARGIN_VOTES"] / df_analysis["TOTAL_POLLED"]) * 100.0
    df_analysis.replace([np.inf, -np.inf], np.nan, inplace=True)

    if VERBOSE: print(f"Successfully calculated victory margins for {len(df_analysis)} constituencies.")

    # Merge + diagnostics
    if VERBOSE: print("Merging margin data with GeoJSON shapes...")
    merged_gdf = gdf_margin.merge(df_analysis[["PC_NAME", "MARGIN_PCT"]], on="PC_NAME", how="left")
    report_merge_diagnostics(
        gdf_margin, df_analysis, "PC_NAME", map_name="GeoJSON (Turnout Mapped)", data_name="Victory Margin CSV"
    )

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    merged_gdf.plot(
        column="MARGIN_PCT",
        ax=ax,
        legend=True,
        cmap="Reds",
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "Victory Margin (% of Total Votes)",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )
    ax.set_title("Victory Margin per Constituency (2024)", fontdict={"fontsize": 18, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Runners for these plots
# -----------------------------
if __name__ == "__main__":
    # Reuse the one-time GeoJSON load; if not loaded, do it here
    if "gdf_constituencies" not in globals():
        gdf_constituencies = load_geojson(feature_to_add_file="geo_utils/kaziranga.txt")

    if gdf_constituencies is not None:
        # 4) NOTA
        if VERBOSE: print("Generating NOTA votes heatmap...")
        fig_nota = plot_nota_map(gdf=gdf_constituencies)
        _maybe_save(fig_nota, "nota_votes_map.png")

        # 5) Alliance
        if VERBOSE: print("Generating alliance heatmap...")
        fig_alliance = plot_alliance_map(gdf=gdf_constituencies)
        _maybe_save(fig_alliance, "alliance_map.png")

        # 6) Victory Margin
        if VERBOSE: print("Generating victory margin heatmap...")
        fig_margin = plot_victory_margin_map(gdf=gdf_constituencies)
        _maybe_save(fig_margin, "victory_margin_map.png")
    else:
        if VERBOSE: print("Error: 'gdf_constituencies' GeoDataFrame not found or failed to load.")


from typing import Optional

# -----------------------------
# Geometry helper
# -----------------------------
def fill_holes(geom):
    """Removes internal holes from polygons to fix black spots."""
    try:
        if geom.type == "Polygon":
            return Polygon(geom.exterior)
        elif geom.type == "MultiPolygon":
            filled_polygons = [Polygon(p.exterior) for p in geom.geoms]
            return MultiPolygon(filled_polygons)
        return geom
    except Exception:
        return geom


# -----------------------------
# Plot 7: % of seats won by women (state-level)
# -----------------------------
def plot_women_seats_map(
    csv_file: str = file_24,
    gdf: Optional[gpd.GeoDataFrame] = None,
    feature_to_add_file: str = "geo_utils/kaziranga.txt",
):
    """Plot percentage of seats won by women per state."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING WOMEN'S SEATS WON MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson(feature_to_add_file=feature_to_add_file)
        if gdf is None:
            return None

    gdf_women = gdf.copy()

    # Load CSV
    try:
        df = pd.read_csv(csv_file, header=2)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    required_csv_cols = ["State /UT", "Constituency Type", "Over total seats in State/UT"]
    if not all(col in df.columns for col in required_csv_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_csv_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    df = df[df["Constituency Type"] == "State Total"].copy()
    df = df[["State /UT", "Over total seats in State/UT"]].copy()
    df.rename(
        columns={"State /UT": "STATE_NAME", "Over total seats in State/UT": "PCT_WOMEN_WON"},
        inplace=True,
    )

    df["STATE_NAME"] = df["STATE_NAME"].apply(clean_state_name)
    df["PCT_WOMEN_WON"] = pd.to_numeric(df["PCT_WOMEN_WON"], errors="coerce")
    df.dropna(subset=["STATE_NAME", "PCT_WOMEN_WON"], inplace=True)

    if VERBOSE: print(f"Successfully loaded and cleaned {len(df)} state data rows from CSV.")

    # Merge + diagnostics
    if VERBOSE: print("Merging CSV data with GeoJSON shapes on STATE_NAME...")
    merged_gdf = gdf_women.merge(df, on="STATE_NAME", how="left")
    report_merge_diagnostics(
        gdf_women, df, "STATE_NAME", map_name="GeoJSON States", data_name="Women's Seats CSV"
    )

    # Geometry processing -> dissolve constituencies into states
    if VERBOSE: print("\nDissolving constituency boundaries into states for plotting...")
    if VERBOSE: print("Applying buffer(0) to fix potential geometry errors...")
    merged_gdf["geometry"] = merged_gdf.geometry.buffer(0)

    state_gdf = merged_gdf.dissolve(by="STATE_NAME", aggfunc={"PCT_WOMEN_WON": "first"})

    if "" in state_gdf.index:
        state_gdf = state_gdf.drop("")
        if VERBOSE: print("Dropped blank state shape from dissolved data.")

    if VERBOSE: print("Fixing internal holes (black spots) in state polygons...")
    state_gdf["geometry"] = state_gdf["geometry"].apply(fill_holes)

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    state_gdf.plot(
        column="PCT_WOMEN_WON",
        ax=ax,
        legend=True,
        cmap="RdYlGn",
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "% of Seats Won by Women in State/UT",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )

    ax.set_title("Percentage of Seats Won by Women per State/UT", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Plot 8: % of female electorate (state-level)
# -----------------------------
def plot_female_electorate_pct_map(
    csv_file: str = file_23,
    gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Plot percentage of female electorate by state."""
    if VERBOSE: print(f"\n{'='*60}\nGENERATING FEMALE ELECTORATE PERCENTAGE MAP\n{'='*60}\n")

    # Load GeoJSON if not provided
    if gdf is None:
        gdf = load_geojson()
        if gdf is None:
            return None

    gdf_state = gdf.copy()

    # Load CSV (actual header on 2nd line)
    try:
        df = pd.read_csv(csv_file, header=1)
    except FileNotFoundError:
        if VERBOSE: print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        if VERBOSE: print(f"Error reading CSV file: {e}")
        return None

    required_cols = ["STATE/UT", "% OF WOMEN ELECTORS OVER TOTAL ELECTORS"]
    if not all(col in df.columns for col in required_cols):
        if VERBOSE: print(f"Error: CSV file missing one or more required columns: {required_cols}")
        if VERBOSE: print(f"Found columns: {df.columns.tolist()}")
        return None

    df_agg = df[required_cols].copy()
    df_agg.rename(
        columns={
            "STATE/UT": "STATE_NAME",
            "% OF WOMEN ELECTORS OVER TOTAL ELECTORS": "PCT_FEMALE_ELECTORS",
        },
        inplace=True,
    )
    df_agg["STATE_NAME"] = df_agg["STATE_NAME"].apply(clean_state_name).replace(STATE_NAME_MAP)
    df_agg["PCT_FEMALE_ELECTORS"] = pd.to_numeric(df_agg["PCT_FEMALE_ELECTORS"], errors="coerce")
    df_agg.dropna(subset=["STATE_NAME", "PCT_FEMALE_ELECTORS"], inplace=True)
    df_agg.drop_duplicates(subset=["STATE_NAME"], inplace=True)

    if VERBOSE: print(f"Successfully loaded and cleaned {len(df_agg)} state data rows from CSV.")

    # Merge + diagnostics
    if VERBOSE: print("Merging CSV data with GeoJSON shapes on STATE_NAME...")
    merged_gdf = gdf_state.merge(df_agg, on="STATE_NAME", how="left")
    report_merge_diagnostics(
        gdf_state, df_agg, "STATE_NAME", map_name="GeoJSON States", data_name="Female Electorate CSV"
    )

    # Geometry processing -> dissolve constituencies into states
    if VERBOSE: print("\nDissolving constituency boundaries into states for plotting...")
    if VERBOSE: print("Applying buffer(0) to fix potential geometry errors...")
    merged_gdf["geometry"] = merged_gdf.geometry.buffer(0)
    state_gdf = merged_gdf.dissolve(by="STATE_NAME", aggfunc={"PCT_FEMALE_ELECTORS": "first"})

    if "" in state_gdf.index:
        state_gdf = state_gdf.drop("")
        if VERBOSE: print("Dropped blank state shape from dissolved data.")

    if VERBOSE: print("Fixing internal holes (black spots) in state polygons...")
    state_gdf["geometry"] = state_gdf["geometry"].apply(fill_holes)

    # Plot
    if VERBOSE: print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    state_gdf.plot(
        column="PCT_FEMALE_ELECTORS",
        ax=ax,
        legend=True,
        cmap="YlGnBu",
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={
            "color": "#cccccc",
            "edgecolor": "#777777",
            "hatch": "///",
            "label": "Data Not Available",
        },
        legend_kwds={
            "label": "% of Total Electors who are Female",
            "orientation": "horizontal",
            "pad": 0.05,
            "shrink": 0.7,
        },
    )
    ax.set_title("Percentage of Female Electorate per State/UT (2024)", fontdict={"fontsize": 18, "fontweight": "bold"})
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# -----------------------------
# Optional saver & runner
# -----------------------------
def _maybe_save(fig, path: str):
    if fig is not None:
        plt.savefig(path, bbox_inches="tight", dpi=150)
        if VERBOSE: print(f"\nSuccess! Map saved as '{path}'")
        plt.show()
    else:
        if VERBOSE: print(f"Skipped saving '{path}' (no figure).")


if __name__ == "__main__":
    # Reuse the one-time GeoJSON load; if not loaded, do it here
    if "gdf_constituencies" not in globals():
        gdf_constituencies = load_geojson(feature_to_add_file="geo_utils/kaziranga.txt")

    if gdf_constituencies is not None:
        if VERBOSE: print("Generating women seats map...")
        fig_women = plot_women_seats_map(gdf=gdf_constituencies)
        _maybe_save(fig_women, "women_seats_won_map.png")

        if VERBOSE: print("Generating female electorate percentage heatmap...")
        fig_female = plot_female_electorate_pct_map(gdf=gdf_constituencies)
        _maybe_save(fig_female, "female_electorate_pct_map.png")
    else:
        if VERBOSE: print("Error: 'gdf_constituencies' GeoDataFrame not found or failed to load.")
