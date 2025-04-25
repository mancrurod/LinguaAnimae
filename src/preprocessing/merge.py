"""
Merge all cleaned CSVs inside a specified subfolder of 'data/processed/' into a single CSV file.

The merged file will:
- Remove the original 'id' columns temporarily to avoid duplication.
- Preserve the 'verse_id' field.
- Generate a new sequential numeric 'id' as the first column.
- Merge CSVs following the correct biblical order based on file naming (e.g., 1_genesis_cleaned.csv, 2_exodo_cleaned.csv).
- Save the result inside the same folder, named after the subfolder (e.g., bible_rv60.csv).

Args:
    None

Returns:
    None
"""

# =====================
# === IMPORTS =========
# =====================

import pandas as pd
from pathlib import Path

# ========================
# === CONSTANTS ==========
# ========================

PROCESSED_DIR = Path("data/processed")

# =============================
# === MERGING FUNCTION ========
# =============================

def combine_cleaned_csvs(processed_subdir: Path) -> None:
    """
    Combine all cleaned CSV files inside a given subfolder of 'data/processed/' into a single CSV.
    The final file will have a new global 'id' column and will preserve the 'verse_id' field.
    Original 'id' columns from individual files will be removed during merging (only for the unified file).

    Args:
        processed_subdir (Path): Path to the subfolder inside 'data/processed/' (e.g., Path("data/processed/bible_rv60")).

    Returns:
        None
    """
    if not processed_subdir.exists():
        print(f"❌ The directory {processed_subdir} does not exist.")
        return

    all_csvs = list(processed_subdir.glob("*_cleaned.csv"))
    if not all_csvs:
        print(f"❌ No cleaned CSV files found in {processed_subdir}")
        return

    # Sort files based on the numeric prefix in the filename
    def extract_prefix(file: Path) -> int:
        try:
            return int(file.stem.split("_")[0])
        except (IndexError, ValueError):
            return float('inf')  # Put malformed filenames at the end

    all_csvs = sorted(all_csvs, key=extract_prefix)

    print(f"📦 Found {len(all_csvs)} cleaned CSV files in {processed_subdir}")

    df_list = []

    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            
            # Drop the existing 'id' column temporarily to avoid duplication
            if "id" in df.columns:
                df = df.drop(columns=["id"])

            df_list.append(df)
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")

    if not df_list:
        print("⚠️ No valid DataFrames to merge.")
        return

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Insert new sequential 'id' as the first column
    combined_df.insert(0, "id", range(1, len(combined_df) + 1))

    # Define the output file name based on the folder name
    folder_name = processed_subdir.name
    output_path = processed_subdir / f"{folder_name}.csv"

    # Save the unified DataFrame
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Combined CSV saved to: {output_path.relative_to(PROCESSED_DIR)}")

# ======================
# === ENTRY POINT ======
# ======================

if __name__ == "__main__":
    # Example usage. You can replace 'bible_rv60' with any other subfolder you want to merge.
    target_folder = PROCESSED_DIR / "bible_rv60"
    combine_cleaned_csvs(target_folder)