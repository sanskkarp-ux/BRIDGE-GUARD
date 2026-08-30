"""Pilot-phase inspection of a raw NBI delimited file.

Not a cleaning/parsing pipeline yet — this is deliberately just eyes-on-data:
shape, columns, missingness patterns, and a sanity check that our assumed
"overall condition" convention (min of deck/superstructure/substructure)
actually matches what FHWA itself computes in the file.
"""

from pathlib import Path

import pandas as pd

KEY_COLUMNS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109",
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B",
    "YEAR_BUILT_027",
    "YEAR_RECONSTRUCTED_106",
    "BRIDGE_CONDITION",
    "LOWEST_RATING",
]


def load_raw(path: Path) -> pd.DataFrame:
    """Load an NBI delimited file. Everything as string — no numeric coercion
    yet, because condition columns use 'N' for not-applicable and we don't
    want pandas silently turning that into NaN vs. a real missing value."""
    return pd.read_csv(path, dtype=str, quotechar="'")


def basic_report(df: pd.DataFrame) -> None:
    print("shape:", df.shape)
    print()
    print("null counts (key columns):")
    print(df[KEY_COLUMNS].isnull().sum())
    print()
    print("BRIDGE_CONDITION value counts:")
    print(df["BRIDGE_CONDITION"].value_counts(dropna=False))
    print()
    print("DECK_COND_058 value counts:")
    print(df["DECK_COND_058"].value_counts(dropna=False))


def validate_lowest_rating_convention(df: pd.DataFrame) -> pd.DataFrame:
    """Check whether min(deck, superstructure, substructure) equals FHWA's
    own LOWEST_RATING column, for non-culvert bridges. Returns mismatches."""
    non_culvert = df[df["CULVERT_COND_062"] == "N"].copy()
    for col in ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "LOWEST_RATING"]:
        non_culvert[col + "_n"] = pd.to_numeric(non_culvert[col], errors="coerce")

    non_culvert["computed_min"] = non_culvert[
        ["DECK_COND_058_n", "SUPERSTRUCTURE_COND_059_n", "SUBSTRUCTURE_COND_060_n"]
    ].min(axis=1)

    mismatches = non_culvert[non_culvert["computed_min"] != non_culvert["LOWEST_RATING_n"]]
    return mismatches


if __name__ == "__main__":
    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "2023" / "DE23.txt"
    df = load_raw(raw_path)
    basic_report(df)

    mismatches = validate_lowest_rating_convention(df)
    print()
    print(f"min(deck,super,sub) vs FHWA LOWEST_RATING mismatches: {len(mismatches)}")
