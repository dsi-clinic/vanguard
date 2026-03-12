"""Module for plotting and merging Perivascular Density (PVD) metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent


def plot_pvd(
    csv_name: str, labels_path: str, title_prefix: str, output_name: str
) -> pd.DataFrame | None:
    """Merge PVD results with clinical labels and plot the distribution.

    Args:
        csv_name: Path to the PVD results CSV.
        labels_path: Path to the labels CSV.
        title_prefix: Cohort name for plot titles.
        output_name: Filename to save the plot.

    Returns:
        A merged DataFrame containing join_id, PVD, and pCR, or None if empty.
    """
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = OUTPUT_DIR / csv_path
    if not csv_path.exists():
        print(f"File not found: {csv_name}")
        return None

    results_df = pd.read_csv(csv_path)
    labels_df = pd.read_csv(labels_path)

    results_df.columns = [c.lower().strip() for c in results_df.columns]
    labels_df.columns = [c.lower().strip() for c in labels_df.columns]

    def clean_id(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.replace("ISPY2_", "")
            .str.replace("DUKE_", "")
            .str.strip()
        )

    results_df["join_id"] = clean_id(results_df.iloc[:, 0])
    labels_df["join_id"] = clean_id(labels_df.iloc[:, 0])

    results_df = results_df.drop_duplicates(subset=["join_id"])
    labels_df = labels_df.drop_duplicates(subset=["join_id"])

    merged_data = results_df.merge(labels_df, on="join_id", how="inner")

    print(f"\n--- {title_prefix} Merge Diagnostics ---")
    print(f"Unique IDs in results: {len(results_df)}")
    print(f"Unique IDs in labels:  {len(labels_df)}")
    print(f"Matches found:         {len(merged_data)}")

    if merged_data.empty:
        print(f"No overlapping IDs found for {title_prefix}.")
        return None

    pvd_col = [c for c in merged_data.columns if "pvd" in c][0]
    pcr_col = (
        "pcr"
        if "pcr" in merged_data.columns
        else [c for c in merged_data.columns if "pcr" in c][0]
    )

    plot_data = merged_data[merged_data[pvd_col] > 0].copy()
    group_counts = plot_data[pcr_col].value_counts()
    print(f"Counts per pCR group:\n{group_counts}")

    plt.figure(figsize=(10, 6))
    use_kde = all(group_counts > 1)

    sns.histplot(
        data=plot_data,
        x=pvd_col,
        hue=pcr_col,
        kde=use_kde,
        element="step",
        palette="viridis",
    )

    plt.title(f"{title_prefix} Cohort (n={len(plot_data)}): PVD by pCR Outcome")
    plt.xlabel("Peritumoral Vessel Density (Voxels)")
    plt.ylabel("Frequency")

    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    plt.savefig(output_path)
    print(f"{title_prefix} plot saved to {output_name}")
    print(plot_data.groupby(pcr_col)[pvd_col].describe())

    return plot_data[["join_id", pvd_col, pcr_col]]


if __name__ == "__main__":
    LABELS_FILE = "/net/projects2/vanguard/MAMA-MIA-syn60868042/pcr_labels.csv"

    duke_data = plot_pvd("pvd_results_duke.csv", LABELS_FILE, "Duke", "duke_pvd.png")
    ispy2_data = plot_pvd(
        "pvd_results_ispy2.csv", LABELS_FILE, "ISPY2", "ispy2_pvd.png"
    )

    if duke_data is not None and ispy2_data is not None:
        combined_features = pd.concat([duke_data, ispy2_data], ignore_index=True)
        combined_features.to_csv(
            OUTPUT_DIR / "combined_vanguard_features.csv", index=False
        )
