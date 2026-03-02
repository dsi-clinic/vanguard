import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_pvd(csv_name, labels_path, title_prefix, output_name):
    if not os.path.exists(csv_name):
        print(f"File not found: {csv_name}")
        return None

    results_df = pd.read_csv(csv_name)
    labels_df = pd.read_csv(labels_path)

    results_df.columns = [c.lower().strip() for c in results_df.columns]
    labels_df.columns = [c.lower().strip() for c in labels_df.columns]

    results_df['join_id'] = results_df.iloc[:, 0].astype(str).str.replace('ISPY2_', '').str.replace('DUKE_', '').str.strip()
    labels_df['join_id'] = labels_df.iloc[:, 0].astype(str).str.replace('ISPY2_', '').str.replace('DUKE_', '').str.strip()

    results_df = results_df.drop_duplicates(subset=['join_id'])
    labels_df = labels_df.drop_duplicates(subset=['join_id'])

    df = pd.merge(results_df, labels_df, on='join_id', how='inner')
    
    print(f"\n--- {title_prefix} Merge Diagnostics ---")
    print(f"Unique IDs in results: {len(results_df)}")
    print(f"Unique IDs in labels:  {len(labels_df)}")
    print(f"Matches found:         {len(df)}")

    if df.empty:
        print(f"No overlapping IDs found for {title_prefix}.")
        return None

    pvd_col = [c for c in df.columns if 'pvd' in c][0]
    pcr_col = 'pcr' if 'pcr' in df.columns else [c for c in df.columns if 'pcr' in c][0]
    
    df = df[df[pvd_col] > 0]
    group_counts = df[pcr_col].value_counts()
    print(f"Counts per pCR group:\n{group_counts}")

    plt.figure(figsize=(10, 6))
    use_kde = all(group_counts > 1)
    
    sns.histplot(data=df, x=pvd_col, hue=pcr_col, kde=use_kde, element="step", palette="viridis")
    
    plt.title(f"{title_prefix} Cohort (n={len(df)}): PVD by pCR Outcome")
    plt.xlabel("Peritumoral Vessel Density (Voxels)")
    plt.ylabel("Frequency")
    
    plt.savefig(output_name)
    print(f"{title_prefix} plot saved to {output_name}")
    print(df.groupby(pcr_col)[pvd_col].describe())
    
    return df[['join_id', pvd_col, pcr_col]]

if __name__ == "__main__":
    LABELS_FILE = "/net/projects2/vanguard/MAMA-MIA-syn60868042/pcr_labels.csv"
    
    duke_df = plot_pvd("pvd_results_duke.csv", LABELS_FILE, "Duke", "duke_pvd.png")
    ispy2_df = plot_pvd("pvd_results_ispy2.csv", LABELS_FILE, "ISPY2", "ispy2_pvd.png")

    if duke_df is not None and ispy2_df is not None:
        combined = pd.concat([duke_df, ispy2_df], ignore_index=True)
        combined.to_csv("combined_vanguard_features.csv", index=False)