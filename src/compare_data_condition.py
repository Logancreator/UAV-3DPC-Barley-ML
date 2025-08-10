import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import cal_indices
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Tuple
from matplotlib.backends.backend_pdf import PdfPages

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by replacing inf/-inf with NaN and filling NaN with 0."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def read_and_prepare_data(pc_file_paths: List[str], ortho_file_paths: List[str], index_name: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str]]:
    """Read CSV files and prepare data with Object_ID for grouping."""
    if len(pc_file_paths) != len(ortho_file_paths):
        raise ValueError("pc_file_paths and ortho_file_paths must have the same length.")

    num_periods = len(pc_file_paths)
    pc_data, ortho_data, periods = [], [], []

    for pc_file, ortho_file in zip(pc_file_paths, ortho_file_paths):
        df_ortho = clean_data(pd.read_csv(ortho_file))
        if index_name not in df_ortho.columns or 'Object_ID' not in df_ortho.columns:
            raise KeyError(f"'{index_name}' or 'Object_ID' not found in {ortho_file}")
        ortho_data.append(df_ortho[['Object_ID', index_name]])

        df_pc = clean_data(pd.read_csv(pc_file))
        if index_name not in df_pc.columns or 'Object_ID' not in df_pc.columns:
            raise KeyError(f"'{index_name}' or 'Object_ID' not found in {pc_file}")
        pc_data.append(df_pc[['Object_ID', index_name]])

        period = os.path.basename(ortho_file).split('_')[-1].replace('.csv', '')
        periods.append(period)

    return pc_data, ortho_data, periods

def split_by_condition(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into normal (1-64) and high nitrogen (65-288) based on Object_ID."""
    normal = df[df['Object_ID'].between(1, 64)][df.columns[-1]].values
    high_n = df[df['Object_ID'].between(65, 288)][df.columns[-1]].values
    return normal, high_n

def compare_data(pc_data: List[pd.DataFrame], ortho_data: List[pd.DataFrame], index_name: str, 
                 periods: List[str], pdf: PdfPages) -> None:
    """Compare 3DPC and ortho data with condition grouping and save to a single PDF."""
    num_periods = len(periods)
    colors = sns.color_palette("viridis", n_colors=num_periods)[:num_periods]

    # Split data by condition
    pc_normal, pc_high_n = [], []
    ortho_normal, ortho_high_n = [], []
    for pc_df, ortho_df in zip(pc_data, ortho_data):
        pc_n, pc_hn = split_by_condition(pc_df)
        ortho_n, ortho_hn = split_by_condition(ortho_df)
        pc_normal.append(pc_n)
        pc_high_n.append(pc_hn)
        ortho_normal.append(ortho_n)
        ortho_high_n.append(ortho_hn)

    # Statistical comparison
    def statistical_comparison(data1: List[np.ndarray], data2: List[np.ndarray], label: str) -> List[dict]:
        stats_data = []
        for i, period in enumerate(periods):
            mean1, std1 = np.mean(data1[i]), np.std(data1[i])
            mean2, std2 = np.mean(data2[i]), np.std(data2[i])
            corr, _ = pearsonr(data1[i], data2[i])
            t_stat, p_value = ttest_ind(data1[i], data2[i])
            stats_data.append({
                "Period": period, "Mean_3DPC": mean1, "Std_3DPC": std1, "Mean_Ortho": mean2, 
                "Std_Ortho": std2, "Correlation": corr, "T-stat": t_stat, "P-value": p_value
            })
        return stats_data

    normal_stats = statistical_comparison(pc_normal, ortho_normal, "Normal")
    high_n_stats = statistical_comparison(pc_high_n, ortho_high_n, "High Nitrogen")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"Comparison of {index_name} (3DPC vs. Ortho)", fontsize=16)

    # Boxplot for Normal Conditions
    box_data_normal = [pc_normal[i] for i in range(num_periods)] + [ortho_normal[i] for i in range(num_periods)]
    labels = [f"{p}\n3DPC" for p in periods] + [f"{p}\nOrtho" for p in periods]
    box_colors = [colors[i] for i in range(num_periods)] * 2
    sns.boxplot(data=box_data_normal, palette=box_colors, width=0.5, ax=axes[0, 0])
    axes[0, 0].set_xticklabels(labels, rotation=45)
    axes[0, 0].set_title("Normal Conditions")
    axes[0, 0].set_ylabel(f"{index_name} Values")
    for i in range(num_periods):
        x_pc, x_ortho = i * 2, i * 2 + 1
        y_pc = max(pc_normal[i]) * 1.05 if len(pc_normal[i]) > 0 else 0
        y_ortho = max(ortho_normal[i]) * 1.05 if len(ortho_normal[i]) > 0 else 0
        axes[0, 0].text(x_pc, y_pc, f"{normal_stats[i]['Mean_3DPC']:.2f}±{normal_stats[i]['Std_3DPC']:.2f}", 
                        ha='center', fontsize=8)
        axes[0, 0].text(x_ortho, y_ortho, f"{normal_stats[i]['Mean_Ortho']:.2f}±{normal_stats[i]['Std_Ortho']:.2f}", 
                        ha='center', fontsize=8)
        x_mid = i * 2 + 0.5
        y_mid = max(y_pc, y_ortho) * 0.9
        sig_text = f"p={normal_stats[i]['P-value']:.3f}*" if normal_stats[i]['P-value'] < 0.05 else f"p={normal_stats[i]['P-value']:.3f}"
        axes[0, 0].text(x_mid, y_mid, sig_text, ha='center', fontsize=8)

    # Boxplot for High Nitrogen Conditions
    box_data_high_n = [pc_high_n[i] for i in range(num_periods)] + [ortho_high_n[i] for i in range(num_periods)]
    sns.boxplot(data=box_data_high_n, palette=box_colors, width=0.5, ax=axes[0, 1])
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].set_title("High Nitrogen Conditions")
    axes[0, 1].set_ylabel(f"{index_name} Values")
    for i in range(num_periods):
        x_pc, x_ortho = i * 2, i * 2 + 1
        y_pc = max(pc_high_n[i]) * 1.05 if len(pc_high_n[i]) > 0 else 0
        y_ortho = max(ortho_high_n[i]) * 1.05 if len(ortho_high_n[i]) > 0 else 0
        axes[0, 1].text(x_pc, y_pc, f"{high_n_stats[i]['Mean_3DPC']:.2f}±{high_n_stats[i]['Std_3DPC']:.2f}", 
                        ha='center', fontsize=8)
        axes[0, 1].text(x_ortho, y_ortho, f"{high_n_stats[i]['Mean_Ortho']:.2f}±{high_n_stats[i]['Std_Ortho']:.2f}", 
                        ha='center', fontsize=8)
        x_mid = i * 2 + 0.5
        y_mid = max(y_pc, y_ortho) * 0.9
        sig_text = f"p={high_n_stats[i]['P-value']:.3f}*" if high_n_stats[i]['P-value'] < 0.05 else f"p={high_n_stats[i]['P-value']:.3f}"
        axes[0, 1].text(x_mid, y_mid, sig_text, ha='center', fontsize=8)

    # Scatter Plot for Normal Conditions (3DPC vs. Ortho)
    for i, period in enumerate(periods):
        axes[1, 0].scatter(pc_normal[i], ortho_normal[i], color=colors[i], marker='o', alpha=0.6, 
                           label=f"{period}", s=50)
        if len(pc_normal[i]) > 1 and len(ortho_normal[i]) > 1:  # Ensure enough data for regression
            X, y = pc_normal[i].reshape(-1, 1), ortho_normal[i]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            axes[1, 0].plot(X, y_pred, color=colors[i], lw=2, linestyle='--')
    axes[1, 0].set_xlabel(f"3DPC {index_name}")
    axes[1, 0].set_ylabel(f"Ortho {index_name}")
    axes[1, 0].set_title("Normal Conditions (3DPC vs. Ortho)")
    axes[1, 0].legend(loc='upper left', fontsize=8)

    # Scatter Plot for High Nitrogen Conditions (3DPC vs. Ortho)
    for i, period in enumerate(periods):
        axes[1, 1].scatter(pc_high_n[i], ortho_high_n[i], color=colors[i], marker='o', alpha=0.6, 
                           label=f"{period}", s=50)
        if len(pc_high_n[i]) > 1 and len(ortho_high_n[i]) > 1:  # Ensure enough data for regression
            X, y = pc_high_n[i].reshape(-1, 1), ortho_high_n[i]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            axes[1, 1].plot(X, y_pred, color=colors[i], lw=2, linestyle='--')
    axes[1, 1].set_xlabel(f"3DPC {index_name}")
    axes[1, 1].set_ylabel(f"Ortho {index_name}")
    axes[1, 1].set_title("High Nitrogen Conditions (3DPC vs. Ortho)")
    axes[1, 1].legend(loc='upper left', fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    base_path = r"E:\Git\phenomics\barley\project\uav\data"
    pdf_path = os.path.join(base_path, "vegetation_indices_comparison.pdf")

    # RGB Data
    ortho_file_paths_rgb = [
        os.path.join(base_path, "rgb_zonal_stats_VIs_02-06-24.csv"),
        os.path.join(base_path, "rgb_zonal_stats_VIs_30-06-24.csv"),
        os.path.join(base_path, "rgb_zonal_stats_VIs_21-07-24.csv"),
        os.path.join(base_path, "rgb_zonal_stats_VIs_11-08-24.csv")
    ]
    pc_file_paths_rgb = [
        os.path.join(base_path, "rgb_threeD_indices_02-06-24.csv"),
        os.path.join(base_path, "rgb_threeD_indices_30-06-24.csv"),
        os.path.join(base_path, "rgb_threeD_indices_21-07-24.csv"),
        os.path.join(base_path, "rgb_threeD_indices_11-08-24.csv")
    ]
    dummy_rgb = cal_indices.get_rgb_VIs(np.array([0]), np.array([0]), np.array([0]))
    rgb_index_names = [name + "_mean" for name in dummy_rgb.keys()]

    # MSI Data
    ortho_file_paths_msi = [
        os.path.join(base_path, "msi_zonal_stats_VIs_02-06-24.csv"),
        os.path.join(base_path, "msi_zonal_stats_VIs_30-06-24.csv"),
        os.path.join(base_path, "msi_zonal_stats_VIs_21-07-24.csv"),
        os.path.join(base_path, "msi_zonal_stats_VIs_11-08-24.csv")
    ]
    pc_file_paths_msi = [
        os.path.join(base_path, "msi_threeD_indices_02-06-24.csv"),
        os.path.join(base_path, "msi_threeD_indices_30-06-24.csv"),
        os.path.join(base_path, "msi_threeD_indices_21-07-24.csv"),
        os.path.join(base_path, "msi_threeD_indices_11-08-24.csv")
    ]
    dummy_msi = cal_indices.get_msi_VIs(np.array([0]), np.array([0]), np.array([0]), 
                                        np.array([0]), np.array([0]), np.array([0]))
    msi_index_names = [name + "_mean" for name in dummy_msi.keys()]

    with PdfPages(pdf_path) as pdf:
        # Process RGB indices
        for index_name in rgb_index_names:
            print(f"Processing {index_name} (RGB)...")
            pc_data, ortho_data, periods = read_and_prepare_data(pc_file_paths_rgb, ortho_file_paths_rgb, index_name)
            compare_data(pc_data, ortho_data, index_name, periods, pdf)

        # Process MSI indices
        for index_name in msi_index_names:
            print(f"Processing {index_name} (MSI)...")
            pc_data, ortho_data, periods = read_and_prepare_data(pc_file_paths_msi, ortho_file_paths_msi, index_name)
            compare_data(pc_data, ortho_data, index_name, periods, pdf)

    print(f"All comparisons saved to {pdf_path}")