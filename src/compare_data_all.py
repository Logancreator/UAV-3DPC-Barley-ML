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
    """
    Clean DataFrame by replacing inf/-inf with NaN and filling NaN with 0.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def read_and_prepare_data(pc_file_paths: List[str], ortho_file_paths: List[str], index_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Read CSV files from multiple periods for point cloud and orthophoto data and prepare for comparison.
    """
    if len(pc_file_paths) != len(ortho_file_paths):
        raise ValueError("pc_file_paths and ortho_file_paths must have the same length (one file per period).")

    num_periods: int = len(pc_file_paths)
    pc_data: List[np.ndarray] = []
    ortho_data: List[np.ndarray] = []
    periods: List[str] = []

    for pc_file, ortho_file in zip(pc_file_paths, ortho_file_paths):
        df_ortho: pd.DataFrame = clean_data(pd.read_csv(ortho_file))
        if index_name not in df_ortho.columns:
            raise KeyError(f"'{index_name}' not found in {ortho_file}")
        ortho_data.append(df_ortho[index_name].values)

        df_pc: pd.DataFrame = clean_data(pd.read_csv(pc_file))
        if index_name not in df_pc.columns:
            raise KeyError(f"'{index_name}' not found in {pc_file}")
        pc_data.append(df_pc[index_name].values)

        period: str = os.path.basename(ortho_file).split('_')[-1].replace('.csv', '')
        periods.append(period)

    return pc_data, ortho_data, periods

def compare_data(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                 index_name: str = "Vegetation Index", periods: Optional[List[str]] = None,  
                 show_pdf: bool = False, save_pdf: bool = False, pdf_pages: Optional[PdfPages] = None) -> None:
    """
    Perform statistical, visualization, error, and regression analysis on multi-period point cloud 
    and orthophoto vegetation indices, distributing stats into boxplot and scatter plot.
    """
    if not isinstance(pc_data, list) or not isinstance(ortho_data, list) or len(pc_data) != len(ortho_data):
        raise ValueError("pc_data and ortho_data must be lists of the same length.")
    
    num_periods: int = len(pc_data)
    if periods is None:
        periods = [f"Period {i+1}" for i in range(num_periods)]
    elif len(periods) != num_periods:
        raise ValueError("Length of periods must match the number of data periods.")

    colors: List[str] = sns.color_palette("viridis", n_colors=num_periods)[:num_periods]

    # 1. Statistical Comparison
    def statistical_comparison(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                              periods: List[str]) -> List[dict]:
        stats_data: List[dict] = []
        for i, period in enumerate(periods):
            mean_pc: float = np.mean(pc_data[i])
            std_pc: float = np.std(pc_data[i])
            mean_ortho: float = np.mean(ortho_data[i])
            std_ortho: float = np.std(ortho_data[i])
            corr, _ = pearsonr(pc_data[i], ortho_data[i])
            t_stat, p_value = ttest_ind(pc_data[i], ortho_data[i])
            
            stats_data.append({
                "Period": period,
                "Mean_PC": mean_pc,
                "Std_PC": std_pc,
                "Mean_Ortho": mean_ortho,
                "Std_Ortho": std_ortho,
                "Correlation": corr,
                "T-stat": t_stat,
                "P-value": p_value
            })
        return stats_data

    # 2. Error and Regression Analysis for Stats
    def compute_metrics(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                        periods: List[str], stats_data: List[dict]) -> List[dict]:
        for i, period in enumerate(periods):
            rmse_error: float = np.sqrt(mean_squared_error(pc_data[i], ortho_data[i]))
            mae: float = mean_absolute_error(pc_data[i], ortho_data[i])
            bias: float = np.mean(pc_data[i] - ortho_data[i])
            
            X: np.ndarray = pc_data[i].reshape(-1, 1)
            y: np.ndarray = ortho_data[i].reshape(-1, 1)
            model: LinearRegression = LinearRegression()
            model.fit(X, y)
            y_pred: np.ndarray = model.predict(X)
            rmse_reg: float = np.sqrt(mean_squared_error(y, y_pred))
            r2: float = r2_score(y, y_pred)
            
            stats_data[i].update({
                "RMSE_Error": rmse_error,
                "MAE": mae,
                "Bias": bias,
                "R²": r2,
                "Slope": model.coef_[0][0],
                "Intercept": model.intercept_[0]
            })
        return stats_data

    # 3. Visualization Comparison
    def visualization_comparison(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                                index_name: str, periods: List[str], stats_data: List[dict]) -> None:
        markers: List[str] = ['o', 's']  # Circle for PC, Square for Ortho
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Boxplot with Mean and Std
        box_data: List[np.ndarray] = []
        labels: List[str] = []
        box_colors: List[str] = []
        for i, period in enumerate(periods):
            box_data.append(pc_data[i])
            box_data.append(ortho_data[i])
            labels.append(f"{period}\nPC")
            labels.append(f"{period}\nOrtho")
            box_colors.extend([colors[i], colors[i]])
        
        sns.boxplot(data=box_data, palette=box_colors, width=0.5, ax=axes[0])
        axes[0].set_xticklabels(labels, rotation=45)
        axes[0].set_ylabel(f"{index_name} Values")
        axes[0].set_title(f"Boxplot Comparison ({index_name})")

        for i in range(num_periods):
            x_pc: float = i * 2
            x_ortho: float = i * 2 + 1
            y_pc: float = max(pc_data[i]) * 1.05
            y_ortho: float = max(ortho_data[i]) * 1.05
            axes[0].text(x_pc, y_pc, f"{stats_data[i]['Mean_PC']:.2f}±{stats_data[i]['Std_PC']:.2f}", 
                         ha='center', fontsize=8, color="black")
            axes[0].text(x_ortho, y_ortho, f"{stats_data[i]['Mean_Ortho']:.2f}±{stats_data[i]['Std_Ortho']:.2f}", 
                         ha='center', fontsize=8, color="black")
            x_mid: float = i * 2 + 0.5
            y_mid: float = max(max(pc_data[i]), max(ortho_data[i])) * 0.9
            sig_text: str = f"p={stats_data[i]['P-value']:.3f}*" if stats_data[i]['P-value'] < 0.05 else f"p={stats_data[i]['P-value']:.3f}"
            axes[0].text(x_mid, y_mid, sig_text, ha='center', fontsize=8)

        # Scatter Plot with Regression Metrics
        for i, period in enumerate(periods):
            axes[1].scatter(pc_data[i], ortho_data[i], color=colors[i], marker=markers[0], 
                            alpha=0.6, label=f"{period} PC", s=50)
            axes[1].scatter(pc_data[i], ortho_data[i], color=colors[i], marker=markers[1], 
                            alpha=0.6, label=f"{period} Ortho", s=50)
            X: np.ndarray = pc_data[i].reshape(-1, 1)
            y: np.ndarray = ortho_data[i].reshape(-1, 1)
            model: LinearRegression = LinearRegression()
            model.fit(X, y)
            y_pred: np.ndarray = model.predict(X)
            axes[1].plot(X, y_pred, color=colors[i], lw=2, linestyle='--')

        axes[1].set_xlabel(f"Point Cloud {index_name}")
        axes[1].set_ylabel(f"Orthophoto {index_name}")
        axes[1].set_title(f"Regression Analysis ({index_name})")

        text_str: str = "\n".join(
            f"{period}: y={stats_data[i]['Slope']:.2f}x+{stats_data[i]['Intercept']:.2f}, "
            f"R²={stats_data[i]['R²']:.2f}, RMSE={stats_data[i]['RMSE_Error']:.2f}, "
            f"MAE={stats_data[i]['MAE']:.2f}, Bias={stats_data[i]['Bias']:.2f}"
            for i, period in enumerate(periods)
        )
        axes[1].text(0.95, 0.05, text_str, transform=axes[1].transAxes, fontsize=8, 
                     verticalalignment='bottom', horizontalalignment='right', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[1].legend(loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_pdf and pdf_pages is not None:
            pdf_pages.savefig(fig)
        if show_pdf:
            plt.show()
        plt.close(fig)  # Close the figure to free memory

    # 4. Error Analysis
    def error_analysis(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                      periods: List[str]) -> None:
        for i, period in enumerate(periods):
            rmse: float = np.sqrt(mean_squared_error(pc_data[i], ortho_data[i]))
            mae: float = mean_absolute_error(pc_data[i], ortho_data[i])
            bias: float = np.mean(pc_data[i] - ortho_data[i])

    # 5. Regression Analysis
    def regression_analysis(pc_data: List[np.ndarray], ortho_data: List[np.ndarray], 
                           periods: List[str]) -> None:
        for i, period in enumerate(periods):
            X: np.ndarray = pc_data[i].reshape(-1, 1)
            y: np.ndarray = ortho_data[i].reshape(-1, 1)
            model: LinearRegression = LinearRegression()
            model.fit(X, y)
            y_pred: np.ndarray = model.predict(X)

            rmse: float = np.sqrt(mean_squared_error(y, y_pred))
            r2: float = r2_score(y, y_pred)
            mae: float = mean_absolute_error(y, y_pred)

    # Execute and collect data
    stats_data = statistical_comparison(pc_data, ortho_data, periods)
    stats_data = compute_metrics(pc_data, ortho_data, periods, stats_data)
    visualization_comparison(pc_data, ortho_data, index_name, periods, stats_data)
    error_analysis(pc_data, ortho_data, periods)
    regression_analysis(pc_data, ortho_data, periods)

# Example usage with multiple index names
if __name__ == "__main__":
    global base_path
    base_path: str = r"E:\Git\phenomics\barley\project\uav\data"
    
    # Create a single PDF file for all visualizations
    pdf_filename: str = os.path.join(base_path, "combined_vegetation_indices_comparison.pdf")
    with PdfPages(pdf_filename) as pdf_pages:
        ##### For RGB
        ortho_file_paths: List[str] = [
            os.path.join(base_path, "rgb_zonal_stats_VIs_02-06-24.csv"),
            os.path.join(base_path, "rgb_zonal_stats_VIs_30-06-24.csv"),
            os.path.join(base_path, "rgb_zonal_stats_VIs_21-07-24.csv"),
            os.path.join(base_path, "rgb_zonal_stats_VIs_11-08-24.csv")
        ]
        pc_file_paths: List[str] = [
            os.path.join(base_path, "rgb_threeD_indices_02-06-24.csv"),
            os.path.join(base_path, "rgb_threeD_indices_30-06-24.csv"),
            os.path.join(base_path, "rgb_threeD_indices_21-07-24.csv"),
            os.path.join(base_path, "rgb_threeD_indices_11-08-24.csv")
        ]
        
        # Define multiple index names to process
        dummy_indices = cal_indices.get_rgb_VIs(np.array([0]), np.array([0]), np.array([0]))
        index_names = [name + "_mean" for name in list(dummy_indices.keys())]
        print(index_names)
        
        # Loop through each index name for RGB
        for index_name in index_names:
            print(f"\nProcessing {index_name}...")
            pc_data, ortho_data, periods = read_and_prepare_data(pc_file_paths, ortho_file_paths, index_name)
            compare_data(pc_data, ortho_data, index_name=index_name, periods=periods, 
                         show_pdf=False, save_pdf=True, pdf_pages=pdf_pages)
            print(f"\n--------------------------")
        
        ##### For MSI
        ortho_file_paths: List[str] = [
            os.path.join(base_path, "msi_zonal_stats_VIs_02-06-24.csv"),
            os.path.join(base_path, "msi_zonal_stats_VIs_30-06-24.csv"),
            os.path.join(base_path, "msi_zonal_stats_VIs_21-07-24.csv"),
            os.path.join(base_path, "msi_zonal_stats_VIs_11-08-24.csv")
        ]
        pc_file_paths: List[str] = [
            os.path.join(base_path, "msi_threeD_indices_02-06-24.csv"),
            os.path.join(base_path, "msi_threeD_indices_30-06-24.csv"),
            os.path.join(base_path, "msi_threeD_indices_21-07-24.csv"),
            os.path.join(base_path, "msi_threeD_indices_11-08-24.csv")
        ]
        
        # Define multiple index names to process
        dummy_indices = cal_indices.get_msi_VIs(np.array([0]), np.array([0]), np.array([0]), 
                        np.array([0]), np.array([0]), np.array([0]))
        index_names = [name + "_mean" for name in list(dummy_indices.keys())]
        print(index_names)
        
        # Loop through each index name for MSI
        for index_name in index_names:
            print(f"\nProcessing {index_name}...")
            pc_data, ortho_data, periods = read_and_prepare_data(pc_file_paths, ortho_file_paths, index_name)
            compare_data(pc_data, ortho_data, index_name=index_name, periods=periods, 
                         show_pdf=False, save_pdf=True, pdf_pages=pdf_pages)
            print(f"\n--------------------------")
    
    print(f"Saved all visualizations to {pdf_filename}")