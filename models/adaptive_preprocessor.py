import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup (assuming script is run from its directory) ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("当前工作路径已设置为：", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("当前工作路径：", os.getcwd())
# --- End Setup ---

class AdaptiveDataPreprocessor:
    def __init__(self, final_file, sheet_number, target_col, time_points=4, corr_threshold=0.99, vif_threshold=20, alpha=0.05):
        self.final_file = final_file
        self.sheet_number = sheet_number
        self.target_col = target_col
        self.time_points = time_points
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.alpha = alpha

        # Data representations
        self.X_wide_initial = None  # Keep original wide features for validation
        self.X_wide = None          # Wide-format feature data (samples x (features * time_points)), gets filtered
        self.X_by_time = []         # List of DFs, one per time point (samples x features), gets filtered
        self.X_long = None          # Stacked format ((samples * time_points) x features)
        self.y = None               # Target variable (samples x 1)
        self.y_long = None          # Target variable repeated for long format ((samples * time_points) x 1)

        # Feature tracking
        self.base_feature_names = []  # Base names (e.g., 'NDVI', 'EVI')
        self.wide_feature_map = {}    # Map: base_feature -> list of wide_feature_names
        self.features_to_keep_wide = []  # List of final wide feature names to keep
        self.features_per_timepoint = None

        # Analysis results
        self.use_pearson = False
        self.dropped_by_correlation = set()  # Store base features dropped
        self.dropped_by_vif = set()          # Store wide features dropped

        logging.info("Preprocessor initialized.")

    def _check_files(self):
        if not os.path.exists(self.final_file):
            raise FileNotFoundError(f"Final file not found: {self.final_file}")
        logging.info("Input file checked successfully.")

    def load_and_prepare_data(self):
        """Loads data from a single file, handles missing values, and creates different data views."""
        self._check_files()

        # Load data from the single Excel file
        df_data = pd.read_excel(self.final_file,sheet_name= self.sheet_number)
        
        # Check if target column exists
        if self.target_col not in df_data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in {self.final_file}")
        
        # Extract feature columns (exclude target, date, plot_id, Object_ID, Unnamed: 0)
        feature_cols = [col for col in df_data.columns if col.endswith(('_1', '_2', '_3', '_4', '_5'))]
        self.X_wide = df_data[feature_cols]
        self.X_wide_initial = self.X_wide.copy()  # Store original for validation
        logging.info(f"Loaded feature data (wide): {self.X_wide.shape}")

        # Extract target data
        self.y = df_data[self.target_col]
        logging.info(f"Loaded target data '{self.target_col}': {self.y.shape}")

        # Basic Alignment Check
        if len(self.X_wide) != len(self.y):
            raise ValueError(f"Feature data ({len(self.X_wide)} rows) and target data ({len(self.y)} rows) have different lengths.")

        # Handle missing values
        missing_ratio_X = self.X_wide.isnull().mean()
        if missing_ratio_X.max() > 0.5:
            logging.warning(f"Feature columns have > 50% missing values: {missing_ratio_X[missing_ratio_X > 0.5].index.tolist()}. Imputing with mean.")
        if missing_ratio_X.max() > 0:
            self.X_wide.fillna(self.X_wide.mean(), inplace=True)
            logging.info("Missing values in features imputed with column means.")

        if self.y.isnull().any():
            raise ValueError("Missing values detected in target variable. Please clean the target data.")

        # --- Feature Structure ---
        total_features = self.X_wide.shape[1]
        self.features_per_timepoint = total_features // self.time_points

        if total_features % self.time_points != 0:
            raise ValueError(f"Total features ({total_features}) is not evenly divisible by time points ({self.time_points}).")
        logging.info(f"Assuming {self.features_per_timepoint} features per time point.")

        # Create base feature names
        self.base_feature_names = [f"base_feature_{i+1}" for i in range(self.features_per_timepoint)]

        # Build wide_feature_map
        self.wide_feature_map = {base_name: [] for base_name in self.base_feature_names}
        for t in range(self.time_points):
            start_idx = t * self.features_per_timepoint
            end_idx = (t + 1) * self.features_per_timepoint
            cols_for_t = self.X_wide.columns[start_idx:end_idx].tolist()
            for i, base_name in enumerate(self.base_feature_names):
                self.wide_feature_map[base_name].append(cols_for_t[i])

        # Create X_by_time
        self.X_by_time = []
        n_samples = self.X_wide.shape[0]
        for t in range(self.time_points):
            start_idx = t * self.features_per_timepoint
            end_idx = (t + 1) * self.features_per_timepoint
            cols_for_t = self.X_wide.columns[start_idx:end_idx].tolist()
            df_t = self.X_wide[cols_for_t].copy()
            df_t.columns = self.base_feature_names
            self.X_by_time.append(df_t)

        logging.info(f"Created X_by_time with {len(self.X_by_time)} DataFrames, each with shape ~({n_samples}, {len(self.base_feature_names)})")

        # Create X_long and y_long
        if self.X_by_time:
            self.X_long = pd.concat(self.X_by_time, ignore_index=True)
            self.y_long = pd.concat([self.y] * len(self.X_by_time), ignore_index=True)
            logging.info(f"Created X_long (stacked): {self.X_long.shape}")
            logging.info(f"Created y_long (stacked): {self.y_long.shape}")
        else:
            logging.error("Cannot create X_long because X_by_time is empty.")
            raise ValueError("Failed to create X_by_time.")

        # Initialize features to keep
        self.features_to_keep_wide = self.X_wide.columns.tolist()

    def check_normality(self):
        """Checks normality of base features (using X_long) to decide correlation method."""
        if self.X_long is None:
            logging.warning("X_long not available, cannot perform normality check. Defaulting to Spearman.")
            self.use_pearson = False
            return

        normal_columns = []
        for col in self.X_long.columns:
            data_clean = self.X_long[col].dropna()
            if len(data_clean) >= 3:
                stat, p_value = shapiro(data_clean)
                if p_value > self.alpha:
                    normal_columns.append(col)
            else:
                logging.warning(f"Not enough data points ({len(data_clean)}) for normality test on base feature '{col}'. Skipping.")

        if len(self.X_long.columns) > 0:
            self.use_pearson = len(normal_columns) > len(self.X_long.columns) / 2
            logging.info(f"Normality check on base features (X_long): {len(normal_columns)}/{len(self.X_long.columns)} are normally distributed (alpha={self.alpha}).")
        else:
            self.use_pearson = False
            logging.warning("No columns in X_long for normality check.")

        logging.info(f"Correlation method set to: {'Pearson' if self.use_pearson else 'Spearman'}")

    def _get_avg_target_corr(self, base_feature_name):
        """Calculates the average absolute correlation between a base feature's time instances and the target."""
        corrs = []
        if base_feature_name not in self.wide_feature_map:
            logging.warning(f"Base feature '{base_feature_name}' not found in wide_feature_map.")
            return 0

        for wide_feature_name in self.wide_feature_map[base_feature_name]:
            if wide_feature_name in self.X_wide.columns:
                feature_data = self.X_wide[wide_feature_name]
                valid_idx = ~feature_data.isna() & ~self.y.isna()
                if valid_idx.sum() < 2:
                    continue

                try:
                    if self.use_pearson:
                        corr, _ = pearsonr(feature_data[valid_idx], self.y[valid_idx])
                    else:
                        corr, _ = spearmanr(feature_data[valid_idx], self.y[valid_idx])
                    if not np.isnan(corr):
                        corrs.append(abs(corr))
                    else:
                        logging.warning(f"NaN correlation obtained for {wide_feature_name} and target. Skipping.")
                except Exception as e:
                    logging.error(f"Error calculating correlation for {wide_feature_name}: {e}")
            else:
                logging.warning(f"Feature {wide_feature_name} not found in X_wide.")

        return np.mean(corrs) if corrs else 0

    def _select_features_by_correlation(self):
        """Identifies highly correlated base feature pairs and drops the one with lower target correlation."""
        if self.X_long is None:
            logging.error("X_long is not available. Cannot perform correlation-based feature selection.")
            return

        if self.X_long.shape[1] < 2:
            logging.info("Not enough base features (< 2) to perform correlation selection.")
            return

        logging.info(f"Starting correlation-based feature selection (threshold={self.corr_threshold})...")
        corr_method = 'pearson' if self.use_pearson else 'spearman'
        corr_matrix_long = self.X_long.corr(method=corr_method).abs()
        upper_tri = corr_matrix_long.where(np.triu(np.ones(corr_matrix_long.shape), k=1).astype(bool))

        base_features_to_drop = set()
        for feat1 in upper_tri.columns:
            for feat2 in upper_tri.index:
                if feat2 != feat1 and upper_tri.loc[feat2, feat1] > self.corr_threshold:
                    if feat1 in base_features_to_drop or feat2 in base_features_to_drop:
                        continue
                    avg_corr1 = self._get_avg_target_corr(feat1)
                    avg_corr2 = self._get_avg_target_corr(feat2)
                    logging.debug(f"High correlation ({upper_tri.loc[feat2, feat1]:.3f}) between base features: '{feat1}' and '{feat2}'")
                    logging.debug(f"  Avg |Target Corr| for '{feat1}': {avg_corr1:.3f}")
                    logging.debug(f"  Avg |Target Corr| for '{feat2}': {avg_corr2:.3f}")

                    if avg_corr1 >= avg_corr2:
                        base_features_to_drop.add(feat2)
                    else:
                        base_features_to_drop.add(feat1)

        self.dropped_by_correlation = base_features_to_drop
        logging.info(f"Identified {len(self.dropped_by_correlation)} base features to drop due to high correlation: {self.dropped_by_correlation}")

        initial_wide_features = self.features_to_keep_wide[:]
        self.features_to_keep_wide = []
        removed_count = 0
        for base_feature, wide_names in self.wide_feature_map.items():
            if base_feature not in self.dropped_by_correlation:
                self.features_to_keep_wide.extend([wn for wn in wide_names if wn in initial_wide_features])
            else:
                removed_count += len([wn for wn in wide_names if wn in initial_wide_features])

        logging.info(f"Removed {removed_count} wide features based on base feature correlation analysis.")
        logging.info(f"Features remaining after correlation filter: {len(self.features_to_keep_wide)}")

    def _select_features_by_vif(self):
        """Iteratively removes features with high VIF within each time point's data."""
        logging.info(f"Starting VIF-based feature selection (threshold={self.vif_threshold})...")

        if not self.X_by_time:
            logging.error("X_by_time is empty. Cannot perform VIF selection.")
            return

        current_features_to_keep = set(self.features_to_keep_wide)
        all_vif_dropped_wide = set()

        for t, df_t_original in enumerate(self.X_by_time):
            time_point_label = f"Time Point {t+1}"
            logging.info(f"\n--- Processing {time_point_label} ---")
            wide_names_t = [wide_names[t] for base_feat, wide_names in self.wide_feature_map.items() if t < len(wide_names)]
            features_for_vif_t = [f for f in wide_names_t if f in current_features_to_keep]

            if len(features_for_vif_t) < 2:
                logging.warning(f"{time_point_label}: Skipping VIF check, only {len(features_for_vif_t)} features remaining.")
                continue

            X_subset = self.X_wide[features_for_vif_t].copy()

            while True:
                if X_subset.shape[1] < 2:
                    logging.warning(f"{time_point_label}: Less than 2 features remaining, stopping VIF iteration.")
                    break

                if (X_subset.std() == 0).any():
                    constant_cols = X_subset.columns[X_subset.std() == 0]
                    logging.warning(f"{time_point_label}: Constant columns found: {constant_cols.tolist()}. Removing them before VIF calculation.")
                    X_subset.drop(columns=constant_cols, inplace=True)
                    for col in constant_cols:
                        if col in current_features_to_keep:
                            current_features_to_keep.remove(col)
                            all_vif_dropped_wide.add(col)
                    if X_subset.shape[1] < 2:
                        logging.warning(f"{time_point_label}: Less than 2 features after removing constant columns.")
                        break

                scaler = StandardScaler()
                try:
                    X_scaled = scaler.fit_transform(X_subset)
                except ValueError as e:
                    logging.error(f"{time_point_label}: Error during scaling: {e}. Stopping VIF for this time point.")
                    break

                try:
                    vif_data = pd.DataFrame()
                    vif_data["Feature"] = X_subset.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
                    vif_data = vif_data.sort_values("VIF", ascending=False)
                except Exception as e:
                    logging.error(f"{time_point_label}: Error calculating VIF: {e}. Stopping VIF for this time point.")
                    break

                max_vif = vif_data["VIF"].iloc[0]
                if max_vif <= self.vif_threshold:
                    logging.info(f"{time_point_label}: VIF check passed. Max VIF = {max_vif:.2f} <= {self.vif_threshold}.")
                    break
                else:
                    drop_feature = vif_data["Feature"].iloc[0]
                    logging.info(f"{time_point_label}: Dropping '{drop_feature}' (VIF = {max_vif:.2f})")
                    X_subset = X_subset.drop(columns=drop_feature)
                    if drop_feature in current_features_to_keep:
                        current_features_to_keep.remove(drop_feature)
                        all_vif_dropped_wide.add(drop_feature)

            logging.info(f"{time_point_label}: Finished VIF processing. Features remaining for this time point: {X_subset.shape[1]}")

        self.dropped_by_vif = all_vif_dropped_wide
        self.features_to_keep_wide = sorted(list(current_features_to_keep))
        logging.info(f"Total wide features dropped by VIF: {len(self.dropped_by_vif)}")
        logging.info(f"Features remaining after VIF filter: {len(self.features_to_keep_wide)}")

    def preprocess(self):
        """Runs the full preprocessing pipeline."""
        logging.info("="*30 + " Starting Preprocessing Pipeline " + "="*30)
        self.load_and_prepare_data()
        initial_feature_count = self.X_wide_initial.shape[1]
        logging.info(f"Initial number of wide features: {initial_feature_count}")

        self.check_normality()
        self._select_features_by_correlation()
        self._select_features_by_vif()

        self.X_wide = self.X_wide[self.features_to_keep_wide]
        logging.info(f"Final number of wide features selected: {len(self.features_to_keep_wide)}")
        logging.info("="*30 + " Preprocessing Pipeline Finished " + "="*30)

        return self.features_to_keep_wide

    def get_processed_data(self):
        """Returns the final processed wide feature matrix and the target variable."""
        if self.X_wide is None or self.y is None or not self.features_to_keep_wide:
            raise ValueError("Preprocessing not completed or resulted in no features. Call preprocess() first.")

        if not all(f in self.X_wide.columns for f in self.features_to_keep_wide):
            logging.warning("Mismatch between features_to_keep_wide and X_wide columns. Re-filtering X_wide.")
            self.X_wide = self.X_wide_initial[self.features_to_keep_wide].copy()
            self.X_wide.fillna(self.X_wide.mean(), inplace=True)

        X_processed = self.X_wide
        y_processed = self.y

        logging.info("\n" + "-"*20 + " Validation Summary " + "-"*20)
        logging.info(f"Initial wide features: {self.X_wide_initial.shape[1]}")
        logging.info(f"Final wide features: {X_processed.shape[1]}")
        logging.info(f"Base features dropped by correlation: {len(self.dropped_by_correlation)}")
        logging.info(f"Wide features dropped by VIF: {len(self.dropped_by_vif)}")

        try:
            corr_method = 'pearson' if self.use_pearson else 'spearman'
            if self.X_wide_initial.shape[1] > 1:
                corr_before = self.X_wide_initial.corr(method=corr_method).abs()
                upper_tri_before = corr_before.where(np.triu(np.ones(corr_before.shape), k=1).astype(bool))
                max_corr_before = upper_tri_before.max().max()
                logging.info(f"Max feature correlation BEFORE filtering: {max_corr_before:.4f}")
            else:
                logging.info("Not enough initial features to calculate 'before' correlation.")

            if X_processed.shape[1] > 1:
                corr_after = X_processed.corr(method=corr_method).abs()
                upper_tri_after = corr_after.where(np.triu(np.ones(corr_after.shape), k=1).astype(bool))
                max_corr_after = upper_tri_after.max().max()
                logging.info(f"Max feature correlation AFTER filtering:  {max_corr_after:.4f}")
            else:
                logging.info("Not enough final features to calculate 'after' correlation.")
        except Exception as e:
            logging.error(f"Could not perform correlation validation: {e}")

        logging.info("-"*58)
        return X_processed, y_processed

# --- Usage Example ---
if __name__ == "__main__":
    FINAL_FILE = "file1.xlsx"  # Make sure this file exists
    SHEET_NUMBER ="2d"
    TARGET_COL = "GrainYield"  # Make sure this column exists in the file
    output = 'FS1_2d_GY.csv'
    try:
        preprocessor = AdaptiveDataPreprocessor(
            final_file=FINAL_FILE,
            sheet_number=SHEET_NUMBER,
            target_col=TARGET_COL,
            time_points=4,
            corr_threshold=0.99,
            vif_threshold=20,
            alpha=0.05
        )

        final_feature_list = preprocessor.preprocess()

        if final_feature_list:
            X_final, y_final = preprocessor.get_processed_data()

            print("\n--- Final Processed Data Shapes ---")
            print(f"Features (X_final): {X_final.shape}")
            print(f"Target (y_final):   {y_final.shape}")
            print("\n--- Final Selected Features ---")
            print(final_feature_list)
            print("\n--- First 5 rows of processed data ---")
            print(X_final.head())

            # Combine X_final and y_final and save to CSV
            final_data = pd.concat([X_final, y_final], axis=1)
            final_data.to_csv(output, index=False)
            print("\nFinal processed data saved.")
        else:
            print("\nPreprocessing resulted in no features being selected.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the input file exists in the correct location.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check data integrity, column names, and parameters.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()