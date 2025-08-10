import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor, RANSACRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')

# --- Set working directory ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("Current working directory set to:", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("Current working directory:", os.getcwd())

# 1. Data loading and preprocessing
def load_and_preprocess_data(filepath):
    """Load data and perform preprocessing"""
    data = pd.read_csv(filepath)
    
    # Handle missing values
    if data.isnull().any().any():
        print("Missing values detected, filling with median...")
        data = data.fillna(data.median(numeric_only=True))
    
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)
    
    return X, y

def adjusted_r2_score(y_true, y_pred, n_features, n_samples):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Define the scorer for scikit-learn
def adjusted_r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return adjusted_r2_score(y, y_pred, X.shape[1], len(y))

# Feature selection function
def feature_selection(X, y):
    """Perform feature selection using recursive feature elimination"""
    print("\n=== Starting feature selection ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Use Gradient Boosting as the base model
    estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Remove 1% of features per iteration
    step = max(1, int(X_scaled.shape[1] * 0.01))
    
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=10,
        scoring=adjusted_r2_scorer,
        min_features_to_select=10,
        n_jobs=-1
    )
    
    try:
        selector.fit(X_scaled, y)
    except Exception as e:
        print(f"Feature selection failed: {str(e)}")
        return X.columns  # Fallback to all features if selection fails
    
    # Get selected features
    selected_features = X_scaled.columns[selector.support_]
    print(f"Original feature count: {X_scaled.shape[1]}, Selected feature count: {len(selected_features)}")
    print("Important features:", selected_features.tolist())
    
    # Plot feature selection results
    plt.figure(figsize=(10, 6))
    n_scores = len(selector.cv_results_['mean_test_score'])
    plt.plot(range(selector.min_features_to_select, selector.min_features_to_select + n_scores), 
             selector.cv_results_['mean_test_score'])
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross Validation Score (Adjusted R²)")
    plt.title("Feature Selection Performance")
    plt.savefig('feature_selection.png')
    plt.close()
    
    return selected_features

def train_and_optimize(X_train, y_train, X_test, y_test):
    """Train and optimize regression models"""
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'SVR': SVR(),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Bayesian Ridge': BayesianRidge(),
        'KNN Regression': KNeighborsRegressor(),
        'PLSR': PLSRegression(),
        'MLP': MLPRegressor(random_state=42),
        'ElasticNet': ElasticNet(),
        'CatBoost': CatBoostRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Passive Aggressive': PassiveAggressiveRegressor(random_state=42),
        'Theil Sen': TheilSenRegressor(random_state=42),
        'Huber': HuberRegressor(),
        'RANSAC': RANSACRegressor(random_state=42)
    }
    
    # Store results
    results = {}
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring=adjusted_r2_score)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            print(f"Cross-validation Adjusted R²: {cv_mean:.4f} (±{cv_std:.4f})")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            results[name] = {
                'CV_R2_mean': cv_mean,
                'CV_R2_std': cv_std,
                'Test_R2': adjusted_r2_score(y_test, y_pred, X_test.shape[1], len(y_test)),
                'Test_MSE': mean_squared_error(y_test, y_pred),
                'Test_MAE': mean_absolute_error(y_test, y_pred),
                'Model': model
            }
            print(f"Test set Adjusted R²: {results[name]['Test_R2']:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = {
                'CV_R2_mean': None,
                'CV_R2_std': None,
                'Test_R2': None,
                'Test_MSE': None,
                'Test_MAE': None,
                'Model': None,
                'Error': str(e)
            }
    
    return results

# 4. Optuna hyperparameter optimization
def optimize_with_optuna(X_train, y_train, model_name, n_trials=100):
    """Perform hyperparameter optimization using Optuna"""
    
    def objective(trial):
        if model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 5, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=42)
            
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 5, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            model = XGBRegressor(**params, random_state=42)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 5, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
            }
            model = LGBMRegressor(**params, random_state=42)

        elif model_name == 'SVR':
            params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            }
            model = SVR(**params)
        
        elif model_name == 'Ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            }
            model = Ridge(**params)

        elif model_name == 'KNN Regression':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            }
            model = KNeighborsRegressor(**params)

        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'thread_count': trial.suggest_int('thread_count', 1, 8),
                'random_seed': trial.suggest_int('random_seed', 1, 1000)
            }
            model = CatBoostRegressor(**params, random_state=42)
        
        score = cross_val_score(model, X_train, y_train, cv=10, scoring=adjusted_r2_scorer, n_jobs=-1).mean()
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nBest {model_name} parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    return study.best_params

def stratified_group_split(data, condition_col, genotype_col, test_size=0.2, random_state=42):
    """
    Perform stratified split based on Condition column, ensuring no overlap of Genotypes between train and test sets.
    
    Parameters:
        data: pandas DataFrame, containing the dataset
        condition_col: str, name of the Condition column
        genotype_col: str, name of the Genotype column
        test_size: float, proportion of test set
        random_state: int, random seed
    
    Returns:
        train_data: Training set DataFrame
        test_data: Test set DataFrame
    """
    # Ensure required columns exist
    if condition_col not in data.columns or genotype_col not in data.columns:
        raise ValueError("Condition or Genotype column not found!")
    
    # Group by Genotype, get Condition values (assuming consistent Condition per group)
    grouped = data.groupby(genotype_col)[condition_col].first().reset_index()
    
    # Perform stratified split based on Condition
    train_groups, test_groups = train_test_split(
        grouped,
        test_size=test_size,
        stratify=grouped[condition_col],
        random_state=random_state
    )
    
    # Extract full train and test sets based on Genotype groups
    train_data = data[data[genotype_col].isin(train_groups[genotype_col])]
    test_data = data[data[genotype_col].isin(test_groups[genotype_col])]
    
    # Verify Condition distribution
    train_condition_dist = train_data[condition_col].value_counts(normalize=True)
    test_condition_dist = test_data[condition_col].value_counts(normalize=True)
    print("\nTraining set Condition distribution:")
    print(train_condition_dist)
    print("\nTest set Condition distribution:")
    print(test_condition_dist)
    
    # Verify no overlapping Genotypes
    train_genotypes = set(train_data[genotype_col])
    test_genotypes = set(test_data[genotype_col])
    overlapping_genotypes = train_genotypes.intersection(test_genotypes)
    if overlapping_genotypes:
        raise ValueError(f"Overlapping Genotypes found: {overlapping_genotypes}")
    else:
        print("\nVerification passed: No overlapping Genotypes between train and test sets")
    
    # Visualize Condition distribution
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    train_condition_dist.plot(kind='bar', title='Train Condition Distribution')
    plt.subplot(1, 2, 2)
    test_condition_dist.plot(kind='bar', title='Test Condition Distribution')
    plt.tight_layout()
    plt.savefig('condition_distribution.png')
    plt.close()
    
    return train_data, test_data

def load_and_split_data(filepath, sheet_name, target_col, random_state, condition_col='Condition', genotype_col='Genotype'):
    """
    Load data and perform stratified group split
    """
    # Load data
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    if data.isnull().any().any():
        print("Missing values detected, filling with median...")
        data = data.fillna(data.median(numeric_only=True))
    check_replicate(data)
    data = data[[col for col in data.columns if col.endswith(('_1', '_2', '_3', '_4', target_col, condition_col, genotype_col))]]
    # Split dataset
    train_data, test_data = stratified_group_split(
        data,
        condition_col=condition_col,
        genotype_col=genotype_col,
        test_size=0.3,
        random_state=random_state
    )
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_col, condition_col, genotype_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col, condition_col, genotype_col])
    y_test = test_data[target_col]
    
    return X_train, X_test, y_train, y_test

def check_replicate(data):
    # Group by Condition and Genotype, count replicates
    replicate_counts = data.groupby(['Condition', 'Genotype']).size().reset_index(name='Replicate_Count')

    # Summarize replicate counts by Condition
    condition_summary = replicate_counts.groupby('Condition')['Replicate_Count'].value_counts().unstack(fill_value=0)

    # Print replicate count statistics
    print("\nReplicate count statistics for Genotypes under each Condition:")
    print(condition_summary)

    # Print detailed information
    conditions = data['Condition'].unique()
    for condition in conditions:
        print(f"\nCondition: {condition}")
        condition_data = replicate_counts[replicate_counts['Condition'] == condition]
        print(f"  Total Genotype count: {len(condition_data)}")
        print(f"  Replicate count distribution:\n{condition_data['Replicate_Count'].value_counts().sort_index()}")
        
        # List Genotypes with non-standard replicate counts
        non_standard = condition_data[condition_data['Replicate_Count'] != 2]
        if not non_standard.empty:
            print(f"  Genotypes with non-2 replicates:")
            for _, row in non_standard.iterrows():
                print(f"    Genotype {row['Genotype']}: {row['Replicate_Count']} replicates")

# 5. Main function
def main():
    dataset_random_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 100, 123, 200, 300, 400, 500, 1234, 2025, 12345]
    all_results = []

    for dataset_random_state in dataset_random_states:
        data_path = 'file2.xlsx'
        sheet_name = "3d_TKW"
        target_col = "1000GrainWeight"

        # Load and split data
        X_train, X_test, y_train, y_test = load_and_split_data(
            data_path,
            sheet_name=sheet_name,
            target_col=target_col,
            random_state=dataset_random_state
        )

        # Feature selection
        selected_features = feature_selection(X_train, y_train)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Standardize features
        scaler = StandardScaler()
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)

        # Train and evaluate base models
        results = train_and_optimize(X_train_selected, y_train, X_test_selected, y_test)

        # Save model results
        results_df = pd.DataFrame(results).T
        results_df.drop(columns=['Model'], inplace=True)
        results_df.rename(columns={'Test_R2': 'Test_Adjusted_R2', 'CV_R2_mean': 'CV_Adjusted_R2_mean'}, inplace=True)
        results_file = f"model_performance_{sheet_name}_rs{dataset_random_state}.csv"
        results_df.to_csv(results_file)
        print(f"\nModel results saved to {results_file}")

        results_df['random_state'] = dataset_random_state
        all_results.append(results_df)

        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['Test_R2'] if results[x]['Test_R2'] is not None else -np.inf)
        print(f"\nSelected best model for optimization: {best_model_name}")

        # Hyperparameter optimization
        best_params = optimize_with_optuna(X_train_selected, y_train, best_model_name)

        # Train final model with optimal parameters
        if best_model_name == 'Random Forest':
            final_model = RandomForestRegressor(**best_params, random_state=42)
        elif best_model_name == 'XGBoost':
            final_model = XGBRegressor(**best_params, random_state=42)
        elif best_model_name == 'LightGBM':
            final_model = LGBMRegressor(**best_params, random_state=42)
        elif best_model_name == 'SVR':
            final_model = SVR(**best_params)
        elif best_model_name == 'Ridge':
            final_model = Ridge(**best_params)
        elif best_model_name == 'KNN Regression':
            final_model = KNeighborsRegressor(**best_params)
        elif best_model_name == 'CatBoost':
            final_model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
        else:
            print(f"Unsupported model: {best_model_name}, skipping final training")
            continue

        # Train final model
        final_model.fit(X_train_selected, y_train)
        y_pred_final = final_model.predict(X_test_selected)
        final_test_r2 = adjusted_r2_score(y_test, y_pred_final, X_test_selected.shape[1], len(y_test))
        print(f"Final model {best_model_name} Test Adjusted R²: {final_test_r2:.4f}")

        # Save final model
        model_filename = f"final_model_{best_model_name.lower().replace(' ', '_')}_rs{dataset_random_state}.joblib"
        joblib.dump(final_model, model_filename)
        print(f"Model saved to {model_filename}")

    # Summarize all results
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(f"all_results_summary_{sheet_name}.csv")
    print(f"\nAll random seed results summarized and saved to all_results_summary_{sheet_name}.csv")

if __name__ == "__main__":
    main()