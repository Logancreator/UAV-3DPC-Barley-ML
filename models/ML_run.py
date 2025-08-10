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


# --- 设置工作目录 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("当前工作路径已设置为：", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("当前工作路径：", os.getcwd())

# 1. 数据加载与预处理
def load_and_preprocess_data(filepath):
    """加载数据并进行预处理"""
    data = pd.read_csv(filepath)
    
    # 处理缺失值
    if data.isnull().any().any():
        print("发现缺失值，使用中位数填补...")
        data = data.fillna(data.median(numeric_only=True))
    
    # 分离特征和目标
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # 分类变量编码
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)
    
    return X, y


# def adjusted_r2_scorer(estimator, X, y):
#     y_pred = estimator.predict(X)
#     return adjusted_r2_score(y, y_pred, X.shape[1], len(y))
# adjusted_r2_scorer = make_scorer(adjusted_r2_scorer, greater_is_better=True)

# # 2. 特征选择
# def feature_selection(X, y):
#     """使用递归特征消除进行特征选择"""
#     print("\n=== 开始特征选择 ===")
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
#     # 使用随机森林作为基础模型
#     estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
#     # 设置每次迭代删除1%的特征
#     step = max(1, int(X_scaled.shape[1] * 0.01))
    
#     selector = RFECV(
#         estimator=estimator,
#         step=step,
#         cv=10,
#         scoring=adjusted_r2_scorer,
#         min_features_to_select=10,
#         n_jobs=-1)
    
#     selector.fit(X_scaled, y)
    
#     # 获取选择的特征
#     selected_features = X_scaled.columns[selector.support_]
#     print(f"原始特征数: {X_scaled.shape[1]}, 选择后特征数: {len(selected_features)}")
#     print("重要特征:", selected_features.tolist())
    
#     # 绘制特征选择结果
#     plt.figure(figsize=(10, 6))
#     n_scores = len(selector.cv_results_['mean_test_score'])
#     plt.plot(range(selector.min_features_to_select, selector.min_features_to_select + n_scores), 
#             selector.cv_results_['mean_test_score'])
#     plt.xlabel("Number of Features Selected")
#     plt.ylabel("Cross Validation Score (Adjusted R²)")
#     plt.title("Feature Selection Performance")
#     plt.savefig('feature_selection.png')
#     plt.close()
    
#     return selected_features

def adjusted_r2_score(y_true, y_pred, n_features, n_samples):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Define the scorer for scikit-learn
def adjusted_r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return adjusted_r2_score(y, y_pred, X.shape[1], len(y))

# Feature selection function
def feature_selection(X, y):
    """使用递归特征消除进行特征选择"""
    print("\n=== 开始特征选择 ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 使用随机森林作为基础模型
    estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # 设置每次迭代删除1%的特征
    step = max(1, int(X_scaled.shape[1] * 0.01))
    
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=10,
        scoring=adjusted_r2_scorer,  # Use the correctly defined scorer
        min_features_to_select=10,
        n_jobs=-1
    )
    
    try:
        selector.fit(X_scaled, y)
    except Exception as e:
        print(f"特征选择失败: {str(e)}")
        return X.columns  # Fallback to all features if selection fails
    
    # 获取选择的特征
    selected_features = X_scaled.columns[selector.support_]
    print(f"原始特征数: {X_scaled.shape[1]}, 选择后特征数: {len(selected_features)}")
    print("重要特征:", selected_features.tolist())
    
    # 绘制特征选择结果
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
    """训练和优化回归模型"""
    
    # 定义要测试的模型
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
    
    # 存储结果
    results = {}
    
    for name, model in models.items():
        print(f"\n=== 训练 {name} ===")
        
        try:
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring=adjusted_r2_score)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            print(f"交叉验证 R²: {cv_mean:.4f} (±{cv_std:.4f})")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估
            y_pred = model.predict(X_test)

            results[name] = {
                                'CV_R2_mean': cv_mean,  # Now represents adjusted R²
                                'CV_R2_std': cv_std,
                                'Test_R2': adjusted_r2_score(y_test, y_pred, X_test.shape[1], len(y_test)),  # Replace r2_score
                                'Test_MSE': mean_squared_error(y_test, y_pred),
                                'Test_MAE': mean_absolute_error(y_test, y_pred),
                                'Model': model
                            }
            print(f"测试集 Adjusted R²: {results[name]['Test_R2']:.4f}")
            
            
        except Exception as e:
            print(f"训练 {name} 时出错: {str(e)}")
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

# 4. Optuna超参数优化
def optimize_with_optuna(X_train, y_train, model_name, n_trials=100):
    
    """使用Optuna进行超参数优化"""
    
    def objective(trial):
        if model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 5, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])}
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
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),  # 邻近点数量
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),  # 权重类型
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # 算法类型
            }
            model = KNeighborsRegressor(**params)

        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),  # 迭代次数
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),  # 学习率
                'depth': trial.suggest_int('depth', 4, 10),  # 树的深度
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10),  # 叶子节点的 L2 正则化系数
                'border_count': trial.suggest_int('border_count', 32, 255),  # 分割特征时考虑的边界数量
                'thread_count': trial.suggest_int('thread_count', 1, 8),  # 线程数
                'random_seed': trial.suggest_int('random_seed', 1, 1000)  # 随机种子
            }
            model = CatBoostRegressor(**params, random_state=42)
        

        score = cross_val_score(model, X_train, y_train, cv=10, scoring=adjusted_r2_scorer, n_jobs=-1).mean()
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n最佳 {model_name} 参数:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # # 可视化优化过程
    # fig = plot_optimization_history(study)
    # fig.write_image(f"{model_name.lower().replace(' ', '_')}_optimization_history.png")
    
    # fig = plot_param_importances(study)
    # fig.write_image(f"{model_name.lower().replace(' ', '_')}_param_importance.png")
    
    return study.best_params

def stratified_group_split(data, condition_col, genotype_col, test_size=0.2, random_state=42):
    """
    根据Condition列进行分层划分，并确保同一个Genotype不出现在训练集和测试集中。
    
    参数:
        data: pandas DataFrame，包含数据集
        condition_col: str，Condition列名
        genotype_col: str，Genotype列名
        test_size: float，测试集比例
        random_state: int，随机种子
    
    返回:
        train_data: 训练集DataFrame
        test_data: 测试集DataFrame
    """
    # 确保输入数据包含必要的列
    if condition_col not in data.columns or genotype_col not in data.columns:
        raise ValueError("Condition或Genotype列不存在！")
    
    # 按Genotype分组，获取每组的Condition值（假设每组的Condition一致）
    grouped = data.groupby(genotype_col)[condition_col].first().reset_index()
    
    # 对Genotype进行分层划分，基于Condition
    train_groups, test_groups = train_test_split(
        grouped,
        test_size=test_size,
        stratify=grouped[condition_col],
        random_state=random_state
    )
    
    # 根据划分的Genotype组，提取完整的训练集和测试集
    train_data = data[data[genotype_col].isin(train_groups[genotype_col])]
    test_data = data[data[genotype_col].isin(test_groups[genotype_col])]
    
    # 验证Condition分布
    train_condition_dist = train_data[condition_col].value_counts(normalize=True)
    test_condition_dist = test_data[condition_col].value_counts(normalize=True)
    print("\n训练集Condition分布：")
    print(train_condition_dist)
    print("\n测试集Condition分布：")
    print(test_condition_dist)
    
    # 验证Genotype不重复
    train_genotypes = set(train_data[genotype_col])
    test_genotypes = set(test_data[genotype_col])
    overlapping_genotypes = train_genotypes.intersection(test_genotypes)
    if overlapping_genotypes:
        raise ValueError(f"发现Genotype重复：{overlapping_genotypes}")
    else:
        print("\n验证通过：训练集和测试集的Genotype无重复")
    
    # 可视化Condition分布
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
    加载数据并进行分层分组划分
    """
    # 加载数据
    data = pd.read_excel(filepath,sheet_name=sheet_name)
    if data.isnull().any().any():
        print("发现缺失值，使用中位数填补...")
        data = data.fillna(data.median(numeric_only=True))
    check_replicate(data)
    data = data[[col for col in data.columns if col.endswith(('_1', '_2', '_3', '_4', target_col, condition_col, genotype_col))]]
    # 划分数据集
    train_data, test_data = stratified_group_split(
        data,
        condition_col=condition_col,
        genotype_col=genotype_col,
        test_size=0.3,
        random_state=random_state)
    
    # 分离特征和目标
    X_train = train_data.drop(columns=[target_col,condition_col,genotype_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col,condition_col,genotype_col])
    y_test = test_data[target_col]
    
    return X_train, X_test, y_train, y_test

def check_replicate(data):
    # if 'Condition' not in data.columns or 'Genotype' not in data.columns:
    #     print("错误：Condition 或 Genotype 列不存在！")
    # exit()

    # 按 Condition 和 Genotype 分组，统计每组的重复次数
    replicate_counts = data.groupby(['Condition', 'Genotype']).size().reset_index(name='Replicate_Count')

    # 按 Condition 汇总，统计重复次数分布
    condition_summary = replicate_counts.groupby('Condition')['Replicate_Count'].value_counts().unstack(fill_value=0)

    # 打印重复次数统计
    print("\n每种 Condition 下 Genotype 的重复次数统计：")
    print(condition_summary)

    # 打印详细信息
    conditions = data['Condition'].unique()
    for condition in conditions:
        print(f"\nCondition: {condition}")
        condition_data = replicate_counts[replicate_counts['Condition'] == condition]
        print(f"  总 Genotype 数: {len(condition_data)}")
        print(f"  重复次数分布:\n{condition_data['Replicate_Count'].value_counts().sort_index()}")
        
        # 列出非 2 次重复的 Genotype
        non_standard = condition_data[condition_data['Replicate_Count'] != 2]
        if not non_standard.empty:
            print(f" 非 2 次重复的 Genotype:")
            for _, row in non_standard.iterrows():
                print(f"    Genotype {row['Genotype']}: {row['Replicate_Count']} 次")
# 5. 主函数
def main():
    dataset_random_states=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 100, 123, 200, 300, 400, 500, 1234, 2025, 12345]
    all_results = []
    for dataset_random_state in dataset_random_states:
        data_path = 'file2.xlsx'
        sheet_name = "3d_TKW"
        target_col = "1000GrainWeight"
        X_train, X_test, y_train, y_test = load_and_split_data(data_path, sheet_name=sheet_name, target_col=target_col, random_state=dataset_random_state)
        print(X_train)

        # # Feature Selection
        selected_features = feature_selection(X_train, y_train)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 基础模型训练
        print("\n=== 基础模型比较 ===")
        scaler = StandardScaler()
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)
        results = train_and_optimize(X_train_selected, y_train, X_test_selected, y_test)
        
        # 保存结果
        results_df = pd.DataFrame(results).T
        results_df.drop(columns=['Model'], inplace=True)
        results_df.rename(columns={'Test_R2': 'Test_Adjusted_R2', 'CV_R2_mean': 'CV_Adjusted_R2_mean'}, inplace=True)
        results_file = f"model_performance_{sheet_name}_rs{dataset_random_state}.csv"
        results_df.to_csv(results_file)
        print("\n模型结果已保存到 model_performance")

        results_df['random_state'] = dataset_random_state
        all_results.append(results_df)
        
        # 选择最佳模型进行优化
        best_model_name = max(results, key=lambda x: results[x]['Test_R2'])
        print(f"\n选择最佳模型进行优化: {best_model_name}")
        
        # 超参数优化
        best_params = optimize_with_optuna(X_train_selected, y_train, best_model_name)
        
        # 用最优参数训练最终模型
        if best_model_name == 'Random Forest':
            final_model = RandomForestRegressor(**best_params, random_state=42)
        elif best_model_name == 'XGBoost':
            final_model = XGBRegressor(**best_params, random_state=42)
        elif best_model_name == 'LightGBM':
            final_model = LGBMRegressor(**best_params, random_state=42)
        elif best_model_name == 'Ridge':
            final_model = Ridge(**best_params)
        elif best_model_name == 'KNN Regression':
            final_model = KNeighborsRegressor(**best_params)
        elif best_model_name == 'CatBoost':
            final_model = CatBoostRegressor(**best_params,random_state=42)
        final_model.fit(X_train_selected, y_train)
        
        # 评估最终模型
        y_pred = final_model.predict(X_test_selected)
        final_r2 = adjusted_r2_score(y_test, y_pred, X_test_selected.shape[1], len(y_test))
        print(f"\n优化后模型 R²: {final_r2:.4f}")
        break
        # if hasattr(final_model, 'feature_importances_'):
        #     plt.figure(figsize=(10, 6))
        #     feat_importances = pd.Series(final_model.feature_importances_, index=selected_features)
        #     feat_importances.nlargest(10).plot(kind='barh')
        #     plt.title("Top 10 Important Features")
        #     plt.savefig('feature_importance.png')
        #     plt.close()
        # else:
        #     print(f"{best_model_name} 不支持特征重要性可视化")
    final_results = pd.concat(all_results)
    final_results.groupby('Model').agg({'Test_R2': ['mean', 'std']}).to_csv('model_stability.csv')

if __name__ == "__main__":
    main()