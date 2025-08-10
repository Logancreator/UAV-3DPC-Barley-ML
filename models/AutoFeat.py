import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 导入 Autofeat
from autofeat import AutoFeatRegressor

print("--- 加载和准备数据 ---")
# 1. 加载数据
diabetes = load_diabetes()
X = pd.read_csv(r"E:\Git\phenomics\barley\project\uav\models\dataset_3d.csv").loc[:,['R_mean_msi_threeD_indices_02-06-24', 'G_mean_msi_threeD_indices_02-06-24','B_mean_msi_threeD_indices_02-06-24', 'redE_mean_msi_threeD_indices_02-06-24','nir_mean_msi_threeD_indices_02-06-24',
        'R_mean_msi_threeD_indices_30-06-24', 'G_mean_msi_threeD_indices_30-06-24','B_mean_msi_threeD_indices_30-06-24', 'redE_mean_msi_threeD_indices_30-06-24','nir_mean_msi_threeD_indices_30-06-24',
        'R_mean_msi_threeD_indices_21-07-24', 'G_mean_msi_threeD_indices_21-07-24','B_mean_msi_threeD_indices_21-07-24', 'redE_mean_msi_threeD_indices_21-07-24','nir_mean_msi_threeD_indices_21-07-24',
        'R_mean_msi_threeD_indices_11-08-24', 'G_mean_msi_threeD_indices_11-08-24','B_mean_msi_threeD_indices_11-08-24', 'redE_mean_msi_threeD_indices_11-08-24','nir_mean_msi_threeD_indices_11-08-24']]
y = pd.read_excel(r"E:\Git\phenomics\barley\project\uav\models\Agronomic_data_final.xlsx").loc[:,"GrainYield"]

print(f"原始数据集特征数量: {X.shape[1]}")
print("原始特征名称:", X.columns.tolist())
print(f"数据集大小: {X.shape[0]} 样本")

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

print("\n--- 初始化和拟合 Autofeat ---")
# 3. 初始化 AutoFeatRegressor
#    - verbose=1: 显示详细的运行过程信息
#    - feateng_steps: 特征工程的迭代次数 (默认 3)
#    - 其他参数可以调整，例如 'transformations', 'units' 等
afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)

# 4. 拟合 Autofeat 模型 (自动进行特征生成和选择)
#    注意：Autofeat 内部可能会进行数据缩放，但原始 X_train 保持不变
#    传入 Pandas DataFrame 可以保留特征名称，方便查看公式
afreg.fit(X_train, y_train)

print("\n--- 转换数据 ---")
# 5. 转换训练集和测试集
#    transform 方法会应用在 fit 过程中学到的最佳特征转换
X_train_tr = afreg.transform(X_train)
X_test_tr = afreg.transform(X_test)

print(f"Autofeat 处理后的训练集特征数量: {X_train_tr.shape[1]}")
print(f"Autofeat 处理后的测试集特征数量: {X_test_tr.shape[1]}")

# 6. 查看部分生成的特征 (可选)
print("\n--- 查看 Autofeat 结果 ---")
print("部分生成的特征公式:")
# afreg.feature_formulas_ 存储了可读的特征公式
# afreg.new_feat_cols_ 存储了新特征的列名 (可能不太直观)
formulas = afreg.feature_formulas_
limit = 10
count = 0
for i, f in formulas.items():
    if count < limit:
        print(f"  特征 {i}: {f}")
        count += 1
    else:
        print(f"  ... (总共 {len(formulas)} 个特征)")
        break

# 如果想看 DataFrame 形式的特征名 (可能较长)
if isinstance(X_train_tr, pd.DataFrame):
    print("\n部分转换后的特征列名 (DataFrame):")
    print(X_train_tr.columns[:10].tolist())
    if X_train_tr.shape[1] > 10:
        print("...")

print("\n--- 性能评估对比 ---")
# 7. 评估性能

# 7.a. 在原始特征上训练和评估
print("评估: 原始特征 + 线性回归")
model_orig = LinearRegression()
model_orig.fit(X_train, y_train)
y_pred_orig = model_orig.predict(X_test)
r2_orig = r2_score(y_test, y_pred_orig)
print(f"  测试集 R² 分数 (原始特征): {r2_orig:.4f}")

# 7.b. 在 Autofeat 生成的特征上训练和评估
print("评估: Autofeat 特征 + 线性回归")
model_tr = LinearRegression()
# 注意：如果 transform 返回的是 numpy 数组，可能需要先转回 DataFrame
# 但 Autofeat 通常会尽量保持 Pandas DataFrame 格式 (如果输入是 DataFrame)
model_tr.fit(X_train_tr, y_train)
y_pred_tr = model_tr.predict(X_test_tr)
r2_tr = r2_score(y_test, y_pred_tr)
print(f"  测试集 R² 分数 (Autofeat 特征): {r2_tr:.4f}")

print("\n--- 总结 ---")
print(f"原始特征数: {X_train.shape[1]}")
print(f"Autofeat 特征数: {X_train_tr.shape[1]}")
print(f"原始特征 R²: {r2_orig:.4f}")
print(f"Autofeat 特征 R²: {r2_tr:.4f}")

if r2_tr > r2_orig:
    print("\nAutofeat 生成的特征在这个例子中提升了线性模型的性能。")
elif r2_tr < r2_orig:
     print("\nAutofeat 生成的特征在这个例子中降低了线性模型的性能（可能需要调整参数或数据集不适合）。")
else:
     print("\nAutofeat 生成的特征在这个例子中性能与原始特征相当。")