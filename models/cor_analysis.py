import pandas as pd
import os

# --- 设置工作目录 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("当前工作路径已设置为：", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("当前工作路径：", os.getcwd())

# 读取数据
df = pd.read_excel("file1.xlsx", sheet_name="3d")

# 根据Replication分为两组
rep1 = df[df['Replication'] == 1]
rep2 = df[df['Replication'] == 2]

# 确保两组数据的Genotype顺序一致，并按照Genotype进行排序
rep1 = rep1.set_index('Genotype').sort_index()
rep2 = rep2.set_index('Genotype').sort_index()

# 选择数值型列进行相关性分析
numeric_columns = rep1.select_dtypes(include=['float64', 'int64']).columns

# 计算每对数值型变量的相关性（斯皮尔曼相关系数）
corr_results = {}
for col in numeric_columns:
    if col in rep2.columns:  # 确保列在rep2中存在
        corr = rep1[col].corr(rep2[col])
        corr_results[col] = corr

# 将结果转换为DataFrame
corr_df = pd.DataFrame.from_dict(corr_results, orient='index', columns=['Spearman_Correlation'])

# 输出相关性结果
print("\n重复1和重复2之间的斯皮尔曼相关性：")
print(corr_df)

# 保存结果到Excel文件
output_file = "correlation_results.xlsx"
corr_df.to_excel(output_file, sheet_name="Correlation_Results")
print(f"\n相关性结果已保存到 {output_file}")