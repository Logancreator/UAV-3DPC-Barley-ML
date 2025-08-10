import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns  # Added for enhanced styling
import os

# 设置工作目录
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("当前工作路径已设置为：", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("当前工作路径：", os.getcwd())

# 读取数据
df = pd.read_excel("cor_results.xlsx", sheet_name="VIs")

# 检查数据
print(df.head())

# 设置Seaborn美化风格
sns.set_style("whitegrid")  # Clean white background with subtle grid
plt.rcParams['font.family'] = 'Arial'  # Use Arial for a professional look
plt.rcParams['font.size'] = 12  # Default font size for consistency

# 创建画布
plt.figure(figsize=(8, 6), dpi=100)  # Adjusted size for better proportion, higher DPI for clarity

# 自定义箱型图样式
boxprops = dict(linestyle='-', linewidth=2, color='black', facecolor=sns.color_palette("Set2")[0], alpha=0.7)
medianprops = dict(linestyle='-', linewidth=2, color='darkred')
whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
capprops = dict(linestyle='-', linewidth=1.5, color='black')
flierprops = dict(marker='o', markersize=5, markerfacecolor='gray', alpha=0.5)  # Style outliers

# 绘制箱型图
boxplot_dict = plt.boxplot(
    [df['2d'], df['3d']],
    labels=['2D', '3D'],  # Capitalized for better readability
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    flierprops=flierprops,
    patch_artist=True
)

# 设置标题和标签
plt.title('Correlation Coefficients of Two Replications: 2D vs. 3D', fontsize=16, weight='bold', pad=15)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Dimension', fontsize=14)

# 自定义刻度样式
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12)

# 调整网格样式
plt.grid(True, linestyle='--', alpha=0.3, color='gray')

# 进行t检验
t_stat, p_value = stats.ttest_ind(df['2d'], df['3d'], nan_policy='omit')  # Handle NaN values
p_value = float("{:.2f}".format(p_value))
# 判断显著性
alpha = 0.05
significance = ""
if p_value < 0.001:
    significance = "***"
elif p_value <= 0.01:
    significance = "**"
elif p_value <= 0.05:
    significance = "*"
else:
    significance = "ns"

# 计算y轴最大值用于标注位置
max_y = max(max(df['2d'].dropna()), max(df['3d'].dropna()))
y_annot = max_y * 0.9  # Slightly above max value
x_annot = 1.5  # Centered between boxes

# 添加显著性标注
plt.text(x_annot, y_annot, significance, fontsize=16, ha='center', va='center', color='black', weight='bold')

# 添加统计信息注释
stats_text = f't = {t_stat:.2f}, p = {p_value:.2f}'
plt.annotate(
    stats_text,
    xy=(0.02, 0.95),  # Top-left corner
    xycoords='axes fraction',
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
)

# 优化布局
plt.tight_layout()

# 保存高分辨率图像
plt.savefig('boxplot_2d_vs_3d.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 打印统计结果
print(f"t检验结果: t统计量 = {t_stat:.2f}, P值 = {p_value:.2f}")
if p_value <= alpha:
    print("2D和3D之间的差异显著 (P <= 0.05)")
else:
    print("2D和3D之间的差异不显著 (P > 0.05)")