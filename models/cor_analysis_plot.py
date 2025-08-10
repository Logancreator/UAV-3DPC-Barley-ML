import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns  # Added for enhanced styling
import os

# Set working directory
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("Current working directory set to:", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("Current working directory:", os.getcwd())

# Read data
df = pd.read_excel("cor_results.xlsx", sheet_name="VIs")

# Check data
print(df.head())

# Set Seaborn styling
sns.set_style("whitegrid")  # Clean white background with subtle grid
plt.rcParams['font.family'] = 'Arial'  # Use Arial for a professional look
plt.rcParams['font.size'] = 12  # Default font size for consistency

# Create canvas
plt.figure(figsize=(8, 6), dpi=100)  # Adjusted size for better proportion, higher DPI for clarity

# Customize boxplot style
boxprops = dict(linestyle='-', linewidth=2, color='black', facecolor=sns.color_palette("Set2")[0], alpha=0.7)
medianprops = dict(linestyle='-', linewidth=2, color='darkred')
whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
capprops = dict(linestyle='-', linewidth=1.5, color='black')
flierprops = dict(marker='o', markersize=5, markerfacecolor='gray', alpha=0.5)  # Style outliers

# Plot boxplot
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

# Set title and labels
plt.title('Correlation Coefficients of Two Replications: 2D vs. 3D', fontsize=16, weight='bold', pad=15)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Dimension', fontsize=14)

# Customize tick styles
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12)

# Adjust grid style
plt.grid(True, linestyle='--', alpha=0.3, color='gray')

# Perform t-test
t_stat, p_value = stats.ttest_ind(df['2d'], df['3d'], nan_policy='omit')  # Handle NaN values
p_value = float("{:.2f}".format(p_value))

# Determine significance
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

# Calculate max y-value for annotation position
max_y = max(max(df['2d'].dropna()), max(df['3d'].dropna()))
y_annot = max_y * 0.9  # Slightly above max value
x_annot = 1.5  # Centered between boxes

# Add significance annotation
plt.text(x_annot, y_annot, significance, fontsize=16, ha='center', va='center', color='black', weight='bold')

# Add statistical info annotation
stats_text = f't = {t_stat:.2f}, p = {p_value:.2f}'
plt.annotate(
    stats_text,
    xy=(0.02, 0.95),  # Top-left corner
    xycoords='axes fraction',
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
)

# Optimize layout
plt.tight_layout()

# Save high-resolution image
plt.savefig('boxplot_2d_vs_3d.png', dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# Print statistical results
print(f"T-test results: t-statistic = {t_stat:.2f}, P-value = {p_value:.2f}")
if p_value <= alpha:
    print("The difference between 2D and 3D is significant (P <= 0.05)")
else:
    print("The difference between 2D and 3D is not significant (P > 0.05)")
