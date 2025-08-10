import pandas as pd
import os

# --- Set working directory ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("Current working directory set to:", os.getcwd())
except NameError:
    print("Running interactively or __file__ not defined. Assuming current directory is correct.")
    print("Current working directory:", os.getcwd())

# Read data
df = pd.read_excel("file1.xlsx", sheet_name="3d")

# Split into two groups based on Replication
rep1 = df[df['Replication'] == 1]
rep2 = df[df['Replication'] == 2]

# Ensure both groups have the same Genotype order and sort by Genotype
rep1 = rep1.set_index('Genotype').sort_index()
rep2 = rep2.set_index('Genotype').sort_index()

# Select numeric columns for correlation analysis
numeric_columns = rep1.select_dtypes(include=['float64', 'int64']).columns

# Calculate the correlation for each numeric variable (Spearman correlation coefficient)
corr_results = {}
for col in numeric_columns:
    if col in rep2.columns:  # Ensure column exists in rep2
        corr = rep1[col].corr(rep2[col])
        corr_results[col] = corr

# Convert results to DataFrame
corr_df = pd.DataFrame.from_dict(corr_results, orient='index', columns=['Spearman_Correlation'])

# Output correlation results
print("\nSpearman correlation between Replication 1 and Replication 2:")
print(corr_df)

# Save results to Excel
output_file = "correlation_results.xlsx"
corr_df.to_excel(output_file, sheet_name="Correlation_Results")
print(f"\nCorrelation results saved to {output_file}")