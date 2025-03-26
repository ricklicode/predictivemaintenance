#!/usr/bin/env python
"""
Simple script to analyze the UC Predictive Maintenance dataset without complex preprocessing.
"""

import pandas as pd
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('simple_results', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data_path = 'uc_pred_mait_ds.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Machine failures: {df['Machine failure'].sum()} out of {len(df)} samples ({df['Machine failure'].mean()*100:.2f}%)")

# Check failure modes
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for mode in failure_modes:
    print(f"{mode} failures: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)")

# Calculate some basic statistics for numeric columns
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Compare statistics for failed vs non-failed machines
print("\nStatistics for numeric features by machine failure status:")
for col in numeric_cols:
    failed_stats = df[df['Machine failure'] == 1][col].describe()
    nonfailed_stats = df[df['Machine failure'] == 0][col].describe()
    
    print(f"\n{col}:")
    print(f"  Failed machines - Mean: {failed_stats['mean']:.2f}, Std: {failed_stats['std']:.2f}, Min: {failed_stats['min']:.2f}, Max: {failed_stats['max']:.2f}")
    print(f"  Non-failed machines - Mean: {nonfailed_stats['mean']:.2f}, Std: {nonfailed_stats['std']:.2f}, Min: {nonfailed_stats['min']:.2f}, Max: {nonfailed_stats['max']:.2f}")

# Count product types and their failure rates
product_types = df['Product ID'].str[0].value_counts()
print("\nProduct Type Distribution:")
for product_type, count in product_types.items():
    failures = df[df['Product ID'].str[0] == product_type]['Machine failure'].sum()
    failure_rate = failures / count * 100
    print(f"  {product_type}: {count} products, {failures} failures ({failure_rate:.2f}% failure rate)")

# Look at feature correlations with machine failure
correlations = df[numeric_cols + ['Machine failure']].corr()['Machine failure'].drop('Machine failure')
print("\nCorrelations with Machine Failure:")
for feature, corr in correlations.sort_values(ascending=False).items():
    print(f"  {feature}: {corr:.4f}")

# Let's look at some specific failure patterns for each mode
print("\nAnalyzing specific patterns for each failure mode:")

# Tool wear failure (TWF) analysis
twf_df = df[df['TWF'] == 1]
print(f"\nTool Wear Failure (TWF) - {len(twf_df)} cases:")
print(f"  Average tool wear: {twf_df['Tool wear [min]'].mean():.2f} min (vs {df[df['TWF'] == 0]['Tool wear [min]'].mean():.2f} min for non-TWF)")

# Heat dissipation failure (HDF) analysis
hdf_df = df[df['HDF'] == 1]
temp_diff = hdf_df['Process temperature [K]'] - hdf_df['Air temperature [K]']
print(f"\nHeat Dissipation Failure (HDF) - {len(hdf_df)} cases:")
print(f"  Average temperature difference: {temp_diff.mean():.2f} K")
print(f"  Average rotational speed: {hdf_df['Rotational speed [rpm]'].mean():.2f} rpm")

# Power failure (PWF) analysis
pwf_df = df[df['PWF'] == 1]
# Calculate power (torque * rotational speed in rad/s)
power = pwf_df['Torque [Nm]'] * pwf_df['Rotational speed [rpm]'] * (2 * np.pi / 60)
print(f"\nPower Failure (PWF) - {len(pwf_df)} cases:")
print(f"  Average power: {power.mean():.2f} W")
print(f"  Min power: {power.min():.2f} W")
print(f"  Max power: {power.max():.2f} W")

# Overstrain failure (OSF) analysis
osf_df = df[df['OSF'] == 1]
tool_torque = osf_df['Tool wear [min]'] * osf_df['Torque [Nm]']
print(f"\nOverstrain Failure (OSF) - {len(osf_df)} cases:")
print(f"  Average tool wear * torque: {tool_torque.mean():.2f} min*Nm")

# Random failures (RNF) analysis
rnf_df = df[df['RNF'] == 1]
print(f"\nRandom Failure (RNF) - {len(rnf_df)} cases:")
print(f"  Represents {rnf_df['RNF'].sum() / df['Machine failure'].sum() * 100:.2f}% of all failures")

# Save detailed stats to file
with open('simple_results/dataset_analysis.txt', 'w') as f:
    f.write("Predictive Maintenance Dataset Analysis\n")
    f.write("======================================\n\n")
    f.write(f"Dataset: {data_path}\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Machine failures: {df['Machine failure'].sum()} ({df['Machine failure'].mean()*100:.2f}%)\n\n")
    
    f.write("Failure modes:\n")
    for mode in failure_modes:
        f.write(f"- {mode}: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)\n")
    
    f.write("\nProduct Type Distribution:\n")
    for product_type, count in product_types.items():
        failures = df[df['Product ID'].str[0] == product_type]['Machine failure'].sum()
        failure_rate = failures / count * 100
        f.write(f"- {product_type}: {count} products, {failures} failures ({failure_rate:.2f}% failure rate)\n")
    
    f.write("\nCorrelations with Machine Failure:\n")
    for feature, corr in correlations.sort_values(ascending=False).items():
        f.write(f"- {feature}: {corr:.4f}\n")
    
    f.write("\nDetailed Statistics by Feature and Failure Status:\n")
    for col in numeric_cols:
        f.write(f"\n{col}:\n")
        for status, label in [(1, "Failed Machines"), (0, "Non-failed Machines")]:
            stats = df[df['Machine failure'] == status][col].describe()
            f.write(f"  {label}:\n")
            for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                f.write(f"    {stat_name}: {stats[stat_name]:.2f}\n")

print("\nResults saved to 'simple_results' directory")
print("Analysis complete!") 