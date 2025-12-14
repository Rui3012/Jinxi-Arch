#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jinxi enclosure analysis (converted from a Jupyter notebook).

Notes
- This script was auto-converted from an .ipynb file.
- Colab-specific lines (e.g., `!pip install`, `google.colab.files.download`) were removed.
- Comments and user-facing strings were converted to English (best-effort), and all Chinese characters were removed.

Run
  python jinxi_converted.py

If you see missing-package errors, install dependencies, e.g.:
  pip install pandas numpy openpyxl scikit-learn matplotlib seaborn
"""



import argparse
from pathlib import Path

def _parse_args():
    p = argparse.ArgumentParser(description="Jinxi enclosure analysis (converted from notebook)")
    p.add_argument("--input", "-i", default="Jinxi_Model_Input.xlsx",
                   help="Path to the input Excel file (default: Jinxi_Model_Input.xlsx)")
    p.add_argument("--outdir", "-o", default="outputs",
                   help="Directory to write outputs (default: ./outputs)")
    return p.parse_args()

ARGS = _parse_args()
INPUT_XLSX = str(Path(ARGS.input).expanduser().resolve())
OUTDIR = Path(ARGS.outdir).expanduser().resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Notebook cell 0 ---
"""
============================================================================
Stage A: Exploratory Data Analysis (Exploratory Data Analysis)
- Google Colab============================================================================
"""

# ============================================================================
# 1.
# ============================================================================

# (Colab)
# (Install dependencies separately: pip install -r requirements.txt)

# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# 2.
# ============================================================================

print("=" * 80)
print("1:")
print("=" * 80)
print(f"Using input Excel: {INPUT_XLSX}")

# (Colab-only) removed: google.colab files downloader
# Input file is provided via --input (see ARGS above)

# 
filename = INPUT_XLSX
print(f"Loaded: {filename}")

# ============================================================================
# 3.
# ============================================================================

print("\n" + "=" * 80)
print("Stage A: Exploratory Data Analysis")
print("=" * 80)

# 
df = pd.read_excel(filename)

print(f"\n[Dataset overview]")
print(f"Total sites: {len(df)}")
print(f" (Enclosure): {len(df[df['Type']=='Enclosure'])}")
print(f" (Mound): {len(df[df['Type']=='Mound'])}")

# 
env_vars = ['TWI', 'VDCN', 'Slope', 'TRI', 'DEM', 'HubDist']
var_labels = {
    'TWI': 'Topographic Wetness Index (TWI)',
    'VDCN': 'Vertical Distance to Channel (m)',
    'Slope': 'Slope (degrees)',
    'TRI': 'Terrain Ruggedness Index',
    'DEM': 'Elevation (m)',
    'HubDist': 'Distance to River (m)'
}

# ============================================================================
# 4. (Type)
# ============================================================================

print("\n" + "=" * 80)
print("[:Enclosure vs Mound]")
print("=" * 80)

# ( vs )
comparison = pd.DataFrame({
    'Enclosure_Mean': df[df['Type']=='Enclosure'][env_vars].mean(),
    'Enclosure_Median': df[df['Type']=='Enclosure'][env_vars].median(),
    'Mound_Mean': df[df['Type']=='Mound'][env_vars].mean(),
    'Mound_Median': df[df['Type']=='Mound'][env_vars].median()
})
print("\n vs :")
print(comparison.round(2))

# ============================================================================
# 5. :Mann-Whitney U Test
# ============================================================================

print("\n" + "=" * 80)
print("[Mann-Whitney U ]")
print("=" * 80)
print("H0:")
print("H1: (p < 0.05)")
print("-" * 80)

u_test_results = []

for var in env_vars:
    enclosure_data = df[df['Type']=='Enclosure'][var]
    mound_data = df[df['Type']=='Mound'][var]

# # Mann-Whitney U
    stat, p_value = mannwhitneyu(enclosure_data, mound_data, alternative='two-sided')

# # (Cohen's d)
    mean_diff = enclosure_data.mean() - mound_data.mean()
    pooled_std = np.sqrt((enclosure_data.std()**2 + mound_data.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

# #
    significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

    u_test_results.append({
        'Variable': var,
        'U_Statistic': stat,
        'P_Value': p_value,
        'Significance': significance,
        'Cohens_d': cohens_d,
        'Enclosure_Mean': enclosure_data.mean(),
        'Mound_Mean': mound_data.mean(),
        'Mean_Diff': mean_diff
    })

    print(f"{var_labels[var]}")
    print(f"U: {stat:.2f}")
    print(f"P: {p_value:.6f} {significance}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"Enclosure: {enclosure_data.mean():.2f}, Mound: {mound_data.mean():.2f}")
    print(f": {mean_diff:.2f}\n")

# DataFrame
u_test_df = pd.DataFrame(u_test_results)

print("\n[]")
sig_vars = u_test_df[u_test_df['P_Value'] < 0.05]['Variable'].tolist()
print(f" (p<0.05): {', '.join(sig_vars) if sig_vars else ''}")
print(f"{len(sig_vars)}/{len(env_vars)}")

# ============================================================================
# 6. :(6)
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, var in enumerate(env_vars):
    ax = axes[idx]

# #
    data_to_plot = [
        df[df['Type']=='Enclosure'][var].values,
        df[df['Type']=='Mound'][var].values
    ]

# #
    bp = ax.boxplot(data_to_plot,
                     labels=['Enclosure\n(n=44)', 'Mound\n(n=60)'],
                     patch_artist=True,
                     widths=0.6)

# #
    colors = ['#ff9999', '#66b3ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

# #
    ax.set_title(var_labels[var], fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

# #
    p_val = u_test_df[u_test_df['Variable']==var]['P_Value'].values[0]
    sig = u_test_df[u_test_df['Variable']==var]['Significance'].values[0]
    if sig != 'ns':
        y_max = max([d.max() for d in data_to_plot])
        y_range = y_max - min([d.min() for d in data_to_plot])
        ax.text(1.5, y_max + y_range*0.05, f'p={p_val:.4f} {sig}',
                ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('Stage_A_Boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("OK")

# ============================================================================
# 7. Analysis:Spearman Correlation Matrix
# ============================================================================

print("\n" + "=" * 80)
print("[Spearman ]")
print("=" * 80)

# SpearmanCorrelation coefficient
corr_matrix = df[env_vars].corr(method='spearman')
print("\nCorrelation coefficient:")
print(corr_matrix.round(3))

# (|r| > 0.85)
print("\n[Multicollinearity check]")
high_corr_pairs = []
for i in range(len(env_vars)):
    for j in range(i+1, len(env_vars)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.85:
            high_corr_pairs.append((env_vars[i], env_vars[j], corr_val))
            print(f"WARN  {env_vars[i]} <-> {env_vars[j]}: r = {corr_val:.3f}")

if not high_corr_pairs:
    print("OK (|r| < 0.85)")
else:
    print(f"\n: {len(high_corr_pairs)}")

# 
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Spearman Correlation Matrix\n(Environmental Variables)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('Stage_A_Correlation.png', dpi=300, bbox_inches='tight')
plt.show()
print("OK")

# ============================================================================
# 8.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# UTest results
u_test_df.to_csv('Stage_A_UTest_Results.csv', index=False)
print("OK UTest results: Stage_A_UTest_Results.csv")

# 
comparison.to_csv('Stage_A_Descriptive_Stats.csv')
print("OK : Stage_A_Descriptive_Stats.csv")

# ============================================================================
# 9. A
# ============================================================================

print("\n" + "=" * 80)
print("[A ]")
print("=" * 80)

print(f"\n1. Sample overview:")
print(f"   - Total sites: 104 (Enclosure: 44, Mound: 60)")
print(f"- : 100% ()")

print(f"\n2. Variables with significant differences (p<0.05):")
if sig_vars:
    for var in sig_vars:
        row = u_test_df[u_test_df['Variable']==var].iloc[0]
        direction = "Enclosure > Mound" if row['Mean_Diff'] > 0 else "Enclosure < Mound"
        print(f"   - {var}: p={row['P_Value']:.6f}, d={row['Cohens_d']:.3f} ({direction})")
else:
    print("- Variables with significant differences")

print(f"\n3. :")
if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"   - {pair[0]} <-> {pair[1]}: r={pair[2]:.3f}")
    print(f"\n WARN : DEM (VDCN), TRI (Slope)")
else:
    print("-")

print(f"\n4. :")
print(f"- Enclosure sites (p<0.001)")
print(f"- Enclosure sites (p<0.001)")
print(f"- Enclosure sites (p<0.05)")
print(f"- TWI() (p=0.96)")
print(f"\n   NOTE : 'controlled proximity'")
print(f"")

print(f"\n5. :")
if len(sig_vars) >= 2:
    print("OK 2, B()")
    print(f"OK : TWI, VDCN, Slope, HubDist (DEMTRI)")
else:
    print("WARN , ")

print("\n" + "=" * 80)
print("A !")
print("=" * 80)

# ============================================================================
# 10.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# (Colab-only) removed: google.colab files downloader

# Download all generated files
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)

print("\nOK")
print("=" * 80)

# --- Notebook cell 1 ---
"""
============================================================================
Stage B: Synchronic Random Forest Analysis (Synchronic Random Forest Analysis)
- Google Colab============================================================================
: vs: + + Analysis============================================================================
"""

# ============================================================================
# 1.
# ============================================================================

# (Install dependencies separately: pip install -r requirements.txt)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# 
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# 2.
# ============================================================================

print("=" * 80)
print("1:")
print("=" * 80)
print(f"Using input Excel: {INPUT_XLSX}")

# (Colab-only) removed: google.colab files downloader
# Input file is provided via --input (see ARGS above)

filename = INPUT_XLSX
print(f"Loaded: {filename}")

# ============================================================================
# 3.
# ============================================================================

print("\n" + "=" * 80)
print("Stage B: Synchronic Random Forest Analysis")
print("=" * 80)

# 
df = pd.read_excel(filename)

print(f"\n[Dataset overview]")
print(f"Total sites: {len(df)}")
print(f" (Enclosure): {len(df[df['Type']=='Enclosure'])}")
print(f" (Mound): {len(df[df['Type']=='Mound'])}")

# 
# AAnalysis, DEMTRI
X_vars = ['TWI', 'VDCN', 'Slope', 'HubDist']
var_labels = {
    'TWI': 'TWI (Wetness)',
    'VDCN': 'VDCN (Flood Risk)',
    'Slope': 'Slope (Defense)',
    'HubDist': 'Distance to River'
}

print(f"\n[]")
print(f"A, DEM(VDCNr=0.976)TRI(Sloper=0.934)")
print(f"(n={len(X_vars)}):")
for var in X_vars:
    print(f"  - {var}: {var_labels[var]}")

# 
X = df[X_vars].values
y = (df['Type'] == 'Enclosure').astype(int).values  # 1=Enclosure, 0=Mound

print(f"\n[]")
print(f": {X.shape} (104 × 4)")
print(f": {y.sum()} Enclosures, {len(y)-y.sum()} Mounds")

# ============================================================================
# 4.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
rf_model = RandomForestClassifier(
    n_estimators=200,           # ()
    max_depth=None,             # 
    min_samples_split=5,        # 
    min_samples_leaf=2,         # 
    class_weight='balanced',    # 44:60
    random_state=42,            # 
    n_jobs=-1                   # CPU
)

print(":")
print(f"  - n_estimators: {rf_model.n_estimators}")
print(f"  - class_weight: {rf_model.class_weight}")
print(f"  - random_state: {rf_model.random_state}")

# 5
print(f"\n[5]")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# 
cv_results = cross_validate(rf_model, X, y, cv=skf, scoring=scoring,
                            return_train_score=True)

print(f"\n( ± ):")
print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
print(f"  F1 Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

# ()
print(f"\n[]")
rf_model.fit(X, y)
print("OK")

# ============================================================================
# 5.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
y_pred = rf_model.predict(X)

# 
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"\n:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Enclosure)")
print(f"Recall: {recall:.4f} (Enclosure)")
print(f"  F1 Score:  {f1:.4f}")

# 
cm = confusion_matrix(y, y_pred)
print(f"\n:")
print(f"Mound Enclosure")
print(f"Mound {cm[0,0]:>6} {cm[0,1]:>6}")
print(f"Enclosure {cm[1,0]:>6} {cm[1,1]:>6}")

# 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Mound', 'Enclosure'],
            yticklabels=['Mound', 'Enclosure'],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Type', fontsize=12, fontweight='bold')
plt.ylabel('Actual Type', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix\nRandom Forest Classification', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('Stage_B_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nOK")

# ============================================================================
# 6. Analysis()
# ============================================================================

print("\n" + "=" * 80)
print("[Analysis]")
print("=" * 80)

# 
feature_importance = rf_model.feature_importances_

# DataFrame
importance_df = pd.DataFrame({
    'Variable': X_vars,
    'Importance': feature_importance,
    'Label': [var_labels[v] for v in X_vars]
}).sort_values('Importance', ascending=False)

print("\n:")
print("-" * 60)
for idx, row in importance_df.iterrows():
    bar = "" * int(row['Importance'] * 100)
    print(f"{row['Variable']:10} ({row['Label']:20}): {row['Importance']:.4f} {bar}")

# 
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(importance_df)), importance_df['Importance'],
                color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])

# 
for i, (idx, row) in enumerate(importance_df.iterrows()):
    plt.text(row['Importance'] + 0.01, i, f"{row['Importance']:.3f}",
             va='center', fontsize=11, fontweight='bold')

plt.yticks(range(len(importance_df)), importance_df['Label'], fontsize=11)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.title('Random Forest Feature Importance\n(Enclosure vs Mound Classification)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlim(0, max(importance_df['Importance']) * 1.15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('Stage_B_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nOK")

# ============================================================================
# 7. :H1 vs H0
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
top_var = importance_df.iloc[0]['Variable']
top_importance = importance_df.iloc[0]['Importance']

print("\n:")
print("H1 (Hypothesis): TWI VDCN")
print("H0 (Hypothesis): Slope")

print(f"\n:")
print(f": {top_var} ( = {top_importance:.3f})")

if top_var == 'TWI':
    print("\nOK H1 ()")
    print("(TWI)")
    print("  ''")
elif top_var == 'VDCN':
    print("\nOK H1 ()")
    print("(VDCN)")
    print("  '' - ")
elif top_var == 'Slope':
    print("\nOK H0")
    print("(Slope)")
    print("  ''")
else:
    print("\nWARN")
    print("(HubDist)")
    print("  ''")

# TWIVDCN
hydraulic_importance = importance_df[importance_df['Variable'].isin(['TWI', 'VDCN'])]['Importance'].sum()
defense_importance = importance_df[importance_df['Variable'] == 'Slope']['Importance'].sum()

print(f"\n:")
print(f"(TWI + VDCN): {hydraulic_importance:.3f}")
print(f"(Slope): {defense_importance:.3f}")

if hydraulic_importance > defense_importance:
    print("\nNOTE : >")
    print("")
else:
    print("\nNOTE : >")
    print("")

# ============================================================================
# 8.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
importance_df.to_csv('Stage_B_Feature_Importance.csv', index=False)
print("OK : Stage_B_Feature_Importance.csv")

# 
cv_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Mean': [cv_results['test_accuracy'].mean(),
             cv_results['test_precision'].mean(),
             cv_results['test_recall'].mean(),
             cv_results['test_f1'].mean()],
    'Std': [cv_results['test_accuracy'].std(),
            cv_results['test_precision'].std(),
            cv_results['test_recall'].std(),
            cv_results['test_f1'].std()]
})
cv_summary.to_csv('Stage_B_CrossValidation_Results.csv', index=False)
print("OK : Stage_B_CrossValidation_Results.csv")

# 
cm_df = pd.DataFrame(cm,
                     columns=['Predicted_Mound', 'Predicted_Enclosure'],
                     index=['Actual_Mound', 'Actual_Enclosure'])
cm_df.to_csv('Stage_B_Confusion_Matrix.csv')
print("OK : Stage_B_Confusion_Matrix.csv")

# ============================================================================
# 9. B
# ============================================================================

print("\n" + "=" * 80)
print("[B ]")
print("=" * 80)

print(f"\n1. :")
print(f"   - : {cv_results['test_accuracy'].mean():.2%} (±{cv_results['test_accuracy'].std():.2%})")
if cv_results['test_accuracy'].mean() > 0.65:
    print(f"OK , ")
else:
    print(f"WARN , ")

print(f"\n2. :")
for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
    print(f"   {i}. {row['Variable']:10} {row['Importance']:.3f} ({row['Label']})")

print(f"\n3. :")
if top_var in ['TWI', 'VDCN']:
    print(f"OK (Niche Construction Theory)")
    print(f"OK Enclosure sites")
    print(f"   OK ''")
else:
    print(f"WARN")
    print(f"WARN")

print(f"\n4. :")
print(f"- C: Analysis")
print(f"   - Phase II()''")
print(f"- Analysis{top_var}")

print("\n" + "=" * 80)
print("B !")
print("=" * 80)

# ============================================================================
# 10.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# (Colab-only) removed: google.colab files downloader

# Download all generated files
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)

print("\nOK")
print("=" * 80)

# --- Notebook cell 2 ---
"""
============================================================================
C:Analysis (Diachronic Trajectory Analysis)- Google Colab============================================================================
:Phase II()"": + Analysis:("Phase 2; Phase 3")============================================================================
"""

# ============================================================================
# 1.
# ============================================================================

# (Install dependencies separately: pip install -r requirements.txt)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# 2.
# ============================================================================

print("=" * 80)
print("1:")
print("=" * 80)
print(f"Using input Excel: {INPUT_XLSX}")

# (Colab-only) removed: google.colab files downloader
# Input file is provided via --input (see ARGS above)

filename = INPUT_XLSX
print(f"Loaded: {filename}")

# ============================================================================
# 3. ()
# ============================================================================

print("\n" + "=" * 80)
print("C:Analysis")
print("=" * 80)

# 
df = pd.read_excel(filename)

print(f"\n[Dataset overview]")
print(f"Total sites: {len(df)}")
print(f"\nPhase:")
print(df['Phase'].value_counts().sort_index())

# 
X_vars = ['TWI', 'VDCN', 'Slope', 'HubDist']
var_labels = {
    'TWI': 'TWI (Wetness)',
    'VDCN': 'VDCN (Flood Risk)',
    'Slope': 'Slope (Defense)',
    'HubDist': 'Distance to River'
}

# ============================================================================
# 4. :
# ============================================================================

def expand_phases(df):
    """
:"Phase 2; Phase 3" -> Phase 2 + Phase 3    """
    expanded_rows = []

    for idx, row in df.iterrows():
        phase_str = str(row['Phase'])

# # Unknown
        if 'Unknown' in phase_str or phase_str == 'nan':
            continue

# # Phase()
        phases = [p.strip() for p in phase_str.replace(';', ',').split(',')]

# # Phase
        for phase in phases:
            if phase in ['Phase 1', 'Phase 2', 'Phase 3']:
                new_row = row.copy()
                new_row['Phase_Single'] = phase
                expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

# 
df_expanded = expand_phases(df)

print(f"\n[Dataset overview]")
print(f": {len(df_expanded)} (: {len(df)})")
print(f"\n():")
for phase in ['Phase 1', 'Phase 2', 'Phase 3']:
    phase_df = df_expanded[df_expanded['Phase_Single'] == phase]
    n_total = len(phase_df)
    n_enclosure = len(phase_df[phase_df['Type'] == 'Enclosure'])
    n_mound = len(phase_df[phase_df['Type'] == 'Mound'])
    print(f"{phase}: {n_total:3} (Enclosure: {n_enclosure}, Mound: {n_mound})")

# ============================================================================
# 5.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

phases = ['Phase 1', 'Phase 2', 'Phase 3']
phase_labels = {
    'Phase 1': 'Phase I',
    'Phase 2': 'Phase II',
    'Phase 3': 'Phase III'
}

# 
phase_results = {}

for phase in phases:
    print(f"\n{'='*60}")
    print(f"{phase}")
    print(f"{'='*60}")

# #
    phase_df = df_expanded[df_expanded['Phase_Single'] == phase]

# #
    if len(phase_df) < 10:
        print(f"WARN {phase} (n={len(phase_df)}), ")
        continue

# #
    X_phase = phase_df[X_vars].values
    y_phase = (phase_df['Type'] == 'Enclosure').astype(int).values

    print(f": {len(phase_df)}")
    print(f"  Enclosure: {y_phase.sum()}")
    print(f"  Mound: {len(y_phase) - y_phase.sum()}")

# #
    rf = RandomForestClassifier(
        n_estimators=100,  # 
        max_depth=5,       # 
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

# # , 
    if len(phase_df) >= 20:
        n_splits = min(5, len(phase_df) // 10)  # 
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X_phase, y_phase, cv=skf, scoring='accuracy')
        print(f": {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# #
    rf.fit(X_phase, y_phase)

# #
    importance = rf.feature_importances_

# #
    phase_results[phase] = {
        'model': rf,
        'importance': importance,
        'n_samples': len(phase_df),
        'n_enclosure': y_phase.sum(),
        'n_mound': len(y_phase) - y_phase.sum()
    }

    print(f"\n:")
    for var, imp in zip(X_vars, importance):
        print(f"  {var:10}: {imp:.4f}")

# ============================================================================
# 6.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
importance_comparison = pd.DataFrame({
    phase: phase_results[phase]['importance']
    for phase in phases if phase in phase_results
}, index=X_vars).T

print("\n:")
print(importance_comparison.round(4))

# 
importance_comparison.to_csv('Stage_C_Importance_Comparison.csv')
print("\nOK : Stage_C_Importance_Comparison.csv")

# ============================================================================
# 7. ()
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ========== 1: ==========
ax1 = axes[0]

colors = {'TWI': '#ff6b6b', 'VDCN': '#4ecdc4', 'Slope': '#45b7d1', 'HubDist': '#96ceb4'}
markers = {'TWI': 'o', 'VDCN': 's', 'Slope': '^', 'HubDist': 'D'}

for var in X_vars:
    values = [phase_results[p]['importance'][X_vars.index(var)]
              for p in phases if p in phase_results]
    phases_valid = [p for p in phases if p in phase_results]

    ax1.plot(range(len(phases_valid)), values,
             marker=markers[var], markersize=10, linewidth=2.5,
             label=var_labels[var], color=colors[var])

# #
    for i, val in enumerate(values):
        ax1.text(i, val + 0.015, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

ax1.set_xticks(range(len(phases_valid)))
ax1.set_xticklabels([phase_labels[p] for p in phases_valid], fontsize=11)
ax1.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
ax1.set_title('Diachronic Trajectory of Feature Importance\n(Phase I -> Phase II -> Phase III)',
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=10, frameon=True, shadow=True)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max([max(phase_results[p]['importance']) for p in phases_valid]) * 1.15)

# ========== 2: vs ==========
ax2 = axes[1]

# 
hydraulic_trajectory = []
defense_trajectory = []

for phase in phases:
    if phase in phase_results:
        imp = phase_results[phase]['importance']
        hydraulic = imp[X_vars.index('TWI')] + imp[X_vars.index('VDCN')] + imp[X_vars.index('HubDist')]
        defense = imp[X_vars.index('Slope')]
        hydraulic_trajectory.append(hydraulic)
        defense_trajectory.append(defense)

phases_valid = [p for p in phases if p in phase_results]

# 
x_pos = np.arange(len(phases_valid))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, hydraulic_trajectory, width,
                label='Hydraulic Factors\n(TWI + VDCN + HubDist)',
                color='#3498db', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, defense_trajectory, width,
                label='Defense Factor\n(Slope)',
                color='#e74c3c', alpha=0.8)

# 
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels([phase_labels[p] for p in phases_valid], fontsize=11)
ax2.set_ylabel('Combined Importance', fontsize=12, fontweight='bold')
ax2.set_title('Hydraulic vs Defense: Diachronic Comparison',
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='best', fontsize=10, frameon=True, shadow=True)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('Stage_C_Diachronic_Trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nOK")

# ============================================================================
# 8. :Phase II ""
# ============================================================================

print("\n" + "=" * 80)
print("[:Phase II ]")
print("=" * 80)

if 'Phase 2' in phase_results:
# # Phase 2VDCN
    vdcn_idx = X_vars.index('VDCN')
    twi_idx = X_vars.index('TWI')

    vdcn_phase2 = phase_results['Phase 2']['importance'][vdcn_idx]
    twi_phase2 = phase_results['Phase 2']['importance'][twi_idx]
    hydraulic_phase2 = hydraulic_trajectory[phases_valid.index('Phase 2')]

    print(f"\nPhase II () :")
    print(f"  VDCN:     {vdcn_phase2:.4f}")
    print(f"  TWI:      {twi_phase2:.4f}")
    print(f": {hydraulic_phase2:.4f}")

# #
    print(f"\nVDCN:")
    for phase in phases_valid:
        vdcn_val = phase_results[phase]['importance'][vdcn_idx]
        marker = "<-" if vdcn_val == max([phase_results[p]['importance'][vdcn_idx]
                                               for p in phases_valid]) else ""
        print(f"  {phase_labels[phase]:25}: {vdcn_val:.4f}{marker}")

    print(f"\n:")
    for i, phase in enumerate(phases_valid):
        hydraulic_val = hydraulic_trajectory[i]
        marker = "<-" if hydraulic_val == max(hydraulic_trajectory) else ""
        print(f"  {phase_labels[phase]:25}: {hydraulic_val:.4f}{marker}")

# #
    vdcn_is_max = (vdcn_phase2 == max([phase_results[p]['importance'][vdcn_idx]
                                        for p in phases_valid]))
    hydraulic_is_max = (hydraulic_phase2 == max(hydraulic_trajectory))

    print(f"\n{'='*60}")
    print("[]")
    print(f"{'='*60}")

    if vdcn_is_max and hydraulic_is_max:
        print("OKOKOK H1!")
        print("Phase II()''")
        print("VDCNPhase II")
    elif vdcn_is_max or hydraulic_is_max:
        print("OK H1")
        print("Phase II")
    else:
        print(" H1")
        print("Phase II")
        max_phase = phases_valid[hydraulic_trajectory.index(max(hydraulic_trajectory))]
        print(f"{max_phase}")
else:
    print("WARN Phase 2 , ")

# ============================================================================
# 9.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

sample_stats = pd.DataFrame({
    'Phase': phases_valid,
    'Total_Sites': [phase_results[p]['n_samples'] for p in phases_valid],
    'Enclosures': [phase_results[p]['n_enclosure'] for p in phases_valid],
    'Mounds': [phase_results[p]['n_mound'] for p in phases_valid]
})
sample_stats['Enclosure_Ratio'] = (sample_stats['Enclosures'] /
                                   sample_stats['Total_Sites'] * 100).round(1)

print("\n:")
print(sample_stats.to_string(index=False))

sample_stats.to_csv('Stage_C_Sample_Statistics.csv', index=False)
print("\nOK : Stage_C_Sample_Statistics.csv")

# ============================================================================
# 10. C
# ============================================================================

print("\n" + "=" * 80)
print("[C ]")
print("=" * 80)

print(f"\n1. :")
print(f"- : {len(df)}")
print(f"- : {len(df_expanded)}")
print(f"-")

print(f"\n2. :")
for phase in phases_valid:
    stats = phase_results[phase]
    print(f"   {phase_labels[phase]:25}: {stats['n_samples']} "
          f"(Enclosure: {stats['n_enclosure']}, Mound: {stats['n_mound']})")

print(f"\n3. :")
if 'Phase 2' in phase_results:
    if vdcn_is_max:
        print(f"OK Phase IIVDCN ({vdcn_phase2:.3f})")
        print(f"   OK ''")
    else:
        print(f"WARN Phase IIVDCN")
        print(f"WARN")

print(f"\n4. :")
print(f"-")
print(f"-")

print(f"\n5. :")
print(f"- D:")
print(f"   - {len(df[df['Phase']=='Unknown'])}")

print("\n" + "=" * 80)
print("C !")
print("=" * 80)

# ============================================================================
# 11.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# (Colab-only) removed: google.colab files downloader

# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)
# (Colab-only) removed: files.download(...)

print("\nOK")
print("=" * 80)

# --- Notebook cell 3 ---
"""
============================================================================
Stage D: Environmental Affinity Analysis (Environmental Affinity Analysis)
- Google Colab============================================================================
:: +:Analysis, ============================================================================
"""

# ============================================================================
# 1.
# ============================================================================

# (Install dependencies separately: pip install -r requirements.txt)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# 2.
# ============================================================================

print("=" * 80)
print("1:")
print("=" * 80)
print(f"Using input Excel: {INPUT_XLSX}")

# (Colab-only) removed: google.colab files downloader
# Input file is provided via --input (see ARGS above)

filename = INPUT_XLSX
print(f"Loaded: {filename}")

# ============================================================================
# 3.
# ============================================================================

print("\n" + "=" * 80)
print("Stage D: Environmental Affinity Analysis")
print("=" * 80)

# 
df = pd.read_excel(filename)

# 
X_vars = ['TWI', 'VDCN', 'Slope', 'HubDist']

print(f"\n[Dataset overview]")
print(f"Total sites: {len(df)}")
print(f"\nPhase:")
print(df['Phase'].value_counts().sort_index())

# 
df['Has_Date'] = ~df['Phase'].str.contains('Unknown', na=False)
df_dated = df[df['Has_Date']].copy()
df_unknown = df[~df['Has_Date']].copy()

print(f"\n: {len(df_dated)}")
print(f": {len(df_unknown)}")

# ============================================================================
# 4. (Phase 2 vs Phase 3 )
# ============================================================================

print(f"\n[]")
print(f"WARN :Phase 1(Phase 1)")
print(f"Phase 2 vs Phase 3")

def extract_single_phase(phase_str):
    """Phase 2Phase 3"""
    phase_str = str(phase_str).strip()
    if phase_str == 'Phase 2':
        return 'Phase 2'
    elif phase_str == 'Phase 3':
        return 'Phase 3'
    else:
        return None

df_dated['Primary_Phase'] = df_dated['Phase'].apply(extract_single_phase)
df_dated = df_dated[df_dated['Primary_Phase'].notna()]

print(f": {len(df_dated)}")
print(f"\n:")
for phase in ['Phase 2', 'Phase 3']:
    n = len(df_dated[df_dated['Primary_Phase'] == phase])
    print(f"{phase}: {n:3}")

# ============================================================================
# 5. (Phase 2 vs Phase 3)
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# 
X_train = df_dated[X_vars].values
y_train = df_dated['Primary_Phase'].values

# ()
phase_map = {'Phase 2': 0, 'Phase 3': 1}
phase_map_inv = {0: 'Phase 2', 1: 'Phase 3'}
y_train_encoded = np.array([phase_map[p] for p in y_train])

print(f": X={X_train.shape}, y={y_train_encoded.shape}")

# 
rf_chrono = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 
print(f"\n[]")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_chrono, X_train, y_train_encoded,
                            cv=skf, scoring='accuracy')
print(f": {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

if cv_scores.mean() < 0.4:
    print("\nWARN :(<40%)")
    print("")
    print(", ")

# 
rf_chrono.fit(X_train, y_train_encoded)
print("\nOK")

# 
print(f"\n():")
for var, imp in zip(X_vars, rf_chrono.feature_importances_):
    print(f"  {var:10}: {imp:.4f}")

# ============================================================================
# 6. Predict undated sites
# ============================================================================

print("\n" + "=" * 80)
print("[Predict undated sites]")
print("=" * 80)

if len(df_unknown) == 0:
    print("WARN")
else:
# #
    X_unknown = df_unknown[X_vars].values

# #
    y_pred = rf_chrono.predict(X_unknown)
    y_proba = rf_chrono.predict_proba(X_unknown)

# # DataFrame
    predictions = []
    for i, (idx, row) in enumerate(df_unknown.iterrows()):
        pred_phase = phase_map_inv[y_pred[i]]
        proba = y_proba[i]
        max_proba = proba.max()

        predictions.append({
            'Site_ID': row['SiteID'],
            'Site_Name': row['Name'],
            'Type': row['Type'],
            'Predicted_Phase': pred_phase,
            'Confidence': max_proba,
            'Prob_Phase2': proba[0],
            'Prob_Phase3': proba[1]
        })

    pred_df = pd.DataFrame(predictions)

# #
    pred_df = pred_df.sort_values('Confidence', ascending=False)

    print(f"\nPrediction results({len(pred_df)}):\n")
    print("=" * 90)
    print(f"{'Site_ID':<8} {'Name':<20} {'Type':<10} {'Predicted':<10} {'Confidence':<12} {'P2':<6} {'P3':<6}")
    print("=" * 80)

    for _, row in pred_df.iterrows():
        print(f"{row['Site_ID']:<8} {row['Site_Name']:<20} {row['Type']:<10} "
              f"{row['Predicted_Phase']:<10} {row['Confidence']:.2%}{'':>6} "
              f"{row['Prob_Phase2']:.2f}  {row['Prob_Phase3']:.2f}")

# # Confidence grouping
    print(f"\n{'='*90}")
    print("[Confidence grouping]")
    print(f"{'='*90}")

    high_conf = pred_df[pred_df['Confidence'] >= 0.7]
    med_conf = pred_df[(pred_df['Confidence'] >= 0.5) & (pred_df['Confidence'] < 0.7)]
    low_conf = pred_df[pred_df['Confidence'] < 0.5]

    print(f"High confidence (≥70%): {len(high_conf)}")
    print(f"(50-70%): {len(med_conf)}")
    print(f"Low confidence (<50%): {len(low_conf)}")

# #
    print(f"\n:")
    for phase in ['Phase 2', 'Phase 3']:
        n = len(pred_df[pred_df['Predicted_Phase'] == phase])
        print(f"{phase}: {n}")

# #
    pred_df.to_csv('Stage_D_Predictions.csv', index=False)
    print(f"\nOK Prediction results: Stage_D_Predictions.csv")

# ============================================================================
# 7. :
# ============================================================================

if len(df_unknown) > 0:
    print("\n" + "=" * 80)
    print("[]")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # ========== 1:() ==========
    ax1 = axes[0, 0]

    colors_type = {'Enclosure': '#ff6b6b', 'Mound': '#4ecdc4'}
    for site_type in ['Enclosure', 'Mound']:
        mask = pred_df['Type'] == site_type
        if mask.sum() > 0:
# # Phase 2 vs Phase 3
            ax1.scatter(pred_df[mask]['Prob_Phase2'],
                       np.random.uniform(0, 1, mask.sum()),  # y
                       s=pred_df[mask]['Confidence']*300,
                       alpha=0.6,
                       c=colors_type[site_type],
                       label=site_type,
                       edgecolors='black',
                       linewidth=0.5)

    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax1.text(0.25, 0.95, 'Phase 2', fontsize=12, fontweight='bold',
            ha='center', transform=ax1.transAxes)
    ax1.text(0.75, 0.95, 'Phase 3', fontsize=12, fontweight='bold',
            ha='center', transform=ax1.transAxes)
    ax1.set_xlabel('P(Phase 2) <- -> P(Phase 3)', fontweight='bold')
    ax1.set_ylabel('Random Jitter (for visualization)', fontweight='bold')
    ax1.set_title('Prediction Probability Distribution\n(Size = Confidence)',
                  fontweight='bold', pad=15)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

# # ========== 2: ==========
    ax2 = axes[0, 1]

    ax2.hist(pred_df['Confidence'], bins=10, color='#3498db',
            alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2,
               label='50% Threshold')
    ax2.axvline(0.7, color='orange', linestyle='--', linewidth=2,
               label='70% Threshold')
    ax2.set_xlabel('Prediction Confidence', fontweight='bold')
    ax2.set_ylabel('Number of Sites', fontweight='bold')
    ax2.set_title('Distribution of Prediction Confidence',
                  fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

# # ========== 3:Type ==========
    ax3 = axes[1, 0]

    pred_by_type = pred_df.groupby(['Type', 'Predicted_Phase']).size().unstack(fill_value=0)
    pred_by_type.plot(kind='bar', ax=ax3, color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                     alpha=0.8, width=0.6)
    ax3.set_xlabel('Site Type', fontweight='bold')
    ax3.set_ylabel('Number of Sites', fontweight='bold')
    ax3.set_title('Predicted Phase by Site Type', fontweight='bold', pad=15)
    ax3.legend(title='Predicted Phase', loc='best')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    ax3.grid(axis='y', alpha=0.3)

# # ========== 4:High confidence ==========
    ax4 = axes[1, 1]
    ax4.axis('off')

    if len(high_conf) > 0:
        table_data = []
        for _, row in high_conf.head(10).iterrows():
            table_data.append([
                f"{row['Site_ID']}",
                f"{row['Site_Name'][:12]}...",
                f"{row['Predicted_Phase'].replace('Phase ', 'P')}",
                f"{row['Confidence']:.1%}"
            ])

        table = ax4.table(cellText=table_data,
                         colLabels=['ID', 'Name', 'Phase', 'Conf.'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15, 0.4, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

# #
        for i in range(len(table_data) + 1):
            if i == 0:
                for j in range(4):
                    table[(i, j)].set_facecolor('#3498db')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(4):
                    table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

        ax4.set_title(f'High Confidence Predictions (Top {min(len(high_conf), 10)})',
                     fontweight='bold', pad=20, fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No high confidence\npredictions available',
                ha='center', va='center', fontsize=14, color='gray')
        ax4.set_title('High Confidence Predictions', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('Stage_D_Prediction_Visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nOK")

# ============================================================================
# 8. ()
# ============================================================================

print("\n" + "=" * 80)
print("[()]")
print("=" * 80)

y_train_pred = rf_chrono.predict(X_train)

print("\n:")
print(classification_report(y_train_encoded, y_train_pred,
                           target_names=['Phase 2', 'Phase 3'],
                           digits=3))

# 
cm = confusion_matrix(y_train_encoded, y_train_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Phase 2', 'Phase 3'],
           yticklabels=['Phase 2', 'Phase 3'])
plt.xlabel('Predicted Phase', fontweight='bold')
plt.ylabel('Actual Phase', fontweight='bold')
plt.title('Confusion Matrix (Training Set)', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('Stage_D_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nOK")

# ============================================================================
# 9. D
# ============================================================================

print("\n" + "=" * 80)
print("[D ]")
print("=" * 80)

print(f"\n1. :")
print(f"- : {cv_scores.mean():.2%}")
if cv_scores.mean() < 0.5:
    print(f"WARN (33%)")
    print(f"WARN")
else:
    print(f"OK")

print(f"\n2. Prediction results:")
if len(df_unknown) > 0:
    print(f"- {len(pred_df)}")
    print(f"- High confidence(≥70%): {len(high_conf)}")
    print(f"- (50-70%): {len(med_conf)}")
    print(f"- Low confidence(<50%): {len(low_conf)}")
else:
    print(f"-")

print(f"\n3. :")
print(f"WARN Phase 2()vs Phase 3(-)")
print(f"WARN Phase 1()")
print(f"WARN Analysis")
print(f"WARN (, )")
print(f"   WARN 'X'")
print(f"   WARN 'X'")

print(f"\n4. :")
print(f"OK")
print(f"OK")
print(f"OK")

print("\n" + "=" * 80)
print("D !")
print("Analysis!")
print("=" * 80)

# ============================================================================
# 10.
# ============================================================================

print("\n" + "=" * 80)
print("[]")
print("=" * 80)

# (Colab-only) removed: google.colab files downloader


