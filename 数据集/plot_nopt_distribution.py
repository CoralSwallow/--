"""
LSC 训练集与测试集 nopt 分布对比可视化
生成：直方图 + 核密度图（KDE）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 数据清洗函数（与建模时保持一致）====================
def clean_numeric_column(df, col):
    """提取数值列中的数字部分（例如 48.00,[41] -> 48.00）"""
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ==================== 加载数据 ====================
print("加载数据...")
df_train = pd.read_csv('train.csv', encoding='utf-8-sig')
df_test = pd.read_csv('test.csv', encoding='utf-8-sig')

# 需要清理的数值列（包括目标列）
numeric_cols = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY', 'nopt']
for col in numeric_cols:
    clean_numeric_column(df_train, col)
    clean_numeric_column(df_test, col)

# 删除目标变量缺失的行
df_train = df_train.dropna(subset=['nopt']).copy()
df_test = df_test.dropna(subset=['nopt']).copy()
print(f"训练集有效样本: {len(df_train)}")
print(f"测试集有效样本: {len(df_test)}")

# 提取目标变量
y_train = df_train['nopt']
y_test = df_test['nopt']

# 统计信息
print("\n训练集 nopt 统计:")
print(y_train.describe())
print("\n测试集 nopt 统计:")
print(y_test.describe())

# ==================== 绘图设置 ====================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# -------------------- 图1：直方图（叠加）--------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30, alpha=0.6, label='Training Set', color='blue', edgecolor='black')
plt.hist(y_test, bins=30, alpha=0.6, label='Test Set', color='orange', edgecolor='black')
# 添加均值线
train_mean = y_train.mean()
test_mean = y_test.mean()
plt.axvline(train_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Train Mean = {train_mean:.2f}%')
plt.axvline(test_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Test Mean = {test_mean:.2f}%')
plt.xlabel('Optical Efficiency η_opt (%)')
plt.ylabel('Frequency')
plt.title('Histogram of η_opt Distribution')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# -------------------- 图2：核密度图（KDE）--------------------
plt.subplot(1, 2, 2)
sns.kdeplot(y_train, label='Training Set', shade=True, color='blue', linewidth=2)
sns.kdeplot(y_test, label='Test Set', shade=True, color='orange', linewidth=2)
plt.xlabel('η_opt (%)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (KDE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('nopt_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 可选：单独保存两个图
# 直方图单独保存
plt.figure(figsize=(8,5))
plt.hist(y_train, bins=30, alpha=0.6, label='Training Set', color='blue', edgecolor='black')
plt.hist(y_test, bins=30, alpha=0.6, label='Test Set', color='orange', edgecolor='black')
plt.axvline(train_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Train Mean = {train_mean:.2f}%')
plt.axvline(test_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Test Mean = {test_mean:.2f}%')
plt.xlabel('η_opt (%)')
plt.ylabel('Frequency')
plt.title('Histogram of η_opt')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('nopt_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

# KDE图单独保存
plt.figure(figsize=(8,5))
sns.kdeplot(y_train, label='Training Set', shade=True, color='blue', linewidth=2)
sns.kdeplot(y_test, label='Test Set', shade=True, color='orange', linewidth=2)
plt.xlabel('η_opt (%)')
plt.ylabel('Density')
plt.title('KDE of η_opt')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('nopt_kde.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存: nopt_distribution_comparison.png, nopt_histogram.png, nopt_kde.png")