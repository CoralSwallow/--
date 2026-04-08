import pandas as pd
import numpy as np

def clean_numeric_column(df, col):
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 加载当前数据
df_train = pd.read_csv('train.csv', encoding='utf-8-sig')
df_test = pd.read_csv('test.csv', encoding='utf-8-sig')

# 清洗目标列（与脚本一致）
for col in ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY', 'nopt']:
    clean_numeric_column(df_train, col)
    clean_numeric_column(df_test, col)

# 查看目标变量统计
print("训练集 nopt 统计：")
print(df_train['nopt'].describe())
print("\n测试集 nopt 统计：")
print(df_test['nopt'].describe())

# 查看训练集和测试集的目标值范围是否一致
print(f"\n训练集 nopt 范围: {df_train['nopt'].min():.2f} - {df_train['nopt'].max():.2f}")
print(f"测试集 nopt 范围: {df_test['nopt'].min():.2f} - {df_test['nopt'].max():.2f}")

# 检查是否有异常大的值或缺失值
print(f"\n训练集 nopt 缺失数: {df_train['nopt'].isna().sum()}")
print(f"测试集 nopt 缺失数: {df_test['nopt'].isna().sum()}")

# 检查分类特征的取值是否合理
print("\n训练集 Type_OC 取值：", df_train['Type_OC'].unique())
print("测试集 Type_OC 取值：", df_test['Type_OC'].unique())