"""
LSC 光学效率预测 - 最简可靠版
基于原始特征，无派生特征，无特征选择，XGBoost 随机搜索调优
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载与清洗 ====================
def clean_numeric_column(df, col):
    """提取数值列中的数字部分（例如 48.00,[41] -> 48.00）"""
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

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
print(f"训练集: {len(df_train)} 行, 测试集: {len(df_test)} 行")

# ==================== 2. 特征定义 ====================
# 仅使用原始数值特征
numeric_features = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY']
categorical_features = ['Type_OC', 'Host', 'Processing']

# 只保留存在的列
numeric_features = [col for col in numeric_features if col in df_train.columns]
categorical_features = [col for col in categorical_features if col in df_train.columns]

# 填充分类变量缺失值
for col in categorical_features:
    df_train[col] = df_train[col].fillna('unknown').astype(str)
    df_test[col] = df_test[col].fillna('unknown').astype(str)

# ==================== 3. 预处理管道 ====================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 应用预处理
X_train = preprocessor.fit_transform(df_train[numeric_features + categorical_features])
X_test = preprocessor.transform(df_test[numeric_features + categorical_features])
y_train = df_train['nopt']
y_test = df_test['nopt']

print(f"预处理后特征维度: {X_train.shape[1]}")

# ==================== 4. XGBoost 超参数调优（轻度）====================
print("\n超参数调优 (RandomizedSearchCV) ...")
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
search = RandomizedSearchCV(xgb_model, param_dist, n_iter=30, cv=5, scoring='r2',
                            random_state=42, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
print(f"最佳参数: {search.best_params_}")
best_xgb = search.best_estimator_

# ==================== 5. 评估 ====================
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n最终测试集性能:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ==================== 6. 可视化 ====================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True nopt (%)')
plt.ylabel('Predicted nopt (%)')
plt.title(f'XGBoost: True vs Predicted (R² = {r2:.3f})')
plt.tight_layout()
plt.savefig('xgb_reliable.png', dpi=150)
plt.show()

# 特征重要性（获取特征名称）
cat_feature_names = list(preprocessor.named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_feature_names
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1][:15]
plt.figure(figsize=(10,6))
plt.title('Top 15 Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_importance_reliable.png', dpi=150)
plt.show()

# ==================== 7. 保存模型 ====================
# 保存完整 pipeline（包括预处理和模型）
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_xgb)
])
full_pipeline.fit(df_train[numeric_features + categorical_features], y_train)
joblib.dump(full_pipeline, 'lsc_reliable_pipeline.pkl')
print("\n模型已保存为 lsc_reliable_pipeline.pkl")

# 保存预测结果
results = pd.DataFrame({'True': y_test, 'Pred': y_pred})
results.to_csv('predictions_reliable.csv', index=False)
print("预测结果已保存")