"""
LSC 光学效率预测 - 针对分布不一致的测试集优化版
策略：对数变换 + 样本加权 + XGBoost 调优
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
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
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("加载数据...")
df_train = pd.read_csv('train.csv', encoding='utf-8-sig')
df_test = pd.read_csv('test.csv', encoding='utf-8-sig')

numeric_cols = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY', 'nopt']
for col in numeric_cols:
    clean_numeric_column(df_train, col)
    clean_numeric_column(df_test, col)

df_train = df_train.dropna(subset=['nopt']).copy()
df_test = df_test.dropna(subset=['nopt']).copy()
print(f"训练集: {len(df_train)} 行, 测试集: {len(df_test)} 行")

# ==================== 2. 特征定义 ====================
numeric_features = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY']
categorical_features = ['Type_OC', 'Host', 'Processing']
numeric_features = [col for col in numeric_features if col in df_train.columns]
categorical_features = [col for col in categorical_features if col in df_train.columns]

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
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train = preprocessor.fit_transform(df_train[numeric_features + categorical_features])
X_test = preprocessor.transform(df_test[numeric_features + categorical_features])

# ==================== 4. 目标变量对数变换 ====================
y_train = df_train['nopt']
y_test = df_test['nopt']
y_train_log = np.log1p(y_train)   # log(1+y)
y_test_log = np.log1p(y_test)

# ==================== 5. 计算样本权重（让高值样本获得更高权重）====================
# 权重与 y_train 值成正比（例如权重 = 1 + y_train/10），避免过大的权重导致不稳定
weights = 1 + y_train / 10
weights = np.clip(weights, 0.5, 3.0)   # 限制权重范围

# ==================== 6. XGBoost 超参数调优（使用对数目标）====================
print("\n超参数调优 (RandomizedSearchCV) ...")
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}
# 注意：RandomizedSearchCV 不支持直接传递样本权重，将在搜索后手动训练最佳模型时使用权重
# 这里先不设置权重，只搜索超参数
search = RandomizedSearchCV(xgb_model, param_dist, n_iter=50, cv=5, scoring='r2',
                            random_state=42, n_jobs=-1, verbose=1)
search.fit(X_train, y_train_log)   # 注意：使用对数目标进行搜索
print(f"最佳参数: {search.best_params_}")
best_params = search.best_params_

# ==================== 7. 使用最佳参数和样本权重训练最终模型 ====================
print("\n使用样本权重训练最终模型...")
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train_log, sample_weight=weights)

# 预测（对数空间）
y_pred_log = final_model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # 还原到原始尺度

# ==================== 8. 评估 ====================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n最终测试集性能 (对数变换+权重):")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ==================== 9. 可视化 ====================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True nopt (%)')
plt.ylabel('Predicted nopt (%)')
plt.title(f'XGBoost (log+weight): R² = {r2:.3f}')
plt.tight_layout()
plt.savefig('xgb_improved.png', dpi=150)
plt.show()

# ==================== 10. 保存模型 ====================
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', final_model)
])
full_pipeline.fit(df_train[numeric_features + categorical_features], np.log1p(df_train['nopt']))
joblib.dump(full_pipeline, 'lsc_improved_pipeline.pkl')
print("模型已保存为 lsc_improved_pipeline.pkl")