"""
LSC 光学效率预测建模脚本
训练集：train.csv
测试集：test.csv
目标变量：nopt (%)
作者：Coral
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
print("正在加载数据...")
# 使用从补充材料提取的训练集和测试集
df_train = pd.read_csv('train.csv', encoding='utf-8-sig')
df_test = pd.read_csv('test.csv', encoding='utf-8-sig')

# 查看基本信息
print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")
print("训练集列名:", df_train.columns.tolist())
print("测试集列名:", df_test.columns.tolist())

# 目标列名 'nopt'
target_col = 'nopt'
if target_col not in df_train.columns:
    raise KeyError(f"训练集中找不到目标列 '{target_col}'，现有列名: {df_train.columns.tolist()}")

# ==================== 2. 数据清洗 ====================
print("\n正在清洗数据...")

# 删除目标变量缺失的行（测试集也需要删除，因为我们需要真实值来评估）
df_train = df_train.dropna(subset=[target_col]).copy()
df_test = df_test.dropna(subset=[target_col]).copy()

print(f"训练集清洗后: {len(df_train)} 行")
print(f"测试集清洗后: {len(df_test)} 行")

# 检查缺失比例（可选）
print("\n训练集缺失比例:")
print(df_train.isnull().mean().sort_values(ascending=False))

# ==================== 3. 特征工程 ====================
print("\n正在进行特征工程...")

# 定义特征列（根据实际列名）
numeric_features = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax', 'QY']
categorical_features = ['Type_OC', 'Host', 'Processing']

# 只保留存在的列
numeric_features = [col for col in numeric_features if col in df_train.columns]
categorical_features = [col for col in categorical_features if col in df_train.columns]

print(f"数值特征: {numeric_features}")
print(f"分类特征: {categorical_features}")

# 强制数值特征列为数值类型（无法转换的变为 NaN）
for col in numeric_features:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# 处理分类变量缺失值（用 'unknown' 填充）
for col in categorical_features:
    df_train[col] = df_train[col].fillna('unknown').astype(str)
    df_test[col] = df_test[col].fillna('unknown').astype(str)

# 构造新特征（如 Stokes 位移）
if 'Ep' in df_train.columns and 'Ap' in df_train.columns:
    df_train['Stokes_shift'] = df_train['Ep'] - df_train['Ap']
    df_test['Stokes_shift'] = df_test['Ep'] - df_test['Ap']
    numeric_features.append('Stokes_shift')
    print("添加新特征: Stokes_shift")

if 'Amax' in df_train.columns and 'Amin' in df_train.columns:
    df_train['Abs_bandwidth'] = df_train['Amax'] - df_train['Amin']
    df_test['Abs_bandwidth'] = df_test['Amax'] - df_test['Amin']
    numeric_features.append('Abs_bandwidth')
    print("添加新特征: Abs_bandwidth")

# 分离特征和目标
X_train = df_train[numeric_features + categorical_features]
y_train = df_train[target_col].copy()   # 确保是纯数值

X_test = df_test[numeric_features + categorical_features]
y_test = df_test[target_col].copy()

# 检查
print("y_train 示例:", y_train.head())
print("y_train 类型:", y_train.dtype)

print(f"特征矩阵形状: 训练集 {X_train.shape}, 测试集 {X_test.shape}")

# # ==================== 4. 数据划分 ====================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
# print("训练集 nopt 范围:", y_train.min(), "-", y_train.max())
# print("测试集 nopt 范围:", y_test.min(), "-", y_test.max())
# print("训练集 nopt 分位数:\n", y_train.quantile([0.25, 0.5, 0.75]))
# print("测试集 nopt 分位数:\n", y_test.quantile([0.25, 0.5, 0.75]))

# ==================== 5. 创建预处理管道 ====================
# 数值特征：先中位数填充，再标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征：先填充 'missing'，再独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==================== 6. 随机森林模型 ====================
print("\n训练随机森林模型...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"随机森林 - RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}")

# ==================== 7. XGBoost 模型====================
try:
    import xgboost as xgb
    print("\n训练 XGBoost 模型...")
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"XGBoost - RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}, R²: {r2_xgb:.4f}")
except ImportError:
    print("xgboost 未安装，跳过 XGBoost 模型。")

# ==================== 8. 神经网络模型 ====================
print("\n训练神经网络模型...")
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import Dense, Dropout # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore
    from sklearn.preprocessing import StandardScaler

    # 预处理数据（假设 preprocessor 已定义）
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 目标变量标准化
    from sklearn.preprocessing import StandardScaler
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # 输入维度
    input_dim = X_train_processed.shape[1]

    # 构建模型
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # 训练
    history = model.fit(
        X_train_processed, y_train_scaled,
        validation_split=0.2,
        epochs=300,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # 评估（从 model.evaluate 获取损失和 MAE）
    loss_ann, mae_ann = model.evaluate(X_test_processed, y_test_scaled, verbose=0)   # 关键：获取 mae_ann

    # 预测并逆变换
    y_pred_scaled = model.predict(X_test_processed).flatten()
    y_pred_ann = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 计算 RMSE 和 R²
    rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
    r2_ann = r2_score(y_test, y_pred_ann)

    print(f"ANN - RMSE: {rmse_ann:.4f}, MAE: {mae_ann:.4f}, R²: {r2_ann:.4f}")

    # 绘制训练历史
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History (Loss)')

    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='train')
    plt.plot(history.history['val_mae'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training History (MAE)')
    plt.tight_layout()
    plt.show()

except ImportError:
    print("TensorFlow 未安装，跳过神经网络模型。")
except Exception as e:
    print(f"ANN 训练出错: {e}")

# ==================== 9. 模型对比可视化 ====================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, label=f'RF (R²={r2_rf:.3f})')
if 'y_pred_xgb' in locals():
    plt.scatter(y_test, y_pred_xgb, alpha=0.6, label=f'XGB (R²={r2_xgb:.3f})')
if 'y_pred_ann' in locals():
    plt.scatter(y_test, y_pred_ann, alpha=0.6, label=f'ANN (R²={r2_ann:.3f})')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True nopt (%)')
plt.ylabel('Predicted nopt (%)')
plt.legend()
plt.title('True vs Predicted Optical Efficiency')
plt.tight_layout()
plt.show()

# ==================== 10. 特征重要性（随机森林）====================
# 获取特征名称
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))

importances = rf_pipeline.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1][:20]  # 取前20

plt.figure(figsize=(10,8))
plt.title('Top 20 Feature Importances (Random Forest)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==================== 11. 保存结果 ====================
# 保存测试集预测结果
results = pd.DataFrame({
    'True_nopt': y_test,
    'RF_pred': y_pred_rf
})
if 'y_pred_xgb' in locals():
    results['XGB_pred'] = y_pred_xgb
if 'y_pred_ann' in locals():
    results['ANN_pred'] = y_pred_ann

results.to_csv('prediction_results.csv', index=False)
print("\n预测结果已保存到 prediction_results.csv")