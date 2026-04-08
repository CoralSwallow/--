import pandas as pd
import joblib

def load_pipeline(pkl_path='lsc_reliable_pipeline.pkl'):
    """加载保存的模型管道"""
    return joblib.load(pkl_path)

def predict_nopt(pipeline, samples_df):
    """预测光学效率"""
    return pipeline.predict(samples_df)

def main():
    # 加载管道
    pipeline = load_pipeline()
    
    # 准备新数据（假设）
    new_data = pd.DataFrame([
        {'Ap': 520, 'Amin': 450, 'Amax': 650, 'Ep': 580, 'Emin': 500, 'Emax': 700,
        'QY': 88.5, 'Type_OC': 'QD', 'Host': 'polymer', 'Processing': 'bulk'}
    ])
    
    # 预测
    preds = predict_nopt(pipeline, new_data)
    print(f"预测光学效率: {preds[0]:.2f}%")

if __name__ == "__main__":
    main()