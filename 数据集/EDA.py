import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取合并数据
df = pd.read_csv('merged_all_data.csv', encoding='utf-8-sig')

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 列出想要绘制的数值列（请根据实际列名修改）
target_cols = ['η_opt', 'PCE']  # 可能只有部分文献有这些列
spectral_cols = ['Ap', 'Amin', 'Amax', 'Ep', 'Emin', 'Emax']
qy_cols = ['量子产率', 'η_yield']  # 不同文献命名可能不同
other_cols = ['吸收层厚度', '缺陷密度', '电子亲和能']  # 来自 arif.csv

# 合并所有可能存在的数值列
all_num_cols = [col for col in target_cols + spectral_cols + qy_cols + other_cols if col in df.columns]

# 对每个数值列绘制直方图（按文献分组）
for col in all_num_cols:
    plot_df = df[['文献', col]].dropna()
    if len(plot_df) == 0:
        print(f"列 {col} 无有效数据，跳过")
        continue
    
    plt.figure()
    sns.histplot(data=plot_df, x=col, hue='文献', kde=True, bins=20, alpha=0.6)
    plt.title(f'Distribution of {col} by Literature')
    plt.xlabel(col)
    plt.ylabel('Count')
    # 如果确实需要修改图例标题，使用以下方式
    ax = plt.gca()
    if ax.legend_ is not None:
        ax.legend_.set_title('Literature')
    plt.tight_layout()
    plt.savefig(f'hist_{col}.png', dpi=150)
    plt.show()