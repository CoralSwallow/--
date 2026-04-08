import pandas as pd
import numpy as np
from collections import defaultdict
import os

# ==================== 配置区域 ====================
FILE_INFO = {
    'andre.csv': 'André 2024',
    'ayasi.csv': 'Ayasi 2025',
    'garcia_moreno.csv': 'Garcia-Moreno 2024',
    'ferreira.csv': 'Ferreira 2024',
    'shu.csv': '束俊鹏 2019',
    'arif.csv': 'Arif 2025'
}
# =================================================

# ==================== 解析函数 ====================

def parse_andre(filepath):
    """André 2024 (两列键值对)"""
    df = pd.read_csv(filepath, encoding='utf-8-sig', header=None, names=['参数', '值'])
    return df.set_index('参数').T

def parse_ayasi(filepath):
    """Ayasi 2025 (表格格式，前有元数据行)"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 定位表头行（以"模型,网络类型"开头）
    header_idx = next(i for i, line in enumerate(lines) if line.startswith('模型,网络类型'))
    header = lines[header_idx].split(',')
    data = []
    for line in lines[header_idx+1:]:
        if line.startswith('能耗优势'):
            break
        parts = line.split(',')
        if len(parts) < len(header):
            parts += [np.nan] * (len(header) - len(parts))
        data.append(parts[:len(header)])
    return pd.DataFrame(data, columns=header)

def parse_garcia(filepath):
    """Garcia-Moreno 2024 (三列键值对：类别,参数,值)"""
    df = pd.read_csv(filepath, encoding='utf-8-sig', header=None, names=['类别', '参数', '值'])
    df['新列名'] = df['类别'] + '_' + df['参数']
    transposed = df.set_index('新列名')[['值']].T
    return transposed

def parse_ferreira(filepath):
    """Ferreira 2024 (混合格式：元数据 + 表格 + 聚类信息)"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.rstrip('\n') for line in f]

    base_dict = {}
    data_lines = []
    header_line = None
    cluster_info = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '':
            i += 1
            continue
        parts = line.split(',')
        if parts[0] == '数据集':
            if len(parts) >= 3:
                key = parts[1].strip()
                value = parts[2].strip() if len(parts) > 2 else ''
                base_dict[f'数据集_{key}'] = value
            i += 1
        elif parts[0] == '回归模型':
            header_line = parts
            i += 1
            while i < len(lines):
                data_line = lines[i].strip()
                if data_line == '':
                    i += 1
                    continue
                data_parts = data_line.split(',')
                if data_parts[0] in ['聚类方法', '聚类与回归一致性']:
                    break
                if len(data_parts) < len(header_line):
                    data_parts += [''] * (len(header_line) - len(data_parts))
                data_lines.append(data_parts[:len(header_line)])
                i += 1
        elif parts[0] in ['聚类方法', '聚类与回归一致性']:
            if len(parts) >= 2:
                key = parts[0].strip() + '_' + parts[1].strip()
                value = parts[2].strip() if len(parts) > 2 else ''
                cluster_info[key] = value
            i += 1
        else:
            i += 1

    base_dict.update(cluster_info)
    df_data = pd.DataFrame(data_lines, columns=header_line)
    for key, value in base_dict.items():
        df_data[key] = value
    return df_data

def parse_shu(filepath):
    """束俊鹏 2019 (两列键值对)"""
    df = pd.read_csv(filepath, encoding='utf-8-sig', header=None, names=['参数', '值'])
    return df.set_index('参数').T

def parse_arif(filepath):
    """Arif 2025 (分节键值对)"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]

    data_rows = []
    base_dict = {}
    blocks = defaultdict(list)
    current_block = None

    for line in lines:
        parts = line.split(',')
        key = parts[0].strip()
        if current_block is None or key != current_block:
            current_block = key
        blocks[current_block].append(parts)

    for block_key, block_lines in blocks.items():
        if block_key in ['器件结构', '模拟工具', 'ANN结构', '训练算法']:
            for parts in block_lines:
                if len(parts) >= 2:
                    sub_key = parts[1].strip()
                    value = parts[2].strip() if len(parts) > 2 else ''
                    base_dict[f'{block_key}_{sub_key}'] = value
        elif block_key == 'ETL比较':
            materials = block_lines[0][2:]
            prop_names = [line[1] for line in block_lines[1:]]
            prop_values = [line[2:] for line in block_lines[1:]]
            for i, mat in enumerate(materials):
                row_dict = base_dict.copy()
                row_dict['ETL材料'] = mat
                row_dict['CBO (eV)'] = prop_values[0][i]
                row_dict['PCE (%)_ETL筛选'] = prop_values[1][i]
                data_rows.append(row_dict)
        elif block_key == '光源':
            parts = block_lines[0]
            if len(parts) >= 2:
                base_dict['光源1'] = parts[1].strip()
                if len(parts) > 2:
                    base_dict['光源2'] = parts[2].strip()
        elif block_key == '优化参数':
            param_names = [line[1] for line in block_lines]
            param_values = [line[2:] for line in block_lines]
            for light_idx in [0, 1]:
                row_dict = base_dict.copy()
                row_dict['光源'] = base_dict.get('光源1' if light_idx==0 else '光源2', '')
                for i, pname in enumerate(param_names):
                    row_dict[pname] = param_values[i][light_idx]
                data_rows.append(row_dict)
        elif block_key == '优化结果':
            result_names = [line[1] for line in block_lines]
            result_values = [line[2:] for line in block_lines]
            for row in data_rows:
                if '光源' in row:
                    if row['光源'] == base_dict.get('光源1', ''):
                        light_idx = 0
                    elif row['光源'] == base_dict.get('光源2', ''):
                        light_idx = 1
                    else:
                        continue
                    for i, rname in enumerate(result_names):
                        row[rname] = result_values[i][light_idx]
        elif block_key == '性能指标(太阳光)':
            for parts in block_lines:
                if len(parts) >= 2:
                    base_dict[f'太阳光_{parts[1]}'] = parts[2] if len(parts)>2 else ''
        elif block_key == '性能指标(LED)':
            for parts in block_lines:
                if len(parts) >= 2:
                    base_dict[f'LED_{parts[1]}'] = parts[2] if len(parts)>2 else ''
        elif block_key == '参数重要性':
            for parts in block_lines:
                if len(parts) >= 2:
                    base_dict[f'重要性_{parts[1]}'] = parts[2] if len(parts)>2 else ''
        else:
            for parts in block_lines:
                if len(parts) >= 2:
                    sub_key = parts[1].strip()
                    value = parts[2].strip() if len(parts) > 2 else ''
                    base_dict[f'{block_key}_{sub_key}'] = value

    if not data_rows:
        data_rows.append(base_dict.copy())
    else:
        for row in data_rows:
            for k, v in base_dict.items():
                if k not in row:
                    row[k] = v

    return pd.DataFrame(data_rows)

# ==================== 主合并流程 ====================

PARSERS = {
    'andre.csv': parse_andre,
    'ayasi.csv': parse_ayasi,
    'garcia_moreno.csv': parse_garcia,
    'ferreira.csv': parse_ferreira,
    'shu.csv': parse_shu,
    'arif.csv': parse_arif
}

def merge_all_data():
    all_data = []
    for filename, label in FILE_INFO.items():
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在，跳过")
            continue
        print(f"正在解析 {filename}...")
        try:
            df_parsed = PARSERS[filename](filename)
            df_parsed['文献'] = label
            all_data.append(df_parsed)
            print(f"  -> 成功解析，获得 {len(df_parsed)} 行数据")
        except Exception as e:
            print(f"  -> 解析失败: {e}")

    if all_data:
        merged = pd.concat(all_data, ignore_index=True, sort=False)
        merged = merged.replace(r'^\s*$', np.nan, regex=True)
        output_file = 'merged_all_data.csv'
        merged.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 合并完成！共 {len(merged)} 行，已保存至 {output_file}")
        print("请用 Excel 打开该文件（UTF-8编码）。若出现乱码，请使用“数据”->“从文本/CSV”导入并选择 UTF-8 编码。")
    else:
        print("没有成功解析任何文件。")

if __name__ == '__main__':
    merge_all_data()