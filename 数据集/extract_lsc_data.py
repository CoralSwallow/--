import re
import pandas as pd
import pdfplumber

def extract_tables_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    # 定位表格区域
    s1_start = full_text.find("Table S1.")
    s2_start = full_text.find("Table S2.")
    s1_text = full_text[s1_start:s2_start]
    s2_text = full_text[s2_start:]

    # 按行分割
    lines = s1_text.split('\n')
    train_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('Table', 'Entry', 'Materials', 'Optical', 'Supplementary')):
            continue
        # 尝试匹配数值特征（Ap, Amin, ... 等都是数字，可用来判断是否为数据行）
        # 如果行中包含至少6个数字（或小数点），则认为是数据行
        numbers = re.findall(r'\d+\.?\d*', line)
        if len(numbers) >= 6:  # 至少有 Ap, Amin, Amax, Ep, Emin, Emax 这些数值
            # 按空白分割（可以处理多个空格）
            fields = re.split(r'\s+', line)
            # 前几个字段可能是文字，后面是数字，但字段数不确定
            # 我们直接保留，后续再用 CSV 方式二次处理
            train_lines.append(line)

    # 同样处理测试集
    lines = s2_text.split('\n')
    test_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('Table', 'Entry', 'Materials', 'Optical', 'Supplementary')):
            continue
        numbers = re.findall(r'\d+\.?\d*', line)
        if len(numbers) >= 6:
            test_lines.append(line)

    return train_lines, test_lines

def parse_line_to_row(line, has_ref=False):
    # 将行按空白分割，然后尝试将数字部分转成数值
    fields = re.split(r'\s+', line)
    # 字段顺序固定，但可能有缺失值
    # 例如: 1 dye hybrid GLYMO/TMSO/ASB film 578 420 600 613 550 750 98 18.80
    # 但某些字段可能有空格，导致分割后数量不一致，需要基于特征位置进行智能分配
    # 这里简化：假设前5个是文本字段（Entry, Type_OC, Chemical_OC, Host, Processing），之后6个数字是吸收/发射特征，再之后是QY和nopt
    # 如果有引用，最后还有Reference
    # 由于格式复杂，我们采用手动调整的方式，直接使用之前整理好的CSV更稳妥
    return None

def main(pdf_path):
    train_lines, test_lines = extract_tables_from_pdf(pdf_path)
    print(f"提取到训练集行数: {len(train_lines)}")
    print(f"提取到测试集行数: {len(test_lines)}")
    # 这里需要进一步解析每一行，但鉴于复杂度，建议使用手动整理的CSV

if __name__ == "__main__":
    main("补充数据1.pdf")