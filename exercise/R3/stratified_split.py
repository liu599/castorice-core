# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/9/23
# @Author      : liuboyuan
# @Description :
import pandas as pd
from sklearn.model_selection import train_test_split

def print_distribution(df: pd.DataFrame, column: str, title: str):
    """
    计算并打印数据集中指定列的分布情况。

    :param df: 数据集 (DataFrame)
    :param column: 要分析的列名
    :param title: 打印的标题
    """
    distribution = df[column].value_counts(normalize=True).sort_index()
    print(f"\n【{title}】{column}分布：")
    for item, ratio in distribution.items():
        print(f"  {item}: {ratio:.2%}")
    return distribution

# 1. 读取数据
DATA_PATH = "user_profiles.csv"
df = pd.read_csv(DATA_PATH)
tag = "所在城市"
print(f"体会分层抽样【{tag}】")

# 2. 统计“消费水平”在全体数据中的分布（黄金标准）
golden_dist = print_distribution(df, tag, "基准标准")

# 3. 基于“消费水平”分层抽样划分训练集与测试集
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[tag]
)

# 4. 统计训练集和测试集分布
train_dist = print_distribution(train_df, tag, "训练集")
test_dist = print_distribution(test_df, tag, "测试集")

# 【可选】合并展示
print("\n【对比总结】")
summary = pd.DataFrame({
    '全体数据': golden_dist,
    '训练集': train_dist,
    '测试集': test_dist
})
print(summary.applymap(lambda x: f"{x:.2%}"))