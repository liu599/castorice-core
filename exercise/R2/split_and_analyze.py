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
print(f"统计【{tag}】在全体、训练集、测试集的分布, 体会80/20带来的偏差， 观察随机划分带来的分布偏差")
# 2. 显示全体数据的“消费水平”分布（黄金标准）
golden_dist = print_distribution(df, tag, "基准标准")

# 3. 随机划分训练集和测试集
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True
)

# 4. 统计训练集和测试集分布
train_dist = print_distribution(train_df, tag, "训练集")
test_dist = print_distribution(test_df, tag, "测试集")

# 可选：输出结果到csv便于分析
summary_df = pd.DataFrame({
    "全体分布": golden_dist,
    "训练集": train_dist,
    "测试集": test_dist
})
summary_df.to_csv("level_dist_summary.csv", encoding="utf-8-sig")