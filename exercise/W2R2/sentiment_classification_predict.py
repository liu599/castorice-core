# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/9/24
# @Author      : liuboyuan
# @Description : 使用已保存的模型进行情感分类预测

import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

def predict_sentiment(text_list):
    """
    加载模型并对输入的文本列表进行情感预测。
    
    参数:
    text_list (list of str): 需要进行情感预测的文本列表。
    
    返回:
    list of int: 预测结果列表，0表示负面，1表示正面。
    """
    # --- 1. 加载BGE-m3模型 ---
    print("加载BGE-m3模型...")
    try:
        embedding_model = SentenceTransformer("BAAI/bge-m3")
    except Exception as e:
        print(f"加载BGE-m3模型失败: {e}")
        return None

    # --- 2. 加载已训练的逻辑回归模型 ---
    print("加载已训练的逻辑回归模型...")
    try:
        sentiment_model = joblib.load('sentiment_model.joblib')
    except FileNotFoundError:
        print("错误: 未找到'sentiment_model.joblib'。请先运行训练脚本。")
        return None
    except Exception as e:
        print(f"加载逻辑回归模型失败: {e}")
        return None

    # --- 3. 提取文本特征 ---
    print("提取文本嵌入特征...")
    text_embeddings = embedding_model.encode(text_list, show_progress_bar=True)
    print(f"嵌入特征形状: {text_embeddings.shape}")

    # --- 4. 进行预测 ---
    print("进行情感预测...")
    predictions = sentiment_model.predict(text_embeddings)
    
    return predictions

if __name__ == '__main__':
    # --- 示例用法 ---
    new_texts = [
        "我很喜欢这家餐厅",
        "物流速度太慢了，等了半个月才到。",
        "电影的特效很震撼，故事情节也很吸引人。",
        "产品刚用就坏了，质量堪忧。",
        "今天的夕阳真美，心情舒畅。"
    ]

    # 获取预测结果
    predicted_labels = predict_sentiment(new_texts)

    if predicted_labels is not None:
        print("\n--- 预测结果 ---")
        for text, label in zip(new_texts, predicted_labels):
            sentiment = "正面" if label == 1 else "负面"
            print(f"文本: '{text}'\n  -> 预测情感: {sentiment} (标签: {label})\n")