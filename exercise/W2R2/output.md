(test_bed) D:\AIGC\castorice-core>python exercise/W2R2/sentiment_classification_predict.py
2025-09-24 22:55:54.158417: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-09-24 22:55:55.311452: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\sharing\conda\envs\test_bed\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

加载BGE-m3模型...
加载已训练的逻辑回归模型...
提取文本嵌入特征...
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.70it/s] 
嵌入特征形状: (5, 1024)
进行情感预测...

--- 预测结果 ---
文本: '我很喜欢这家餐厅'
  -> 预测情感: 正面 (标签: 1)

文本: '物流速度太慢了，等了半个月才到。'
  -> 预测情感: 负面 (标签: 0)

文本: '电影的特效很震撼，故事情节也很吸引人。'
  -> 预测情感: 正面 (标签: 1)

文本: '产品刚用就坏了，质量堪忧。'
  -> 预测情感: 负面 (标签: 0)

文本: '今天的夕阳真美，心情舒畅。'
  -> 预测情感: 正面 (标签: 1)