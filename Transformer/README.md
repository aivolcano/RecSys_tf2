

### 任务目标
使用Transformer开发排序模型

Transformer提取句子（序列化特征）的能力强于RNN


### 特征工程
* 将 position embedding 和序列中的每个电影movie Embedding相加，得到用户历史行为句子hist_movie_id，反应用户的历史平均兴趣爱好

* 历史历史评分也可以构成序列hist_movie_id，hist_movie_id 乘以 hist_rating，相当于把使用rating作为权重给movie_id加权，实现每个movie embedding 对应不同userID都不同，更能反应用户平均历史行为爱好，让hist_movie_id中 每个movieID权重为1/n 变为 rating（n为hist_movie_id长度）


![feature_extraction](https://user-images.githubusercontent.com/68730894/115144335-bd77ca80-a07e-11eb-99b5-70075fdd37da.png)


### 模型结构
与DeepFM相似，FM换成Transformer，其余部分保持不变

### 模型Pooling
* 每个用户特征均使用tf.keras.layer.Embedding进行编码，embedding_dim等于特征个数的平方根取整。
* 电影序列 sequence_movie_ids 中每部电影和目标电影都使用tf.keras.layers.Embedding编码。Embedding_dim是电影数量的平方根取整。
* 将每部电影的embedding与其电影画像Embedding融合起来，并使用非线性激活函数激活，输出有同等大小的张量。
* 目标电影Embedding与序列影片Embedding进行连接，生成`[batch_size, seq_len, embedding_dim]`的张量。
* 该方法返回两个元素的元组：encoded_transformer_features和encoded_other_features。

### 结果
不使用hist_rating 对 hist_movie_id 进行加权
** TestMAE：0.786 **

使用hist_rating 对 hist_movie_id 进行加权
** TestMAE：0.752 **

### 改进方向
* 最后一个MovieID单独作为一个特征喂给模型，因为有论文表明，最后一个movieid特别重要。

* 对每个userID进行全局负采样

* 增加FM完成类别特征组合，Transformer提取序列化特征，DNN提取类别特征和连续特征
