import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CSV_HEADER = ['user_id', 'sequence_movie_ids', 'sequence_ratings', 'sex', 'age_group',
       'occupation'] # list(df.columns)

def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features):
        movie_ids_string = features["sequence_movie_ids"]
        sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

        # 预测目标是用户观看电影序列的最后一部电影：The last movie id in the sequence is the target movie.
        features["target_movie_id"] = sequence_movie_ids[:, -1]
        features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

        ratings_string = features["sequence_ratings"]
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()

        # 用户观看电影评分序列的最后一部电影作为评分预测的目标：The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features["sequence_ratings"] = sequence_ratings[:, :-1]
        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset

# model input

def create_model_inputs():
    return {"user_id": tf.keras.layers.Input(name="user_id", shape=(1,), dtype=tf.string),
                "sequence_movie_ids": tf.keras.layers.Input(
                    name="sequence_movie_ids", shape=(sequence_length - 1,), dtype=tf.string
                ),
                "target_movie_id": tf.keras.layers.Input(
                    name="target_movie_id", shape=(1,), dtype=tf.string
                ),
                "sequence_ratings": tf.keras.layers.Input(
                    name="sequence_ratings", shape=(sequence_length - 1,), dtype=tf.float32
                ),
                "sex": tf.keras.layers.Input(name="sex", shape=(1,), dtype=tf.string),
                "age_group": tf.keras.layers.Input(name="age_group", shape=(1,), dtype=tf.string),
                "occupation": tf.keras.layers.Input(name="occupation", shape=(1,), dtype=tf.string),
                # 'sex_genres': tf.keras.layers.Input(name='sex_genres', shape=(1, ), dtype=tf.float16)
            }

# Encode input features
def encode_input_features(inputs, include_user_id=True, include_user_features=True,include_movie_features=True):
    encoded_transformer_features = []
    encoded_other_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)

    ## 用户画像编码： user features
    for feature_name in other_feature_names:
        # 在tf中将字符串输入转为整数输入
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        idx = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # 自动计算embedding_dim的维度
        embedding_dims = int(np.sqrt(len(vocabulary)))
        # 上述特征喂给Embedding
        embedding_encoder = tf.keras.layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            trainable=True,
            name=f"{feature_name}_embedding",
        )
        # 将每个特征的Embedding加入到列表中记录下来
        encoded_other_features.append(embedding_encoder(idx))

    ## 所有用户画像的Embedding拼接成一个向量
    if len(encoded_other_features) > 1:
        encoded_other_features = tf.keras.layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## 创建电影的Embedding
    movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
    movie_embedding_dims = int(np.sqrt(len(movie_vocabulary)))
    # 标签编码：在tf中将输入 movieid_823, movieid_821, ... 转为数字
    movie_index_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=movie_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="movie_index_lookup",
    )
    # 降维：使用tf.keras.layers.Embedding把输入特征降维为固定维度
    movie_embedding_encoder = tf.keras.layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        trainable=True,
        name=f"movie_embedding",
    )
    # 所有电影画像的Embedding拼接为一个向量
    genre_vectors = movies[genres].to_numpy()
    movie_genres_lookup = tf.keras.layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(genre_vectors),
        trainable=False,
        name="genres_vector",
    )

    # 构建特征工程 性别和电影类型交叉特征
    # gender_genres = genres
    # gender_genres.append('sex_F')
    # gender_genres.append('sex_M')
    
    # gender_genre_vectors = matrix[gender_genres].to_numpy()
    # gender_genres_lookup = tf.keras.layers.Embedding(input_dim=gender_genre_vectors.shape[0], 

    # 为电影画像(genres)创建一个简单的神经网络学习权重，因为genres embedding的权重不更新(trainabel=False)，所以单独设置神经网络学习权重
    movie_embedding_processor = tf.keras.layers.Dense(
        units=movie_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )

    ## 定义函数对给定的movie_id进行编码
    def encode_movie(movie_id):
        # 标签编码：在tf中将字符串转为整数
        movie_idx = movie_index_lookup(movie_id)
        # 用Embedding向量表示一个movie_id
        movie_embedding = movie_embedding_encoder(movie_idx)
        encoded_movie = movie_embedding
        if include_movie_features:
            movie_genres_vector = movie_genres_lookup(movie_idx)
            # 电影画像 = movie_id embedding + genres embedding
            encoded_movie = movie_embedding_processor(
                tf.keras.layers.concatenate([movie_embedding, movie_genres_vector])
            )
        return encoded_movie

    ## 编码 target_movie_id
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    ## 编码用户观看电影序列 sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # 创建序列的位置编码 positional embedding
    position_embedding_encoder = tf.keras.layers.Embedding(
        input_dim=sequence_length,
        output_dim=movie_embedding_dims,
        trainable=True,
        name="position_embedding",
    )

    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # 找到电影对应的评分序列（sequence ratings）后，跟电影embedding融合者融合
    sequence_ratings = tf.expand_dims(inputs["sequence_ratings"], -1)
    # 给 movie encodings 增加 Positional Embedding，
    # 之后再乘以用户对这些电影的打分
    encoded_sequence_movies_with_poistion_and_rating = tf.keras.layers.Multiply()(
        [(encoded_sequence_movies + encodded_positions), sequence_ratings]
    )

    # 输入Transformer的embedding进行拼接
    for encoded_movie in tf.unstack(encoded_sequence_movies_with_poistion_and_rating, axis=1):
        encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
    encoded_transformer_features.append(encoded_target_movie)

    encoded_transformer_features = tf.keras.layers.concatenate(encoded_transformer_features, axis=1)

    return encoded_transformer_features, encoded_other_features
    
# 未来可升级的地方
# 修改为类别特征和连续特征 类别特征喂给FM，dense特征喂给DNN 

if __name__ == '__main__':
    # 测试
    train_test= get_dataset_from_csv('train_data.csv', shuffle=True, batch_size=265)
    print(next(iter(train_test)))