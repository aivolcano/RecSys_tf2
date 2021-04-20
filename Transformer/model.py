import tensorflow as tf
from utils import create_model_inputs, encode_input_features

include_user_id = False
include_user_features = False
include_movie_features = False

hidden_units = [256, 128, 64] # 使用3层全连接层效果下降，应该是网络层数导致模型发生退化问题
dropout_rate =  0.3
num_heads = 3


def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_user_id, include_user_features, include_movie_features
    )

    # 输出数据进行归一化
    # transform_features = tf.keras.layers.LayerNormalization()(transformer_features)

    # 搭建 Multi-head Attention 层
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    # 残差连接
    x1 = tf.keras.layers.Add()([transformer_features, attention_output])
    # 可将layerNormalization 向前调整到输入层
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x2 = tf.keras.layers.LeakyReLU()(x1)
    x2 = tf.keras.layers.Dense(units=x2.shape[-1])(x2)
    x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
    # Add & Norm 残差连接 & 层归一化
    transformer_features = tf.keras.layers.Add()([x1, x2])
    transformer_features = tf.keras.layers.LayerNormalization()(transformer_features)
    features = tf.keras.layers.Flatten()(transformer_features) #(None, 248)
    # print(features.shape)

    # 包含其他特征
    # if other_features is not None:
    #     features = tf.keras.layers.concatenate(
    #         [features, tf.keras.layers.Reshape([other_features.shape[-1]])(other_features)]
    #     )

        # concated_embeds_value = tf.keras.layers.Reshape([other_features.shape[-1]])(other_features)
        # square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1), keepdims=True)
        # sum_of_square = tf.reduce_sum(tf.pow(concated_embeds_value, 2), axis=1, keepdims=True)  # concated_embeds_value * concated_embeds_value
        # other_features = 0.5 * (square_of_sum - sum_of_square)
        # features = tf.concat([features, other_features], axis=1)

    # Fully-connected layers.
    features_ori = features
    for num_units in hidden_units:
        features = tf.keras.layers.Dense(num_units)(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.LeakyReLU()(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    

    # outputs = dnn_n_resnet(features)
    # 残差连接
    # features_ori = tf.keras.layers.Dense(units=hidden_units[-1], activation=None, trainable=False)(features_ori)
    # features = tf.concat([features, features_ori], axis=-1)
    # features = features + features_ori
    outputs = tf.keras.layers.Dense(units=1)(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = create_model()
    print(model)