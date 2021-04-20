import tensorflow as tf

class ResidualWrapper(tf.keras.Model): 
    '''
    ResidualNet for Deep part：F(x) = f(x) + x = model(inputs) + inputs
    there is some difference between Deep&cross and ResidualWrapper
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta


class Linear(tf.keras.layers.Layer):   
  """ linear part"""
  def __init__(self):
    super(Linear, self).__init__()
    self.dense = tf.keras.layers.Dense(1, activation=None)

  def call(self, inputs, **kwargs):
    result = self.dense(inputs)
    return result

class DNN(tf.keras.layers.Layer):
    """Deep Neural Network part"""
    def __init__(self, hidden_units, activation='relu', dropout=0.2):
        """
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
        input shape (batch_size, units)
        output shape (batch_size, embedding_dim)
		"""
        super(DNN, self).__init__()
        self.dnn_network = [tf.keras.layers.Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = tf.keras.layers.Dropout(dropout)
        # self.bn = tf.keras.layers.LayerNormalization() # BatchNormalization()

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
            x = self.dropout(x)
            # x = self.bn(x)
        return x

class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0.1, embed_reg=1e-4, residual=True):
        """,
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(WideDeep, self).__init__()
        self.residual= residual
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): tf.keras.layers.Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=tf.keras.regularizers.l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = tf.keras.layers.Dense(1, activation=None)

        self.residual_deep = ResidualWrapper(tf.keras.Sequential([
                                    DNN(hidden_units, activation, dnn_dropout),
                                    tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.zeros)
        ]))
        self.dense2 = tf.keras.layers.Dense(1)
        # 门控机制决定 使用wide 或 deep 多少信息量
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.initializers.random_normal(),
                                  trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # dense_input.shape=(batch_size, 13), sparse_inputs.shape=(batch_size, 26)
        # 分类特征Embedding向量拼接：多个向量拼接
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1) # (None, 208), 208=26个特征 * 8维embed_dim
        # 连续特征拼接：一个向量拼接
        x = tf.concat([sparse_embed, dense_inputs], axis=-1) # (None, 221) # 221 = 26 * 8 + 13

        # Wide
        wide_out = self.linear(dense_inputs)  # (None, 1)
        # Deep
        # 将输入DNN的低阶特征 与 DNN提取的高阶特征融合
        if self.residual:
            residual_deep_outputs = self.residual_deep(x)
            deep_out = self.dense2(residual_deep_outputs)  # (None, 1)
        else:
            deep_out = self.dnn_network(x)
            deep_out = self.final_dense(deep_out) # (None, 1)
        # out: 门控机制决定释放多少信息量进入
        outputs = tf.nn.sigmoid((1 - self.w0) * wide_out + self.w0 * deep_out)  # 赋予不同的权重
        return outputs

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()
