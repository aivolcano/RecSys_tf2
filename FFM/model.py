import tensorflow as tf


class FFM_Layer(tf.keras.layers.Layer):
    def __init__(self, dense_feature_columns, sparse_feature_columns, k=32, w_reg=1e-4, v_reg=1e-4):
        """
        :param k: embedding dimension of the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=tf.keras.regularizers.l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v', shape=(self.feature_num, self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=tf.keras.regularizers.l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        stack = dense_inputs
        # 类别特征进行 One-hot encoding
        for i in range(sparse_inputs.shape[1]):
            stack = tf.concat([stack, tf.one_hot(sparse_inputs[:, i],
                                   depth=self.sparse_feature_columns[i]['feat_num'])], axis=-1)
        # first order
        first_order = self.w0 + tf.matmul(tf.concat(stack, axis=-1), self.w)
        # field second order
        second_order = 0
        field_f = tf.tensordot(stack, self.v, axes=[1, 0])
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(tf.multiply(field_f[:, i], field_f[:, j]), axis=1, keepdims=True)

        return first_order + second_order

class FFM(tf.keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        FFM architecture
        :param feature_columns:  a list containing dense and sparse column feature information
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(self.dense_feature_columns, self.sparse_feature_columns,
                             k, w_reg, v_reg)

    def call(self, inputs, *args, **kwargs):
        result_ffm = self.ffm(inputs)
        outputs = tf.nn.sigmoid(result_ffm) # 二分类问题
        # outputs = tf.nn.softmax(result_ffm) # 多分类任务
        # outputs = result_ffm # 回归任务

        return outputs

    def summary(self, *arg, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs])).summary()