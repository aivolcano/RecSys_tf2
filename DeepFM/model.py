import tensorflow as tf

class FM(tf.keras.layers.Layer):
    """ Wide """
    def __init__(self, k=32, w_reg=1e-4, v_reg=1e-4):  
        """
		Factorization Machine
		:param k: A scalar. The dimension of the latent vector. Embedding_dim
		:param w_reg: A scalar. The regularization coefficient of parameter w.
		:param v_reg: A scalar. The regularization coefficient of parameter v.

        Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

        References
            - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
		"""
        super(FM, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        
    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                  initializer='random_uniform',
                                  regularizer=tf.keras.regularizers.l2(self.w_reg),
                                  trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
                                  initializer='random_uniform',
                                  regularizer=tf.keras.regularizers.l2(self.v_reg),
                                  trainable=True)
    
    def call(self, inputs):   
        # first order 一阶
        first_order = self.w0 + tf.matmul(inputs, self.w)
        # second order
        second_order = tf.reduce_sum(tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
                                     tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
        return first_order + second_order
    
class DNN(tf.keras.layers.Layer):   
    """Deep Part"""
    def __init__(self, hidden_units, activation='relu',dnn_dropout=0.1):
        """[summary]

        Args:
            hidden_units ([list]): [List of hidden layer units's numbers]
            activation (str, optional): [Activation function]. Defaults to 'relu'.
            dnn_dropout ([float], optional): [dropout number]. Defaults to 0..
        """
        super(DNN, self).__init__()
        # the number of hidden layers 
        self.dnn_network = [tf.keras.layers.Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = tf.keras.layers.Dropout(dnn_dropout)

    def call(self, inputs):
        x = inputs
        # export each layer to a list, then weighted by attention
        for dnn in self.dnn_network:
            x = dnn(x)
            # x = self.batchnorm(x)
            x = self.dropout(x) # we can add dropout in each hidden layers  BatchNormal()
        return x

class ResidualWrapper(tf.keras.Model): 
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model
        return inputs + delta
    
# DeepFM
class DeepFM(tf.keras.Model):     
    def __init__(self, feature_columns, k=10, hidden_units=(200, 200, 200), dnn_dropout=0.1,residual=False,
                 activation='relu', fm_w_reg=1e-4, fm_v_reg=1e-4, embed_reg=1e-4):   
        """DeepFM

        Args:
            feature_columns ([list]): a list containing dense and sparse column feature information.
            k (int, optional): [fm's latent vector number]. Defaults to 10.
            hidden_units (tuple, optional): [A list of dnn hidden units]. Defaults to (200, 200, 200).
            dnn_dropout ([float], optional): [Dropout rate of dnn]. Defaults to 0..
            activation (str, optional): [Activation function of dnn]. Defaults to 'relu'.
            fm_w_reg ([type], optional): [The regularizer of w in fm]. Defaults to 1e-4.
            fm_v_reg ([type], optional): [The regularizer of v in fm]. Defaults to 1e-4.
            embed_reg ([type], optional): [The regularizer of embedding]. Defaults to 1e-4.
        """
        super(DeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # feature_columns => embedding
        self.embed_layers = {
            'embed_' + str(i): tf.keras.layers.Embedding(input_dim=feat['feat_num'],
                                                        input_length=1,
                                                        output_dim=feat['embed_dim'],
                                                        embeddings_initializer='random_uniform', # pre-trained model hidden state
                                                        embeddings_regularizer=tf.keras.regularizers.l2(embed_reg),
                                                        trainable=True)
            for i,feat in enumerate(self.sparse_feature_columns)
        }
        
        self.fm = FM(k, fm_w_reg, fm_v_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout) 
        self.dense = tf.keras.layers.Dense(1, activation=None)
        #deep + residual block
        self.residual = residual
        self.residual_deep = ResidualWrapper(tf.keras.Sequential([
                                                             DNN(hidden_units, activation, dnn_dropout),
                                                             tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.zeros)
        ]))
        self.dense2 = tf.keras.layers.Dense(1)

        # self.w0 = tf.add_weight(name='fm_deep_weight',shape=(1,), initializer=tf.initializers.random_normal(),
        #                         trainable=True)
    
    def call(self, inputs): 
        dense_inputs, sparse_inputs = inputs
        # dense_inputs.shape=(None, 13) sparse_inputs.shape=(NOne, 26)
        # concat & flatten
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                 for i in range(sparse_inputs.shape[1])], axis=-1) # (none, 1664) 1664=26*64
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1) # (none, 1677+13)
        
        # wide 
        wide_outputs = self.fm(stack) # (None, 1)
        
        if self.residual:
            '''deep + residual block'''
            residual_deep_outputs = self.residual_deep(stack)
            deep_outputs = self.dense2(residual_deep_outputs) # (none, 1)
        else:
            '''Ori deep part'''
            deep_outputs = self.dnn(stack)
            deep_outputs = self.dense(deep_outputs)

        # outputs = tf.nn.sigmoid((1 - self.w0) * wide_outputs + self.w0 * deep_outputs)
        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def summary(self):   
        dense_inputs = tf.keras.layers.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.layers.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()