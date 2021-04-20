import tensorflow as tf
from model import FFM
from utils import create_criteo_dataset
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


file = './criteo_sampled_data.csv'
read_part = True
sample_num = 100000
test_size = 0.2

k = 32

learning_rate = 0.001
batch_size = 500
epochs=100

feature_columns, train, test, val = create_criteo_dataset(file=file,
                                                     read_part=read_part,
                                                     sample_num=sample_num,
                                                     test_size=test_size)
train_X, train_y = train
test_X, test_y = test
val_X, val_y = val

model = FFM(feature_columns=feature_columns, k=k)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=[tf.keras.metrics.AUC()])

model.fit(train_X, train_y,
          epochs=epochs,
          callbacks=[early_stopping],
          batch_size=batch_size,
          class_weight={0:1, 1:3}, # 样本均衡
          validation_split=0.1,
          validation_data=(val_X, val_y))

print('test AUC: %f' % model.evaluate(test_X, test_y)[1])