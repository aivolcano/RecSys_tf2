import tensorflow as tf
from model import WideDeep
from utils import create_criteo_dataset
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# print(gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ------- Hyper Parameters-----------
# file = './criteo_sampled_data.csv'
file = 'E:/BI代码集/L6预测全家桶与机器学习四大神器/DeepModel/Wide&Deep/criteo_sampled_data.csv'
read_part = True
sample_num= 100000
test_size = 0.1

embed_dim = 64
dnn_dropout = 0.5
hidden_units = [256, 128, 64]

learning_rate = 0.01
batch_size = 1024
epochs=20

# -----------------  create dataset---------
feature_columns, train, test, val = create_criteo_dataset(file=file,
                                                          read_part=read_part,
                                                          sample_num=sample_num,
                                                          embed_dim=embed_dim,
                                                          test_size=test_size
                                                  )

train_X, train_y = train
test_X, test_y = test
val_X, val_y = val

# ---------------build model----------
model = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout, residual=True) #
model.summary()

# -------------model checkpoint ---------
check_path = './save/deepfm_weight.epoch_{epoch:4d}.val_loss_{val_loss:.4f}.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,verbose=1, period=5)

# ------------ model evaluate ------------
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=METRICS)

# ---------早停法 -----
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model.fit(train_X,train_y,
        epochs=epochs,
        # callbacks=[early_stopping, checkpoint],
        batch_size=batch_size,
        validation_split=0.1,
        validation_data=(val_X, val_y),
        # class_weight={0:1, 1:3}, # 样本均衡
      )

print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


# ------------- model evaluation in test dataset ----

train_predictions_weighted = model.predict(train_X, batch_size=batch_size)
test_predictions_weighted = model.predict(test_X, batch_size=batch_size)

# ------------- confusion matrix
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
  
weighted_results = model.evaluate(test_X, test_y,
                                           batch_size=batch_size, verbose=0)
for name, value in zip(model.metrics_names, weighted_results):
  print(name, ': ', value)
print()
plot_cm(test_y, test_predictions_weighted)


# ----------AUC-ROC 曲线 ---------
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([10,50])
    # plt.ylim([50,85])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plot_roc('Train weighted', test_y, test_predictions_weighted, color=colors[0], linestyle='-')
plot_roc('Test weighted', train_y, train_predictions_weighted, color=colors[1], linestyle='--')
plt.legend(loc='lower right')