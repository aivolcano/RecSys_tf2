
import tensorflow as tf 
import pandas as pd
import numpy as np
from utils import get_dataset_from_csv
from model import create_model


model = create_model()
# Compile the model.
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

# Read the training data.
train_dataset = get_dataset_from_csv("train_data.csv", shuffle=True, batch_size=512)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=3, restore_best_weights=True)

# Fit the model with the training data.
model.fit(train_dataset, epochs=5,
        #   callbacks=[early_stopping],  # checkpoint

          )

# Read the test data.
test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265) # 512

# Evaluate the model on the test data.
_, rmse = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(rmse, 3)}")
# ===========================Test==============================
# print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])