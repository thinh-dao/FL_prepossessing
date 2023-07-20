import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

base = bases.mobilenetv2_base.MobileNetV2Base()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pretrain(model):
    batch_2 = unpickle("data/data_batch_2")
    batch_3 = unpickle("data/data_batch_3")
    batch_4 = unpickle("data/data_batch_4")
    batch_5 = unpickle("data/data_batch_5")
    key = b'labels' 
    train_labels = np.concatenate((batch_2[key], batch_3[key], batch_4[key], batch_5[key]))
    train_labels = tf.one_hot(train_labels, 10)
    
    key = b'data'
    train_batch_2 = np.dstack((batch_2[key][:, :1024], batch_2[key][:, 1024:2048], batch_2[key][:, 2048:])).reshape(-1, 32, 32, 3)
    train_batch_3 = np.dstack((batch_3[key][:, :1024], batch_3[key][:, 1024:2048], batch_3[key][:, 2048:])).reshape(-1, 32, 32, 3)
    train_batch_4 = np.dstack((batch_4[key][:, :1024], batch_4[key][:, 1024:2048], batch_4[key][:, 2048:])).reshape(-1, 32, 32, 3)
    train_batch_5 = np.dstack((batch_5[key][:, :1024], batch_5[key][:, 1024:2048], batch_5[key][:, 2048:])).reshape(-1, 32, 32, 3)
    train_data = np.vstack((train_batch_2, train_batch_3, train_batch_4, train_batch_5))
    BATCH_SIZE = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(BATCH_SIZE)
    model.fit(train_dataset, epochs=10)
    
    # Get the model weights
    weights = model.get_weights()

    # Save the weights in binary format
    bin_path = 'CIFAR10_B20.bin'
    with open(bin_path, 'wb') as f:
        for weight in weights:
            np.array(weight).astype(np.float32).tofile(f)
    
# head = tf.keras.Sequential(
#     [
#         tf.keras.Input(shape=(32, 32, 3)),
#         tf.keras.layers.Conv2D(6, 5, activation="relu"),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Conv2D(16, 5, activation="relu"),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(units=120, activation="relu"),
#         tf.keras.layers.Dense(units=84, activation="relu"),
#         tf.keras.layers.Dense(units=10, activation="softmax"),
#     ]
# )
# head.compile(loss="categorical_crossentropy", optimizer="sgd")
# print(len(base.layers))
# # Pretrain model
# pretrain(head)

# converter = TFLiteTransferConverter(
#     10, base, heads.KerasModelHead(head), optimizers.SGD(1e-3), train_batch_size=32
# )
# converter.convert_and_save("tflite_model")



base = tf.keras.Sequential(
    [tf.keras.Input(shape=(32, 32, 3)), tf.keras.layers.Lambda(lambda x: x)]
)
base.compile(loss="categorical_crossentropy", optimizer="sgd")
base.save("identity_model", save_format="tf")
head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(6, 5, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, 5, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation="relu"),
        tf.keras.layers.Dense(units=84, activation="relu"),
        tf.keras.layers.Dense(units=10, activation="softmax"),
    ]
)
head.compile(loss="categorical_crossentropy", optimizer="sgd")
base_path = bases.saved_model_base.SavedModelBase("identity_model")
pretrain(head)
converter = TFLiteTransferConverter(
    10, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-3), train_batch_size=32
)
converter.convert_and_save("tflite_model")



