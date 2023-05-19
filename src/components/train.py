import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from src.utils.common import read_yaml
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


class SentimentTraining():

    def __init__(self, parameters) -> None:

        self.dataset_name = parameters['data']['dataset_name']
        self.buffer_size = parameters['train']['buffer_size']
        self.batch_size = parameters['train']['batch_size']
        self.vocab_size = parameters['train']['vocab_size']
        self.output_dim = parameters['train']['output_dim']
        self.epochs = parameters['train']['epochs']
        self.artifact_dir = parameters['train']['artifact_dir']
        self.model_dir = parameters['train']['model_dir']
        self.checkpoint_dir = parameters['train']['checkpoint_dir']
        self.tensorboard_log_dir = parameters['train']['tensorboard_log_dir']
    
    def data_loading(self):

        dataset , info = tfds.load(self.dataset_name, with_info=True, as_supervised=True)
        self.train_ds, self.test_ds = dataset["train"], dataset["test"]
        
        # Shuffling and batching the training and test data
        self.train_ds = self.train_ds.shuffle(self.buffer_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def model_build(self):

        encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=self.vocab_size)
        encoder.adapt(self.train_ds.map(lambda text, label: text))
        embedding_layer = tf.keras.layers.Embedding(input_dim = len(encoder.get_vocabulary()), output_dim = self.output_dim, mask_zero = True)

        Layers = [
          encoder, # text vectorization
          embedding_layer, # embedding
          tf.keras.layers.Bidirectional(
              tf.keras.layers.LSTM(64)
          ),
          tf.keras.layers.Dense(64, activation="relu"),
          tf.keras.layers.Dense(1)
        ]

        self.model = tf.keras.Sequential(Layers)

    def callbacks(self, base_dir="artifact"):

        # tensorboard callbacks - 
        unique_log = time.asctime().replace(" ", "_").replace(":", "")
        tensorboard_log_dir = os.path.join(self.artifact_dir, self.tensorboard_log_dir, unique_log)
        os.makedirs(tensorboard_log_dir, exist_ok=True)

        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

        # ckpt callback
        ckpt_file = os.path.join(self.artifact_dir, self.checkpoint_dir, "model")
        os.makedirs(os.path.join(self.artifact_dir, self.checkpoint_dir), exist_ok=True)

        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_file,
            save_best_only=True
        )

        callback_list = [
                        tb_cb,
                        ckpt_cb
        ]

        return callback_list

    def training(self):
        
        self.data_loading()
        self.model_build()
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # 
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=["accuracy"]
        )
        
        callback_list = self.callbacks()
        
        history = self.model.fit(self.train_ds, epochs=self.epochs, validation_data=self.test_ds,validation_steps=30,callbacks=callback_list)

    def evaluation(self):

        test_loss, test_acc = self.model.evaluate(self.test_ds)

        print(f"test loss: {test_loss}")
        print(f"test accuracy: {test_acc}")

    def get_plot(self, history, metric):
        
        history_obj = history.history
        plt.plot(history_obj[metric])
        plt.plot(history_obj[f'val_{metric}'])
        plt.xlabel("Epochs -->")
        plt.ylabel(f"{metric} -->")
        plt.legend([metric, f'val_{metric}'])





if __name__ == "__main__":
    params = read_yaml("config/config.yaml")
    obj = SentimentTraining(params)
    obj.training()