from __future__ import print_function
from pathlib import Path
from time import strftime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.utils import to_categorical
from sklearn.utils import compute_class_weight
from keras.layers import (Dense, Normalization, Concatenate, Input, Flatten,)
from keras import Model
import tensorflow as tf
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from config.arg_parser import parameter_parser
# from config.arg_parser import parameter_parser
from config.model_metrics import LossHistory
from keras.callbacks import TensorBoard

args = parameter_parser()

warnings.filterwarnings("ignore")

class WIDEVULDET:
    def __init__(self, data, name="", batch_size=args.batch_size, epochs=args.epochs):
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values

        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]

        idxs = np.concatenate([positive_idxs, negative_idxs])

        #Deep
        x_train, x_test, y_train, y_test = train_test_split(self.vectors[idxs], self.labels[idxs],
                                                            test_size=0.2, stratify=self.labels[idxs], random_state=42)

        #Wide
        # x_train_wide, x_test_wide, y_train_wide, y_test_wide = train_test_split(self.vectors[idxs], self.labels[idxs],
        #                                                     test_size=0.2, stratify=self.labels[idxs], random_state=42)


        # x_train_wide, x_test_wide = x_train, x_test

        self.x_train_wide, self.x_train_deep = x_train[:, :50], x_train[:, 50:]

        self.x_test_wide, self.x_test_deep = x_test[:, :50], x_test[:, 50:]

        # x_valid_wide, x_valid_deep = x_train[:, :40], x_train[:, 20:]

        # num_columns = self.x_train_deep.shape[1]
        # num_rows = self.x_train_deep.shape[2]
        #
        # train_shape_deep = self.x_train_deep.shape
        # train_shape_wide = self.x_train_wide.shape
        #
        # test_shape_deep = self.x_test_deep.shape
        # test_shape_wide = self.x_test_wide.shape
        # # num_rows = self.x_train_deep.shape[2]
        #
        # print("x_train_deep shape: ", train_shape_deep)
        # print("x_train_wide shape: ", train_shape_wide)
        # print()
        # print("x_test_deep shape: ", test_shape_deep)
        # print("x_test_wide shape: ", test_shape_wide)

        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        # self.y_train = y_train
        # self.y_test = y_test

        # self.x_train_wide = x_train_wide
        # self.x_test_wide = x_test_wide
        # self.y_train_wide = y_train_wide
        # self.y_test_wide = y_test_wide

        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs

        self.build_model()

        # self.tuner = kt.Hyperband(self.build_model_hp,
        #                      objective='val_accuracy',
        #                      max_epochs=50,
        #                      factor=3,
        #                      directory='hyper_param',
        #                      project_name='Progressive')

        # return tuner

    def build_model(self):
        # print(self.vectors.shape)

        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels)
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}

        input_deep = Input(shape=(self.x_train_deep.shape[1], self.x_train_deep.shape[2]))
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))

        # input_deep = Input(shape=[80])
        # input_wide = Input(shape=[40])

        norm_layer_deep = Normalization()
        norm_layer_wide = Normalization()

        norm_deep = norm_layer_deep(input_deep)
        norm_wide = norm_layer_wide(input_wide)

        hidden1 = Dense(320, activation='relu')(norm_deep)
        hidden2 = Dense(128, activation='relu')(hidden1)

        concat = Concatenate(axis=-1)([hidden2, norm_wide])
        flatten = Flatten()(concat)
        h_final = Dense(32, activation='relu')(flatten)

        output = Dense(2, activation='softmax')(h_final)
        # output = Dense(1)(concat)

        self.model = Model(inputs=[input_deep, input_wide], outputs=[output])

        optimizer = Adam(learning_rate=0.002166277197701016) #1.8840040300e-4) *0.01 /0.00001
        # optimizer = SGD(learning_rate=0.000000000001) #1.8840040300e-4 #*0.01
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # self.model.summary()

    def train(self):
        # Check available GPUs
        gpus = tf.config.list_physical_devices('GPU')

        history = LossHistory()
        tensorboard_cb = TensorBoard(self.get_run_logdir(), profile_batch=(100, 200))
        # with tf.device('GPU:0'):
        # his = self.model.fit(
        #     ([self.x_train, self.x_train_wide ]), self.y_train,
        #     epochs=self.epochs, class_weight=self.class_weight,
        #     validation_data=([self.x_test, self.x_test_wide], self.y_test),
        #     verbose=1, batch_size=self.batch_size,
        #     callbacks=([tensorboard_cb],[history])
        # )

        his = self.model.fit(
            ([self.x_train_deep, self.x_train_wide]), self.y_train,
            epochs=self.epochs, class_weight=self.class_weight,
            validation_data=([self.x_test_deep, self.x_test_wide], self.y_test),
            verbose=1, batch_size=self.batch_size,
            callbacks=([tensorboard_cb], [history])
        )


        self.write_summary_to_log_fil()
        history.loss_plot('epoch')

        pd.DataFrame(his.history).plot(
            figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
            style=["r--", "r--.", "b-", "b-*"]
        )
        plt.show()

        return history

    def test(self):
        predictions = (self.model.predict([self.x_test_deep, self.x_test_wide])).round()

        return predictions, self.y_test

    def write_summary_to_log_fil(self):
        test_logdir = self.get_run_logdir()
        writer = tf.summary.create_file_writer(str(test_logdir))
        with writer.as_default():
            for step in range(1, 1000 + 1):
                tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)
        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)
        texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

    def get_run_logdir(self, log_dir="./logs"):
        return Path(log_dir) / strftime("run_%Y_%m_%d_%H_%M_%S")

