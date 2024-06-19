import os

import joblib
from keras import Sequential
from keras.src.callbacks import TensorBoard
from keras.src.utils import to_categorical
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from tensorflow.keras.layers import Conv1D, Dense, Flatten
from typing import Optional
from keras.src.layers import Dense, Flatten, Conv1D, MaxPooling1D

from src.data.data_preprocesing_repository import predict_new_data, custom_round, most_frequent_number
from src.data.metrics_repository import log_confusion_matrix_scaler, log_confusion_matrix
from src.domain.emg_payload import EMGPayload

saved_model = None
scaler = None


def _dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("label")
    return tf.data.Dataset.from_tensor_slices((dataframe, labels)).shuffle(buffer_size=len(dataframe))


def train_model(data_frame: DataFrame, epochs: Optional[int] = 50):
    print(data_frame.head())
    print(data_frame.shape)
    train_cnn_scaler(data_frame, epochs)


def train_model_cnn(data_frame: DataFrame, epochs: Optional[int] = 10):
    label_encoder = LabelEncoder()
    data_frame['label'] = label_encoder.fit_transform(data_frame['label'])
    val_dataframe = data_frame.sample(frac=0.2, random_state=1337)
    train_dataframe = data_frame.drop(val_dataframe.index)
    creation_time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = f'emg_cnn{creation_time_stamp}'
    tensorboard_callback = TensorBoard(log_dir=f'logs/scalars/{model_name}', histogram_freq=1)
    y = data_frame['label']
    print(
        "Using %d samples for training and %d for validation"
        % (len(train_dataframe), len(val_dataframe))
    )
    train_ds = _dataframe_to_dataset(train_dataframe)
    val_ds = _dataframe_to_dataset(val_dataframe)
    for x, y in train_ds.take(1):
        print("Input:", x)
        print("Target:", y)

    train_ds = train_ds.batch(8)
    val_ds = val_ds.batch(8)
    num_classes = len(y.unique())
    model = keras.Sequential()
    model.add(Conv1D(filters=32, kernel_size=4, activation='sigmoid', input_shape=(4, 1),
                     padding='same'))  # Assuming input data is (4, 1)
    # model.add(Conv1D(filters=16, kernel_size=4, activation='sigmoid', padding='same'))
    # model.add(MaxPooling1D(pool_size=2))  # Optional pooling layer
    model.add(Flatten())

    model.add(Dense(8, activation='sigmoid'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation=tf.nn.softmax))
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # Define the file writer for the confusion matrix.
    file_writer_cm = tf.summary.create_file_writer(f'logs/cm/{model_name}')

    # Define a custom callback to log the confusion matrix.
    cm_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_confusion_matrix(epoch, logs, model, val_ds, label_encoder.classes_,
                                                              file_writer_cm))

    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, cm_callback])

    save_neural_model(model, data_frame, creation_time_stamp)


def train_cnn_scaler(data: DataFrame, epochs: Optional[int] = 50):
    # Load the dataset
    label_encoder = LabelEncoder()

    data['label'] = label_encoder.fit_transform(data['label'])
    creation_time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = f'emg_cnn{creation_time_stamp}'
    tensorboard_callback = TensorBoard(log_dir=f'logs/scalars/{model_name}', histogram_freq=1)
    print(data.head())
    # Separate features and labels
    X = data.drop('label', axis=1)
    y = data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape data for 1D CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convert labels to categorical (one-hot encoding) if it's a multi-class classification
    num_classes = len(y.unique())
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Build the 1D CNN model
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the file writer for the confusion matrix.
    file_writer_cm = tf.summary.create_file_writer(f'logs/cm/{model_name}')

    # Define a custom callback to log the confusion matrix.
    cm_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_confusion_matrix_scaler(epoch, logs, model, X_test, y_test,
                                                                     label_encoder.classes_,
                                                                     file_writer_cm))
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2,
              callbacks=[tensorboard_callback, cm_callback])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    save_neural_model(model, data, creation_time_stamp, scaler)


def save_neural_model(model, data_frame, creation_time_stamp, scaler_to_save=None):
    global saved_model, scaler
    import os
    os.makedirs(f'models/{creation_time_stamp}')
    data_frame.to_csv(f'models/{creation_time_stamp}/dataset.csv', index=False)
    model.save(f'models/{creation_time_stamp}/model.keras')
    keras.utils.plot_model(model, to_file=f'models/{creation_time_stamp}/model_diagram.png', show_shapes=True)
    if scaler_to_save is not None:
        joblib.dump(scaler_to_save, f'models/{creation_time_stamp}/scaler.save')
        scaler = scaler_to_save

    saved_model = model


def load_trained_model(path):
    global saved_model, scaler
    saved_model = tf.keras.models.load_model(path)
    scaler_path = path.replace('model.keras', 'scaler.save')
    if os.path.exists(scaler_path):
        # Load the scaler
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    else:
        # Handle the case where the scaler file does not exist
        scaler = None
        print(f"Scaler file not found at {scaler_path}. Please check the path.")


def model_loaded() -> bool:
    global saved_model
    return saved_model is not None


def predict(data: EMGPayload, raw=None):
    global saved_model, scaler
    if raw is None:
        raw = []
    if scaler is None:
        predictions = saved_model.predict(tf.convert_to_tensor([[data.ch1, data.ch2, data.ch3, data.ch4]]))
        return np.argmax(predictions, axis=1)
    else:
        predictions = predict_new_data(saved_model, raw, scaler)
        # Convert predictions to class labels
        print("Predicted classes:", predictions)
        print("Freq prediction:", most_frequent_number(predictions))
        return [most_frequent_number(predictions)]
