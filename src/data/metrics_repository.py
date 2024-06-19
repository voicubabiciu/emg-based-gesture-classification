import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def _plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(epoch, logs, model, val_ds, class_names, file_writer_cm):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(val_ds)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Extract true labels from the dataset.
    true_labels = []
    for _, y in val_ds.unbatch():
        true_labels.append(y.numpy())
    true_labels = np.array(true_labels)

    # Calculate the confusion matrix.
    cm = confusion_matrix(true_labels, test_pred)

    # Log the confusion matrix as an image summary.
    figure = _plot_confusion_matrix(cm, class_names)
    cm_image = _plot_to_image(figure)
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Function to log confusion matrix
def log_confusion_matrix_scaler(epoch, logs, model, X_test, y_test, class_names, file_writer):
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)
    test_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(test_true, test_pred)
    figure = _plot_confusion_matrix_scaler(cm, class_names)
    cm_image = _plot_to_image(figure)

    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# Function to plot confusion matrix
def _plot_confusion_matrix_scaler(cm, class_names):
    figure = plt.figure(figsize=(10, 10))  # Increased figure size
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names, va='center')

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout(pad=4.0)  # Increased padding to layout
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure




# Function to convert a matplotlib figure to a TensorFlow image
def _plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)  # Added pad_inches
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.close(figure)
    return image

