import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_pneumonia_cnn(input_shape=(28, 28, 1)):
    """
    Creates a lightweight CNN for PneumoniaMNIST binary classification.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Recall(name='recall'), keras.metrics.Precision(name='precision')]
    )

    return model


if __name__ == "__main__":
    model = create_pneumonia_cnn()
    model.summary()