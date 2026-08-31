(
echo import tensorflow as tf
echo from tensorflow import keras
echo from tensorflow.keras import layers
echo.
echo def create_pneumonia_cnn^(input_shape=^(28, 28, 1^), num_classes=1^):
echo     model = keras.Sequential^([
echo         keras.Input^(shape=input_shape^),
echo         layers.Conv2D^(16, ^(3, 3^), activation=^"relu^", padding=^"valid^", name=^"conv1^"^),
echo         layers.MaxPooling2D^(^(2, 2^), name=^"pool1^"^),
echo         layers.Conv2D^(32, ^(3, 3^), activation=^"relu^", padding=^"valid^", name=^"conv2^"^),
echo         layers.MaxPooling2D^(^(2, 2^), name=^"pool2^"^),
echo         layers.Flatten^(name=^"flatten^"^),
echo         layers.Dense^(num_classes, activation=^"sigmoid^", name=^"output^"^)
echo     ]^)
echo.
echo     model.compile^(
echo         optimizer=keras.optimizers.Adam^(learning_rate=0.001^),
echo         loss=^"binary_crossentropy^",
echo         metrics=[^"accuracy^", keras.metrics.Recall^(name=^"recall^"^), keras.metrics.Precision^(name=^"precision^"^)]
echo     ^)
echo.
echo     return model
echo.
echo if __name__ == ^"__main__^":
echo     model = create_pneumonia_cnn^(^)
echo     model.summary^(^)
) > model.py