#Construct the model here
import tensorflow as tf

def construct_model():
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters = 10,
                            kernel_size=3,
                            strides = 1,
                            padding = 'valid',
                            activation = 'relu',
                            input_shape = (224, 224,3)),
        tf.keras.layers.MaxPool2D(pool_size = 2),
        tf.keras.layers.Conv2D(10, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size = 2),
        tf.keras.layers.Conv2D(10, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size = 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
                metrics=["accuracy"])

    #model.summary()
    return model
