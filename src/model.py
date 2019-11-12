from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


class Model:
    """
    Model that will cause overfitting
    """

    def __init___(self):
        self._init_model()

    def _init_model(self, height, width):
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(height, width, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def summary(self):
        self.model.summary()

    def get_model_fit(self, train_data_gen, total_train, epochs, validation_data, validation_steps):
        return self.model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps
        )


class ModelDropout:
    """
    Model with dropout technique
    """
    def __init___(self):
        self._init_model()

    def _init_model(self, height, width):
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                   input_shape=(height, width, 3)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def summary(self):
        self.model.summary()

    def get_model_fit(self, train_data_gen, total_train, epochs, validation_data, validation_steps):
        return self.model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps
        )

