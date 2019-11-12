from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataAugmentation:
    """
    Applies techniques of data augmentation to data sets
    """

    def __init__(self, batch_size, height, width,
                 rescale=1./255, rotation_range=45, width_shift_range=.15, height_shift_range=.15,
                 horizontal_flip=True, zoom_range=0.5):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.image_gen_train = ImageDataGenerator(
                    rescale=rescale,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    horizontal_flip=horizontal_flip,
                    zoom_range=zoom_range
                    )

        self.image_gen_val = ImageDataGenerator(
                    rescale=rescale
                    )

    def generate_train_data(self, train_dir):
        return self.image_gen_train.flow_from_directory(batch_size=self.batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(self.height, self.width),
                                                        class_mode='binary')

    def generate_validation_data(self, validation_dir):
        return self.image_gen_val.flow_from_directory(batch_size=self.batch_size,
                                                      directory=validation_dir,
                                                      target_size=(self.height, self.width),
                                                      class_mode='binary')


