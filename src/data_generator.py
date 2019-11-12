from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator:
    """
    Generates training and data set
    """

    def __init__(self, batch_size, height, width):
        self.train_image_generator = ImageDataGenerator(rescale=1. / 255)
        self.validation_image_generator = ImageDataGenerator(rescale=1. / 255)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def generate_train_data(self, train_dir):
        return self.train_image_generator\
                   .flow_from_directory(batch_size=self.batch_size,
                                        directory=train_dir,
                                        shuffle=True,
                                        target_size=(self.height, self.width),
                                        class_mode='binary')

    def generate_validation_data(self, validation_dir):
        return self.validation_image_generator\
                   .flow_from_directory(batch_size=self.batch_size,
                                        directory=validation_dir,
                                        target_size=(self.height, self.width),
                                        class_mode='binary')

