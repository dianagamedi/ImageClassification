from __future__ import absolute_import, division, print_function, unicode_literals

import os


def get_length_set(set):
    return len(os.listdir(set))


class DataLoader:
    """
    Data loader to upload data set and manipulate data

    :param path_to_zip: path where the zip file containing the data set is located

    """

    def __init__(self, path_to_zip):
        self.path_to_zip = path_to_zip
        self.path = os.path.join(os.path.dirname(self.path_to_zip), 'cats_and_dogs_filtered')
        self.train_dir = os.path.join(self.path, 'train')
        self.validation_dir = os.path.join(self.path, 'validation')

    def get_path(self):
        return self.path

    def get_train_path_category(self, category):
        return os.path.join(self.train_dir, category)

    def get_validation_path_category(self, category):
        return os.path.join(self.validation_dir, category)

