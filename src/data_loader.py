from __future__ import absolute_import, division, print_function, unicode_literals

import os


class DataLoader:
    """
    Data loader to upload data set

    :param path_to_zip: path where the zip file containing the data set is located

    """

    def __init__(self, path_to_zip):
        self.path_to_zip = path_to_zip

    def get_path(self):
        path = os.path.join(os.path.dirname(self.path_to_zip), 'cats_and_dogs_filtered')

        return path
