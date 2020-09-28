from keras import backend as K
import numpy as np
import cv2
from keras.utils import Sequence, plot_model
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
import numpy as np
import cv2
import os


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def central_crop(image, central_fraction):
    """Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start + bbox_h_size, bbox_w_start:bbox_w_start + bbox_w_size]
    return image


def get_processed_image1(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))
    im = preprocess_input(im)
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2, 0, 1))
        im = im.reshape(-1, 3, 299, 299)
    else:
        im = im.reshape(-1, 299, 299, 3)
    return im


# 预处理图片函数
def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = central_crop(im, 1)
    im = cv2.resize(im, (299, 299))
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2, 0, 1))
        im = im.reshape(3, 299, 299)
    else:
        im = im.reshape(299, 299, 3)
    return im


# test 有些失真，可能导致失败
# get_processed_image('../imgs/01-0_1-249&528_393&586-392&584_249&586_250&530_393&528-0_0_25_27_7_26_29-131-21.jpg')


def letterbox_image_v2(image, size):
    image = Image.open(image)
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # x_offset, y_offset = (w - nw) // 2 / 300, (h - nh) // 2 / 300
    new_image = np.array(new_image)
    new_image = preprocess_input(new_image)
    if K.image_data_format() == "channels_first":
        im = np.transpose(new_image, (2, 0, 1))
        im = im.reshape(-1, 3, 224, 224)
    else:
        im = new_image.reshape(-1, 224, 224, 3)
        return im
    return im


def letterbox_image(image, size):
    image = Image.open(image)
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # x_offset, y_offset = (w - nw) // 2 / 300, (h - nh) // 2 / 300
    new_image = np.array(new_image, dtype=np.float64)
    if K.image_data_format() == "channels_first":
        im = np.transpose(new_image, (2, 0, 1))
        im = im.reshape(3, 224, 224)
    else:
        im = new_image.reshape(224, 224, 3)
        return im


def resize_img(filename):
    new_size = (240, 80)
    img = Image.open(filename)
    image_size = img.resize(new_size, Image.ANTIALIAS)
    image_data = np.array(image_size, dtype='float32')
    image_data /= 255.
    return image_data


class MyCustomGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y1 = self.labels[0][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y2 = self.labels[1][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y3 = self.labels[2][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y4 = self.labels[3][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y5 = self.labels[4][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y6 = self.labels[5][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y7 = self.labels[6][idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([np.array(resize_img(filename))
                         for filename in batch_x]), \
               [np.array(batch_y1), np.array(batch_y2), np.array(batch_y3), np.array(batch_y4), np.array(batch_y5), \
                np.array(batch_y6), np.array(batch_y7)
                ]
