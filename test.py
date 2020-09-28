from PIL import Image
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt


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
        im = im.reshape(3, 80, 240)
    else:
        im = new_image.reshape(80, 240, 3)
        return im


# img_data = letterbox_image("img/test0.jpg",(240,80))
#
# print(img_data.shape)
#
# import cv2
# cv2.imshow("image",img_data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
new_size = (240, 80)
img = Image.open("img/test0.jpg")
image_size = img.resize(new_size, Image.ANTIALIAS)
image_size.show()

image_data = np.array(image_size, dtype='float32')
image_data /= 255.
plt.figure(figsize=(10, 10))
plt.imshow(image_data)
plt.show()
