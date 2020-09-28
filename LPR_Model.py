from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose
from keras.models import Model
from keras import backend as K

# --------------------------------------------------#
#   单次卷积
# --------------------------------------------------#
@wraps (Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2 (5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get ('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update (kwargs)
    return Conv2D (*args, **darknet_conv_kwargs)

# ---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update (kwargs)
    return compose (
        DarknetConv2D (*args, **no_bias_kwargs),
        BatchNormalization (),
        LeakyReLU (alpha=0.1))

# ---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D (((1, 0), (1, 0))) (x)
    # 降维了
    x = DarknetConv2D_BN_Leaky (num_filters, (3, 3), strides=(2, 2)) (x)
    for i in range (num_blocks):
        y = DarknetConv2D_BN_Leaky (num_filters // 2, (1, 1)) (x)
        y = DarknetConv2D_BN_Leaky (num_filters, (3, 3)) (y)
        x = Add () ([x, y])
    return x

input = Input([80,240,3])

def LprModel(input):
    # 240x80x32 卷积 + 正则化+ leakRelu
    x = DarknetConv2D_BN_Leaky (32, (3, 3)) (input)
    #  120, 40, 64
    x = resblock_body (x, 64, 1)
    # 60, 20, 128
    x = resblock_body (x, 128, 2)
    # 30 , 10 ,256
    x = resblock_body (x, 256, 8)
    feat1 = x
    # # 26x26x512
    # x = resblock_body (x, 512, 8)
    # feat2 = x
    #
    # # 13x13x1024
    # x = resblock_body (x, 1024, 4)
    feat3 = x
    return Model (input, x)
    # return feat1,feat2,feat3

# transfer_learning test
model = LprModel(input)
model.summary()

from nets.darknet53 import darknet_body

model_darknet53 = darknet_body(input)
model_path = "model_data/darknet53_weights.h5"
model_darknet53.load_weights(model_path)

for i,layer in enumerate(model.layers):
    layer.set_weights(model_darknet53.layers[i].get_weights())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


from PIL import Image
import numpy as np

new_size = (240,80)
img = Image.open("img/test0.jpg")
image_size = img.resize(new_size,Image.ANTIALIAS)
# image_data = np.array(image_size, dtype='float32')
# image_data /= 255.
# image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
# result = model.predict(image_data)
# print(result)