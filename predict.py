from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from LPR_Model import LprModel

from yolo import YOLO, detect_video
from PIL import Image
import glob
import os

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
         "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C",
         "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


input_tensor = Input([80, 240, 3])
model = LprModel(input_tensor)

# 添加全局平均池化层
x = model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有65个类
dense_output1 = Dense(65, activation="softmax")(x)
dense_output2 = Dense(65, activation="softmax")(x)
dense_output3 = Dense(65, activation="softmax")(x)
dense_output4 = Dense(65, activation="softmax")(x)
dense_output5 = Dense(65, activation="softmax")(x)
dense_output6 = Dense(65, activation="softmax")(x)
dense_output7 = Dense(65, activation="softmax")(x)

# 构建我们需要训练的完整模型
model_final = Model(inputs=model.input, \
                    outputs=[dense_output1, dense_output2, dense_output3, dense_output4, dense_output5, dense_output6,
                             dense_output7])

model_final.load_weights("logs/model_last.h5")
from utils.preprocess1_img_v2 import resize_img

# 实例化解释器 yolo
yolo = YOLO()

while True:
    file_name = input("Input the FileName：")
    # 第一步使用YOLO目标检测模型定位车牌
    
    try:
        image = Image.open(file_name)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        # r_image.show()
    # yolo.close_session()

    # 导入目标定位之后分割的车牌图片
    file_imag = glob.glob("img\*.jpg")
    imgfilename = [(os.path.splitext(os.path.basename(file))[0], file) for file in file_imag]
    for ig in imgfilename:
        if "license" in ig[0]:
            img = resize_img(ig[1])
            image_data = np.expand_dims(img, 0)
            prediction = model_final.predict(image_data)
            categories = np.array(chars)
            onehot_enc = OneHotEncoder(sparse=False)
            onehot_enc.fit(categories.reshape(-1, 1))

            y1_hat = onehot_enc.inverse_transform(prediction[0])[0][0]
            y2_hat = onehot_enc.inverse_transform(prediction[1])[0][0]
            y3_hat = onehot_enc.inverse_transform(prediction[2])[0][0]
            y4_hat = onehot_enc.inverse_transform(prediction[3])[0][0]
            y5_hat = onehot_enc.inverse_transform(prediction[4])[0][0]
            y6_hat = onehot_enc.inverse_transform(prediction[5])[0][0]
            y7_hat = onehot_enc.inverse_transform(prediction[6])[0][0]

            print(y1_hat, y2_hat, y3_hat, y4_hat, y5_hat, y6_hat, y7_hat)
        else:
            continue
