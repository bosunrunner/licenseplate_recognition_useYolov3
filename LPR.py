import glob
import os
import numpy as np
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from sklearn.utils import shuffle

files = glob.glob("F:/githuCode/ProjectGit/end-to-end-for-chinese-plate-recognition/data/plate_train/*.jpg")
# 获得每个图片名和文件名
all_files = [(os.path.splitext(os.path.basename(file))[0], file) for file in files]

X = [filename[1] for filename in all_files]

img_name = [img_name[0].split('_') for img_name in all_files]

img_name = np.array(img_name)

plate_nums = img_name[:, 1]

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
         "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C",
         "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

y1 = np.array([plate_label.replace("_", "")[0] for plate_label in plate_nums])
y2 = np.array([plate_label.replace("_", "")[1] for plate_label in plate_nums])
y3 = np.array([plate_label.replace("_", "")[2] for plate_label in plate_nums])
y4 = np.array([plate_label.replace("_", "")[3] for plate_label in plate_nums])
y5 = np.array([plate_label.replace("_", "")[4] for plate_label in plate_nums])
y6 = np.array([plate_label.replace("_", "")[5] for plate_label in plate_nums])
y7 = np.array([plate_label.replace("_", "")[6] for plate_label in plate_nums])

X, y1, y2, y3, y4, y5, y6, y7 = shuffle(X, y1, y2, y3, y4, y5, y6, y7)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

onehot_enc = OneHotEncoder(sparse=False)
categories = np.array(chars)
onehot_enc.fit(categories.reshape(-1, 1))

y1 = onehot_enc.transform(np.array(y1).reshape(-1, 1))
y2 = onehot_enc.transform(np.array(y2).reshape(-1, 1))
y3 = onehot_enc.transform(np.array(y3).reshape(-1, 1))
y4 = onehot_enc.transform(np.array(y4).reshape(-1, 1))
y5 = onehot_enc.transform(np.array(y5).reshape(-1, 1))
y6 = onehot_enc.transform(np.array(y6).reshape(-1, 1))
y7 = onehot_enc.transform(np.array(y7).reshape(-1, 1))

# 训练集和验证集划分以及标签
X_train, X_test, \
y1_train, y1_test, \
y2_trian, y2_test, \
y3_train, y3_test, \
y4_train, y4_test, \
y5_train, y5_test, \
y6_train, y6_test, \
y7_train, y7_test = \
    train_test_split(X, y1, y2, y3, y4, y5, y6, y7, test_size=0.2, random_state=42)

from utils.preprocess1_img_v2 import MyCustomGenerator

batch_size = 4
training_batch_generator = MyCustomGenerator(X_train,
                                             (y1_train, y2_trian, y3_train, y4_train, y5_train, y6_train, y7_train),
                                             batch_size)
test_batch_generator = MyCustomGenerator(X_test, (y1_test, y2_test, y3_test, y4_test, y5_test, y6_test, y7_test),
                                         batch_size)

from LPR_Model import LprModel
from nets.darknet53 import darknet_body

input_tensor = Input([80, 240, 3])
model = LprModel(input_tensor)
# model.summary()

model_darknet53 = darknet_body(input_tensor)
model_path = "model_data/darknet53_weights.h5"
model_darknet53.load_weights(model_path)

for i, layer in enumerate(model.layers):
    layer.set_weights(model_darknet53.layers[i].get_weights())

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
# predictions = Dense(476, activation='sigmoid')(x)

# 构建我们需要训练的完整模型
model_final = Model(inputs=model.input, \
                    outputs=[dense_output1, dense_output2, dense_output3, dense_output4, dense_output5, dense_output6,
                             dense_output7])
model_final.summary()
# model_final.load_weights("logs/ep001-loss24.767-val_loss25.100.h5")

model_final.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 训练参数设置 比较通用的训练模型结构
logging = TensorBoard(log_dir="log2")
checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

history = model_final.fit_generator(training_batch_generator, validation_data=test_batch_generator, epochs=50,
                                    verbose=1,
                                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])

# 模型训练过程
import matplotlib.pyplot as plt


def plot_history(network_history):
    plt.figure()
    plt.xlabel('训练轮次')
    plt.ylabel('损失函数')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['训练集', '验证集'])
    plt.title('训练过程中损失函数变化')
    plt.grid()

    plt.figure()
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['训练集', '验证集'], loc='lower right')
    plt.title('训练过程中准确率变化')
    plt.grid()
    plt.show()

# plot_history(history)
