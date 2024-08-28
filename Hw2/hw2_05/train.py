from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import datetime

# 資料路徑
DATASET_PATH  = 'sample'

# 影像大小
IMAGE_SIZE = (224, 224)

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 16

# 凍結網路層數
FREEZE_LAYERS = 3

# Epoch 數
NUM_EPOCHS = 10

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-final.h5'

# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator()
"""
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
"""
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)
"""
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_batches = train_datagen.flow_from_directory(
        DATASET_PATH + '/train', 
        target_size=IMAGE_SIZE,  
        batch_size=BATCH_SIZE,
        class_mode='categorical')#'binary'

validation_datagen = ImageDataGenerator(rescale=1.0/255)
valid_batches = validation_datagen.flow_from_directory(
        DATASET_PATH + '/valid',  
        target_size=IMAGE_SIZE,  
        batch_size=BATCH_SIZE,
        class_mode='categorical')
"""



# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))


# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

# 輸出整個網路結構
#print(net_final.summary())

#tensorboard
#callback ensures that logs are created and stored. Additionally, enable histogram computation every epoch with histogram_freq=1
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# 訓練模型

history = net_final.fit_generator(train_batches,
                steps_per_epoch = train_batches.samples // BATCH_SIZE,
                validation_data = valid_batches,
                validation_steps = valid_batches.samples // BATCH_SIZE,
                epochs = NUM_EPOCHS,
                callbacks=[tensorboard_callback])

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)

#print(history.history.keys())
#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

fp = open("logs/record.txt", "w")
fp.writelines("epoch\t loss\t accuracy\t val_loss\t val_accuracy\t\n")
for i in range(NUM_EPOCHS):
    fp.writelines("{}\t {:.4f}\t {:.4f}\t\t  {:.4f}\t t {:.4f}\t\n".format( 
        i+1, history.history['loss'][i], history.history['accuracy'][i], history.history['val_loss'][i], history.history['val_accuracy'][i]))
fp.close()

print("accurancy={}".format(history.history['accuracy'][-1]))