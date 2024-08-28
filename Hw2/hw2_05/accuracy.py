from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_eraser
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_batches = test_datagen.flow_from_directory('sample/test',
                                                target_size=(224, 224),
                                                interpolation='bicubic',
                                                class_mode='categorical',
                                                shuffle=True,
                                                batch_size=16)
net5_2 = load_model('model-resnet50-final.h5')
loss2, accuracy2 = net5_2.evaluate(test_batches)
print('accuracy={:.4f}'.format(accuracy2))

net5_4 = load_model('model-resnet50-RE.h5')
loss4, accuracy4 = net5_4.evaluate(test_batches)
print('accuracy={:.4f}'.format(accuracy4))

x_label = ['before', 'after']
accuracy = [accuracy2, accuracy4]
x = np.arange(2)
plt.bar(x, accuracy)
plt.xticks(x, x_label)
#plt.xlabel(x_label)
plt.ylabel('accuracy')
plt.show()
