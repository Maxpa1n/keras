from PIL import Image
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical

'''
sam_path = './new_samples'
im = Image.open(sam_path+'/1.1.0.1.bmp')
im.show()
width, height = im.size
im = im.convert("L")
data = im.getdata()
data = (np.matrix(data, dtype='float'))
new_data = (np.reshape(data, (height, width))/255) - 1
new_im = Image.fromarray(new_data)
plt.imshow(new_data, cmap=plt.cm.binary)
plt.show()
'''
sam_path = './new_samples/'
test_path = './new_test/'


def getimage(i, j, path):
    simple_img = i+'.' + j + '.bmp'
    str(simple_img)
    #print(simple_img)
    im = Image.open(path + simple_img)
    im = im.convert("L")
    width, height = im.size
    data = im.getdata()
    data = (np.matrix(data, dtype='float'))
    new_data = (np.reshape(data, (height, width)) / 255)
    return new_data


def trans_image_train():
    train_im = np.zeros((30, 160, 160))
    k = 0
    for i in range(0, 10):
        for j in range(1, 4):
            first = str(i)
            second = str(j)
            train = np.array([getimage(first, second, sam_path)])
            #print(train.shape)
            train_im[k] = train
            k = k + 1
    return train_im


def trans_image_test():
    test_im = np.zeros((10, 144, 168))
    k = 0
    for i in range(0, 10):
        first = str(i)
        test = np.array([getimage(first, '1', test_path)])
        test_im[k] = test
        k = k+1
    return test_im


test_im = trans_image_test()
train_im = trans_image_train()
a = np.ones((10, 1408))
print(a.shape)
train_im = train_im.reshape((30, 160*160))
train_im = train_im.astype('float32')
test_im = test_im.reshape((10, 144*168))
test_im = np.hstack((test_im, a))
test_im = test_im.astype('float32')
print(a.shape)
print(test_im.shape)

train_lable = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])
test_lable = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

train_lable = to_categorical(train_lable)
test_lable = to_categorical(test_lable)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(160*160, )))
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train_im, train_lable, epochs=5, batch_size=1)

test_loss, test_acc = network.evaluate(test_im, test_lable)

print('test_acc:', test_acc)
print('loss:', test_loss)
