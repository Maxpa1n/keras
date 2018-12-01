import matplotlib.pyplot as plt
from keras.datasets import mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()

print('train_image.shape:', train_image.shape)
print('train_image.ndim:', train_image.ndim)
print('train_image.dtype:', train_image.dtype)

digit = train_image[44]
print('digit.shape:', digit.shape)
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print('train_labels.shape:', train_labels.shape)
print('train_labels.ndim', train_labels.ndim)
print('train_labels:', train_labels)