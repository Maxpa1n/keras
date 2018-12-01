import matplotlib.pyplot as plt
from keras.datasets import mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()

my_slice = train_image[20:100, :, :]
#No.20 to 100 image
print(my_slice.shape)

#my_slice1 = train_image[:, :20, :20]
my_slice1 = train_image[:, 7:-7, 7:-7]
print(my_slice1.shape)
digit = my_slice1[4455]
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

#n = range((60000/128))
batch = train_image[:128] # n = 0
batch = train_image[128:256] # n = 1
for n in range((60000/128)):
    batch = train_image[128*n:128*(n+1)]
