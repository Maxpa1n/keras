from keras.datasets import imdb
import numpy as np
from keras import  models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=1000)

#print(train_data.shape)
#print(len(train_data[0]))
#print(len(train_data[1]))
#print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[12]])


def vectorize_sequence(sequences, dimension=10000):
    print(len(sequences),'and',dimension)
   
    results = np.zeros((len(sequences), dimension), dtype='float16')
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequence(train_data)
print('1')
x_test = vectorize_sequence(test_data)
print('2')
y_train = np.asarray(train_label).astype('float16')
print('3')
y_test = np.asarray(test_label).astype('float16')
print('4')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


models = models.Sequential()
models.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
models.add(layers.Dense(16, activation='relu'))
models.add(layers.Dense(1, activation='sigmoid'))

models.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = models.fit(partial_x_train, partial_y_train, epochs=20, batch_size=50, validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()


test_loss, test_acc = models.evaluate(x_test, y_test)

print('test_acc:', test_acc)
print('loss:', test_loss)