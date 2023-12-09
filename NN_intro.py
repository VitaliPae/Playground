import gzip
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1,28,28)\
            .astype(np.float32)
    
def open_labels(filename):
     with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)
     

X_train = open_images("Data/train-images-idx3-ubyte.gz")
# print(X_train.shape)
# print(X_train)

# plt.imshow(X_train[0], cmap="gray_r")
# plt.show()
y_train = open_labels("Data/train-labels-idx1-ubyte.gz")

#print(y_train)
y_train = y_train == 0
#print(y_train)

model =  Sequential()

model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy")

# print(X_train.reshape(60000, 784))
model.fit(
    X_train.reshape(60000, 784),
    y_train,
    epochs=10,
    batch_size=1000)

# print(y_train[0])
# plt.imshow(X_train[0])
# plt.show()

print(model.predict(X_train[1].reshape(1,784)))
print(model.predict(X_train.reshape(60000,784)))







        