import streamingDataset as sd
import numpy as np
np.random.seed(2808)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from utils import read_section
from keras.optimizers import Adam
from utils import read_section

properties = read_section("part1.ini", "part1")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
img_width = properties["img.width"]
img_height = properties["img.height"]
images_root = properties["images.root.dir"]
json_dir = properties["json.file.dir"]
json_filename = properties["json.file.name"]

json_file_path = json_dir + json_filename

dataset = sd.StreamingDataset(json_file_path, images_root, img_height, img_width)


model = Sequential()
model.add(Conv2D(32, (5, 5), padding="same", activation='relu', input_shape=(img_height,img_width,3)))
model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))

adam = Adam(lr=0.0001)
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()
 
# 9. Fit model on training data
history = model.fit_generator(dataset.generate_training(32), steps_per_epoch=40, epochs=15, \
                verbose=2, validation_data=dataset.generate_test(32), validation_steps=10)

print(history.history) 
# 10. Evaluate model on test data
model.save(model_dir + model_file)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
