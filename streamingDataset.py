from readjson import extract_X_and_y
import cv2
import numpy as np

class StreamingDataset:
    def __init__(self, file_name, width, height):
        self.X = []
        self.y = []
        self.width = width
        self.height = height
        f = open(file_name, "r")
        for line in f:
            a,b = extract_X_and_y(line)
            self.X.append(a)
            self.y.append(b)
        f.close()
        self.create_datasets()

    def create_datasets(self):
        size = len(self.y)
        indices = np.random.permutation(size)
        eighty = int(size*80/100)
        training_idx, test_idx = indices[:eighty], indices[eighty:]
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.training_X, self.test_X = self.X[training_idx], self.X[test_idx]
        self.training_y, self.test_y = self.y[training_idx], self.y[test_idx]


    def generate_training(self, batch_size):
        while True:
            high = len(self.training_X)
            indices = np.random.random_integers(0, high-1, batch_size)
            file_names = self.training_X[indices]
            y = self.training_y[indices]
            ar = np.zeros((batch_size, height, width, 3))
            a = 0
            for file_name in file_names:
                #print(file_name, str(y[a]))
                img = cv2.imread(file_name)
                ar[a,:,:,:] = img
                a += 1
            ar = ar.astype('float32')
            ar /= 255
            yield ar, y
                      
    def generate_test(self, batch_size):
        while True:
            high = len(self.test_X)
            indices = np.random.random_integers(0, high-1, batch_size)
            file_names = self.test_X[indices]
            y = self.test_y[indices]
            ar = np.zeros((batch_size, height, width, 3))
            a = 0
            for file_name in file_names:
                img = cv2.imread(file_name)
                ar[a,:,:,:] = img
                a += 1
            ar = ar.astype('float32')
            ar /= 255
            yield ar, y
     



