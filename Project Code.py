import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


PATH = os.getcwd()

DATA_PATH = os.path.join(PATH, "Dataset")
data_dir_list = os.listdir(DATA_PATH)


print(data_dir_list)



classes_names_list=[]
img_data_list=[]

for dataset in data_dir_list:
    classes_names_list.append(dataset)
    print ('\nGetting images from {} folder'.format(dataset))
    img_list=os.listdir(DATA_PATH+'/'+ dataset)
    for img in img_list:
        input_img=cv2.imread(DATA_PATH + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224, 224))
        (b, g, r)=cv2.split(input_img_resize)
        img=cv2.merge([r,g,b])
        img_data_list.append(img)

num_classes = len(classes_names_list)



img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255


print (img_data.shape)


plt.imshow(img_data[24])
plt.show()


num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape

classes = np.ones((num_of_samples,), dtype='int64')

classes[0:98]=0
classes[98:]=1



classes = to_categorical(classes, num_classes)



X, Y = shuffle(img_data, classes, random_state=456)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=456)

# Check the number of images in THE dataset split.....
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)




#Building the  model....
model = Sequential()

model.add(Conv2D(32, (3, 3),activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, batch_size=32)


print('\nTest Accuracy:', score[1])





