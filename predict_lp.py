from readjson import extract_X_and_y
from utils import read_section
import cv2
from keras.models import load_model
from keras.losses import mean_squared_error

properties = read_section("part1.ini", "part1")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
img_width = int(properties["img.width"])
img_height = int(properties["img.height"])
images_root = properties["images.root.dir"]
json_dir = properties["json.file.dir"]
json_filename = properties["json.file.name"]

json_file_path = json_dir + json_filename
model = load_model(model_dir+model_file)

f = open(json_file_path, "r")
X = []
ar = []
for line in f:
    a,b = extract_X_and_y(line)
    X.append(a)
    ar.append(b)
f.close()

for a in range(0,12):
    file_name = X[a]
    img = cv2.imread(images_root + file_name)
    x1 = int(ar[a][0] * img_width)
    y1 = int(ar[a][1] * img_height)
    x2 = int(ar[a][2] * img_width)
    y2 = int(ar[a][3] * img_height)

    img2 = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
    img2 = img2.astype("float32")
    img2 /= 255
    ar2 = model.predict(img2)
#    err = model.evaluate(img2, ar[a][:])
#    print(err)
    x3 = int(ar2[0][0] * img_width)
    y3 = int(ar2[0][1] * img_height)
    x4 = int(ar2[0][2] * img_width)
    y4 = int(ar2[0][3] * img_height)
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))
    cv2.rectangle(img, (x3,y3), (x4,y4), (0,0,255))
    cv2.imshow(file_name, img)
"""
img3 = cv2.imread("D:/license_plates/smaller/20180619_201140.jpg")
img4 = img3.reshape(1,img3.shape[0], img3.shape[1], img3.shape[2])
img4 = img4.astype("float32")
img4 /= 255
ar2 = model.predict(img4)
#    err = model.evaluate(img2, ar[a][:])
#    print(err)
x3 = int(ar2[0][0] * img_width)
y3 = int(ar2[0][1] * img_height)
x4 = int(ar2[0][2] * img_width)
y4 = int(ar2[0][3] * img_height)
cv2.rectangle(img3, (x3,y3), (x4,y4), (0,0,255))
cv2.imshow("test", img3)
"""
