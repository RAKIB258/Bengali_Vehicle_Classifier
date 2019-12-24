import cv2
import os

images = os.listdir('./data/')

try:
	os.mkdir('./resized')
except:
	pass

for i in images:
    img = cv2.imread('./data/'+i)
    img = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./resized/'+ i, img)
