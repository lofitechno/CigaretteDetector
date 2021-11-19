import argparse
import cv2
import os
import sys

#где хранится исходник
file_path = os.path.dirname(sys.argv[0])

#аргумент командной строки с изображением
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
args = parser.parse_args()
src = args.source

#разбиваем по точке название входного файла
srcsplit =src.split('.')

#считываем изображение
img = cv2.imread(src)

#инициализация параметров детектирующей сети
net = cv2.dnn.readNetFromDarknet('yolov4.cfg.txt', 'yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

#получаем найденные сигареты и отрисовываем их
classIds, scores, boxes = model.detect(img, confThreshold=0.2, nmsThreshold=0.01) #0.6 0.4
for (classId, score, box) in zip(classIds, scores, boxes):
	cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
	              color=(0, 255, 0), thickness=2)


#запись файла
outfilename = file_path+srcsplit[-2] + 'out.'+srcsplit[-1]
cv2.imwrite(outfilename, img)
