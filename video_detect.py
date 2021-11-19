import cv2
import argparse

#аргумент командной строки с изображением
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
args = parser.parse_args()
src = args.source

#захват видео
cap = cv2.VideoCapture(src)

#инициализация параметров детектирующей сети
net = cv2.dnn.readNetFromDarknet('yolov4.cfg.txt', 'yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)


while True:
	read_ok, img = cap.read()
	img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

	classIds, scores, boxes = model.detect(img, confThreshold=0.1, nmsThreshold=0.1)  # 0.6 0.4
	for (classId, score, box) in zip(classIds, scores, boxes):
		cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
		              color=(0, 255, 0), thickness=2)

	cv2.imshow("video", img)

	if cv2.waitKey(1) & 0xFF == ord('x'):
		break