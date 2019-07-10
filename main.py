import cv2
import detection
from jade import *
if __name__ == '__main__':
    img = cv2.imread("images/Aaron_Eckhart_0001.jpg")
    faces = detection.get_faces(img,0.6)
    bboxes = []
    scores = []
    for face in faces:
        bboxes.append([face.x1,face.y1,face.x2,face.y2])
        scores.append(face.confidence)

    image = DrawBoxes(img,bboxes)
    cv2.imshow("result",image)
    cv2.waitKey(0)