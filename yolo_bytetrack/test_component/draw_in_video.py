import cv2
import numpy as np

points = []
polygons = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print((x,y))

cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)

video_path = "input_video/Video_output.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    success, img = cap.read()
    if success:
        key = cv2.waitKey(100)
        for point in points:
            cv2.circle(img, point, 3, (0, 0, 255), -1)
        cv2.polylines(img, np.array(polygons), True, (255, 0, 0), 2)
        
        if key == 13:  # Phím enter
            if len(points) > 1:
                polygons.append(points.copy())
                points.clear()
        elif key == 27:  # Phím ESC để thoát
            break

        cv2.imshow('image', img)
        
cv2.destroyAllWindows()
