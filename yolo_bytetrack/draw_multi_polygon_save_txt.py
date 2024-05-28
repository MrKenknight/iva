import cv2
import numpy as np

points = []
polygons = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        # Vẽ đa giác khi nhấn phím enter
        if len(points) > 1:
            cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)
            cv2.imshow('image', img)
            polygons.append(points.copy())
            points.clear()  # Xóa các điểm sau khi vẽ

cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)

img_name = '20240519_195620.png'

img = cv2.imread(f'input_img/{img_name}')
cv2.imshow('image', img)

while True:
    key = cv2.waitKey(1)
    if key == 13:  # Phím enter
        if len(points) > 1:
            cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)
            cv2.imshow('image', img)
            polygons.append(points.copy())
            points.clear()
    elif key == 27:  # Phím ESC để thoát
        cv2.imwrite(f'output_img/{img_name}', img)
        break

cv2.destroyAllWindows()

polygon_name = img_name.split('.')[0] + '.txt'
with open(f'polygon/{polygon_name}', 'w') as file:
    for polygon in polygons:
        for point in polygon:
            file.write(f"{point[0]}, {point[1]}\n")
        file.write("---\n")