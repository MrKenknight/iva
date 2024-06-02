import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from collections import defaultdict

track_history = defaultdict(lambda: [])
model = YOLO("yolov8s.pt")
names = model.model.names

video_path = "input_video/Video_output.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# video_output = 'demo_ver2.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(f'output_video/{video_output}', fourcc, 45, (w,h), True)


prev_frame_time = 0
new_frame_time = 0

in_zone = set()
out_zone = set()
green_color = (28, 172, 255)
offset = 3
ready = False

in_zone_statement = {"bicycle": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
out_zone_statement = {"bicycle": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def find_side_point(A, B, distance=30):
    AB_vector = np.array(B) - np.array(A)
    mid_point = (np.array(A) + np.array(B)) / 2
    perpendicular_vector = np.array([-AB_vector[1], AB_vector[0]])
    normalized_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    C = mid_point + distance * normalized_perpendicular_vector
    D = mid_point - distance * normalized_perpendicular_vector
    return C.astype(int), D.astype(int)

def vector_cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def check_same_side(A, B, C, D):
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    AD = (D[0] - A[0], D[1] - A[1])
    
    cross1 = vector_cross_product(AB, AC)
    cross2 = vector_cross_product(AB, AD)
    
    if cross1 * cross2 > 0:
        return True  # Cùng phía
    else:
        return False  # Khác phía


line_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)


while cap.isOpened():
    success, frame = cap.read()
    if success:
        if ready:
            # frame = cv2.resize(frame, (1280, 720)) 
            results = model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml', classes = [1, 2, 3, 5, 7])
            boxes = results[0].boxes.xyxy.cpu()
            cv2.line(frame, line_st, line_en, green_color, 2)
            # cv2.polylines(frame, [exterior_coords], isClosed=True, color=green_color, thickness=1)

            if results[0].boxes.id is not None:
                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()
                annotated_frame = results[0].plot()
                frame = annotated_frame
                # Annotator Init
                annotator = Annotator(frame, line_width=0.3)
                for box, cls, track_id in zip(boxes, clss, track_ids):
                    label_name = names[int(cls)]

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)
                    # Plot tracks
                    catch_point = Point(track[-1])

                    if polygon.contains(catch_point):
                        if check_same_side(line_st, line_en, in_zone_point, track[-3]):
                            if track_id not in out_zone:
                                out_zone_statement[label_name] += 1
                            out_zone.add(track_id)
                            
                        else:
                            if track_id not in in_zone:
                                in_zone_statement[label_name] += 1
                            in_zone.add(track_id)


            cv2.putText(frame,('in_zone: ')+ str(len(in_zone)),in_zone_point,cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2, cv2.LINE_AA)
            cv2.putText(frame,('out_zone: ')+ str(len(out_zone)),out_zone_point,cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2, cv2.LINE_AA) 

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        infor = f"fps: {str(fps)}"
        cv2.putText(frame, infor, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color , 1, cv2.LINE_AA)

        for point in line_points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

        cv2.imshow('Frame', frame)
        # out.write(frame)

        clear_screen()
        print('in_zone_statement: ', in_zone_statement)
        print('out_zone_statement: ', out_zone_statement)
        

        key = cv2.waitKey(30)
        if (len(line_points) == 2) & (key == 13):  # Phím enter
            ready = True
            if ready:
                line_st, line_en = line_points[0], line_points[1]
                in_zone_point, out_zone_point = find_side_point(line_st, line_en, 30)

                line = LineString([line_st, line_en])
                offset_distance = 10
                polygon = line.buffer(offset_distance, cap_style=2)  # cap_style=2 for a flat end
                exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
                polygon = Polygon(exterior_coords)


        if key & 0xFF == ord("q"):
            break
    else:
        break

# result.release()
cap.release()
# out.release()
cv2.destroyAllWindows()


