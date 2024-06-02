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

video_path = "input_video/instrusion_example.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_output = 'intrusion_02_06.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output_video/{video_output}', fourcc, 45, (w,h), True)


prev_frame_time = 0
new_frame_time = 0

intrude_zone = set()
green_color = (28, 172, 255)
red_color = (0, 0, 255)
offset = 3
ready = False

intrude_zone_statement = {"bicycle": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

polygons = []
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)


while cap.isOpened():
    success, frame = cap.read()
    if success:
        if ready:
            # frame = cv2.resize(frame, (1280, 720)) 
            results = model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml', classes = [0])
            boxes = results[0].boxes.xyxy.cpu()
            if results[0].boxes.id is not None:
                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()
                # annotated_frame = results[0].plot()
                # frame = annotated_frame
                # Annotator Init
                annotator = Annotator(frame, line_width=1)
                for box, cls, track_id in zip(boxes, clss, track_ids):
                    label_name = names[int(cls)]
                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)
                    # Plot tracks
                    center_x = track[-1][0]
                    footer_y = int(box[3])
                    catch_point = Point([center_x, footer_y])
                    if Polygon(polygons[0]).contains(catch_point):
                        intrude_zone.add(track_id)
                        annotator.box_label(box, color=colors(int(cls), True), label=label_name)


            cv2.putText(frame,('count: ')+ str(len(intrude_zone)),(100,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 2, cv2.LINE_AA) 

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        infor = f"fps: {str(fps)}"
        cv2.putText(frame, infor, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color , 2, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)
        cv2.polylines(frame, np.array(polygons), True, (255, 0, 0), 2)
        

        cv2.imshow('Frame', frame)
        out.write(frame)

        # clear_screen()
        # print('in_zone_statement: ', in_zone_statement)
        # print('out_zone_statement: ', out_zone_statement)
        

        key = cv2.waitKey(30)
        if key == 13:  # Phím enter
            polygons.append(points.copy())
            points.clear()
            ready = True

        if key & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


