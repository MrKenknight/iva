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

video_output = 'demo_ver2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output_video/{video_output}', fourcc, 45, (w,h), True)


prev_frame_time = 0
new_frame_time = 0
down={}
counter_down=set()

in_zone = set()
out_zone = set()
white_color = (255, 255, 255)
offset = 3


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


line_st, line_en = (20,500), (1280-20,500)
in_zone_point, out_zone_point = find_side_point(line_st, line_en, 30)

line = LineString([line_st, line_en])
offset_distance = 10
polygon = line.buffer(offset_distance, cap_style=2)  # cap_style=2 for a flat end
exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
polygon = Polygon(exterior_coords)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # frame = cv2.resize(frame, (1280, 720)) 
        results = model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml', classes = [2, 3, 5, 7])
        boxes = results[0].boxes.xyxy.cpu()
        cv2.line(frame, line_st, line_en, white_color, 2)
        cv2.polylines(frame, [exterior_coords], isClosed=True, color=(0, 255, 0), thickness=1)

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
                # annotator.box_label(box, color=colors(int(cls), True), label=label_name)

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)
                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                # cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                # cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)
                cx, cy = track[-1]
                catch_point = Point(track[-1])
                y = line_st[1]

                if polygon.contains(catch_point):
                    if check_same_side(line_st, line_en, in_zone_point, track[-3]):
                        out_zone.add(track_id)
                    else:
                        in_zone.add(track_id)

                # if y < (cy + offset) and y > (cy - offset):
                #     if track[-3][1] < y:
                #         out_zone.add(track_id)
                #     elif track[-3][1] > y:
                #         in_zone.add(track_id)
                    # cv2.circle(frame,(cx,cy),10,colors(int(cls),True),-1)
                    # cv2.putText(frame,str(len(counter_down)),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        cv2.putText(frame,('in_zone: ')+ str(len(in_zone)),in_zone_point,cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2, cv2.LINE_AA)
        cv2.putText(frame,('out_zone: ')+ str(len(out_zone)),out_zone_point,cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2, cv2.LINE_AA) 

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        infor = f"fps: {str(fps)}"
        cv2.putText(frame, infor, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color , 1, cv2.LINE_AA) 
        cv2.imshow('Framee', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# result.release()
cap.release()
out.release()
cv2.destroyAllWindows()


