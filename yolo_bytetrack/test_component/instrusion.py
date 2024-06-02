import cv2
import numpy as np
import time

from collections import defaultdict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


track_history = defaultdict(lambda: [])
model = YOLO("yolov8s.pt")
names = model.model.names

video_path = "vietnh41/instrusion_example.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

output = 'vietnh41/demo.avi'
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output, codec, fps, (w, h))

prev_frame_time = 0
new_frame_time = 0
down = {}
counter_down = set()
insize_zone = set()

top_left  = (1094, 319)
top_right = (1330, 315)
bot_left  = (1126, 538)
bot_right = (1407, 530)
zone = [top_left, top_right, bot_right, bot_left]
polygon = Polygon(zone)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml', classes = [0])
        boxes = results[0].boxes.xyxy.cpu()
        red_color = (0, 0, 255)

        # Instrusion detection zone
        cv2.polylines(frame, np.array([zone]), True, (255, 0, 0), 2)

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                label_name = names[int(cls)]
                
                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                center_x = int((x1+x2)/2)
                footer_y = y2
                point = Point([center_x, footer_y])
                
                if polygon.contains(point):
                    insize_zone.add(track_id)
                    annotator.box_label(box, color=colors(int(cls), True), label=label_name)
                    # cv2.circle(frame, (center_x,footer_y), 5, (0,255,0), -1)

        cv2.putText(frame,('count: ')+ str(len(insize_zone)),(100,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA) 

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        infor = f"fps: {str(fps)}"
        cv2.putText(frame, infor, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color , 1, cv2.LINE_AA) 
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# result.release()
cap.release()
# out.release()
cv2.destroyAllWindows()