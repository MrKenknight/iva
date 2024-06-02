import cv2
import numpy as np
from ultralytics import YOLO
import time
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

track_history = defaultdict(lambda: [])
model = YOLO("yolov8x.pt")
names = model.model.names


video_path = "input_video/market_square_720_1280.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w,h)

resize = (720, 1280)
# resize = (1280, 720)

output = 'output_video/market_square_720_1280.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 40, resize, True)

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # frame = cv2.resize(frame, resize) 
        results = model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml')
        boxes = results[0].boxes.xyxy.cpu()
        annotated_frame = results[0].plot()
        frame = annotated_frame
        annotator = Annotator(frame, line_width=2)
        red_color = (0, 0, 255)
        # if results[0].boxes.id is not None:
        #     # Extract prediction results
        #     clss = results[0].boxes.cls.cpu().tolist()
        #     track_ids = results[0].boxes.id.int().cpu().tolist()
        #     confs = results[0].boxes.conf.float().cpu().tolist()
        #     frame = annotated_frame
            # Annotator Init

            # for box, cls, track_id in zip(boxes, clss, track_ids):
            #     label_name = names[int(cls)]
            #     # Store tracking history
            #     track = track_history[track_id]
            #     track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            #     if len(track) > 30:
            #         track.pop(0)
            #     # Plot tracks
            #     points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))

            #     cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)

            #     cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)
        # cv2.putText(frame,('count: ')+ str(len(counter_down)),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA) 

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        infor = f"fps: {str(fps)} / count: {str(len(boxes))}"
        # infor = f"fps: {str(fps)}"
        # infor = f"count: {str(len(boxes))}"


        cv2.putText(frame, infor, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color , 2, cv2.LINE_AA) 
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# result.release()
cap.release()
out.release()
cv2.destroyAllWindows()


