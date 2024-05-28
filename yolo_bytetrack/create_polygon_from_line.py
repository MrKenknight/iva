import numpy as np
import cv2
from shapely.geometry import LineString

# Step 1: Define the line
line_points = [(100, 200), (400, 500)]  # Replace with your line coordinates

# Step 2: Create a LineString with Shapely
line = LineString(line_points)
# Step 3: Offset the line
offset_distance = 10
polygon = line.buffer(offset_distance, cap_style=2)  # cap_style=2 for a flat end

# Step 4: Convert the Polygon to a format suitable for OpenCV
exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)

# Step 5: Draw the Polygon with OpenCV
# Create a blank image
height, width = 600, 800  # Replace with your image dimensions
image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the polygon
cv2.polylines(image, [exterior_coords], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the image
cv2.imshow('Polygon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
