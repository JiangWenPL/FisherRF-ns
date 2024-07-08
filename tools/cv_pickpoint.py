import cv2
import numpy as np
import argparse
import os
import glob
import pathlib as Pathlib

# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle at the clicked position
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        points.append((x, y))

        # Display coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{x}, {y}", (x + 10, y + 10), font, 0.5, (0, 255, 0), 1)
        cv2.imshow('image', img)
        print(f"Point selected: ({x}, {y})")

# Function to redraw all points
def redraw_points():
    for point in points:
        cv2.circle(img, point, 3, (255, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{point[0]}, {point[1]}", (point[0] + 10, point[1] + 10), font, 0.5, (0, 255, 0), 1)
    cv2.imshow('image', img)

arg = argparse.ArgumentParser()
arg.add_argument("--path", type=str, default="imgs")
arg.add_argument("--points_path", type=str, default="points")
args = arg.parse_args()

img_paths = glob.glob(os.path.join(args.path, "*.png"))
img_paths.sort()

os.makedirs(args.points_path, exist_ok=True)

# Create a window and set a mouse callback function
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', click_event)

points = []  # List to store points in format (x, y)

# Display the image
idx = 0
img = cv2.imread(img_paths[idx])
cv2.imshow('image', img)

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord('b'):
        if points:
            points.pop()  # Remove the last point
            img = cv2.imread(img_paths[idx])  # Reload the image
            redraw_points()  # Redraw all points
            print("Latest point cleared.")
    elif key == ord('s'):
        points_np = np.array(points)
        img_filename = img_paths[idx]
        # find the stem
        stem = Pathlib.Path(img_filename).stem
        np.savez(os.path.join(args.points_path, stem + ".npz"), points=points_np)

        points.clear()
        idx += 1
        img = cv2.imread(img_paths[idx])
        redraw_points()
    elif key == 27:  # Escape key
        break

cv2.destroyAllWindows()
