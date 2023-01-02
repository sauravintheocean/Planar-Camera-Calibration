import argparse
import cv2
import numpy as np
import glob
import yaml

### TAKE IMAGE INPUT AS PASSED ARGUMENT
parser = argparse.ArgumentParser(description = 'This is the Feature Extraction part!')
parser.add_argument('-f', metavar = 'Enter "/file-loc/" having calibration images in jpg format', type = str, nargs = '?', default = '../data/')
args = parser.parse_args()

### GLOBAL
count = 0
color = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
rows = 7
columns = 7
objpoint = np.zeros((rows*columns,3), np.float32)
objpoint[:,:2] = np.mgrid[0:columns, 0:rows].T.reshape(-1,2)

# Arrays to store object points and image points from all the views
objpoints = [] # 3d point in real world space, Z = 0
imgpoints = [] # 2d points in image plane.

# Import calibration images
images = glob.glob(f'{args.f}*.jpg')

# Extract features
for file_name in images:
    ### SOURCE HANDLER
    image = cv2.imread(file_name)
    if image is None:
        print(f'Failed to load image file {file_name}. Pass a valid location as -f <loc>')
        exit(0)
    image_instance = image.copy()

    ### START APPLICATION
    image_instance = cv2.cvtColor(image_instance,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(image_instance, (columns,rows),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        count += 1
        objpoints.append(objpoint)
        corners2 = cv2.cornerSubPix(image_instance,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        image_instance = cv2.drawChessboardCorners(image_instance, (columns,rows), corners2,ret)

    cv2.putText(image_instance,'Press any key for next (exits if no more images)',(30,30), font, 1, color,2)

    # Display loop
    while True:
        cv2.imshow('Feature Extracted Image',image_instance)

        # User Input
        key = cv2.waitKey()
        break

cv2.destroyAllWindows()

# transform the object points and image points into readable yaml list
data = {'objpoints': np.asarray(objpoints).tolist(),
        'imgpoints': np.asarray(imgpoints).tolist()}

# and save it to a file
with open("extracts.yaml", "w") as f:
    yaml.dump(data, f)
    print(f'extracted feature points from {count} images out of {len(images)} saved at {f.name}')
