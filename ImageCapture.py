import argparse
import cv2

### TAKE IMAGE NAME INPUT AS PASSED ARGUMENT
parser = argparse.ArgumentParser(description = 'This is the capturing program')
parser.add_argument('-n', metavar = 'location and name for images captured', type = str, nargs = '?', default = '../data/cap')
args = parser.parse_args()

try:
    cap = cv2.VideoCapture(0)  # Camera Input Number
except Exception as e:
    print(f'Error in camera capture: {e}')
    exit(0)

### GLOBAL
count = 0
color = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame
    _, image_instance = cap.read()
    image_instance = cv2.cvtColor(image_instance,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Camera', image_instance)

    # User Input
    key = cv2.waitKey(1)

    # Save Image Frame
    if key == ord('s'):
        cv2.imwrite(f'{args.n}{count}.jpg', image_instance)
        count += 1

    # Exit Application
    if key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

    # Print on Image(Not before saving)
    cv2.putText(image_instance,'s - save frame',(50,150), font, 1, color,2)
    cv2.putText(image_instance,'q - exit',(50,200), font, 1, color,2)
