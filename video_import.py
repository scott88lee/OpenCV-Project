import cv2
import os

vid_capture = cv2.VideoCapture(0)
path = "capture"
counter = 1

while True:
  _, frame = vid_capture.read()
  #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #canvas = detect(grey, frame)

  cv2.imshow("Video", frame)
  cv2.imwrite(os.path.join(path, "cap_frame%d.jpg" % counter), frame)
  counter += 1

  if cv2.waitKey(1) & 0xFF == ord(' '):
    break

vid_capture.release()
cv2.destroyAllWindows()