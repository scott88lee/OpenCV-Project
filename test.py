import functions_file as f
import cv2

def test1():
  img = cv2.imread("images/test.jpg")
  #img = cv2.imread("images/test2.jpg")
  faces, img_grey = f.faceDetection(img)

  #Draw rect on face
  for (x, y, w, h) in faces:
    cv2.rectangle(img_grey, (x, y), (x + w, y + h), (0, 255, 0), 2)

  while True:
    cv2.imshow("Test output", img_grey)

    if cv2.waitKey(1) == 13:
      break
  cv2.destroyAllWindows()

test1()