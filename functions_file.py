import numpy as np
import os
import cv2

# FUNCTION 1
def faceDetection(img):
  face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml") # Loading a pretrained HAAR cascade model
  img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Coverting img to Greyscale
  faces_rect = face_cascade.detectMultiScale(img_grey, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

  return faces_rect, img_grey

# FUNCTION 2
def getFaces(img_dir):
  faces = []
  faceID = []

  for path, sub_dir, filenames in os.walk(img_dir):  # Traverse DIR
    for filename in filenames:
      if filename.startswith("test") or filename.startswith("."):
        print("Skipped: ", filename)
        continue  # SKIP; Sys/Test files

      img_dirID = os.path.basename(path)
      img_fullpath = os.path.join(path, filename)
      img = cv2.imread(img_fullpath)
      print(img_fullpath)

      faces_rect, img_grey = faceDetection(img)

      if img is None:  # Loads image into Function 1
        print("No image found: ", img_fullpath)

      if (len(faces_rect) == 0):
        print("No face detected: ", img_fullpath)
        continue
      elif (len(faces_rect) > 1):
        print("More than one face detected: ", img_fullpath)
        continue

      (x, y, w, h) = faces_rect[0]
      crop_img = img_grey[y: y + w, x: x + h]
      faces.append(crop_img)
      faceID.append(int(img_dirID))
  print("End of Function 2.")
  return faces, faceID

# FUNCTION 3
def f3_recognizer(faces, faceID):
  recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
  recognizer.train(faces, np.array(faceID))  # Train
  return recognizer

# FUNCTION 4
def highlight(img, face):
  (x, y, w, h) = face  # Co-ordinates
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3) # Draw Rect

# FUNCTION 5
def addLabel(img, text, x, y):  # Add text to image
  cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 4, (250, 0, 0))
