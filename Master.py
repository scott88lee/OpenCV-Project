import numpy as np
import os
import cv2

# FUNCTION 1
def faceDetection(img):
  # Loading a pretrained HAAR cascade model
  face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

  # Coverting img to Greyscale
  img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Detecting multiscale images
  faces_rect = face_cascade.detectMultiScale(img_grey, scaleFactor=1.2, minNeighbors=6, minSize=(30,30))

  return faces_rect, img_grey

# TESTING FUNC 1
def test1():
  #img = cv2.imread("images/test.jpg")
  #img = cv2.imread("images/test2.jpg")
  img = cv2.imread("images/1/flrmat.jpeg")
  #img = cv2.imread("images/1/matty.jpeg")
  faces, img_grey = faceDetection(img)

  #Draw rect on face
  for (x, y, w, h) in faces:
    cv2.rectangle(img_grey, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
  while True:
    cv2.imshow("Test output", img_grey)

    if cv2.waitKey(1) == 13:
      break
  cv2.destroyAllWindows()
# TESTING FUNC 1


############################
# FUNCTION 2
def getFaces(img_dir):
  faces = []
  faceID = []

  for path, sub_dir, filenames in os.walk(img_dir): #Traverse DIR
    for filename in filenames:
      if filename.startswith("test") or filename.startswith("."):
        print("Skipped: ", filename)
        continue  # SKIP; Sys/Test files

      img_dirID = os.path.basename(path)
      img_fullpath = os.path.join(path, filename)

      img = cv2.imread(img_fullpath)

      if img is None:  # Loads image into Function 1
        print("No image found: ", img_fullpath)
      
      faces_rect, img_grey = faceDetection(img)
      
      if (len(faces_rect) == 0):
        print("No face detected: ", img_fullpath)
        continue

      if (len(faces_rect) > 1):
        print("More than one face detected: ", img_fullpath)
        continue

      (x, y, w, h) = faces_rect[0]
      crop_img = img_grey[y: y + w, x: x + h]
        
      faces.append(crop_img)
      faceID.append(int(img_dirID))
  
  print("End of Function 2.")
  return faces, faceID

# FUNCTION 3
def recognizer(faces, faceID):
  # Declare  
  recognizer = cv2.face.LBPHFaceRecognizer_create()
  # Train
  recognizer.train(faces, np.array(faceID))
  return recognizer

# FUNCTION 4
def highlight(img, face):
  (x, y, w, h) = face  # Co-ordinates
  # Draw Rect
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)

# FUNCTION 5
def label(img, text, x, y): # Add text to image
  cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (250, 0, 0))


def train():
  faces, faceID = getFaces("images")
  trainer = recognizer(faces, faceID)

  trainer.write("model.yml")

# Run once to train
train()

def main():
  name = {1: "Matt Damon", 2: "Emma Stone"}
