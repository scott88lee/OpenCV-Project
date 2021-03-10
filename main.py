import functions_file as f
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read("models/trained.yml")

name = {1: "Matt Damon", 2: "Emma Stone",
        3: "Clooney", 4: "Jenn", 5: "Nat", 6: "Scott"}

#img = cv2.imread("images/test3.jpg")
img = cv2.imread("images/test9.jpg")
faces, img_grey = f.faceDetection(img) # Function 1

#Draw rect on face
for face in faces:
  (x, y, w, h) = face
  subject = img_grey[y: y + w, x: x + h]

    #cv2.rectangle(img_grey, (x, y), (x + w, y + h), (0, 255, 0), 2)
  label, confidence = recognizer.predict(subject)
  text = name[label] + ": " + str(confidence)
  if confidence > 50:
    text = "Unknown"

  f.highlight(img, face)
  f.addLabel(img, text, x, y)
 
while True:
  cv2.imshow("Test output", img)

  if cv2.waitKey(1) == 13:
    break
cv2.destroyAllWindows()
