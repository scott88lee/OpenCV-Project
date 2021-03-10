import functions_file as f
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/trained.yml")
v_cap = cv2.VideoCapture(0)

name = {1: "Matt Damon", 2: "Emma Stone", 3: "Clooney",
         4: "Jenn", 5: "Nat", 6: "Scott", 7: "Sarah"}

while True:
  _, frame = v_cap.read()
  faces_detected, img_grey = f.faceDetection(frame)

  for face in faces_detected:
    (x, y, w, h) = face
    subject = img_grey[y: y + w, x: x + h]

    label, confidence = recognizer.predict(subject) # Making the predition
    predicted_name = name[label] + ": " + str(confidence) + " confidence."

    f.highlight(frame, face) #Function 4

    if confidence > 50:
      continue
    else:
      #Function 5
      f.addLabel(frame, str(predicted_name), x, y)
    
  cv2.imshow("Feed", frame)

  if cv2.waitKey(1) & 0xFF == ord(' '):
    break
v_cap.release()
cv2.destroyAllWindows()
