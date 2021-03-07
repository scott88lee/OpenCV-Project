# import the libraries
import os
import face_recognition


images = os.listdir('images')

image_to_be_matched = face_recognition.load_image_file('my_image.jpg')