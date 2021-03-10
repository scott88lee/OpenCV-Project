import functions_file as f

faces, facesID = f.getFaces("images")
trainer = f.f3_recognizer(faces, facesID)
trainer.write("models/trained.yml")