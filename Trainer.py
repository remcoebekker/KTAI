import os
import cv2
from PIL import Image
import numpy as np


class Trainer:


    def __init__(self):
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_samples = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def __getImagesAndLabels(self, path):
        # We create a list of image paths
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        # We loop through the images
        for imagePath in imagePaths:
            # We convert the image to the grayscale
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            # We put the image into an array
            img_numpy = np.array(PIL_img, 'uint8')
            # And we extract the id of the image file name
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            # We extract the faces, if any, from the array
            faces = self.face_samples.detectMultiScale(img_numpy, minNeighbors=10)
            # And we loop through the faces
            for (x, y, w, h) in faces:
                # And for each face, we add the rectangle of the array to the face samples
                face_samples.append(img_numpy[y:y + h, x:x + w])
                # And we lock in the corresponding id
                ids.append(id)
        # And we return the face samples and corresponding ids
        return face_samples, ids

    def train(self, training_path:str, training_model_file_name:str):
        # Extract the lists of faces and corresponding ids from the training snapshots
        print("gettings images and labels")
        faces, ids = self.__getImagesAndLabels(training_path)
        print("done gettings images and lables")
        # We train the learner on the extracted faces
        print("trainig...")
        self.__recognizer.train(faces[1::2], np.array(ids[1::2]))
        print("testings...")
        # get model performance
        total:int = len(ids[0::2])
        good:int = 0
        test_faces = faces[0::2]
        for index, correct_id in enumerate(ids[0::2]):
            id, confidence = self.__recognizer.predict(test_faces[index])
            if(id == correct_id and confidence < 70):
                good+=1
        print("Percentage good = ",(good/total)*100)   

        # Save the learned model into the trainer.yml file
        self.__recognizer.write(training_model_file_name)


