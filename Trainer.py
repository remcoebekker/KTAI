import os
import cv2
from PIL import Image
import numpy as np


class Trainer:

    def __init__(self, min_neighbors: int):
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors=min_neighbors)
        self.face_samples = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def __getImagesAndLabels(self, path, frame_training_count: int):
        # We create a list of image paths only jpg
        image_paths = [os.path.join(path, f) for f in list(filter(lambda name: name.endswith(".jpg"),
                                                                  os.listdir(path)))]
        face_samples = []
        ids = []
        ids_counts = {}

        # We loop through the images
        for imagePath in image_paths:
            # We convert the image to the grayscale
            pil_img = Image.open(imagePath).convert('L')  # grayscale
            # We put the image into an array
            img_numpy = np.array(pil_img, 'uint8')
            # And we extract the id of the image file name
            user_id = int(os.path.split(imagePath)[-1].split(".")[1])
            # We extract the faces, if any, from the array
            # remove image that are to small (incorrect faces like noses)
            faces = self.face_samples.detectMultiScale(img_numpy, minNeighbors=8, minSize=(100, 100))
            # And we loop through the faces
            for (x, y, w, h) in faces:
                id_count = ids_counts.get(user_id, 0)
                if id_count < frame_training_count:
                    ids_counts[user_id] = id_count + 1
                    # And for each face, we add the rectangle of the array to the face samples
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    # And we lock in the corresponding id
                    ids.append(user_id)

        # And we return the face samples and corresponding ids
        return face_samples, ids

    def train(self, training_path: str, training_model_file_name: str, frame_training_count: int):
        # Extract the lists of faces and corresponding ids from the training snapshots
        print("getting images and labels")
        faces, ids = self.__getImagesAndLabels(training_path, frame_training_count)
        print("done getting images and labels")
        # We train the learner on the extracted faces
        print("training...")
        self.__recognizer.train(faces, np.array(ids))

        # Save the learned model into the trainer.yml file
        self.__recognizer.write(training_model_file_name)
