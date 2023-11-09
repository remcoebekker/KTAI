import cv2
import os


class Dataset_collector:


    def __init__(self):
        self.__faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def are_training_faces_already_collected_from_videos(self, training_path:str, videos:list, number_of_sampled_frames)->bool:
        # Only jpg
        files = list(filter(lambda x: x.endswith(".jpg"), os.listdir(training_path)))
        if len(files) == len(videos) * number_of_sampled_frames:
            return True

        return False

    def collect_training_faces_from_videos(self, training_path:str, videos:list, number_of_sampled_frames, minNeighbors:int):
        # We loop through the list of training videos
        for i in range(0, len(videos)):
            cap = cv2.VideoCapture(videos[i])

            # Check if video is successfully opened...
            if (cap.isOpened() == False):
                # Oops...there is a problem with opening the file
                raise IOError("Error opening the video!")

            # Each video corresponds to one identity and we link a unique integer to that identity
            face_id = str(i)
            # Now we go through the video and extract up to 500 face snapshots and write these to the
            # training library
            count = 0
            at_end = False
            while (cap.isOpened() & at_end == False):
                # We read a frame from video until the end of the file is hit or we have enough snapshots
                ret, img = cap.read()
                if (ret == False):
                    at_end = True
                else:
                    # The image is turned into a gray scale image for easier face detection
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Detect the faces from the image
                    faces = self.__faceCascade.detectMultiScale(gray, 1.3, minNeighbors)
                    # We loop through all the faces being detected
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        count += 1
                        # Save the captured face snapshot into the training folder
                        cv2.imwrite(training_path + "//User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                    # We take 500 snapshots into account for each identity
                    if count >= number_of_sampled_frames:
                        at_end = True
            # Close the file
            cap.release()
