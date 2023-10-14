import cv2
import Visualizer
import pandas as pd


class FaceRecognizer:

    def __init__(self, identities: list, training_model_file_name: str, sequences:pd.DataFrame):
        self.__face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.__face_recognizer.read(training_model_file_name)
        cascade_path = "haarcascade_frontalface_default.xml"
        self.__faceCascade = cv2.CascadeClassifier(cascade_path)
        self.__identities = identities
        identities.append("Unknown")
        self.__visualizer = Visualizer.Visualizer(self.__identities)
        self.__sequences = sequences

    def recognize_face_in_webcam(self) -> list:
        # We will keep track of how many times an identity was recognized
        identities_count = list()
        for i in range(0, len(self.__identities)):
            identities_count.append(0)
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height
        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, frame = cam.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.__identify_faces(self.__font,
                                      gray,
                                      frame,
                                      minW,
                                      minH,
                                      identities_count)
                self.__visualizer.visualize(self.__identities, identities_count, identities_count)
                cv2.imshow('camera', frame)
                k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def recognize_faces_of_identities_in_video(self, video: str) -> pd.DataFrame:
        identities_count_per_sequence = pd.DataFrame(columns=["identity", "sequence", "appearances"])
        for i in range(0, len(self.__identities)):
            for j in range (0, len(self.__sequences)):
                identities_count_per_sequence.loc[len(identities_count_per_sequence)] = [self.__identities[i],
                                                                                         self.__sequences.loc[j, "sequence"],
                                                                                         0]

        # print(identities_count_per_sequence)
        # We will keep track of how many times an identity was recognized
        identities_count = list()
        for i in range(0, len(self.__identities)):
            identities_count.append(0)

        identity_timeline_appearances = pd.DataFrame(columns=["frame", "identity", "value", "color"])
        frame_count = 2100

        # We open the video to test how well the faces are recognized
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Determine the dimensions of the video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Check if video is opened successfully
        if not cap.isOpened():
            raise IOError("Error opening video stream!")

        # Read until video is completed
        at_end = False
        while cap.isOpened() & at_end == False:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else: 
                at_end = True
            print(frame_count)
            # Read every second frame
            if frame_count % 3 == 0:
                # Loop through indentities to update stat
                for i in range(0, len(self.__identities)):
                    identity_timeline_appearances.loc[len(identity_timeline_appearances.index)] = [frame_count, self.__identities[i], 1, False]
                self.__identify_faces(frame,
                                      0.1 * frame_height,
                                      0.1 * frame_width,
                                      identities_count,
                                      identity_timeline_appearances, frame_count,
                                      identities_count_per_sequence)
                # Display the resulting frame
                k = self.__visualizer.visualize(frame, identities_count, identity_timeline_appearances)
                # space: 32
                # 2 left arrow
                # 3 right arrow
                if k == 32:
                    self.pause(frame, identities_count, identity_timeline_appearances, cap, frame_count, frame_height, frame_width, identities_count_per_sequence)

                if k == 25 or k == 27:
                    at_end = True

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        self.__visualizer.release()

        return identities_count_per_sequence

    def pause(self, frame, identities_count, identity_timeline_appearances, cap, frame_count, frame_height, frame_width, identities_count_per_sequence):
        while True:
            k = self.__visualizer.visualize(frame, identities_count, identity_timeline_appearances)
            if k == 2 and frame_count >= 3:
                frame_count -= 3
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                self.__identify_faces(frame,
                                        0.1 * frame_height,
                                        0.1 * frame_width,
                                        identities_count,
                                        identity_timeline_appearances, frame_count,
                                        identities_count_per_sequence)                
            if k == 3:
                frame_count += 3
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                self.__identify_faces(frame,
                                        0.1 * frame_height,
                                        0.1 * frame_width,
                                        identities_count,
                                        identity_timeline_appearances, frame_count,
                                        identities_count_per_sequence)
            if k == 32:
                break;



    def __identify_faces(self, img, minH, minW, identities_count, identity_timeline_appearances, frame_count, identity_count_per_sequence):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__faceCascade.detectMultiScale(gray, minNeighbors=10, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            id, confidence = self.__face_recognizer.predict(gray[y:y + h, x:x + w])
            print(self.__identities[id], " ", confidence)
            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 70):
                identities_count[id] = identities_count[id] + 1
                id = self.__identities[id]
                identity_timeline_appearances.loc[(identity_timeline_appearances["frame"] == frame_count) &
                                                  (identity_timeline_appearances["identity"] == id) &
                                                  (identity_timeline_appearances["value"] == 1),
                                                  "color"] = True
                seqnum= self.__sequences.query("startframe <= " + str(frame_count) + " and endframe >= " + str(frame_count))["sequence"].iloc[0]
                # print(seqnum)
                identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                 "appearances"] = identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                 "appearances"] + 1
                self.__visualizer.visualize_face_in_frame(img, id, 100 - ((confidence/70) * 100), x, y, w, h)
            else:
                identities_count[3] = identities_count[3] + 1
                id = "Unknown"
                identity_timeline_appearances.loc[(identity_timeline_appearances["frame"] == frame_count) &
                                                  (identity_timeline_appearances["identity"] == id) &
                                                  (identity_timeline_appearances["value"] == 1),
                                                  "color"] = True
                seqnum= self.__sequences.query("startframe <= " + str(frame_count) + " and endframe >= " + str(frame_count))["sequence"].iloc[0]
                identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                 "appearances"] = identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                 "appearances"] + 1
                self.__visualizer.visualize_face_in_frame(img, id, 100 - ((confidence/70) * 100), x, y, w, h)
            


