import cv2
import Visualizer
import pandas as pd


class FaceRecognizer:

    def __init__(self, identities: list, training_model_file_name: str, sequences: pd.DataFrame, visualize: bool):
        self.__face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.__face_recognizer.read(training_model_file_name)
        cascade_path = "haarcascade_frontalface_default.xml"
        self.__faceCascade = cv2.CascadeClassifier(cascade_path)
        self.__identities = identities
        identities.append("Unknown")
        self.__visualize = visualize
        self.__sequences = sequences

        if visualize:
            self.__visualizer = Visualizer.Visualizer(identities)

    def recognize_face_in_webcam(self, confidence_level: int):
        # We will keep track of how many times an identity was recognized
        identities_count = list()
        for i in range(0, len(self.__identities)):
            identities_count.append(0)
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        # Define min window size to be recognized as a face
        min_w = 0.1 * cam.get(3)
        min_h = 0.1 * cam.get(4)

        while True:
            ret, frame = cam.read()
            if ret:
                self.__identify_faces_in_webcam(frame, min_w, min_h, identities_count, confidence_level)
                cv2.imshow('camera', frame)
                k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
                if k == 115:
                    cv2.imwrite("webcam.png", frame)

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def recognize_faces_of_identities_in_video(self, video: str, test_video_sampling_speed: int,
                                               starting_frame: int, confidence_level: int) -> pd.DataFrame:
        # For each identity in each sequence of the video, we count the number of 'appearances'.
        # We will return this dataframe at the end for accuracy analysis
        # Here we initialize all the identities for all sequences to 0.
        identities_count_per_sequence = pd.DataFrame(columns=["identity", "sequence", "appearances"])
        for i in range(0, len(self.__identities)):
            for j in range(0, len(self.__sequences)):
                identities_count_per_sequence.loc[len(identities_count_per_sequence)] = [self.__identities[i],
                                                                                         self.__sequences.loc[
                                                                                             j, "sequence"],
                                                                                         0]

        # We will keep track of how many times an identity was recognized over all sequences
        # Here we initialize these counts to 0.
        identities_count = list()
        for i in range(0, len(self.__identities)):
            identities_count.append(0)

        # we also keep track of the appearances of identities per frame...
        identity_timeline_appearances = pd.DataFrame(columns=["frame", "identity", "value", "color"])

        # We open the video to test how well the faces are recognized
        cap = cv2.VideoCapture(video)
        # We can control the starting frame for testing purposes
        frame_count = starting_frame
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

            # Process every other frame depending on the sampling speed
            if frame_count % test_video_sampling_speed == 0:
                # Loop through indentities to update the statistics for the timeline appearances
                for i in range(0, len(self.__identities)):
                    identity_timeline_appearances.loc[len(identity_timeline_appearances.index)] = [frame_count,
                                                                                                   self.__identities[i],
                                                                                                   1, False]

                # We identify the faces in the frame
                self.__identify_faces(frame,
                                      0.1 * frame_height,
                                      0.1 * frame_width,
                                      identities_count,
                                      identity_timeline_appearances,
                                      frame_count,
                                      identities_count_per_sequence,
                                      confidence_level)

                # If we've turned visualization on, we tell the visualizer to show the frame and the current stats
                if self.__visualize:
                    k = self.__visualizer.visualize(frame, identities_count, identity_timeline_appearances)
                    # space: 32
                    # 2 left arrow
                    # 3 right arrow
                    if k == 32:
                        self.pause(frame, identities_count, identity_timeline_appearances, cap, frame_count,
                                   frame_height, frame_width, identities_count_per_sequence, confidence_level)
                    if k == 25 or k == 27:
                        at_end = True

        # When everything done, release the video capture object
        cap.release()

        if self.__visualize:
            # Closes all the frames
            self.__visualizer.release()

        return identities_count_per_sequence

    def pause(self, frame, identities_count, identity_timeline_appearances, cap, frame_count, frame_height, frame_width,
              identities_count_per_sequence, confidence_level):
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
                                      identities_count_per_sequence,
                                      confidence_level)
            if k == 3:
                frame_count += 3
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                self.__identify_faces(frame,
                                      0.1 * frame_height,
                                      0.1 * frame_width,
                                      identities_count,
                                      identity_timeline_appearances, frame_count,
                                      identities_count_per_sequence,
                                      confidence_level)
            if k == 32:
                break

    def __identify_faces(self, img, minH, minW, identities_count, identity_timeline_appearances, frame_count,
                         identity_count_per_sequence, confidence_level):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 24 neighbors wont detect icons as faces
        faces = self.__faceCascade.detectMultiScale(gray, minNeighbors=24, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            user_id, confidence = self.__face_recognizer.predict(gray[y:y + h, x:x + w])
            # If confidence is below the threshold then, we've got a match
            if confidence < confidence_level:
                identities_count[user_id] = identities_count[user_id] + 1
                user_id = self.__identities[user_id]
                identity_timeline_appearances.loc[(identity_timeline_appearances["frame"] == frame_count) &
                                                  (identity_timeline_appearances["identity"] == user_id) &
                                                  (identity_timeline_appearances["value"] == 1),
                                                  "color"] = True
                # print(self.__sequences)
                seqnum = \
                    self.__sequences.query(
                        "startframe <= " + str(frame_count) + " and endframe >= " + str(frame_count))[
                        "sequence"].iloc[0]
                identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == user_id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                "appearances"] = identity_count_per_sequence.loc[
                                                                     (identity_count_per_sequence[
                                                                          "identity"] == user_id) &
                                                                     (identity_count_per_sequence[
                                                                          "sequence"] == seqnum),
                                                                     "appearances"] + 1

                if self.__visualize:
                    self.__visualizer.visualize_face_in_frame(img, user_id,
                                                              100 - ((confidence / confidence_level) * 100), x, y, w, h)
            else:
                # print("We are adding an unknown")
                identities_count[3] = identities_count[3] + 1
                user_id = "Unknown"
                identity_timeline_appearances.loc[(identity_timeline_appearances["frame"] == frame_count) &
                                                  (identity_timeline_appearances["identity"] == user_id) &
                                                  (identity_timeline_appearances["value"] == 1),
                                                  "color"] = True
                seqnum = \
                    self.__sequences.query(
                        "startframe <= " + str(frame_count) + " and endframe >= " + str(frame_count))[
                        "sequence"].iloc[0]
                identity_count_per_sequence.loc[(identity_count_per_sequence["identity"] == user_id) &
                                                (identity_count_per_sequence["sequence"] == seqnum),
                                                "appearances"] = identity_count_per_sequence.loc[
                                                                     (identity_count_per_sequence[
                                                                          "identity"] == user_id) &
                                                                     (identity_count_per_sequence[
                                                                          "sequence"] == seqnum),
                                                                     "appearances"] + 1
                if self.__visualize:
                    self.__visualizer.visualize_face_in_frame(img, user_id,
                                                              100 - ((confidence / confidence_level) * 100), x, y, w, h)

    def __identify_faces_in_webcam(self, img, minH, minW, identities_count, confidence_level):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__faceCascade.detectMultiScale(gray, minNeighbors=10, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            user_id, confidence = self.__face_recognizer.predict(gray[y:y + h, x:x + w])
            # If confidence is less them 100 ==> "0" : perfect match
            if confidence < confidence_level:
                identities_count[user_id] = identities_count[user_id] + 1
                user_id = self.__identities[user_id]
                self.__visualizer.visualize_face_in_frame(img, user_id,
                                                          100 - ((confidence / confidence_level) * 100), x, y, w, h)
            else:
                identities_count[3] = identities_count[3] + 1
                user_id = "Unknown"
                self.__visualizer.visualize_face_in_frame(img, user_id,
                                                          100 - ((confidence / confidence_level) * 100), x, y, w, h)
