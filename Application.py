import Dataset_collector
import Trainer
import FaceRecognizer
import pandas as pd
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pylab as plt
import tabulate
matplotlib.use('TkAgg')

# The three  identities to be trained on
TRAINING_IDENTITIES = ["Leonie", "Remco", "Richard"]
# The three mp4 files of the three identities to be trained on
TRAINING_VIDEOS = ["./Video/AT_Leonie.mp4", "./Video/AT_Remco.mp4", "./Video/AT_Richard.mp4"]
# The separate training folder, in which the extracted training face shots are recorded
TRAINING_FOLDER = "training"
# The test video on which we test how well the identities are recognized and with which we perform stress-tests
TEST_VIDEO = "./Video/RHDHV#221207_Video_Analytics translator_V02.mp4"
# Determines how fast we go through the test video. 2 means we process every other frame, 3 means we process every
# other third frame. Etc.
TEST_VIDEO_SAMPLING_SPEED = 2
# Determines how many frames we are sampling from the training videos
NUMBER_OF_FRAMES_SAMPLED = 500


def run(webcam_testing:bool):
    """
    Main function for the overall flow of the application.
    """

    # We instantiate a Dataset collector object which will extract images from the training videos of the faces
    collector = Dataset_collector.Dataset_collector()
    # We check whether or not the faces are already collected from the videos
    if collector.are_training_faces_already_collected_from_videos(TRAINING_FOLDER, TRAINING_VIDEOS, NUMBER_OF_FRAMES_SAMPLED) == False:
        # This is not the case, so we collect the faces from the videos
        print("We need to collect the faces from the training videos...this will take a few minutes")
        collector.collect_training_faces_from_videos(TRAINING_FOLDER, TRAINING_VIDEOS, NUMBER_OF_FRAMES_SAMPLED)

    # Next up is instantiating the Trainer based and having it learn what the faces look like, in other words
    # learning the identity embeddings
    trainer = Trainer.Trainer()
    trainer.train(TRAINING_FOLDER, "trainer.yml", NUMBER_OF_FRAMES_SAMPLED)

    # Now we ware ready to put the trained model to the test. We instantiate the face_recognizer object
    # Are we webcam testing?
    if webcam_testing:
        print("We are webcam testing...")
        face_recognizer = FaceRecognizer.FaceRecognizer(TRAINING_IDENTITIES, "trainer.yml", get_sequences(), True)
        identities_count = face_recognizer.recognize_face_in_webcam()
    else:
        print("We are testing on a test video...")
        face_recognizer = FaceRecognizer.FaceRecognizer(TRAINING_IDENTITIES, "trainer.yml", get_sequences(), True)
        sequence_results = face_recognizer.recognize_faces_of_identities_in_video(TEST_VIDEO,
                                                                              TEST_VIDEO_SAMPLING_SPEED,
                                                                              1800)

        # We output the accuracy for this training frame count
        print(tabulate.tabulate(get_accuracy_table(sequence_results, TEST_VIDEO_SAMPLING_SPEED)))

def get_sequences():
    # The following triples represent the start frame, the end frame and the number of frames in between in which the
    # identities appear.
    sequences = pd.DataFrame(columns=["sequence", "startframe", "endframe", "identity"])
    sequences.loc[len(sequences)] = [1, 2, 934, "Remco"]
    sequences.loc[len(sequences)] = [2, 935, 2228, "Leonie"]
    sequences.loc[len(sequences)] = [3, 2229, 3492, "Richard"]
    sequences.loc[len(sequences)] = [4, 3493, 4000, "Blank"]
    return sequences


def run_hyper_parameter_1_test():
    print("Hyper parameter 1 stress-test: changing the frame training count")
    # We instantiate the trainer
    trainer = Trainer.Trainer()

    # We will test the variation in accuracy when we change the number of frames per identity that we train on
    frame_training_count = [20, 50, 100, 500]

    # We loop through the different frame training counts and for each train the trainer with the specified number
    # of training frames.
    for i in range(0, len(frame_training_count)):
        print("Frame training count = " + str(frame_training_count[i]))
        trainer.train(TRAINING_FOLDER, "trainer.yml", frame_training_count[i])

        # Now we are ready to put the trained model to the test
        # We instantiate the face_recognizer object and have it identify the faces in a video in which all three
        # trained faces appear
        face_recognizer = FaceRecognizer.FaceRecognizer(TRAINING_IDENTITIES, "trainer.yml", get_sequences(), False)
        sequence_results = face_recognizer.recognize_faces_of_identities_in_video(TEST_VIDEO,
                                                                                  TEST_VIDEO_SAMPLING_SPEED,
                                                                                  0)

        # We output the accuracy for this training frame count
        print(tabulate.tabulate(get_accuracy_table(sequence_results, TEST_VIDEO_SAMPLING_SPEED)))

def get_accuracy_table(sequence_results, test_video_sampling_speed: int):
    # We turn the sequence results into a table with the accuracy per sequence
    table = [["Sequence", "Accuracy"]]
    for index, row in get_sequences().iterrows():
        actual_appearances = (row["endframe"] - row["startframe"]) / test_video_sampling_speed
        seqnum = row["sequence"]
        identity = row["identity"]
        sequence_result = sequence_results.query("sequence == @seqnum and identity == @identity")
        if len(sequence_result) > 0:
            predicted_appearances = sequence_result["appearances"].item()
            table.append([row["sequence"], format(predicted_appearances / actual_appearances, "0.2%")])
    return table


def pick_handler(event):
     if isinstance(event.artist, PolyCollection):
        patch : PolyCollection = event.artist
        start_x = patch.get_paths()[0].vertices[0][0]
        end_x = patch.get_paths()[0].vertices[2][0]
        print('onpick1 patch: start:', start_x," end: ", end_x)

def visualizev2(identity_timeline_appearances : pd.DataFrame):
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('pick_event', pick_handler)

    ListOfIdentities = identity_timeline_appearances["identity"].drop_duplicates().tolist()
    ax.set_yticks(list(map(lambda x: 15+(x*10),[*range(0, len(ListOfIdentities))])), labels=ListOfIdentities)     # Modify y-axis tick labels

    bars = []
    # Loop through all identities
    for i, name in enumerate(ListOfIdentities):
        appearances_by_identities = identity_timeline_appearances.loc[(identity_timeline_appearances["identity"] == name)]
        # Get all points
        for index in appearances_by_identities.index:
            bars.append((appearances_by_identities["start_frame"][index],
                         appearances_by_identities["frame_amount"][index]))
        ax.broken_barh(bars, ((i+1)*10, 9), facecolors='tab:blue', picker=True)
        bars = []

    ax.set_ylim(5, 55)
    ax.set_xlim(0, 200)
    ax.set_xlabel('frames')
    ax.grid(False)                                       # Make grid lines visible

    plt.show()

# If this module is run, it will call the run function
if __name__ == "__main__":
    #run_hyper_parameter_1_test()
    run(False)

    
