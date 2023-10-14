import Dataset_collector
import Trainer
import FaceRecognizer
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pylab as plt
matplotlib.use('TkAgg')

def run():
    """
    Main function for the overall flow of the application.
    """


    # First step is learning the algorithm what the faces look like
    # We use three mp4 files of the three identities to be trained on
    training_identities = ["Leonie", "Remco", "Richard"]
    # The following triples represent the start frame, the end frame and the number of frames in between in which the
    # identities appear. Since we process every other frame, the frames and appearances are multiples of 2.
    sequences = pd.DataFrame(columns = ["sequence", "startframe", "endframe", "identity"])
    sequences.loc[len(sequences)] = [1, 2, 934, "Remco"]
    sequences.loc[len(sequences)] = [2, 936, 2228, "Leonie"]
    sequences.loc[len(sequences)] = [3, 2230, 3492, "Richard"]
    training_videos = ["./Video/AT_Leonie.mp4", "./Video/AT_Remco.mp4", "./Video/AT_Richard.mp4"]
    training_folder = "training"
    # We instantiate a Dataset collector object which will extract images from the training videos of the faces
    collector = Dataset_collector.Dataset_collector()
    #dataset = collector.collect_training_faces_from_videos(training_folder, training_videos)

    # Next up is instantiating the Trainer based ad having it learn what the faces look like, in other words
    # learning the identity embeddings
    trainer = Trainer.Trainer()
    trainer.train(training_folder, "trainer.yml")

    # Now we ware ready to put the trained model to the test
    # We instantiate the face_recognizer object and have it identify the faces in a video in which all three
    # trained faces appear
    face_recognizer = FaceRecognizer.FaceRecognizer(training_identities, "trainer.yml", sequences)
  #  identities_count = face_recognizer.recognize_face_in_webcam(training_identities)
    sequence_results = face_recognizer.recognize_faces_of_identities_in_video(
        "./Video/RHDHV#221207_Video_Analytics translator_V02.mp4")

    table = [["Sequence", "Accuracy"]]
    for index, row in sequences.iterrows():
        actual_appearances = (row["endframe"] - row["startframe"]) / 2
        seqnum = row["sequence"]
        identity = row["identity"]
        sequence_result = sequence_results.query("sequence == @seqnum and identity == @identity")
        if len(sequence_result) > 0:
            predicted_appearances = sequence_result["appearances"].item()
            table.append([row["sequence"], format(predicted_appearances / actual_appearances, "0.2%")])

    # print(tabulate.tabulate(table))
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
    run()
    
