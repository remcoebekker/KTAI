import cv2
import matplotlib
import numpy as np
import matplotlib.pylab as plt
matplotlib.use('TkAgg')
import io
from PIL import Image
import plotly_express as px
import plotly.io as pio
from matplotlib.collections import PolyCollection
import pandas as pd

class Visualizer:


    def __init__(self, identities):
        self.__identities = identities

        self.__fig1 = plt.figure(1)
        plt.xlabel("Identities")
        plt.ylabel("Frequency of appearance")
        plt.title("Identity appearance frequencies")
        plt.close(self.__fig1)

        self.__fig2 = plt.figure(2)
        plt.xlabel("Time")
        plt.ylabel("Identity")
        plt.title("Identity appearance in time")
        plt.close(self.__fig2)

    def fig2img(self, fig):
        img_bytes = pio.to_image(fig, format="png", scale=10)
        return img_bytes

    def visualize(self, frame, identity_frequencies, identity_timeline_appearances):
        frame = cv2.resize(frame, (1000, 640), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("video", frame)

        self.__fig1 = px.bar(x = self.__identities, y = identity_frequencies, labels = {
                     "x": "Identity",
                     "y": "Frequency of appearance"}, title = 'Frequency of appearance', template = "plotly_white")
        self.__fig1.update_traces(marker_color='green')
        fig1_bytes = self.__fig1.to_image(format="png", scale=1)
        buf = io.BytesIO(fig1_bytes)
        img1 = Image.open(buf)
        buf.seek(0)
        img1 = Image.open(buf)

        self.__fig2 = px.bar(identity_timeline_appearances, x="value", y="identity", orientation='h',
                             hover_data=[identity_timeline_appearances.index], height=300,
                             labels={
                                 "value": "Frame",
                                 "identity": "Identity"
                             }, title='Identity appearances in time', template="plotly_white")

        self.__fig2.update_traces(marker_color=['white' if i == False else 'green' for i in identity_timeline_appearances.color], showlegend=False)
        fig2_bytes = self.__fig2.to_image(format="png", scale=1)
        buf = io.BytesIO(fig2_bytes)
        img2 = Image.open(buf)
        buf.seek(0)
        img2 = Image.open(buf)

        cv2.imshow("stats1", np.asarray(img1))
        cv2.imshow("stats2", np.asarray(img2))
        cv2.imshow("video", frame)
        
        k = cv2.waitKey(50)

        if k == 115:
            cv2.imwrite("stats1.png", np.asarray(img1))
            cv2.imwrite("stats2.png", np.asarray(img2))
            cv2.imwrite("video.png", frame)

        return k

# WIP   
    def pick_handler(event):
        if isinstance(event.artist, PolyCollection):
            patch : PolyCollection = event.artist
            start_x = patch.get_paths()[0].vertices[0][0]
            end_x = patch.get_paths()[0].vertices[2][0]
            print('onpick1 patch: start:', start_x," end: ", end_x)
    # Visualization with matplot using broken bar chart 
    # Mabye usefull to intercept click event then jump tho frame?
    def visualizev2(identity_timeline_appearances : pd.DataFrame):
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('pick_event', Visualizer.pick_handler)

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

    def release(self):
        cv2.destroyAllWindows()

    def visualize_face_in_frame(self, img, identity, confidence, x, y, w, h):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        confidence = "  {0}%".format(round(confidence))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(identity), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)