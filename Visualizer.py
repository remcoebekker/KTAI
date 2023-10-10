import cv2
import matplotlib
import numpy as np
import matplotlib.pylab as plt
matplotlib.use('TkAgg')
import io
from PIL import Image
import plotly_express as px
import plotly.io as pio



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

        k = cv2.waitKey(1)

        return k

    def release(self):
        cv2.destroyAllWindows()

    def visualize_face_in_frame(self, img, identity, confidence, x, y, w, h):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        confidence = "  {0}%".format(round(confidence))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(identity), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)