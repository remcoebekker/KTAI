This document describes the steps that need to be taken to get the software running that has been needed to produce the report.

1. Start-up Anaconda
2. Import in Anaconda the "KTAI_FaceRecognition.yaml" file which sets up the right environment for the software. Alternatively, you could set up a new environment in Anaconda, in which case the following steps need to be performed:
a. Create new environment
b. Open terminal
c. pip install opencv-contrib-python
d. conda install -c conda-forge notebook
e. pip install Pillow
f. pip install matplotlib
g. pip install plotly-express
h. pip install tabulate
4. Download the the files in the repository and put them in a folder of your own choice, for instance "KTAI - Face recognition"
5. Under this folder create 2 new sub folders: "Training" and "Video"
6. You have received a link to the training and testing videos. Download these videos (4 in total) and store them in the "Video" sub folder
7. In Anaconda click on the newly imported (or created) environment and select "Open with Jupyter Notebook"
8. In the Jupyter Notebook environment navigate towards the folder just created, for instance "KTAI - Face recognition"
9. Click on "KTAI.ipynb"
10. First execute the "import Application" and "Application.run(False)" lines
11. After a few minutes of setting up, three new windows will be created (you may have to click on them bring them to the front) in which you will see the application in action: extracting face shots from the training material, then training the model on the training material and then testing the model on the testing video while displaying the frequency of recognized identities and the number of identities recognized in time. Afterwards it outputs the accuracy output per sequence.
12. Next execute the "import Application" and "Application.run_hyper_parameter_1_test()" lines
13. After a few minutes of setting up, it will output the accuracy of identity recognition for different numbers of face shots being used for training purposes.
14. Finally, execute the "import Application" and "Application.run(True)" lines
15. After a few minutes of setting up, it will show a webcam window, that will show the detection of your own face and whether or not it recognizes your face as one of the identities trained on. If you want to stop testing, click on the Esc button.

To-list voor 17-11:
2. Uitleggen van algoritmes (Matthew)
3. Met Webcam spelen: jij zit niet in de videos: wanneer herkent het algoritme jou als 1 van de identiteiten en als g            gezicht --> robuustheid (Matthew)
5. Clean up the code and document
6. End-to-end test
7. ReadMe afmaken

Nice-to-have
8. Spelen met extra hyper parameters
9. Toevoegen vergelijking resultaten met literatuur
10. Zorgen dat de windows naar de voorgrond worden geduwd en dat de sizing goed is

Done voor 17-11:
1. Analyseren van de 3 opvallende gekke resultaten (Matthew)
2. Zorgen dat de juiste Anacaonda omgeving als back-up in github staat (Remco)
3. Volledige end-to-end test doen (Remco)
4. Verwerken feedback (opening aanpassen, stuk verplaatsen naar methods, onderzoeksvragen, code methodes eruit) (Remco)
5. Met Webcam spelen: ik zit wel in de videos: wanneer herkent het algoritme mij niet meer als 1 van identiteiten:               zonnebril, petje, bril op (Remco)

Notes:
Op het moment gebruiken we dit algoritme om gezichten te herkennen: LBPHFaceRecognizer.
Meer info hierover: https://pyimagesearch.com/2021/05/03/face-recognition-with-local-binary-patterns-lbps-and-opencv/
https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
