# KTAI
Repo for face recognition assingment

Done voor 17-10:
1. Introspectie van code + overleaf inclusief verbeteringen / uitbreidingen (Matthew)
2. Check op gebruik andere face recognizer dan in opdracht tekst (Remco) -> Antwoord is dat we vrij zijn om te kiezen.
4. Checken of we voldoen aan alle requirements uit de opdrachtbeschrijving (Remco)
    - Code kan opgeleverd worden via een Public Github Repo; beeldmateriaal zullen we dan separaat moeten delen met de 
      docenten; to do: Public Githup repo maken
    - Overleaf report van maximaal 5 pagina's (exclusief bijlagen)
      - To do: toevoegen Short description of the problem (including **historical context**) and the starting material
      - To do: toevoegen **overview of the related literature on this subject**.
      - To do: toevoegen A discussion section in which you summarize the results, **compare the results with the literature 
        from the introduction, indicate possible limitations of your work, and provide suggestions for future research**
      - To do: **List the references that you used in the report.**
   - Er staat niet keihard dat het in Jupyter notebook moet draaien, maar lijkt me toch verstandig om daarvan uit te 
     gaan. Als je in Anaconda de juiste omgeving creëert dan draait de applicatie in Jupyter notebook.
      - To do: de windows worden niet automatisch naar de voorgrond geduwd en de afmetingen zijn nog niet helemaal goed
      - To do: de juiste Anaconda omgeving aanmaken en deze als back-up saven en in github zetten
      - To do: een keer een volledige test te doen: van het opbouwen van de omgeving t/m draaien in Jupyter notebook
    - Onder de benodigde libraries wordt PyTorch genoemd voor het bouwen van deep neural networks. Ik heb de vraag uitgezet       bij Daniel of we inderdaad ons eigen deep neural network moeten bouwen. -> Antwoord is dat we vrij zijn om te kiezen.
6. Analyseren van 1 opvallend gek resultaat (Matthew)
7. Spelen met 1 hyper parameter (aantal frames voor training) en effect accuracy (Remco)

Done voor 20-10:
1. Lijstje bedenken met experimenten die we uit willen gaan voeren in de komende 4 weken (Remco):
   - Icoontjes / noise voor gezichten en robuustheid checken
   - Met Webcam spelen: jij zit niet in de videos: wanneer herkent het algoritme jou als 1 van de identiteiten en als gezicht --> robuustheid
   - Met Webcam spelen: ik zit wel in de videos: wanneer herkent het algoritme mij niet meer als 1 van identiteiten: zonnebril, petje, bril op
3. Afmaken eerste versie van het overleaf rapport:
   - Achtergrond van opencv / haar cascades (Matthew)
   - Opnemen referentie met 8 verbeterpunten voor optimalisatie trainingsmateriaal (Matthew)

To-list voor 17-11:
1. Verwerken feedback
2. Analyseren van de 3 opvallende gekke resultaten (Matthew)
3. Uitvoeren lijst met experimenten
   - Icoontjes / noise voor gezichten en robuustheid checken
   - Met Webcam spelen: jij zit niet in de videos: wanneer herkent het algoritme jou als 1 van de identiteiten en als g            gezicht --> robuustheid
   - Met Webcam spelen: ik zit wel in de videos: wanneer herkent het algoritme mij niet meer als 1 van identiteiten:               zonnebril, petje, bril op
4. Spelen met extra hyper parameters
5. Toevoegen historical context aan report
6. Toevoegen related literature aan report
7. Toevoegen vergelijking resultaten met literatuur
8. Zorgen dat de juiste Anacaonda omgeving als back-up in github staat (Remco)
9. Volledige end-to-end test doen (Remco)
10. Zorgen dat de windows naar de voorgrond worden geduwd en dat de sizing goed is
...

Notes:
Op het moment gebruiken we dit algoritme om gezichten te herkennen: LBPHFaceRecognizer.
Meer info hierover: https://pyimagesearch.com/2021/05/03/face-recognition-with-local-binary-patterns-lbps-and-opencv/
https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html


