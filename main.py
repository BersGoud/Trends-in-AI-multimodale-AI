# main.py
from Beeldherkenning import BeeldHerkenning
from Taalmodel import TaalModel

def main(image_path):
    # Initialiseer beeldherkenning en taalmodel
    beeldherkenning = BeeldHerkenning()
    taalmodel = TaalModel()
    
    # Extract features van de afbeelding
    kenmerken = beeldherkenning.extract_features(image_path)
    print("Extracted features:", kenmerken)

    # Genereer beschrijving op basis van de kenmerken
    beschrijving = taalmodel.genereer_beschrijving(kenmerken)
    print("Gegenereerde beschrijving:", beschrijving)

if __name__ == "__main__":
    # Voorbeeldafbeelding pad (vervang met je eigen afbeeldingsbestand)
    image_path = "test.jpg"
    main(image_path)