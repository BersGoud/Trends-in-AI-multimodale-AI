from Beeldherkenning import BeeldHerkenning
from Taalmodel import TaalModel

def main(image_paths):
    beeldherkenning = BeeldHerkenning()
    taalmodel = TaalModel()

    feature_list = []

    # Loop door de opgegeven afbeeldingspaden
    for image_path in image_paths:
        try:
            kenmerken, labels = beeldherkenning.extract_features(image_path)  # Get both features and labels
            feature_list.append(kenmerken)
        except FileNotFoundError:
            print(f"Afbeelding niet gevonden: {image_path}")
            continue  # Ga verder met de volgende afbeelding

    # Controleer of er voldoende kenmerken zijn om PCA te passen
    if len(feature_list) < 2:
        print("Niet genoeg afbeeldingen om PCA toe te passen. Zorg voor ten minste twee afbeeldingen.")
        return  # Stop de uitvoering

    # Train de PCA op de verzamelde features
    beeldherkenning.fit_pca(feature_list)

    laatste_afbeelding = image_paths[-1]
    kenmerken, labels = beeldherkenning.extract_features(laatste_afbeelding)

    # Genereer beschrijving (pass only features if labels are not used)
    beschrijving = taalmodel.genereer_beschrijving(kenmerken)
    print("Gegenereerde beschrijving:", beschrijving)

if __name__ == "__main__":
    # Voorbeeldafbeeldingen (zorg ervoor dat deze bestanden bestaan)
    image_paths = ["test.jpg", "test2.jpg", "test3.jpg"]
    main(image_paths)