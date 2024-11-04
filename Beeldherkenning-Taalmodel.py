from flask import Flask, render_template, request, redirect
import os
import face_recognition
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from PIL import Image  # Import Pillow

app = Flask(__name__)

# Load GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text_model = GPT2LMHeadModel.from_pretrained('gpt2')
text_model.eval()

# Load TMDB Celeb 10K dataset
df = pd.read_parquet("hf://datasets/ashraq/tmdb-celeb-10k/data/train-00000-of-00001-d95dffd623223e73.parquet")

# Prepare for face recognition
known_face_encodings = []
known_face_names = []
known_face_biographies = []
known_face_images = []

# Function to load celebrity images and names for face recognition
BASE_IMAGE_PATH = "static/uploads"

# Functie om celebrity data te laden
def load_celebrity_data():
    for index, row in df.iterrows():
        image_filename = row['profile_path']  # Dit moet het bestandspad zijn zoals 'idYjWWH7LbXFxQlqJes0DWbztpf.jpg'
        image_path = os.path.join(BASE_IMAGE_PATH, image_filename)  # Combineer met de basisdirectory
        print(f"Trying to load image from: {image_path}")  # Log het pad
        if os.path.exists(image_path):  # Zorg ervoor dat het bestand bestaat
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(row['name'])
                known_face_biographies.append(row['biography'])
                known_face_images.append(image_path)
                print(f"Loaded {row['name']}'s image and encoding.")
            else:
                print(f"No face encodings found for {row['name']}.")
        else:
            print(f"Image not found for {row['name']} at path: {image_path}.")

def generate_description(person_name):
    input_text = f"{person_name}. Dit is een beroemd figuur. Geef hier een korte biografie over dit persoon en zijn werk."
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        output = text_model.generate(
            input_ids, 
            max_length=150, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True
        )
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    return description

def compress_image(image_path):
    """Compress the image to 423x424 pixels."""
    with Image.open(image_path) as img:
        # Resize the image using LANCZOS filter
        img = img.resize((423, 424), Image.LANCZOS)
        img.save(image_path)  # Save the image back

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            # Sla de geüploade afbeelding op in de static/uploads map
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, "uploaded_image.jpg")
            file.save(file_path)

            # Compress the uploaded image
            compress_image(file_path)  # Compress the image to 423x424

            # Laad de geüploade afbeelding
            unknown_image = face_recognition.load_image_file(file_path)
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)

            # Debug output
            print(f"Found {len(unknown_face_encodings)} face encodings in the uploaded image.")

            if unknown_face_encodings:
                for unknown_encoding in unknown_face_encodings:
                    print(f"Unknown Encoding: {unknown_encoding}")  # Print de onbekende encoding
                    results = face_recognition.compare_faces(known_face_encodings, unknown_encoding, tolerance=0.5)  # Pas tolerantie aan
                    print(f"Comparison Results: {results}")  # Log de vergelijking resultaten
                    if True in results:
                        matched_index = results.index(True)
                        person_name = known_face_names[matched_index]
                        biography = known_face_biographies[matched_index]
                        image_path = known_face_images[matched_index]
                        description = generate_description(person_name)
                        return render_template("index.html", description=description, image_path=image_path, biography=biography)

                return render_template("index.html", description="Geen beroemde persoon herkend in de afbeelding.", image_path=file_path)

            else:
                return render_template("index.html", description="Geen gezichten gevonden in de afbeelding.", image_path=file_path)

    return render_template("index.html", description="", image_path="", biography="")

@app.route("/clear")
def clear():
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)