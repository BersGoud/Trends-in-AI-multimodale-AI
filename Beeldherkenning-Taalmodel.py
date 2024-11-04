from flask import Flask, render_template, request, redirect
import os
import face_recognition
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

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
def load_celebrity_data():
    for index, row in df.iterrows():
        # Load the celebrity image
        image_path = row['profile_path']  # Adjust based on your dataset path
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(row['name'])
                known_face_biographies.append(row['biography'])
                known_face_images.append(image_path)

load_celebrity_data()

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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            # Save file locally
            file_path = "uploaded_image.jpg"
            file.save(file_path)

            # Load uploaded image
            unknown_image = face_recognition.load_image_file(file_path)
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)

            # Check if we found any faces in the image
            if unknown_face_encodings:
                # Compare with known faces
                for unknown_encoding in unknown_face_encodings:
                    results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
                    if True in results:
                        matched_index = results.index(True)
                        person_name = known_face_names[matched_index]
                        biography = known_face_biographies[matched_index]
                        image_path = known_face_images[matched_index]
                        description = generate_description(person_name)
                        return render_template("index.html", description=description, image_path=image_path, biography=biography)

            return render_template("index.html", description="No famous person recognized.", image_path=file_path)

    return render_template("index.html", description="", image_path="", biography="")

@app.route("/clear")
def clear():
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
