from flask import Flask, render_template, request, redirect
import torch
from PIL import Image
import face_recognition
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

app = Flask(__name__)

# Load GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text_model = GPT2LMHeadModel.from_pretrained('gpt2')
text_model.eval()

# Load your famous people's images and encodings
known_face_encodings = []
known_face_names = []

# Directory where you store famous people's images
dataset_dir = "famous_people_dataset"  # Change this to your dataset directory

# Load images and create encodings
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Name from the file

def generate_description(person_name):
    input_text = f"{person_name}. This is a famous historical figure. Provide a brief biography about this person."
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
                        description = generate_description(person_name)
                        return render_template("index.html", description=description, image_path=file_path)

            return render_template("index.html", description="No famous person recognized.", image_path=file_path)

    return render_template("index.html", description="", image_path="")

@app.route("/clear")
def clear():
    # Logic to clear session or uploaded data
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
