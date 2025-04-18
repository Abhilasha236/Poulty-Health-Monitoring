from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("best_model.keras")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# You may have class names like this:
class_names = ['Coccidiosis', 'Healthy','Salmonella' ,'Newcastle Disease']  # change as per your dataset

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Preprocess the image as per your training
            img = image.load_img(img_path, target_size=(224, 224))  # change size if needed
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)
            prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
