import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from recommendation_system import foodBasedRecommendation

app = Flask(__name__)
CORS(app)

class_labels = ['sate', 'mie_goreng', 'bakso', 'sop_buntut', 'martabak_telur', 'rawon', 'nasi_goreng', 'cumi_goreng',
                'ayam_semur', 'opor_ayam', 'ayam_bakar', 'bubur', 'rendang', 'nasi_uduk', 'iga_bakar', 'telur_rebus',
                'gado_gado', 'telur_dadar', 'ayam_goreng', 'nasi_tumpeng', 'ikan_goreng', 'gulai_ikan', 'soto']
# nutrition_data = pd.read_csv("nutritions.csv")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/classification', methods=['POST'])
def food_image_classification():
    try:
        print("Request received")
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        model = load_model("classification-model.keras")

        img = load_img(file_path, target_size=(299, 299))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        print(predicted_label)

        # nutrition = nutrition_data[nutrition_data['name'].str.lower().str.strip() == predicted_label.lower().strip()]
        # if nutrition.empty:
        #     return jsonify({"error": f"Nutrition data not found for {predicted_label}"}), 404
        #
        # nutrition_dict = nutrition.iloc[0].to_dict()

        return jsonify(
            {"predicted_label": predicted_label, "confidence": float(confidence),  # "nutritions": nutrition_dict
             })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["POST"])
def tes_app():
    return jsonify({"message": "Hello World"})

@app.route("/recommendations/content-based", methods=["POST"])
def food_recommendation():
    try : 
        foodName = request.get_json().get('foodName')
        food_recommendation_system = foodBasedRecommendation()
        recommendation = food_recommendation_system(foodName = foodName)
        return recommendation

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080)
