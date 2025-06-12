import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from recommendation_system import foodBasedRecommendation, goalBasedRecommendation

app = Flask(__name__)
CORS(app)

scaler = joblib.load("models/scaler_nutrients.joblib")

class_labels = ['ayam_bakar', 'ayam_goreng', 'ayam_semur', 'bakso', 'bubur', 'cumi_goreng', 'gado_gado', 'gulai_ikan', 'iga_bakar', 'ikan_goreng', 'martabak_telur', 'mie_goreng', 'nasi_goreng', 'nasi_tumpeng', 'nasi_uduk', 'opor_ayam', 'rawon', 'rendang', 'sate', 'sop_buntut', 'soto', 'telur_dadar', 'telur_rebus']

nutrition_data = pd.read_csv("./data/df_food_exported.csv")

nutrients_cols = ['kalori',
    'protein',
    'lemak',
    'karbohidrat',
    'Cholesterol_g',
    'saturated_fat_g',
    'fiber_g',
    'sugars_g',
    'sodium_mg',
    'iron_mg',
    'zinc_mg',
    'calcium_mg',
    'vitamin_b12_mcg',
    'vitamin_a_mcg',
    'vitamin_b_mcg',
    'vitamin_c_mcg',
    'vitamin_d_mcg',
    'vitamin_e_mcg']

nutrition_data[nutrients_cols] = scaler.inverse_transform(nutrition_data[nutrients_cols])

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

        model = load_model("./models/classification-model.keras")

        img = load_img(file_path, target_size=(299, 299))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        print(predicted_label)

        normalized_label = predicted_label.replace("_", " ").lower().strip()

        nutrition = nutrition_data[nutrition_data['item'].str.lower(
        ).str.replace("_", " ").str.strip() == normalized_label]

        if nutrition.empty:
            return jsonify({"error": f"Nutrition data not found for {predicted_label}"}), 404

        nutrition_dict = nutrition.iloc[0].to_dict()

        return jsonify(
            {"predicted_label": predicted_label, "confidence": float(confidence),  "nutritions": nutrition_dict
             })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["POST"])
def tes_app():
    return jsonify({"message": "Hello World"})


@app.route("/recommendations/content-based", methods=["POST"])
def food_recommendation():
    try:
        foodName = request.get_json().get('foodName')
        food_recommendation_system = foodBasedRecommendation()
        recommendation = food_recommendation_system(foodName=foodName)
        return recommendation

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/recommendations/goal-based", methods=["POST"])
def goal_based_recommendation():
    try:
        userGoal = request.get_json().get('userGoal')
        food_recommendation_system_based_goal = goalBasedRecommendation()
        recommendation = food_recommendation_system_based_goal(
            userGoal=userGoal)
        return recommendation

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080)
