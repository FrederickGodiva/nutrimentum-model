import pandas as pd
import joblib
from flask import jsonify

def food_recommendations(foodName, similarity_data, scaler,items, k=5):
    sim_scores = similarity_data[foodName]

    recommendations = (
        sim_scores
        .sort_values(ascending=False)
        .drop(foodName, errors='ignore')
        .head(k)
        .reset_index()
        .rename(columns={foodName: 'similarity_score'})
        .merge(items, on='item')
    )
    
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
    
    recommendations[nutrients_cols] = scaler.inverse_transform(recommendations[nutrients_cols])
    return recommendations

class foodBasedRecommendation:
        
    def __init__(self):
        self.cosine_sim_nutrients_df = joblib.load('models/cosine_sim_nutrients_df.joblib')
        self.cosine_sim_keyword_df = joblib.load('models/cosine_sim_keyword_df.joblib')
        self.scaler = joblib.load("models/scaler_nutrients.joblib")
        self.food_data = pd.read_csv("data/df_food_exported.csv")

    def __call__(self, foodName: str):
        try:
            nutrients_based_recommendation = food_recommendations(foodName, self.cosine_sim_nutrients_df, self.scaler, self.food_data)
            flavor_based_recommendation = food_recommendations(foodName, self.cosine_sim_keyword_df, self.scaler, self.food_data)

            nutrient_based_recommendation = nutrients_based_recommendation.to_dict(orient="records")
            flavor_based_recommendation = flavor_based_recommendation.to_dict(orient="records")

            result = {
                "nutrient_based": nutrient_based_recommendation,
                "flavor_based": flavor_based_recommendation
            }

            return jsonify(result) 
        except Exception as E:
            print(E)

