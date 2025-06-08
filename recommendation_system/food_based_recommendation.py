import pandas as pd
import joblib
from flask import Flask, request, jsonify

def recipe_recommendations(foodName, similarity_data, items, k=5):
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
    return recommendations

class foodBasedRecommendation:
        
    def __init__(self):
        self.cosine_sim_nutrients_df = joblib.load('model/cosine_sim_nutrients_df.joblib')
        self.cosine_sim_keyword_df = joblib.load('model/cosine_sim_keyword_df.joblib')
        self.food_data = pd.read_csv("data/df_food_exported.csv")

    def __call__(self, foodName: str):
        try:
            nutrients_based_recommendation = recipe_recommendations(foodName, self.cosine_sim_nutrients_df, self.food_data[["item"]])
            flavor_based_recommendation = recipe_recommendations(foodName, self.cosine_sim_keyword_df, self.food_data[['item', 'flavor_profile']])

            nutrient_based_recommendation = nutrients_based_recommendation.to_dict(orient="records")
            flavor_based_recommendation = flavor_based_recommendation.to_dict(orient="records")

            result = {
                "nutrient_based": nutrient_based_recommendation,
                "flavor_based": flavor_based_recommendation
            }

            return jsonify(result) 
        except Exception as E:
            print(E)

