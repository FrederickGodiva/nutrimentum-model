import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_goal_cosine_similarity(scaler, user_goal, df_goal):
    # scale the goals
    userGoals = scaler.transform([user_goal])

    # create user goals
    goals_cols = ['weight_management', 'muscle_development', 'energy_boost', 'heart_health', 'immunity_strength']
    df_user_goals = pd.DataFrame(userGoals, columns=goals_cols)
    df_user_goals['item'] = 'userGoals'

    df_temp = df_goal.copy()
    df_temp = pd.concat([df_temp, df_user_goals], ignore_index=True)

    goals_matrix = df_temp.drop(columns=['item'])

    cosine_sim_goals = cosine_similarity(goals_matrix)
    cosine_sim_goals_df = pd.DataFrame(cosine_sim_goals, index=df_temp['item'], columns=df_temp['item'])

    return cosine_sim_goals_df

def food_recommendations(foodName, similarity_data, items, scaler, k=5):
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

    goals_cols = ['weight_management', 'muscle_development', 'energy_boost', 'heart_health', 'immunity_strength']
    recommendations[goals_cols] = scaler.inverse_transform(recommendations[goals_cols])
    return recommendations

class goalBasedRecommendation:
        
    def __init__(self):
        self.scaler = joblib.load('models/scaler_goals.joblib')
        self.df_goals = joblib.load('models/df_goals.joblib')
        self.food_data = pd.read_csv("data/df_food_exported.csv")

    def __call__(self, userGoal:list):
        try:
            cosine_sim_goals = create_goal_cosine_similarity(self.scaler, userGoal, self.df_goals)
            recommendation =  food_recommendations("userGoals", cosine_sim_goals, self.food_data, self.scaler)

            return recommendation.to_json(orient="records")
        except Exception as E:
            print(E)