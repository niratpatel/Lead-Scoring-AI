# model_loader.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class HighPerformanceLeadScoringModel:
    def __init__(self, random_state=42): self.random_state = random_state; self.pipeline = None
    def create_features(self, df):
        df_processed = df.copy()
        df_processed['lead_date'] = pd.to_datetime(df_processed['lead_date'])
        df_processed['days_since_creation'] = (datetime.now() - df_processed['lead_date']).dt.days
        df_processed['submission_to_view_ratio'] = df_processed['form_submissions'] / (df_processed['page_views'] + 1)
        df_processed['income_x_credit'] = (df_processed['income_inr'] / 1_000_000) * df_processed['credit_score']
        df_processed['total_engagement'] = (df_processed['page_views'] + df_processed['form_submissions'] * 2 + df_processed['previous_interactions'])
        return df_processed
    def predict_score(self, X_new):
        probabilities = self.pipeline.predict_proba(X_new)[:, 1]; return probabilities * 100

class EnhancedReranker:
    def __init__(self):
        self.intent_keywords = { 'ready to buy': 25, 'book now': 25, 'immediate': 20, 'loan pre-approved': 20, 'urgent': 18, 'booking': 18, 'schedule a call': 15, 'site visit': 15, 'final price': 12, 'interested': 8, 'budget': 8, 'details': 5, 'floor plan': 5, 'looking for': 3, 'not interested': -30, 'just browsing': -25, 'researching': -20, 'too expensive': -20, 'not sure': -15, 'just checking': -15, 'later': -10 }

    def get_reranked_score(self, comment_text, initial_score):
        if pd.isna(comment_text):
            return initial_score
        adjustment = sum(s for kw, s in self.intent_keywords.items() if kw in comment_text.lower())
        capped_adjustment = max(min(adjustment, 25), -25)
        return np.clip(initial_score + capped_adjustment, 0, 100)

model = None
reranker = None

def load_models():
    global model, reranker
    try:
        model = joblib.load("high_performance_model.pkl")
        reranker = EnhancedReranker()
        print("Models loaded successfully via model_loader.")
        return True
    except Exception as e:
        print(f"FATAL ERROR in model_loader: Could not load model. Error: {e}")
        return False

def get_scores(lead_data):
    if not model or not reranker: raise Exception("Models are not loaded.")
    
    lead_df = pd.DataFrame([lead_data])
    
    # Add server-side columns
    lead_df['lead_date'] = pd.to_datetime(datetime.now())
    lead_df['day_of_week'] = datetime.now().strftime('%A')
    lead_df['hour_of_day'] = datetime.now().hour
    lead_df['last_interaction_channel'] = 'Website'
    
    # Predict
    X_new = model.create_features(lead_df)
    initial_score = model.predict_score(X_new)[0]

    final_score = reranker.get_reranked_score(lead_data['comments'], initial_score)
    
    return initial_score, final_score
