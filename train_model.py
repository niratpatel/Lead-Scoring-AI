import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# --- IMPORT ADDED HERE ---
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighPerformanceLeadScoringModel:
    """
    This is a high-performance model designed to maximize predictive power on noisy data.
    It uses a sophisticated pipeline with three key stages:
    1. Advanced Feature Engineering.
    2. Automatic Feature Selection to isolate the strongest signals.
    3. A weighted Ensemble (Voting Classifier) for robust and accurate predictions.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.selected_features = None
        self.score_thresholds = {'hot': 80, 'warm': 60, 'cold': 40, 'nurture': 20}

    def create_features(self, df):
        df_processed = df.copy()
        df_processed['profession'] = df_processed['profession'].fillna('Other')
        df_processed['income_inr'] = df_processed.groupby('profession')['income_inr'].transform(lambda x: x.fillna(x.median()))
        df_processed['credit_score'] = df_processed.groupby('profession')['credit_score'].transform(lambda x: x.fillna(x.median()))
        df_processed.fillna({'family_background': 'Unknown', 'property_type_preference': 'No_Preference', 'income_inr': df['income_inr'].median()}, inplace=True)
        df_processed['lead_date'] = pd.to_datetime(df_processed['lead_date'])
        df_processed['days_since_creation'] = (datetime.now() - df_processed['lead_date']).dt.days
        df_processed['submission_to_view_ratio'] = df_processed['form_submissions'] / (df_processed['page_views'] + 1)
        df_processed['income_x_credit'] = (df_processed['income_inr'] / 1_000_000) * df_processed['credit_score'].fillna(df['credit_score'].median())
        df_processed['total_engagement'] = (df_processed['page_views'] + df_processed['form_submissions'] * 2 + df_processed['previous_interactions'])
        return df_processed

    def train(self, df, target_col='high_intent'):
        logger.info("Starting training process for the high-performance model...")
        features_df = self.create_features(df)
        X = features_df.drop(columns=[target_col, 'email', 'phone_number', 'comments', 'lead_date'], errors='ignore')
        y = features_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=self.random_state, stratify=y)
        
        preprocessor = self._create_preprocessor(X_train)

        feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            threshold='median'
        )

        voting_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=self.random_state, class_weight='balanced', solver='liblinear', C=1)),
                ('rf', RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_estimators=150, max_depth=5, min_samples_leaf=10)),
                ('gb', GradientBoostingClassifier(random_state=self.random_state, n_estimators=150, max_depth=3, learning_rate=0.05))
            ],
            voting='soft', weights=[1, 2, 2]
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('classifier', voting_clf)
        ])

        logger.info("Training the full pipeline (preprocess -> select features -> train ensemble)...")
        self.pipeline.fit(X_train, y_train)

        self._evaluate(X_test, y_test)
        self.save()

    def _create_preprocessor(self, X_sample):
        continuous_features = X_sample.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_sample.select_dtypes(include=['object']).columns.tolist()
        return ColumnTransformer(transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), continuous_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))]), categorical_features)
        ], remainder='passthrough')

    def predict_score(self, X_new):
        probabilities = self.pipeline.predict_proba(X_new)[:, 1]
        return probabilities * 100

    def _evaluate(self, X_test, y_test):
        logger.info("--- Evaluating Final Model Performance ---")
        final_scores = self.predict_score(X_test)
        
        test_auc = roc_auc_score(y_test, final_scores)
        y_pred_binary = (final_scores >= 50).astype(int)
        
        # --- CALCULATE ALL METRICS HERE ---
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        # --- LOG ALL METRICS HERE ---
        logger.info(f"\n--- Final Model Performance ---\n"
                    f"Test Accuracy: {accuracy:.4f}\n"
                    f"Test AUC: {test_auc:.4f}\n"
                    f"Test Precision (at score > 50): {precision:.4f}\n"
                    f"Test Recall (at score > 50): {recall:.4f}\n"
                    f"Test F1-Score (at score > 50): {f1:.4f}")
        
    def save(self, filepath='high_performance_model.pkl'):
        joblib.dump(self, filepath)
        logger.info(f"High-performance model saved to {filepath}")

class EnhancedReranker:
    def __init__(self):
        self.intent_keywords = {
            'ready to buy': 25, 'book now': 25, 'immediate': 20, 'loan pre-approved': 20, 'urgent': 18,
            'booking': 18, 'schedule a call': 15, 'site visit': 15, 'final price': 12, 'interested': 8,
            'budget': 8, 'details': 5, 'floor plan': 5, 'looking for': 3, 'not interested': -30,
            'just browsing': -25, 'researching': -20, 'too expensive': -20, 'not sure': -15,
            'just checking': -15, 'later': -10
        }
    
    def rerank(self, df, ml_scores):
        adjustments = df['comments'].apply(lambda c: max(min(sum(s for kw, s in self.intent_keywords.items() if pd.notna(c) and kw in c.lower()), 25), -25))
        return np.clip(ml_scores + adjustments, 0, 100), adjustments

def main():
    logger.info("--- Starting High-Performance Lead Scoring Pipeline ---")
    df = pd.read_csv('lead_scoring_data_enhanced.csv')
    
    model = HighPerformanceLeadScoringModel()
    model.train(df)
    
    logger.info("\n--- Testing Reranking on Sample Data ---")
    reranker = EnhancedReranker()
    sample_df = df.sample(15, random_state=42).reset_index(drop=True)
    
    X_sample = model.create_features(sample_df)
    ml_scores = model.predict_score(X_sample)
    
    final_scores, adjustments = reranker.rerank(sample_df, ml_scores)
    
    recs = ['HOT' if s >= 80 else 'WARM' if s >= 60 else 'COLD' for s in final_scores]

    results = pd.DataFrame({
        'Comments': sample_df['comments'].str[:40], 'ML_Score': ml_scores.round(1),
        'Adj': adjustments.values, 'Final_Score': final_scores.round(1), 'Recommendation': recs
    })
    
    logger.info("\nSample Predictions with Reranking:\n" + results.to_string(index=False))
    logger.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()