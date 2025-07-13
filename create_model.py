# create_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import logging

# --- THIS IS THE CRITICAL STEP ---
# We import the class from the file where it will live permanently.
# This "bakes" the correct location into the saved .pkl file.
from model_loader import HighPerformanceLeadScoringModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    This script trains the model and saves it with the correct import path,
    solving the joblib loading error permanently.
    """
    logger.info("--- Starting Model Training and Creation Process ---")
    df = pd.read_csv('lead_scoring_data_enhanced.csv')
    
    # We create an instance of the class we imported
    model = HighPerformanceLeadScoringModel()
    
    logger.info("Starting training process...")
    features_df = model.create_features(df)
    X = features_df.drop(columns=['high_intent', 'email', 'phone_number', 'comments', 'lead_date'], errors='ignore')
    y = features_df['high_intent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Define pipeline directly here for simplicity
    continuous_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), continuous_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))]), categorical_features)
    ], remainder='passthrough')

    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        threshold='median'
    )

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=1)),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=150, max_depth=5, min_samples_leaf=10)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=3, learning_rate=0.05))
        ],
        voting='soft', weights=[1, 2, 2]
    )
    
    # Assign the trained pipeline to the model instance
    model.pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', feature_selector),
        ('classifier', voting_clf)
    ])
    
    logger.info("Training the full pipeline...")
    model.pipeline.fit(X_train, y_train)

    # --- SAVE THE MODEL ---
    # Now, when we save `model`, joblib records that its class
    # `HighPerformanceLeadScoringModel` is found in the file `model_loader`.
    joblib.dump(model, 'high_performance_model.pkl')
    logger.info("✅ Model has been successfully created and saved with the correct import path.")
    logger.info("You can now run the FastAPI app.")

if __name__ == "__main__":
    main()