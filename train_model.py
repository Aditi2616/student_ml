import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_model():
    print("Loading data...")
    try:
        df = pd.read_csv('student_data.csv')
    except FileNotFoundError:
        print("student_data.csv not found. Please run generate_dataset.py first.")
        return

    X = df.drop('ExamScore', axis=1)
    y = df['ExamScore']

    # Define categorical and numerical features
    categorical_features = ['InternetAccess', 'Extracurricular', 'StressLevel', 'Motivation']
    numerical_features = ['Age', 'StudyHours', 'Attendance', 'Assignments']

    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the Random Forest Regressor...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print(f"Model Evaluation Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

    # Feature Importance (Extracting from the pipeline)
    print("Extracting Feature Importance...")
    try:
        # Get feature names after one-hot encoding
        cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        all_features = numerical_features + list(cat_features)
        
        # Get importances
        rf_model = model.named_steps['regressor']
        importances = rf_model.feature_importances_
        
        # Save feature importance for visualization in Streamlit
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        print("Saved feature_importance.csv.")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    # Save the pipeline
    joblib.dump(model, 'model.pkl')
    print("Pipieline saved successfully to model.pkl.")

if __name__ == "__main__":
    train_and_save_model()
