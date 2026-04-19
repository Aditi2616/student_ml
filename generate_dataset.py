import pandas as pd
import numpy as np

def generate_data(num_samples=1000):
    np.random.seed(42)
    
    # 1. Base Variables
    age = np.random.randint(15, 23, size=num_samples)
    study_hours = np.random.uniform(0, 10, size=num_samples)  # 0 to 10 hours per day
    attendance = np.random.uniform(50, 100, size=num_samples) # 50% to 100%
    assignments = np.random.uniform(0, 100, size=num_samples) # Assignment completion rate
    
    # Categoricals
    internet_access = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.85, 0.15])
    extracurricular = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.6, 0.4])
    stress_level = np.random.choice(['Low', 'Medium', 'High'], size=num_samples, p=[0.3, 0.5, 0.2])
    motivation = np.random.choice(['Low', 'Medium', 'High'], size=num_samples, p=[0.2, 0.5, 0.3])

    # 2. Score Calculation Strategy
    # Base score
    score = np.random.normal(40, 5, size=num_samples)
    
    # Additive effects
    score += study_hours * 2.5
    score += (attendance - 50) * 0.4
    score += assignments * 0.15
    
    # Categorical Effects
    int_effect = np.where(internet_access == 'Yes', np.random.normal(5, 2, num_samples), np.random.normal(-5, 2, num_samples))
    extra_effect = np.where(extracurricular == 'Yes', np.random.normal(2, 1, num_samples), 0)
    
    stress_map = {'Low': 5, 'Medium': 0, 'High': -8}
    stress_effect = pd.Series(stress_level).map(stress_map).values + np.random.normal(0, 2, num_samples)
    
    motivation_map = {'Low': -5, 'Medium': 0, 'High': 7}
    mot_effect = pd.Series(motivation).map(motivation_map).values + np.random.normal(0, 2, num_samples)
    
    score = score + int_effect + extra_effect + stress_effect + mot_effect
    
    # Add some noise
    score += np.random.normal(0, 4, num_samples)
    
    # Clip to 0-100
    score = np.clip(score, 0, 100)
    
    df = pd.DataFrame({
        'Age': age,
        'StudyHours': np.round(study_hours, 1),
        'Attendance': np.round(attendance, 1),
        'Assignments': np.round(assignments, 1),
        'InternetAccess': internet_access,
        'Extracurricular': extracurricular,
        'StressLevel': stress_level,
        'Motivation': motivation,
        'ExamScore': np.round(score, 1)
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic student data...")
    df = generate_data(2000)
    df.to_csv('student_data.csv', index=False)
    print(f"Generated {len(df)} records and saved to student_data.csv.")
    print("Sample data:")
    print(df.head())
