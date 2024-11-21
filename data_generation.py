import numpy as np
import pandas as pd

def generate_synthetic_data():
    # Simulating a dataset with 1000 patients and 100 time steps (e.g., daily measurements)
    patients = 1000
    time_steps = 100

    # Generating random values for some health metrics (e.g., blood pressure, cholesterol)
    blood_pressure = np.random.rand(patients, time_steps) * 50 + 100  # systolic pressure
    cholesterol = np.random.rand(patients, time_steps) * 150 + 150  # cholesterol levels
    age = np.random.randint(30, 80, patients)  # patient age

    # Simulate disease progression based on blood pressure and cholesterol (randomized model)
    disease_progression = blood_pressure * 0.5 + cholesterol * 0.3 + age * 0.1 + np.random.randn(patients, time_steps) * 10

    # Creating a DataFrame
    data = pd.DataFrame({
        'blood_pressure': list(blood_pressure),
        'cholesterol': list(cholesterol),
        'age': list(age),
        'disease_progression': list(disease_progression)
    })

    # Save to CSV for future use
    data.to_csv('synthetic_medical_data.csv', index=False)

if __name__ == "__main__":
    generate_synthetic_data()
