import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ChronicDiseaseProgression").getOrCreate()

def preprocess_data(file_path):
    # Load data using Pandas
    data = pd.read_csv(file_path)

    # Feature extraction: Normalize numerical data (blood_pressure, cholesterol, disease_progression)
    scaler = MinMaxScaler()
    data[['blood_pressure', 'cholesterol', 'disease_progression']] = scaler.fit_transform(
        data[['blood_pressure', 'cholesterol', 'disease_progression']]
    )

    # Convert data to a Spark DataFrame for distributed processing (optional)
    spark_df = spark.createDataFrame(data)

    # Example of some Spark operations (e.g., filtering, grouping, etc.)
    spark_df = spark_df.filter(spark_df['age'] > 30)

    # Show the preprocessed data
    spark_df.show(5)

    return data  # Return the pandas dataframe for model building

if __name__ == "__main__":
    data = preprocess_data('synthetic_medical_data.csv')
