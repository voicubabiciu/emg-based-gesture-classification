from collections import Counter
from datetime import datetime

import math
import numpy as np
import pandas as pd
from influxdb_client import Point, WritePrecision
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def _remove_outliers(df, reference_df, threshold=1.5):
    columns_to_check = ['ch1', 'ch2', 'ch3', 'ch4']
    reference_mean = reference_df[columns_to_check].mean()
    reference_std = reference_df[columns_to_check].std()

    # Log the mean and std deviation
    print("Reference Mean:\n", reference_mean)
    print("Reference Std Dev:\n", reference_std)

    # Calculate z-scores for the entire dataset using reference mean and std
    z_scores = np.abs((df[columns_to_check] - reference_mean) / reference_std)

    # Plot the z-scores to visualize the distribution
    # for column in columns_to_check:
    #     plt.figure()
    #     plt.hist(z_scores[column], bins=50, alpha=0.75)
    #     plt.title(f'Z-Score Distribution for {column}')
    #     plt.xlabel('Z-Score')
    #     plt.ylabel('Frequency')
    #     plt.show()

    # Log the z-scores to see the distribution
    print("Z-Scores:\n", z_scores.describe())

    # Identify outliers
    outliers = z_scores > threshold
    outlier_rows = outliers.any(axis=1)

    # Log some information for debugging
    print(f"Number of rows in original data: {len(df)}")
    print(f"Number of outliers identified: {outlier_rows.sum()}")

    # Return the dataframe without outliers
    return df[~outlier_rows]


def normalize_data(df, reference_df):
    columns_to_check = ['ch1', 'ch2', 'ch3', 'ch4']
    scaler = StandardScaler()
    scaler.fit(reference_df[columns_to_check])
    df[columns_to_check] = scaler.transform(df[columns_to_check])
    return df


def preprocess_data(df: DataFrame):
    record_name = f'emg_dataset{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    json_array = df.to_dict(orient='records')
    points = []
    for record in json_array:
        points.append(Point(record_name)
                      .tag("label", record['label'])
                      .tag("person", record['person'])
                      .tag("type", 'raw')
                      .field("ch1", record['ch1'])
                      .field("ch2", record['ch2'])
                      .field("ch3", record['ch3'])
                      .field("ch4", record['ch4'])
                      .time(datetime.utcfromtimestamp(record['time']), WritePrecision.NS))

    # Get the reference data for 'voicu'
    reference_df = df[df['person'] == 'voicu']

    # Ensure that reference data is not empty
    if reference_df.empty:
        raise ValueError("Reference data for 'voicu' is empty.")

    # Split the dataset by label
    label_groups = df.groupby('label')
    # Apply the function to each group and concatenate the results
    cleaned_data = pd.concat([_remove_outliers(group, reference_df) for _, group in label_groups])

    cleaned_data_json_array = cleaned_data.to_dict(orient='records')
    for record in cleaned_data_json_array:
        points.append(Point(record_name)
                      .tag("label", record['label'])
                      .tag("person", record['person'])
                      .tag("type", 'cleared-outliers')
                      .field("ch1", record['ch1'])
                      .field("ch2", record['ch2'])
                      .field("ch3", record['ch3'])
                      .field("ch4", record['ch4'])
                      .time(datetime.utcfromtimestamp(record['time']), WritePrecision.NS))
    df = cleaned_data.drop(columns=['person', 'time', ])
    return df, points


def most_frequent_number(arr):
    counter = Counter(arr)
    most_common_number, _ = counter.most_common(1)[0]
    return most_common_number


def custom_round(number):
    if number % 1 < 0.5:
        return math.floor(number)
    else:
        return math.ceil(number)


def preprocess_input(new_data, scaler):
    # Ensure new_data is a DataFrame with the same feature names
    if isinstance(new_data, np.ndarray):
        new_data = pd.DataFrame(new_data, columns=scaler.feature_names_in_)

    # Normalize the features
    new_data_scaled = scaler.transform(new_data)

    # Reshape data for 1D CNN
    new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)
    return new_data_scaled


def predict_new_data(model, new_data, scaler):
    # Preprocess the new data
    new_data_processed = preprocess_input(new_data, scaler)

    # Predict with the model
    predictions = model.predict(new_data_processed)

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes
