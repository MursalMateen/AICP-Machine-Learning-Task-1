# AICP-Machine-Learning-Task-1
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('transaction_anomalies_dataset.csv')

# Preprocess the data: fill missing values, encode categorical variables if necessary
df.fillna(df.median(), inplace=True)

# Exploratory Data Analysis: Visualize the distribution of Transaction Amounts
sns.histplot(df['Transaction_Amount'], kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Assume 'Transaction_Amount' and other relevant features are selected for anomaly detection
features = df[['Transaction_Amount', 'Transaction_Volume']]  # Example features

# Use Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(features)

# Predict anomalies
predictions = model.predict(features)
df['anomaly'] = np.where(predictions == -1, 1, 0)

# Visualization of anomalies
sns.scatterplot(x='Transaction_Amount', y='Transaction_Volume', hue='anomaly', data=df)
plt.title('Anomalies in Transactions')
plt.show()

# Assuming you have a column 'true_label' for evaluation
# (Replace 'true_label' with your actual column name if different)
df['true_label'] = (df['Transaction_Amount'] > 10000).astype(int)  # Example condition for labeling anomalies

# Generate classification report
print(classification_report(df['true_label'], df['anomaly'], target_names=['Normal', 'Anomaly']))

# Summary of detected anomalies
num_anomalies = df['anomaly'].sum()
print(f"Number of anomalies detected: {num_anomalies}")
percentage_anomalies = (num_anomalies / len(df)) * 100
print(f"Percentage of transactions classified as anomalies: {percentage_anomalies:.2f}%")
