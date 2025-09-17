import pandas as pd

from config.config import excluded_user_ids

# Load the dataset
df_demographic = pd.read_csv('/Users/alessandrocaruso/Desktop/Master Thesis Files/SMART_all_demographics.csv')

# Remove excluded users if any
df_demographic = df_demographic[~df_demographic['user'].isin(excluded_user_ids)]  # Uncomment if needed
print(df_demographic.columns)
# Group the dataset into depressed (DP) and healthy controls (HC)
df_depressed = df_demographic[df_demographic['type'] == 'p']
df_healthy = df_demographic[df_demographic['type'] != 'p']

# Calculate the number of participants in each group
num_dp = len(df_depressed)
num_hc = len(df_healthy)
total_participants = num_dp + num_hc

# Print basic demographic statistics
print(f"Total number of participants: {total_participants}")
print(f"Number of depressed participants (DP): {num_dp}")
print(f"Number of healthy controls (HC): {num_hc}")

# Age statistics
print("\nAge statistics for DP group:")
print(f"Minimum age: {df_depressed['Age'].min()}")
print(f"Maximum age: {df_depressed['Age'].max()}")
print(f"Average age: {df_depressed['Age'].mean():.2f} years (±{df_depressed['Age'].std():.2f})")

print("\nAge statistics for HC group:")
print(f"Minimum age: {df_healthy['Age'].min()}")
print(f"Maximum age: {df_healthy['Age'].max()}")
print(f"Average age: {df_healthy['Age'].mean():.2f} years (±{df_healthy['Age'].std():.2f})")

# Gender distribution
gender_dp = df_depressed['Sex'].value_counts()
gender_hc = df_healthy['Sex'].value_counts()

print("\nGender distribution:")
print(f"DP group - Females: {gender_dp.get('female', 0)}, Males: {gender_dp.get('male', 0)}")
print(f"HC group - Females: {gender_hc.get('female', 0)}, Males: {gender_hc.get('male', 0)}")

# Occupation distribution
occupation_dp = df_depressed['Current occupation'].value_counts()
occupation_hc = df_healthy['Current occupation'].value_counts()

print("\nOccupation distribution:")
print(f"DP group:\n{occupation_dp}")
print(f"HC group:\n{occupation_hc}")

# Antidepressant usage
antidepressant_dp = df_depressed['antidepressant_type'].value_counts()
antidepressant_hc = df_healthy['antidepressant_type'].value_counts()

print("\nAntidepressant usage:")
print(f"DP group:\n{antidepressant_dp}")
print(f"HC group:\n{antidepressant_hc}")

# Hormonal contraception
contraception_dp = df_depressed['hormonal_contraception'].value_counts()
contraception_hc = df_healthy['hormonal_contraception'].value_counts()

print("\nHormonal contraception usage:")
print(f"DP group: {contraception_dp.get('Yes', 0)}")
print(f"HC group: {contraception_hc.get('Yes', 0)}")

# Sleep drugs usage
sleep_drugs_dp = df_depressed['sleep_drugs'].value_counts()
sleep_drugs_hc = df_healthy['sleep_drugs'].value_counts()

print("\nSleep drugs usage:")
print(f"DP group: {sleep_drugs_dp.get('Yes', 0)}")
print(f"HC group: {sleep_drugs_hc.get('Yes', 0)}")

# ADHD medication usage
adhd_medication_dp = df_depressed['ADHD_medication'].value_counts()
adhd_medication_hc = df_healthy['ADHD_medication'].value_counts()

print("\nADHD medication usage:")
print(f"DP group: {adhd_medication_dp.get('Yes', 0)}")
print(f"HC group: {adhd_medication_hc.get('Yes', 0)}")