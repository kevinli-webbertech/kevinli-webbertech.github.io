import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

def SortDoctorsPractices(df):
    df['taxonomy names'] = df['taxonomy names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_ex=df.explode('taxonomy names')
    df_ex.reset_index(drop=True,inplace=True)
    return df_ex

def remove_deactivated_doctors(df):
    df=df.fillna(0) #to ease dealing with the data
    reactivated_doctors=df[df['NPI Reactivation Date'] != 0] #split the reactivated doctors in a list
    df = df[df['NPI Deactivation Date'] == 0] #delete all doctors who have a deactivation date
    active_doctors=pd.concat([df,reactivated_doctors]) #merge the reactivated with the already active doctors df

    return active_doctors

def active_doctors_count_based_on_practice(df):
    df2=df.groupby('taxonomy names').size().reset_index(name='count of doctors')
    df2.set_index('taxonomy names',inplace=True)
    return df2

def process(file):
    df=pd.read_csv(file)
    df_ex=SortDoctorsPractices(df)
    active_doctors=remove_deactivated_doctors(df_ex)
    active_doctors_count=active_doctors_count_based_on_practice(active_doctors)
    return active_doctors_count

import matplotlib.pyplot as plt
import pandas as pd
import os

def aggregate_and_plot(output_folder='chunks'):
    total_counts = pd.DataFrame()

    for file_name in os.listdir(output_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(output_folder, file_name)
            counts = process(file_path)
            counts = counts.reset_index()

            if total_counts.empty:
                total_counts = counts
            else:
                total_counts = total_counts.merge(counts, on='taxonomy names', how='outer', suffixes=('', '_new'))
                total_counts['count of doctors'] = total_counts[['count of doctors', 'count of doctors_new']].sum(axis=1)
                total_counts = total_counts.drop(columns=['count of doctors_new'])

    # Sort the DataFrame
    total_counts = total_counts.sort_values(by='count of doctors', ascending=False)

    # Create the pie chart as a Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(total_counts['count of doctors'][:20],
           labels=total_counts['taxonomy names'][:20],
           autopct='%1.1f%%',
           startangle=140)
    ax.set_title('Percentage of Doctors by Practice')

    # Return the Figure and DataFrame
    return fig, total_counts
fig, data = aggregate_and_plot(output_folder='chunks')
fig.show()  # Display the plot

import pandas as pd

# List of names to search for
names_to_search = [
    ("elizabeth", "botham"),
    ("michael", "costanzo"),
    ("jenna", "presto"),
    ("daniel", "roling"),
    ("marlyn", "wu"),
    ("robert", "kantor")
]

# File path to the large CSV file
file_path = 'npidata_pfile_20050523-20241013.csv'

# Function to search for a doctor in a large CSV file by processing it in chunks
def doctor_search_large_file(FirstName, LastName, file_path, chunksize=100000):
    """
    Search for a doctor in a large CSV file by processing it in chunks.

    Parameters:
    - FirstName: The first name of the doctor to search for.
    - LastName: The last name of the doctor to search for.
    - file_path: Path to the large CSV file.
    - chunksize: Number of rows to process per chunk (default: 10,000).

    Returns:
    - A DataFrame containing the matching rows, or an empty DataFrame if no matches are found.
    """
    # Convert search terms to lowercase for case-insensitive comparison
    FirstName = FirstName.lower()
    LastName = LastName.lower()

    results = []  # Collect matching rows

    # Read the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Remove deactivated doctors in the chunk
        chunk = remove_deactivated_doctors(chunk)

        # Check if required columns exist
        if 'Provider Last Name (Legal Name)' in chunk.columns and 'Provider First Name' in chunk.columns:
            # Ensure string columns and convert to lowercase
            chunk['Provider Last Name (Legal Name)'] = chunk['Provider Last Name (Legal Name)'].astype(str).str.lower()
            chunk['Provider First Name'] = chunk['Provider First Name'].astype(str).str.lower()

            # Filter rows matching the given first and last name
            matching_rows = chunk[
                (chunk['Provider Last Name (Legal Name)'] == LastName) &
                (chunk['Provider First Name'] == FirstName)
                ]

            if not matching_rows.empty:
                results.append(matching_rows)  # Collect matches

    # Combine results if there are any
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        print(f"The doctor {FirstName} {LastName} does not exist in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame

# Iterate over each name and search in the file.
for first_name, last_name in names_to_search:
    result = doctor_search_large_file(first_name, last_name, file_path)
    if not result.empty:
        print(f"Results for {first_name} {last_name}:")
        print(result)
    else:
        print(f"No results found for {first_name} {last_name}.")
