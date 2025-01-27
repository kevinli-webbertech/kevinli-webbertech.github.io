import pandas as pd
df=pd.read_csv('npidata_pfile_20050523-20241013_fileheader.csv')
print(df.head())

#read the file where the names of taxonomy codes exist
taxonomy_df=pd.read_csv('nucc_taxonomy_241.csv')
print(taxonomy_df.head())

#create a dictionary mapping taxonomy codes
taxonomy_dict=dict(zip(taxonomy_df['Code'],taxonomy_df['Display Name']))

#function to map taxonomy codes in the chunks to names
def map_taxonomy_codes(chunk):
    chunk['taxonomy names']=chunk['codes'].apply(lambda codes:[taxonomy_dict.get(code,'Unknown')for code in codes])
    return chunk

#load the chunks in a smaller sizes
chunk_size = 50000
chunks = pd.read_csv('npidata_pfile_20050523-20241013.csv', chunksize=chunk_size)

taxonomy_columns = [
    'Healthcare Provider Taxonomy Code_1', 'Healthcare Provider Taxonomy Code_2',
    'Healthcare Provider Taxonomy Code_3', 'Healthcare Provider Taxonomy Code_4',
    'Healthcare Provider Taxonomy Code_5', 'Healthcare Provider Taxonomy Code_6',
    'Healthcare Provider Taxonomy Code_7', 'Healthcare Provider Taxonomy Code_8',
    'Healthcare Provider Taxonomy Code_9', 'Healthcare Provider Taxonomy Code_10',
    'Healthcare Provider Taxonomy Code_11', 'Healthcare Provider Taxonomy Code_12',
    'Healthcare Provider Taxonomy Code_13', 'Healthcare Provider Taxonomy Code_14',
    'Healthcare Provider Taxonomy Code_15'
]

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")  # Checkpoint for progress

    # Combine taxonomy codes into one column 'codes'
    chunk['codes'] = chunk[taxonomy_columns].agg(lambda x: ','.join(x.dropna().astype(str)), axis=1)

    # Drop original taxonomy columns
    chunk.drop(columns=taxonomy_columns, inplace=True)

    # Split 'codes' into a list
    chunk['codes'] = chunk['codes'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Apply mapping function
    chunk = map_taxonomy_codes(chunk)

    # Save processed chunk to CSV
    chunk.to_csv(f"Chunk{i+1}.csv", index=False)
    print(f"Chunk {i+1} saved.")


chunk1=pd.read_csv('chunk1.csv')
print(chunk1.head())