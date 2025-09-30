import pandas as pd
df = pd.read_csv("spotify_songs.csv")
regional_data = pd.read_csv("final.csv")

# drop nulls
df.dropna(inplace=True)

# check for duplicates
print(df.duplicated().sum())
print(regional_data.duplicated().sum())


# Clean 'uri' column in regional_data
regional_data['uri'] = regional_data['uri'].str.replace('spotify:track:', '')

df['track_id'] = df['track_id'].str.strip()
regional_data['uri'] = regional_data['uri'].str.strip()


# Merge datasets on 'track_id'
merged_df = pd.merge(df, regional_data, left_on='track_id', right_on = 'uri', how='inner')

print(merged_df.duplicated().sum())

# write merged df to csv
merged_df.to_csv("merged_df.csv", index = False)