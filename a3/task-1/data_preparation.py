import pandas as pd
df = pd.read_csv('spotify_songs.csv')

# Dropping nulls
print(df.isna().sum())
df.dropna(inplace=True)
df.to_csv('spotify_songs_without_null.csv')

# Handling duplicate songs 
print(df[df['track_name'].duplicated()])
# to check artist popularity, dropping songs repeated over different playlists
df_no_duplicates = df.drop_duplicates(subset=['track_name', 'track_album_id'], keep=False)
df_no_duplicates.to_csv('spotify_songs_without_duplicates.csv')


