import pandas as pd
import warnings
warnings.filterwarnings("ignore")

spotify = pd.read_csv('spotify_songs.csv')

# dropping nulls
print(spotify.isna().sum())
spotify.dropna(inplace=True)

# drop duplicates
spotify_no_duplicates = spotify.drop_duplicates(subset=['track_name', 'track_album_id'], keep='last')

# create count above threshold field
spotify_no_duplicates['is_above_threshold'] = spotify_no_duplicates['track_popularity'] >= 75
spotify_no_duplicates['is_above_threshold'] = spotify_no_duplicates['is_above_threshold'].astype(int)
artists = spotify_no_duplicates.groupby(['track_artist']).agg(count_above_threshold=('is_above_threshold', 'sum'))

# write to csv
artists.to_csv('artists.csv')

