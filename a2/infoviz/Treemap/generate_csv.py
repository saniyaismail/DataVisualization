import pandas as pd

data = pd.read_csv("spotify_songs.csv")

data.isnull().sum()

data.dropna(inplace = True)
data.isnull().sum()

#Treemap 1
# Removing duplicate tracks within each genre-subgenre combination
unique_genre_subgenre_tracks = data.drop_duplicates(subset=['playlist_genre', 'playlist_subgenre', 'track_id'])

# Calculating the average popularity by genre and subgenre
genre_subgenre_data = unique_genre_subgenre_tracks.groupby(['playlist_genre', 'playlist_subgenre'])['track_popularity'].mean().reset_index()

# Rank subgenres within each genre so that the highest popularity gets rank 1
genre_subgenre_data['rank'] = genre_subgenre_data.groupby('playlist_genre')['track_popularity'].rank(method='dense', ascending=False).astype(int)

genre_subgenre_data.to_csv('genre_subgenre_data.csv', index=False)

data.duplicated(subset=['playlist_genre', 'playlist_subgenre', 'track_id'], keep = False).sum()

unique_genre_subgenre_tracks.duplicated(subset=['playlist_genre', 'playlist_subgenre', 'track_id'], keep = False).sum()


#Treemap 2
data['year'] = pd.DatetimeIndex(data['track_album_release_date']).year

# Remove duplicate tracks within each year-artist combination
unique_year_artist_tracks = data.drop_duplicates(subset=['year', 'track_artist', 'track_id'])

# Calculate the average popularity by year and artist
year_artist_data = unique_year_artist_tracks.groupby(['year', 'track_artist'])['track_popularity'].mean().reset_index()

# Rank artists within each year by popularity, with rank 1 being the highest popularity
year_artist_data['rank'] = year_artist_data.groupby('year')['track_popularity'].rank(method='dense', ascending=False).astype(int)

# Filter for the top 10 artists per year based on popularity rank
top_year_artist_data = year_artist_data[year_artist_data['rank'] <= 10]

year_avg_popularity = top_year_artist_data.groupby('year')['track_popularity'].mean().reset_index()
year_avg_popularity.columns = ['year', 'year_avg_popularity']

# Merge the average yearly popularity into the top artist data for each year
top_year_artist_data = top_year_artist_data.merge(year_avg_popularity, on='year', how='left')

top_year_artist_data.to_csv('year_artist_data.csv', index=False)

data.duplicated(subset = ['year', 'track_artist', 'track_id'], keep = False).sum()

unique_year_artist_tracks.duplicated(subset = ['year', 'track_artist', 'track_id'], keep = False).sum()



#Treemap 3
# Remove duplicate tracks within each artist-genre pair
unique_tracks = data.drop_duplicates(subset=['playlist_genre', 'track_artist', 'track_id'])

# Calculate the average popularity and unique track count for each artist within each genre
artist_data = unique_tracks.groupby(['playlist_genre', 'track_artist']).agg(
    track_popularity=('track_popularity', 'mean'),
    track_count=('track_id', 'nunique')  # Unique track count for each artist in each genre
).reset_index()

# Rank artists within each genre by popularity
artist_data['rank'] = artist_data.groupby('playlist_genre')['track_popularity'].rank(method='dense', ascending=False).astype(int)

# Filter for the top 5 artists per genre
top_artists_per_genre = artist_data[artist_data['rank'] <= 5]

top_artists_per_genre.to_csv('genre_artist_popularity_trackcount.csv', index=False)

data.duplicated(subset=['playlist_genre', 'track_artist', 'track_id'], keep = False).sum()

unique_tracks.duplicated(subset=['playlist_genre', 'track_artist', 'track_id'], keep = False).sum()