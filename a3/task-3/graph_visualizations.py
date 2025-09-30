import pandas as pd
import itertools
from collections import Counter

# read the artist collaboration data
df = pd.read_csv('final.csv')

df = df[df.artists_num != 'artists_num']
# cleaning the dataset

num_features = ['rank', 'artists_num', 'collab',
      'album_num_tracks','peak_rank',
      'previous_rank', 'weeks_on_chart','streams','danceability',
      'energy', 'key', 'mode', 'loudness','speechiness',
      'acousticness', 'instrumentalness', 'liveness','valence',
      'tempo', 'duration', 'pivot']

date_features = ['release_date','week']

# convert the numerical and date features to the correct datatype
df[num_features] = df[num_features].astype('float')
df[date_features] = df[date_features].apply(pd.to_datetime, format='mixed')


# drop duplicates for artist data
df_no_duplicates = df.drop_duplicates(subset=['artist_names', 'track_name', 'collab'], keep='last')

# list of collaborating artists
df_no_duplicates['artist_list'] = df_no_duplicates['artist_names'].str.split(', ')  # Split comma-separated artist names

# creating edgelist
edges = []
for artists in df_no_duplicates['artist_list']:
    if len(artists) > 1:  # Only consider collaborations
        edges.extend(itertools.combinations(artists, 2))  # Create all pairwise combinations

edge_weights = Counter(edges)

# convert to DataFrame for Gephi
edges_df = pd.DataFrame([(source, target, weight) for (source, target), weight in edge_weights.items()],
                        columns=['Source', 'Target', 'Weight'])

nodes = set(itertools.chain.from_iterable(df_no_duplicates['artist_list']))
nodes_df = pd.DataFrame({'Id': list(nodes), 'Label': list(nodes)})

artist_streams = df.groupby('artist_individual')['streams'].sum().to_dict()
nodes_df['Total_Streams'] = nodes_df['Label'].map(artist_streams)

# filter and keep the artists for which we have popularity data
artists = pd.read_csv('artists.csv')
artist_names_to_keep = set(artists.track_artist.to_list())

nodes_df_filtered = nodes_df[nodes_df['Label'].isin(artist_names_to_keep)]
edges_df_filtered = edges_df[
    (edges_df['Source'].isin(artist_names_to_keep)) &
    (edges_df['Target'].isin(artist_names_to_keep))
]


artists.columns = ['Label', 'Popularity']

nodes_df_filtered = pd.merge(nodes_df_filtered, artists, on='Label', how='left')

# create nodes anf edges df to be used in gephi
nodes_df_filtered.to_csv('filtered_nodes.csv', index=False)
edges_df_filtered.to_csv('filtered_edges.csv', index=False)


# cleaned data used for collaborations in the forecasting model
df_no_duplicates.to_csv('cleaned_regional_data.csv')

