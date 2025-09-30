import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# read the artist collaboration data
df = pd.read_csv('cleaned_regional_data.csv')

# read the artist popularity data
artists = pd.read_csv('artists.csv')
artist_names_to_keep = set(artists.track_artist.to_list())

def all_artists_in_set(artist_names):
    # Split the artist_names by comma and check if all are in the set
    return all(artist.strip() in artist_names_to_keep for artist in artist_names.split(', '))

df_filtered = df[df['artist_names'].apply(lambda x: all_artists_in_set(x))]

# create a map for each artist and the number of collaborations
artist_collab_map = {}
for artist in df_filtered['artist_list']:
    artist = eval(artist)
    if len(artist) > 1:
        for a in artist:
            artist_collab_map[a] = artist_collab_map.get(a, 0) + 1

# read the song charts data
playlists = pd.read_csv('Spotify Most Streamed Songs.csv')

# filter and keep the artists for which we have popularity data
playlists_filtered = playlists[playlists['artist(s)_name'].apply(lambda x: all_artists_in_set(x))]
playlists_filtered['artist(s)_name'].value_counts()

# Function to split rows based on comma-separated artist names
def expand_collaborations(df):
    # Split the artist_names and repeat the other columns for each artist
    expanded_rows = []
    for _, row in df.iterrows():
        artist_names = row['artist(s)_name'].split(',')
        for artist in artist_names:
            new_row = row.copy()
            new_row['artist_names'] = artist.strip()  # Set artist_names to individual artist
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

# add a row for every artist - to ensure every artist is given credit in a collaboration
playlists_filtered_expanded = expand_collaborations(playlists_filtered)

# aggregate
aggregated_df = playlists_filtered_expanded.groupby(
    by=['artist_names', 'released_year']
).agg({
    'in_spotify_playlists': 'sum',
    'in_spotify_charts': 'sum',
    'streams': 'sum',
    'in_apple_playlists': 'sum',
    'in_apple_charts': 'sum',
    'in_deezer_playlists': 'sum',
    'in_deezer_charts': 'sum',
    'in_shazam_charts': 'sum'
}).reset_index()

# Select the columns to normalize (all columns except 'artist_names' and 'released_year')
columns_to_normalize = [
    'in_spotify_playlists', 'in_spotify_charts', 'streams',
    'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
    'in_deezer_charts', 'in_shazam_charts'
]

# convert to float
aggregated_df[columns_to_normalize] = aggregated_df[columns_to_normalize].replace({',': ''}, regex=True).astype(float)

# choose top 10 popular artists to plot
artists_to_plot = artists.sort_values(by='count_above_threshold', ascending=False).head(10)['track_artist'].to_list()


# Filter data for the selected artists
filtered_df = df[df['artist_names'].isin(artists_to_plot)]

# List of the columns to plot trends for
columns_to_plot = [
    'in_spotify_playlists', 'in_spotify_charts', 'streams',
    'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
    'in_deezer_charts', 'in_shazam_charts'
]

fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(15, 20))
axes = axes.flatten()

# Plot for each artist
for idx, artist in enumerate(artists_to_plot):
    artist_data = aggregated_df[aggregated_df['artist_names'] == artist]

    # Loop through each of the columns (charts to plot)
    for col_idx, col in enumerate(columns_to_plot):
        ax = axes[col_idx]
        sns.lineplot(data=artist_data, x='released_year', y=col, label=artist, ax=ax)
        ax.set_title(f"{col}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()

plt.tight_layout()
plt.savefig('3.6.png')
plt.show()



'''
Clustering model
'''

columns_to_normalize = [
    'in_spotify_playlists', 'in_spotify_charts', 'streams',
    'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
    'in_deezer_charts', 'in_shazam_charts'
]

# convert to float
playlists_filtered_expanded[columns_to_normalize] = playlists_filtered_expanded[columns_to_normalize].replace({',': ''}, regex=True).astype(float)

# aggregate based on song features and charts/playlist data across platforms
artist_aggregated_df = playlists_filtered_expanded.groupby(
    by=['artist_names']
).agg({'in_spotify_playlists': 'mean',
 'in_spotify_charts': 'mean',
 'streams': 'mean',
 'in_apple_playlists': 'mean',
 'in_apple_charts': 'mean',
 'in_deezer_playlists': 'mean',
 'in_deezer_charts': 'mean',
 'in_shazam_charts': 'mean',
 'bpm': 'mean',
 'danceability_%': 'mean',
 'valence_%': 'mean',
 'energy_%': 'mean',
 'acousticness_%': 'mean',
 'instrumentalness_%': 'mean',
 'liveness_%': 'mean',
 'speechiness_%': 'mean'}).reset_index()

# drop nulls
artist_aggregated_df.dropna(inplace=True)


# Select features for clustering
features = [
    "in_spotify_playlists", "in_spotify_charts", "streams",
    "in_apple_playlists", "in_apple_charts", "in_deezer_playlists",
    "in_deezer_charts", "in_shazam_charts", "bpm",
    "danceability_%", "valence_%", "energy_%",
    "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%"
]

# Extract relevant features
X = artist_aggregated_df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define the range for the number of clusters
k_values = range(1, 20)

inertia_values = []

# Fit KMeans for each K and store the inertia
for k in k_values:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(X_scaled)
    inertia_values.append(kmeans_model.inertia_)

# Plot the Elbow Method visualization
plt.plot(k_values, inertia_values, 'bo-', markersize=8)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.grid(True)
plt.savefig('3.7.png')
plt.show()

# choose 5 clusters based on elbow method
kmeans_model = KMeans(n_clusters=5, random_state=42)
kmeans_model.fit(X_scaled)

# assign labels
artist_aggregated_df['clusters'] = kmeans_model.labels_


'''
PCA scatter plot visualization
'''
# Reduce to 2 dimensions with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA results and cluster labels to the DataFrame
artist_aggregated_df["pca_1"] = X_pca[:, 0]
artist_aggregated_df["pca_2"] = X_pca[:, 1]

# Scatter plot of the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="pca_1", y="pca_2", hue="clusters", data=artist_aggregated_df, palette="Set2", s=100
)
plt.title("Artist Clusters (PCA Visualization)")
plt.savefig('3.8.png')
plt.show()

'''
Cluster summary heatmap
'''
scaled_features = [f + '_scaled' for f in features]

df = pd.DataFrame(X_scaled, columns=scaled_features)
artist_aggregated_df = pd.concat([df, artist_aggregated_df], axis=1)
scaled_features.append('clusters')

cluster_summary = artist_aggregated_df[scaled_features].groupby("clusters").mean()


# Plot the heatmap
plt.figure(figsize=(20, 6))
sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Cluster Feature Summary Heatmap")
plt.savefig('3.9.png')
plt.show()

# assign each artist to a cluster in a map
artist_cluster_map = dict(zip(artist_aggregated_df['artist_names'], artist_aggregated_df['clusters']))



"""
Forecasting
"""

# get collaboration and cluster data
aggregated_df['cluster'] = aggregated_df['artist_names'].map(artist_cluster_map)
aggregated_df['collab'] = aggregated_df['artist_names'].map(artist_collab_map)

aggregated_df.collab.fillna(0, inplace=True)
aggregated_df.dropna(inplace=True)

aggregated_df.isna().sum()

# prep for df to be used in prophet model
cluster_streams = aggregated_df.groupby(['cluster', 'released_year']).agg({'streams':'sum', 'collab':'sum'}).reset_index()
cluster_streams.rename(columns={'released_year': 'ds', 'streams': 'y'}, inplace=True)
cluster_streams['ds'] = pd.to_datetime(cluster_streams['ds'], format='%Y')

# cap the value to ensure it does not reach infinity
cap_value = cluster_streams['y'].quantile(0.95)
cluster_streams['y'] = cluster_streams['y'].clip(upper=cap_value)

# scaling the features
scaler = StandardScaler()
cluster_streams[['y', 'collab']] = scaler.fit_transform(cluster_streams[['y', 'collab']])

cluster_groups = cluster_streams.groupby('cluster')

# check the error for the models
for cluster, cluster_data in cluster_groups:
    # Train-test split for a cluster
    test = cluster_data[-2:]   # Use the last 2 years for testing
    train = cluster_data[:-2]  # Use everything else for training

    # Train the model
    model = Prophet()
    model.add_regressor('collab')
    model.fit(train)

    # Predict for test years
    future_test = test[['ds', 'collab']]
    forecast_test = model.predict(future_test)

    # Evaluate
    y_true = test['y'].values
    y_pred = forecast_test['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae}, RMSE: {rmse}")



# visualize the predictions
def forecast_cluster(data, cluster_name):
    # Initialize Prophet model
    model = Prophet()
    model.add_regressor('collab')  # Add collaboration as a regressor

    # Fit the model
    model.fit(data)

    # forecasting next 5 years
    future = model.make_future_dataframe(periods=5, freq='Y')

    # Fill 'collaborations' column for future periods with the last value
    future['collab'] = data['collab'].iloc[-1]  # Last known value

    forecast = model.predict(future)

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"Cluster {cluster_name}: Streams Forecast")
    plt.savefig(f'3.{10 + int(cluster_name)}.png')
    plt.show()
    return forecast

# build a model for each cluster
for cluster, cluster_data in cluster_groups:
    forecast_cluster(cluster_data, cluster)

