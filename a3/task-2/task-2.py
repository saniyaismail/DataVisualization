import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet



merged_df = pd.read_csv('merged_df.csv')

# drop duplicates
unique_country_tracks = merged_df.drop_duplicates(subset=['track_id', 'playlist_genre', 'country'])


'''
Dominant genre region wise
'''

# define a colour mapping for each genre
genre_colors = {
    'pop': 'red',
    'rock': 'cyan',
    'rap': 'green',
    'r&b': 'blue',
    'edm': 'magenta',
    'latin': 'orange'
}
genre_order = ['pop', 'rock', 'rap', 'r&b', 'edm', 'latin']

# Find the dominant genre by popularity in each region
genre_dominance = unique_country_tracks.groupby(['country', 'playlist_genre']).agg(
    avg_popularity=('track_popularity', 'mean')
).reset_index()

# Get the dominant genre for each region
dominant_genre = genre_dominance.loc[genre_dominance.groupby('country')['avg_popularity'].idxmax()]

# Choropleth map for dominant genre
fig = px.choropleth(
    dominant_genre,
    locations='country',
    locationmode='country names',
    color='playlist_genre',
    hover_name='country',
    title='Dominant Genre by Popularity in Each Region',
    color_discrete_map=genre_colors,
    category_orders={'playlist_genre': genre_order},
    # color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_geos(showcoastlines=True, coastlinecolor="Gray", showland=True, landcolor="White")
fig.show()



'''
Dominant Genre by Song Production in Each Region
'''
unique_country_tracks['playlist_genre'].unique()

# Find the dominant genre by song production in each country
genre_dominance_production = unique_country_tracks.groupby(['country', 'playlist_genre']).agg(
    total_songs=('track_id', 'nunique')  # Count unique tracks per genre per country
).reset_index()

# Get the dominant genre for each country based on song production
dominant_genre_production = genre_dominance_production.loc[genre_dominance_production.groupby('country')['total_songs'].idxmax()]

# Choropleth map for dominant genre by song production
fig = px.choropleth(
    dominant_genre_production,
    locations='country',
    locationmode='country names',
    color='playlist_genre',
    hover_name='country',
    hover_data=['total_songs'],
    title='Dominant Genre by Song Production in Each Region',
    color_discrete_map=genre_colors,
    category_orders={'playlist_genre': genre_order}
)
fig.update_geos(showcoastlines=True, coastlinecolor="Gray", showland=True, landcolor="White")
fig.show()



'''
Dominant Genre by Streams in Each Region
'''

# Find the dominant genre by streams in each country
genre_dominance_streams = unique_country_tracks.groupby(['country', 'playlist_genre']).agg(
    total_streams=('streams', 'sum')  # Sum of streams per genre per country
).reset_index()

# Get the dominant genre for each country based on streams
dominant_genre_streams = genre_dominance_streams.loc[genre_dominance_streams.groupby('country')['total_streams'].idxmax()]

# Choropleth map for dominant genre by streams
fig = px.choropleth(
    dominant_genre_streams,
    locations='country',
    locationmode='country names',
    color='playlist_genre',
    hover_name='country',
    hover_data = ['total_streams'],
    title='Dominant Genre by Streams in Each Region',
    # color_discrete_sequence=px.colors.qualitative.Set2,
    color_discrete_map=genre_colors,
    category_orders={'playlist_genre': genre_order},
)
fig.update_geos(showcoastlines=True, coastlinecolor="Gray", showland=True, landcolor="White")
fig.show()


merged_df['month'] = pd.to_datetime(merged_df['week']).dt.month
merged_df.duplicated(subset=['track_id', 'playlist_genre', 'month']).sum()
unique_monthly_tracks = merged_df.drop_duplicates(subset=['track_id', 'month', 'playlist_genre'])


'''
Seasonal Trends in Weeks on Chart by Genre
'''

# Aggregate total weeks on chart by month and genre
seasonal_weeks = unique_monthly_tracks.groupby(['month', 'playlist_genre']).agg(
    total_weeks=('weeks_on_chart', 'sum')  # Sum of weeks on chart for each genre in each month
).reset_index()

# Line chart for seasonal trends in weeks on chart
fig = px.line(
    seasonal_weeks,
    x='month',
    y='total_weeks',
    color='playlist_genre',
    title='Seasonal Trends in Weeks on Chart by Genre',
    labels={'month': 'Month', 'total_weeks': 'Total Weeks on Chart'}
)
fig.update_xaxes(tickmode='linear', dtick=1)
fig.show()

seasonal_weeks[seasonal_weeks['playlist_genre'] == 'latin']

unique_monthly_tracks['week'].unique()

# Add events
events = pd.DataFrame({
    'event': ['Grammy Awards', 'Summer Hits', 'Holiday Season'],
    'month': [2, 6, 12]
})

# Line chart for seasonal trends
fig = px.line(
    seasonal_weeks,
    x='month',
    y='total_weeks',
    color='playlist_genre',
    title='Seasonal Trends in Weeks on Chart by Genre',
    labels={'month': 'Month', 'total_weeks': 'Total Weeks on Chart'}
)

# Add vertical lines for events
for _, row in events.iterrows():
    fig.add_vline(
        x=row['month'],
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=row['event'],
        annotation_position="top right"
    )

fig.update_xaxes(tickmode='linear', dtick=1, title='Month')
fig.update_yaxes(title='Total Weeks on Chart')
fig.show()




unique_tracks = merged_df.drop_duplicates(subset=['language', 'track_id', 'playlist_genre'])


# Filter out 'Global' from the country column
country_data = merged_df[merged_df['country'] != 'Global']

# Remove duplicates based on track_id, country, and playlist_genre
unique_country_tracks = country_data.drop_duplicates(subset=['track_id', 'country', 'playlist_genre'])

# Aggregate data by country and genre
country_genre_data = unique_country_tracks.groupby(['country', 'playlist_genre']).agg(
    avg_popularity=('track_popularity', 'mean'),
    total_streams=('streams', 'sum'),
    weeks_on_chart=('weeks_on_chart', 'sum')
).reset_index()

# Pivot data to prepare for clustering
pivot_data = country_genre_data.pivot(index='country', columns='playlist_genre',
                                      values=['avg_popularity', 'total_streams', 'weeks_on_chart']).fillna(0)

# Flatten multi-index columns
pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns]
pivot_data.reset_index(inplace=True)

print("Preprocessed Data Shape:", pivot_data.shape)



'''
PCA and clustering
'''
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data.drop(columns=['country']))

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Add PCA components to the dataset
pivot_data['PCA1'] = pca_data[:, 0]
pivot_data['PCA2'] = pca_data[:, 1]


# Prepare lists to store metrics
inertia_values = []
silhouette_scores = []

# Range of cluster values to test
cluster_range = range(2, 11)  # Starting from 2 clusters since silhouette isn't defined for 1 cluster

# Calculate inertia and silhouette scores for each number of clusters
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_data)  # Use PCA-transformed data

    inertia_values.append(kmeans.inertia_)

    # Calculate silhouette score only if n_clusters > 1
    silhouette_avg = silhouette_score(pca_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot Inertia (Elbow Method)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print metrics
for i, n_clusters in enumerate(cluster_range):
    print(f"Number of Clusters: {n_clusters} | Inertia: {inertia_values[i]:.2f} | Silhouette Score: {silhouette_scores[i]:.2f}")

# Choose the optimal number of clusters (e.g., 4 based on elbow method)
kmeans = KMeans(n_clusters=5, random_state=42)
pivot_data['cluster'] = kmeans.fit_predict(pca_data)


# Scatter plot using PCA components
fig = px.scatter(
    pivot_data,
    x='PCA1',
    y='PCA2',
    color=pivot_data['cluster'].astype(str),
    hover_name='country',
    title='Clustering of Countries by Genre Preferences',
    labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig.update_layout(
    width=900,
    height=600,
    title_font_size=18,
    legend_title_text='Cluster'
)
fig.show()



'''
Comparison of Streams/Popularity/Week on chart: Cross-Genre vs Solo Tracks
'''
# Count the number of unique genres for each track
genre_counts = merged_df.groupby('track_id')['playlist_genre'].nunique().reset_index()

# Create the 'is_cross_genre' column
merged_df['is_cross_genre'] = merged_df['track_id'].map(genre_counts.set_index('track_id')['playlist_genre']) > 1

# Filter out duplicate entries for the same track in different genres/subgenres
cross_genre_tracks = merged_df[merged_df['is_cross_genre'] == True].drop_duplicates(subset=['track_id', 'playlist_genre', 'playlist_subgenre'])

solo_genre_tracks = merged_df[merged_df['is_cross_genre'] == False].drop_duplicates(subset=['track_id', 'playlist_genre', 'playlist_subgenre'])




# Aggregate data to get average popularity and total streams for each group
cross_genre_performance = cross_genre_tracks.groupby('track_id').agg(
    avg_popularity=('track_popularity', 'mean'),
    total_streams=('streams', 'sum'),
    weeks_on_chart=('weeks_on_chart', 'sum')
).reset_index()

solo_genre_performance = solo_genre_tracks.groupby('track_id').agg(
    avg_popularity=('track_popularity', 'mean'),
    total_streams=('streams', 'sum'),
    weeks_on_chart=('weeks_on_chart', 'sum')
).reset_index()

# Combine both datasets for comparison
cross_genre_performance['track_type'] = 'Cross-Genre'
solo_genre_performance['track_type'] = 'Solo Genre'

# Concatenate both datasets
performance_data = pd.concat([cross_genre_performance, solo_genre_performance])

# Plot Comparison of Streams for Cross-Genre vs Solo Tracks
fig = px.box(
    performance_data,
    x='track_type',
    y='total_streams',
    color='track_type',
    title='Comparison of Streams: Cross-Genre vs Solo Tracks',
    labels={'total_streams': 'Total Streams', 'track_type': 'Track Type'}
)
fig.show()

# Plot Comparison of Popularity for Cross-Genre vs Solo Tracks
fig = px.box(
    performance_data,
    x='track_type',
    y='avg_popularity',
    color='track_type',
    title='Comparison of Popularity: Cross-Genre vs Solo Tracks',
    labels={'avg_popularity': 'Average Popularity', 'track_type': 'Track Type'}
)
fig.show()

# Plot Comparison of Weeks on Chart for Cross-Genre vs Solo Tracks
fig = px.box(
    performance_data,
    x='track_type',
    y='weeks_on_chart',
    color='track_type',
    title='Comparison of Weeks on Chart: Cross-Genre vs Solo Tracks',
    labels={'weeks_on_chart': 'Weeks on Chart', 'track_type': 'Track Type'}
)
fig.show()


# Remove duplicates before aggregation
unique_tracks = merged_df.drop_duplicates(subset=['track_id', 'playlist_genre', 'language', 'month'])

# Now aggregate by month, genre, and language
language_genre_trends = unique_tracks.groupby(['month', 'playlist_genre', 'language']).agg(
    total_streams=('streams', 'sum')  # Summing the streams for each language and genre per month
).reset_index()

unique_tracks_2 = merged_df.drop_duplicates(subset=['track_id', 'playlist_genre', 'language'])

merged_df['language'].unique()

filtered_data = unique_tracks_2[unique_tracks_2['language'] != 'Global']

# Determine significant languages based on total streams
significant_languages = (
    filtered_data.groupby('language')['streams']
    .sum()
    .sort_values(ascending=False)
    .head(7)  # Choose top 5 for clarity
    .index
)

significant_languages

# Example of grouping languages into regional subsets
regional_groups = {
    'Asian': ['Korean', 'Japanese', 'Hindi', 'Indonesian'],
    'European': ['French', 'Spanish', 'English', 'Portuguese']
}

grouped_data = {}

# Create subsets of data based on regional groups
for region, languages in regional_groups.items():
    grouped_data[region] = filtered_data[filtered_data['language'].isin(languages)]



'''
Chord diagrams
'''

hv.extension('bokeh')

def create_chord_diagram(data_subset, title, color_mapping):
    # Aggregate data for chord diagram
    chord_data = data_subset.groupby(['language', 'playlist_genre']).agg(
        total_streams=('streams', 'sum')
    ).reset_index()

    # Prepare data for Holoviews
    links = chord_data.rename(columns={'language': 'Source', 'playlist_genre': 'Target', 'total_streams': 'Value'})

    # Create Chord Diagram
    return hv.Chord(links).opts(
        opts.Chord(
            node_color="index",
            edge_color="Source",
            # cmap="Category20",
            cmap = color_mapping,
            labels="index",
            node_size=15,
            title=title,
            height=500,
            width=500
        )
    )
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',  # Colors for languages
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',  # Colors for genres
                 '#bcbd22', '#17becf']
# Generate individual chord diagrams for each group
chord_diagrams = {region: create_chord_diagram(data, f"{region} Language-Genre total streams relationships", color_mapping=custom_colors)
                  for region, data in grouped_data.items()}

# Display all diagrams
hv.Layout(list(chord_diagrams.values())).cols(2)



'''
Forecasting model
'''

# Filter necessary columns and remove duplicates
filtered_data = merged_df[['playlist_genre', 'week', 'track_popularity']].drop_duplicates()

# Ensure 'week' is a datetime object
filtered_data['week'] = pd.to_datetime(filtered_data['week'])

# Aggregate popularity by genre and week
genre_trends = filtered_data.groupby(['playlist_genre', 'week']).agg(
    avg_popularity=('track_popularity', 'mean')  # Aggregate by mean popularity
).reset_index()


# Prepare a dictionary to store validation results
validation_results = {}

# Loop through each genre
for genre in genre_trends['playlist_genre'].unique():
    # Filter data for the current genre
    genre_data = genre_trends[genre_trends['playlist_genre'] == genre]

    # Rename columns for Prophet
    prophet_data = genre_data.rename(columns={'week': 'ds', 'avg_popularity': 'y'})

    # Train-Test Split (80% train, 20% test)
    split_index = int(len(prophet_data) * 0.8)
    train = prophet_data.iloc[:split_index]
    test = prophet_data.iloc[split_index:]

    # Train the Prophet model
    model = Prophet()
    model.fit(train)

    # Predict for test years
    future_test = test[['ds']]  # Ensure `ds` is a DataFrame
    forecast_test = model.predict(future_test)

    # Evaluate
    y_true = test['y'].values
    y_pred = forecast_test['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Store results
    validation_results[genre] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


# Display validation metrics
for genre, metrics in validation_results.items():
    print(f"Genre: {genre}")
    print(f"  - MAE: {metrics['MAE']:.2f}")
    print(f"  - RMSE: {metrics['RMSE']:.2f}")
    print(f"  - MAPE: {metrics['MAPE']:.2f}%\n")


# Forecasting with Prophet
# Create a dictionary to store models and forecasts
genre_forecasts = {}

# Loop through each genre
for genre in genre_trends['playlist_genre'].unique():
    # Filter data for the current genre
    genre_data = genre_trends[genre_trends['playlist_genre'] == genre]

    # Rename columns to fit Prophet's requirements
    prophet_data = genre_data.rename(columns={'week': 'ds', 'avg_popularity': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)

    # Create a dataframe for future dates (e.g., next 12 months)
    future = model.make_future_dataframe(periods=52, freq='W')  # Weekly predictions
    # Forecast future values
    forecast = model.predict(future)

    # Store model and forecast
    genre_forecasts[genre] = {
        'model': model,
        'forecast': forecast
    }

    # Visualize actual vs. forecasted data
    fig = px.line(
        x=forecast['ds'],
        y=forecast['yhat'],
        title=f"Popularity Forecast for {genre}",
        labels={'x': 'Date', 'y': 'Forecasted Popularity'}
    )
    # Add historical data
    fig.add_scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='markers', name='Actual Popularity')
    fig.show()

# Additional Insights
# Combine all forecasts into a single DataFrame for comparative visualization
combined_forecasts = pd.DataFrame()

for genre, data in genre_forecasts.items():
    forecast = data['forecast']
    forecast['genre'] = genre  # Add genre as a column
    combined_forecasts = pd.concat([combined_forecasts, forecast[['ds', 'yhat', 'genre']]])


merged_df['year'] = pd.to_datetime(merged_df['track_album_release_date'], errors='coerce', format='mixed').dt.year

# merged_df['year'].unique()