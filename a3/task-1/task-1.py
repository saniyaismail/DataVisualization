import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# reading spotify an youtube data
spotify_data = pd.read_csv("spotify_songs.csv")
youtube_data = pd.read_csv("Spotify_Youtube.csv")

print("Columns in Spotify 30k Songs Dataset:")
print(spotify_data.columns.tolist())

print("\nColumns in Spotify and YouTube Dataset:")
print(youtube_data.columns.tolist())


# clean artist and track name
spotify_data['track_name'] = spotify_data['track_name'].str.strip().str.lower()
spotify_data['track_artist'] = spotify_data['track_artist'].str.strip().str.lower()

youtube_data['Track'] = youtube_data['Track'].str.strip().str.lower()
youtube_data['Artist'] = youtube_data['Artist'].str.strip().str.lower()

# merge the datasets
merged_data = pd.merge(
    spotify_data,
    youtube_data[['Track', 'Artist', 'Views', 'Likes', 'Comments']],
    left_on=['track_name', 'track_artist'],
    right_on=['Track', 'Artist'],
    how='left'
)

merged_data.drop(['Track', 'Artist'], axis=1, inplace=True)
merged_data.dropna(inplace=True)


"""
Adding column Artist Popularity (count above threshold as defined in Task-3 assignment-1) to the Spotify Dataset
"""

#drop the duplicates
merged_no_duplicates = merged_data.drop_duplicates(subset=['track_name', 'track_album_id'], keep='last')
merged_data.shape

# Create a column indicating whether the track's popularity is above the threshold (>= 75)
merged_no_duplicates['is_above_threshold'] = merged_no_duplicates['track_popularity'] >= 75

# Convert the boolean column to integer (1 for above threshold, 0 otherwise)
merged_no_duplicates['is_above_threshold'] = merged_no_duplicates['is_above_threshold'].astype(int)

# Group by 'track_artist' and count the number of above-threshold tracks for each artist
artist_popularity = (
    merged_no_duplicates.groupby('track_artist')
    .agg(artist_popularity=('is_above_threshold', 'sum'))
    .reset_index()
)

# Merge the artist popularity back into the original merged dataset
merged_data = pd.merge(merged_data, artist_popularity, on='track_artist', how='left')

merged_data.head()
spotify_df  = merged_data


"""
Building Model to Predict the Song Popularity
"""


"""
Dropping the irrelavant columns
"""

print("Columns in New Spotify Dataset:")
print(spotify_df.columns.tolist())

# Listing the columns to drop
drop_columns = ['track_id', 'track_name', 'track_artist', 'track_album_name', 'track_album_id', 'playlist_name', 'playlist_id', 'playlist_subgenre']
spotify_df = spotify_df.drop(columns=drop_columns)

"""
Dropping the null values
"""

spotify_df.dropna(inplace=True)
print(spotify_df.isna().sum())


# Prepare data
genre_summary = spotify_df.groupby('playlist_genre')[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_popularity']].mean()
genres = genre_summary.index
features = genre_summary.columns

# Normalize data
genre_summary_normalized = genre_summary / genre_summary.max()

# Create radar chart
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Repeat the first value to close the circle
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each genre
for genre in genres:
    values = genre_summary_normalized.loc[genre].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=genre)
    ax.fill(angles, values, alpha=0.2)

# Add labels
ax.set_yticks([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=12)
ax.set_title('Radar Chart of Features and Track Popularity by Genre', fontsize=16)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Save the plot as an image
plt.savefig("radar_chart_features_genre.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


"""
One hot encoding the play_list genre column
"""

#printing the unique value in the playlist_genre column
unique_genres = spotify_df['playlist_genre'].unique()
print("Unique genres in the dataset:")
print(unique_genres)

#one hot encoding the playlist_genre column
spotify_df_encoded = pd.get_dummies(spotify_df, columns=['playlist_genre'], drop_first=False)

print(spotify_df_encoded.head())

#replacing the true with 1 and false with 0 in the one hot encoded playlist_genre column
playlist_genre_columns = [col for col in spotify_df_encoded.columns if col.startswith('playlist_genre_')]
spotify_df_encoded[playlist_genre_columns] = spotify_df_encoded[playlist_genre_columns].astype(int)


"""
Adding release_year column instead of release_date
"""

print(spotify_df_encoded['track_album_release_date'].head())

def parse_date(date):
    try:
        return pd.to_datetime(date, errors='coerce')
    except:
        return pd.NaT

spotify_df_encoded['track_album_release_date'] = spotify_df_encoded['track_album_release_date'].apply(parse_date)
spotify_df_encoded['release_year'] = spotify_df_encoded['track_album_release_date'].dt.year
spotify_df_encoded.drop(columns=['track_album_release_date'], inplace=True)

print(spotify_df_encoded['release_year'])


"""
Printing the range of all the columns
"""

print(spotify_df_encoded.describe().loc[['min', 'max']])

print("Columns in New Spotify Dataset:")
print(spotify_df_encoded.columns.tolist())


# Select relevant columns
relevant_columns = ['Views', 'Likes', 'Comments', 'artist_popularity', 'track_popularity']

# Create the pair plot
pairplot = sns.pairplot(spotify_df_encoded[relevant_columns], diag_kind='kde', corner=True, plot_kws={'alpha': 0.5})

# Add title
plt.suptitle('Pair Plot of Social Media Metrics and Popularity', y=1.02, fontsize=16)

# Save the plot as an image
pairplot.savefig("pair_plot_social_media_popularity.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


"""
Plotting the correlation matrix
"""

correlation_matrix = spotify_df_encoded.corr()
correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']
correlation_long.to_csv('correlation_matrix_long.csv', index=False)


# Compute the correlation matrix
correlation_matrix = spotify_df_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

# Add title
plt.title('Correlation Matrix Heatmap')

# Save the plot as an image
plt.savefig("correlation_matrix_heatmap.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


"""
Feature Engineering
"""

# 1. valence_danceability
# (High Positive Correlation between valence and danceability)
spotify_df_encoded['valence_danceability'] = spotify_df_encoded['valence'] * spotify_df_encoded['danceability']

# 2. acousticness_loudness
# (High Negative Correlation between acousticness and loudness)
spotify_df_encoded['acousticness_loudness'] = spotify_df_encoded['acousticness'] * spotify_df_encoded['loudness']

# 3. energy_loudness
# (High Positive Correlation between energy and loudness)
spotify_df_encoded['energy_loudness'] = spotify_df_encoded['energy'] * spotify_df_encoded['loudness']

# 4. acousticness_energy
# (High Negative Correlation between acousticness and energy)
spotify_df_encoded['acousticness_energy'] = spotify_df_encoded['acousticness'] * spotify_df_encoded['energy']

# 5. valence_danceability_energy
# (Positive Correlation between valence, danceability, and energy)
spotify_df_encoded['valence_danceability_energy'] = spotify_df_encoded['valence'] * spotify_df_encoded['danceability'] * spotify_df_encoded['energy']

# 6. acousticness_energy_loudness
# (Negative Correlation between acousticness, energy, and loudness)
spotify_df_encoded['acousticness_energy_loudness'] = spotify_df_encoded['acousticness'] * spotify_df_encoded['energy'] * spotify_df_encoded['loudness']


# 7. Tempo-related Features
def categorize_tempo(tempo):
    if tempo < 90:
        return 'slow'
    elif 90 <= tempo <= 130:
        return 'medium'
    else:
        return 'fast'

spotify_df_encoded['tempo_category'] = spotify_df_encoded['tempo'].apply(categorize_tempo)


# 8. Mood and Emotion Features
# Function to classify mood based on valence and danceability
def classify_mood(row):
    if row['valence'] > 0.5 and row['danceability'] > 0.5:
        return 'happy_energetic'
    elif row['valence'] < 0.5 and row['danceability'] > 0.5:
        return 'sad_energetic'
    elif row['valence'] > 0.5 and row['danceability'] < 0.5:
        return 'happy_slow'
        # return 'sad_slow'

spotify_df_encoded['mood'] = spotify_df_encoded.apply(classify_mood, axis=1)


# 9. Acoustic Complexity
# Creating a 'clarity' feature which represents the interaction between 'loudness' and 'acousticness'
# This feature gives a sense of the acoustic clarity of the track.
spotify_df_encoded['clarity'] = spotify_df_encoded['loudness'] * (1 - spotify_df_encoded['acousticness'])


# 10. Track Length Features
# Creating a 'duration_minutes' feature by converting track duration from milliseconds to minutes
spotify_df_encoded['duration_minutes'] = spotify_df_encoded['duration_ms'] / 60000

# Function to categorize track duration into short, medium, or long
def categorize_duration(duration):
    if duration < 2:
        return 'short'
    elif 2 <= duration <= 3:
        return 'medium'
    else:
        return 'long'

spotify_df_encoded['duration_category'] = spotify_df_encoded['duration_minutes'].apply(categorize_duration)


# 11. Popularity Prediction Features
# Creating a new feature 'popularity_weighted' by calculating a weighted sum of Likes, Comments, and Views
spotify_df_encoded['popularity_weighted'] = spotify_df_encoded['Likes'] * 0.4 + spotify_df_encoded['Comments'] * 0.3 + spotify_df_encoded['Views'] * 0.3

"""
One hot encoding the tempo_category, 'mood', 'duration_category'
"""

spotify_df_encoded1 = pd.get_dummies(spotify_df_encoded, columns=['tempo_category', 'mood', 'duration_category'], drop_first=False)

print(spotify_df_encoded1.head())

clm = [col for col in spotify_df_encoded1.columns if col.startswith(('mood_', 'tempo_category_', 'duration_category_'))]

spotify_df_encoded1[clm] = spotify_df_encoded1[clm].astype(int)

print("Columns in New Spotify Dataset:")
print(spotify_df_encoded1.columns.tolist())


"""
Model
"""



# Create the final dataset with selected columns
columns = [
    'track_popularity','artist_popularity','release_year', 'valence_danceability', 'acousticness_loudness', 'acousticness_energy', 'energy_loudness', 'valence_danceability_energy', 'acousticness_energy_loudness', 'clarity', 'duration_minutes', 
    'popularity_weighted', 'tempo_category_fast', 'tempo_category_medium', 'tempo_category_slow','mood_happy_energetic', 'mood_happy_slow', 
    'mood_sad_energetic', 'duration_category_long', 
    'duration_category_medium', 'duration_category_short'
]

# Extracting the relevant columns
spotify_final_df = spotify_df_encoded1[columns]

# Normalize the features (excluding the target column 'track_popularity')
scaler = MinMaxScaler()
X = spotify_final_df.drop(columns=['track_popularity'])  # Features
X_scaled = scaler.fit_transform(X)  # Normalized features



# Transform the 'track_popularity' column into ordinal categories (0 to 99 -> bins of 10)
bins = list(range(0, 101, 10))  # Create bins for intervals of 10
labels = range(1, 11)  # Labels for categories (1 to 10)
y = pd.cut(spotify_df_encoded1['track_popularity'], bins=bins, labels=labels, include_lowest=True)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Base models for stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42))
]

# Define the stacking classifier
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of Stacking Classifier: {accuracy}")
print("Classification Report for Stacking Classifier:")
print(classification_report(y_test, y_pred))

# Feature importance visualization
# Note: Feature importance is not directly available for stacking, so we can visualize the base models separately
for model_name, model in base_learners:
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title(f"Feature Importance for {model_name}")
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.show()






# Define the columns to use for the model (including playlist genres)
columns = [
    'track_popularity', 'artist_popularity', 'release_year', 'valence_danceability', 'acousticness_loudness',
    'acousticness_energy', 'energy_loudness', 'valence_danceability_energy', 'acousticness_energy_loudness',
    'clarity', 'duration_minutes', 'popularity_weighted', 'tempo_category_fast', 'tempo_category_medium',
    'tempo_category_slow', 'mood_happy_energetic', 'mood_happy_slow', 'mood_sad_energetic',
    'playlist_genre_edm', 'playlist_genre_latin', 'playlist_genre_pop', 'playlist_genre_r&b', 'playlist_genre_rap', 'playlist_genre_rock'
]

# Extracting the relevant columns from the dataset
spotify_final_df = spotify_df_encoded1[columns]

# Normalize the features (excluding the target column 'track_popularity')
scaler = MinMaxScaler()
X = spotify_final_df.drop(columns=['track_popularity'])  # Features
X_scaled = scaler.fit_transform(X)  # Normalized features

# Transform the 'track_popularity' column into ordinal categories (0 to 99 -> bins of 10)
bins = list(range(0, 101, 10))  # Create bins for intervals of 10
labels = range(1, 11)  # Labels for categories (1 to 10)
y = pd.cut(spotify_df_encoded1['track_popularity'], bins=bins, labels=labels, include_lowest=True)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of Random Forest Classifier: {accuracy}")
print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test, y_pred))

# Feature importance visualization for Random Forest model
feature_importance = rf_model.feature_importances_

# Sort the feature importance in ascending order
sorted_idx = feature_importance.argsort()

# Plot the feature importance
plt.figure(figsize=(12, 8))

# Plot bars in ascending order with a color palette
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx], color='#1f77b4')

# Set y-ticks to be the features in the sorted order
plt.yticks(range(len(feature_importance)), [X.columns[i] for i in sorted_idx], fontsize=12)

# Labeling the axes and title
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Importance for Predicting Song Popularity (Random Forest)', fontsize=16)

# Save the plot as an image
plt.savefig("feature_importance_rf.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.arange(1, 11),
    yticklabels=np.arange(1, 11)
)
plt.title('Confusion Matrix for Song Popularity Prediction')
plt.xlabel('Predicted Popularity')
plt.ylabel('Actual Popularity')

# Save the plot as an image
plt.savefig("confusion_matrix_song_popularity.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()