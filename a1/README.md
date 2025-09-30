 # DataVisualization

## Code

`dataset_preparation.py` consists of the code used to cleanup the data.
1. Songs with any null features were dropped
2. Duplicate songs appearing in several playlists were dropped
3. Duplicate songs appearing more than once in the same album were dropped

`correlation.py` consists of the code used to prepare the heatmap in Fig 1.1<br>
Python was preferred over tableau for this visualization as code was required to calculate the correlation between features.

Run `dataset_preparation.py` to generate the csv files required for `correlation.py` and the tableau workbooks.

All the data files (.csv) have been uploaded in the Data files folder. Upload the respective csv files for each task to use the tableau workbook of the task.

## Images

1. <b>Task 1</b>
  - 1.1 - The heatmap is chosen to depict correlations between variables as it provides a clear, visual representation of relationships. The X and Y axes represent the different variables, and the color gradients highlight the strength of correlations, with cooler colors indicating weaker correlations and warmer colors representing stronger ones.
  - 1.2 - Scatter plot depicting the positive correlation between the valence and danceablity. X axis - Danceability, Y axis - Valence
  - 1.3 - Box plot depicting the valence for different acoustic range for different genres. X axis - Acoustic Range, Y axis - Valence. The marks (boxes and whiskers) provide a clear view of central tendency, spread, and outliers, while different colors for genres for easy identification
         1.3.1 - edm
         1.3.2 - latin
         1.3.3 - pop
         1.3.4 - r&b
         1.3.5 - rap
         1.3.6 - rock   
  - 1.4 - Bar chart depicting avg valence and danceability for different genre
          1.4.1 - valence vs genre. X axis - Genre, Y axis - Valence
          1.4.2 - danceability vs genre. X axis - Genre, Y axis - Danceability
  - 1.5 - Line graph depicting avg valence and danceability over year
          1.5.1 - valence vs genre. X axis - Year, Y axis - Valence
          1.5.2 - danceability vs genre. X axis - Year, Y axis - Danceability
  - 1.6 - Scatter plot depicting the positive correlation between the Loudness and Energy. X axis - Loudness, Y axis - Energy
  - 1.7 - Box plot depicting the Loudness for different genres. X axis - Genre, Y axis - Loudness. The marks (boxes and whiskers) provide a clear view of central tendency, spread, and outliers, while different colors for genres for easy identification
  - 1.8 - Box plot depicting the Energy for different genres. X axis - Genre, Y axis - Energy. The marks (boxes and whiskers) provide a clear view of central tendency, spread, and outliers, while different colors for genres for easy identification
  - 1.9 - Line chart depicting avg loudness and avg for different year for different genre
         1.9.1 - edm
         1.9.2 - latin
         1.9.3 - pop
         1.9.4 - r&b
         1.9.5 - rap
         1.9.6 - rock
2. <b>Task 2</b>
  - 2.1 - Bar chart depicting the number of unique songs released by each genre, highlighting which genres are most prolific in terms of music production.<br>
          2.1.1 - Marks: Bars
          2.1.2 - X-axis: Playlist Genre 
          2.1.3 - Y-axis: Distinct count of Track ID (Unique Song Count) 
          2.1.4 - Color: Different colors for each genre to distinguish them
    
  - 2.2 - The line chart displays the number of songs released each year by genre, revealing trends in music production over time.
2.2.1 - Marks: Lines.
2.2.2 - X-axis: Year of Track Album Release Date.
2.2.3 - Y-axis:Y-axis: Distinct count of Track ID (Unique Song Count).

- 2.3 - This line chart shows the average popularity of songs in each genre over time, illustrating which genres have gained or lost popularity.<br>
2.3.1 - Marks: Lines.
2.3.2 - X-axis: Year of Track Album Release Date
2.3.3 - Y-axis: Average Track Popularity.
2.3.4 - Color: Separate colors for each genre to distinguish trends.

- 2.4 - Treemaps displaying the most popular songs of 2019 across multiple genres 
2.4.1 - Marks: Tiles.
2.4.2 - Color: Represents average track popularity (darker for higher popularity).

- 2.5 - This bar chart shows the average track popularity for each genre, highlighting which genres have the most popular songs on average.<br>
2.5.1 - Marks: Bars.
2.5.2 - X-axis: Playlist Genre.
2.5.3 - Y-axis: Average Track Popularity.
2.5.4 - Color: Each bar is colored differently to represent different genres, making them easily distinguishable.

- 2.6 - The box plot shows the distribution of track popularity within each genre, illustrating the range of popularity from minimum to maximum, along with the median.<br>
2.6.1 - Marks: Boxes and whiskers.
2.6.2 - X-axis: Playlist Genre.
2.6.3 - Y-axis: Track Popularity.
2.6.4 - Color: Separate colors for each genre to distinguish them.

- 2.7 - This treemap visualizes the contribution of subgenres to their parent genre's popularity, with larger tiles representing more tracks and darker colors showing higher popularity.<br>
2.7.1 - Marks: Tiles.
2.7.2 - Size: Count of Tracks in each subgenre.
2.7.3 - Color: Represents average track popularity (darker for higher popularity).

- 2.8 - This box plot represents the distribution of song popularity within albums, focusing on albums with more than 13 tracks to assess consistency or variation in popularity.<br>
2.8.1 - Marks: Boxes and whiskers.
2.8.2 - X-axis: Track Album Name.
2.8.3 - Y-axis: Average Track Popularity.
2.8.4 - Color: Different colors for each album to represent popularity distribution.
3. <b>Task 3</b>
  - 3.1 - Bar Chart depicting the top 20 artists based on average track popularity. X axis - Track popularity, Y axis - Artist name
  - 3.2 - Bar Chart depicting the top 20 artists based on count above threshold. X axis - Count above threshold, Y axis - Artist name
         Count above threshold is a calculated field quantify how many tracks an artist has with a popularity score greater than a specific threshold (75th percentile of track popularity)

Marks and channels used for 3.1 and 3.2 - Bars, grouped by artist and length of the bar shows popularity/count, with artists labeled on the Y-axis.
Reason: Bar charts make it easy to compare the top artists based on the selected measure (average popularity, count above threshold).

  - 3.3 - Scatter plot for count above threshold vs track count per artist. Each point represents an artist. Artists have been grouped into tiers (lesser known, mid-tier, popular) <br>
          • count above threshold ≥ 13 - top tier artist <br>
          • count above threshold ≥ 2 - mid tier artist <br>
          • count above threshold < 2 - lesser known artist <br>
  - 3.4 - Scatter plot for average track popularity vs track count per artist. Each point represents an artist. Artists have been grouped into tiers (lesser known, mid-tier, popular) <br>
          • average track popularity ≥ 75 - top tier artist <br>
          • average track popularity ≥ 50 - mid tier artist <br>
          • average track popularity < 50 - lesser known artist <br>
  - 3.5 - Scatter plot for relationship between the two different popularity measures, grouped by artist tier. X axis - Track popularity, Y axis - count above threshold. Each point represents an artist.

Marks and channels used for 3.3, 3.4, 3.5 - Points, grouped by artist and position on the X and Y axes for the artist's popularity/track count, grouped by tiers. Each tier is represented by a colour.
Reason: Scatter plots clearly show trends and outliers, highlighting relationships between popularity measures and track count.

  - 3.6 - Heatmap for songs by top artists, genre-wise. Darker the colour more the songs released by the artist for the genre.

Marks and channels used for 3.6 - Colour gradient, grouped by artist and genre, darker colour intensity represents more songs.
Reason: Heatmaps effectively show song distribution across genres, emphasizing concentration of releases by artists

  - 3.6.1 to 3.6.6 - Tree map for the top songs genre wise, grouped by artist. Darker the colour, more the top songs released by the artist for the genre. <br>
          Note - In the report, only 3.6.1 and 3.6.2 were displayed to contrast between two genres <br>
          3.6.1 - EDM <br>
          3.6.2 - R&B <br>
          3.6.3 - Latin <br>
          3.6.4 - Pop <br>
          3.6.5 - Rap <br>
          3.6.6 - Rock <br>
Marks and channels used for 3.6.1 to 3.6.6 - coloured blocks representing artists' song counts and colour intensity reflects the number of top songs by an artist in a genre.
Reason: Treemaps give an overview of artists' dominance within genres.

  - 3.7  - Plot of 3 musical features for top artists (acousticness, danceability, loudness) taken as an average over all the songs of the artist. Polygon marks are chosen to help visualize how an artist’s song features compare to others.

Marks and channels used for 3.7 - Polygons, with vertices representing feature averages (acousticness, danceability, loudness) and different colours used for different features
Reason: Polygons allow easy comparison of multiple musical features for different artists.

  - 3.8 - Line chart of artist popularity trends for the top 7 artists. Line charts are ideal for visualizing changes over time, helping us see trends in how artists maintain or lose popularity. By focusing on a few top artists, this chart avoids overcrowding and clearly shows shifts over years.
  - 3.9 - Line chart for number of popular songs released year-wise, grouped by artist popularity tier

Marks and channels used for 3.8 and 3.9 - Lines, with points representing artist popularity over time or the number of popular songs per year. Colour is used in 3.9 to differentiate between different artist tiers.
Reason: Line charts are best for showing trends over time
     
  
