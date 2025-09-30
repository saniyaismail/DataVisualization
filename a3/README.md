# Visual Analytics workflow

## Installation

To install the required Python modules, run the following command:

```bash
pip install -r requirements.txt
```

Install [gephi](https://gephi.org/) 

## Project Structure

- `task-1/` - Code for task 1
- `task-2/` - Code for task 2
- `task-3/` - Code for task 3
- `images/` - Contains images generated during the visualization processes.
- `README.md` - Project README
- `requirements.txt` - Dependencies to be installed
- `Report.pdf` - Project report
- `demo-video.mp4` - Demo video
- `A3-presentation.pptx` - PPT used in the video

## Task - 1
The code for this task can be found in `task-1/`.

This task focuses on identifying the features that affect song popularity. A model was built to predict the popularity of the song.

### Steps to Execute

### 1. Parallel coordinates plot
- The data cleanup script is available in `data_preparation.py`, where the nulls and duplicates are dropped.
- `index.html` - HTML file for displaying the parallel coordinates plot.
- `script.js` - Contains the JavaScript code for data loading and plot interactions.

- To view the plot, start a simple HTTP server in the folder with the HTML and the Javascript file:

  ```bash
  python -m http.server 8000
  ```

  Open your browser and go to [http://localhost:8000](http://localhost:8000).

  Alternatively, you can use the VSCode Live Server extension.

- **User Interactions**:
  1. **Brushing**: Click and drag on any axis to filter the range of values shown for that attribute. Multiple axes can be brushed simultaneously.
  2. **Axis Reordering**: Drag and rearrange axes to compare specific attributes more effectively.


### 2. Other visualizations and model
- Run `task-1.py` to process `spotify_songs.csv` and `Spotify_Youtube.csv` and produce the images `1..png` to `1..png`. The script also produces `correlation_matrix_long.csv`.



For task-1:

The input files are:
- `spotify_songs.csv`: Dataset used for calculating artist popularity.
- `Spotify_Youtube.csv`: Dataset with youtube and spotify data

The generated file is:
- `correlation_matrix_long.csv`

The images are:
- Unrolled visual analytics feedback loop: `1.1.png`.
- Parallel coordinates plot: `1.2.png` to `1.6.png`.
- Radar chart: `1.7.png`.
- Pair Plot: `1.8.png`.
- Correlation matrix heatmap: `1.9.png`.
- Classification report of Stacking classifier: `1.10.png`
- Classification report of Random Forest classifier: `1.11.png`
- Confusion matrix: `1.12.png`.
- Feature importance random forest: `1.13.png` .


## Task - 2
The code for this task can be found in `task-2/`.

The focus of this task is on extracting deeper insights into the dynamics of genre trends, identifying patterns, and leveraging machine learning models for forecasting.


### Steps to Execute

### 1. Merge datasets
Run `merged_dataset.py` to merge `spotify_songs.csv` and `final.csv` and produce `merged_df.csv`.

### 2. Visualizations and model
Run `task-2.py` to produce the visualzation images `2.2.png` to `2..png` and train the forecasting model.

For task-2:

The input files are:
- `spotify_songs.csv`: Dataset used for calculating artist popularity.
- `final.csv`: Dataset with regional data

The generated file is:
- `merged_df.csv`

The images are:
- Unrolled visual analytics feedback loop: `2.1.png`
- Choropleth showing Dominant Genre by Popularity: `2.2.png`
- Choropleth showing Dominant Genre by Song Production: `2.3.png`
- Choropleth showing Dominant Genre by Total Number of Streams: `2.4.png`
- Line charts to show the seasonal trends in weeks on chart by genre: `2.5.png`
- Cross-genre vs solo-genre analysis based on popularity: `2.6.png`
- Cross-genre vs solo-genre analysis based on total streams: `2.7.png`
- Cross-genre vs solo-genre analysis based on number of weeks on charts a track has streamed: `2.8.png`
- Elbow method to find optimal number of clusters: `2.9.png`
- Silhouette scores to find optimal number of clusters: `2.10.png`
- Countries clustered based on their genre preferences: `2.11.png`
- Chord diagram showing language genre relationship based on the number of streams: Asian languages: `2.12.png`
- Chord diagram showing language genre relationship based on the number of streams: European languages: `2.13.png`
- Pop genre node selected to display its connections with the European languages: `2.14.png`
- Spanish language node selected to highlight its relation to different genres: `2.15.png`
- rb popularity forecast: `2.16.png`
- rap popularity forecast: `2.17.png`
- rock popularity forecast: `2.18.png`
- pop popularity forecast: `2.19.png`
- edm popularity forecast: `2.20.png`
- latin popularity forecast: `2.21.png`



## Task - 3
The code for this task can be found in `task-3/`.

This task focuses on analyzing artist popularity and collaborations and provides visualizations and forecasts. 

### Steps to Execute  

### 1. Generate Artist Popularity Data  
Run `count_above_threshold.py` to process `spotify_songs.csv` and generate `artists.csv`.  
This file contains the calculated artist popularity metrics required for further analysis.  

### 2. Visualize Artist Collaboration Network  
Run `graph_visualizations.py` to process collaboration data and generate the following:  
  - `cleaned_regional_data.csv`: Artist collaboration data used in the forecasting model.  
  - `filtered_nodes.csv` and `filtered_edges.csv`: Input files for graph visualizations in Gephi.  

### 3. **Graph Visualizations**:  
  The Gephi project (`collab_graph.gephi`) contains the network visualizations of artist collaborations using `filtered_nodes.csv` and `filtered_edges.csv`.  
  - **Graph Images**:  
    - `3.2.png`, `3.3.png`, `3.4.png`, and `3.5.png`: Various views of the artist collaboration network.  
    - `3.5.png` is a zoomed-in version of `3.5.1.png` for better clarity.  

### 4. Generate Clustering and Forecasting Visualizations  
Run `models.py` to generate the following visualizations:  
- **Line Graph (Artist Popularity Over Time)**:  
  - `3.6.png`: Trends in artist popularity over time.  

- **Clustering Model Visualizations**:  
  - `3.7.png`, `3.8.png`, and `3.9.png`: Visual representations of artist clusters.  

- **Forecasting Model Visualizations**:  
  - `3.10.png`, `3.11.png`, `3.12.png`, `3.13.png`, and `3.14.png`: Forecasting results for each cluster.  


For task-3:

The input files are:
- `spotify_songs.csv`: Dataset used for calculating artist popularity.  
- `final.csv`: File for collaboration analysis and graph generation.  
- `Spotify Most Streamed Songs.csv`: File with playlist and chart data across different platforms.

The files generated are:
- `artists.csv`: Contains artist popularity metrics.  
- `cleaned_regional_data.csv`: Collaboration data for forecasting.  
- `filtered_nodes.csv` and `filtered_edges.csv`: Files for Gephi network visualizations.  

The images generated are:
- Unrolled visual analytics feedback loop: `3.1.png` 
- Collaboration graphs: `3.2.png` to `3.5.png` (and `3.5.1.png`).  
- Line graph (popularity trends): `3.6.png`.  
- Clustering visualizations: `3.7.png` to `3.9.png`.  
- Forecasting visualizations: `3.10.png` to `3.14.png`.  

Gephi Project:  
- `collab_graph.gephi`: View graphs of artist collaborations in Gephi.

