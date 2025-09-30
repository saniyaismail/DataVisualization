# Scientific and Information Visualization

This project involves creating scientific and information visualizations using various Python libraries and tools.
## Installation

To install the required Python modules, run the following command:

```bash
pip install -r requirements.txt
```

Install [gephi](https://gephi.org/) 

## Project Structure

- `infoviz/` - Code for information visualization tasks.
- `scivis/` - Code for scientific visualization tasks.
- `images/` - Contains images generated during the visualization processes.
- `gifs/` - Contains GIFs generated during the visualization processes.
- `README.md` - Project README
- `requirements.txt` - Dependencies to be installed
- `Report.pdf` - Project report

The images in `images/` are labelled as follows:

- 1.jpg - Hurricane Ida
- 1.1.1 - 1.1.13 : Quiver Plot
- 1.2.1 - 1.2.22 : Colour Mapping
- 1.3.1 - 1.3.24 : Contour Mapping
- 2.1.1 - 2.1.11 : Node Link
- 2.2.1 - 2.2.8 : Parallel Coordinates Plot
- 2.3.1 - 2.3.8 : Treemap

---

## Scientific Visualization

This section focuses on visualizing meteorological data during the months of August, September, and October 2021. The data was chosen to visualize Hurricane Ida. The selected dates—August 26, August 29, September 1, September 4, and September 10—capture the storm's progression and aftermath.

### Quiver Plot

Code for generating quiver plots is in the `quiver_plot/` subfolder under `scivis/`.

- **NetCDF Files** in the `data/` folder:
  - `th_2021.nc` - Contains wind direction data.
  - `vs_2021.nc` - Contains wind speed data.

- **Code Workflow**:
  - The script `sampling.py` reads the NetCDF files using the `xarray` library and saves data for specific dates into separate files. This script must be run first before `quiver_plot.py` is run.

  - In the script `quiver_plot.py` Data is further processed and saved into a `pandas.DataFrame` for analysis. The following grid sampling experiments are done:
    1. **Simple Subsampling** - Choosing every nth data point.
    2. **Meshgrid Sampling** - Interpolates wind vectors onto a defined grid.

  - Quiver plots for all sampled dates are created using `matplotlib` and `cartopy` in the `quiver_plot.py`. 
  - Two types of quiver plots are generated:
    1. Quiver plots with same-sized vectors, colored by magnitude using the Viridis colormap.
    2. Quiver plots with magnitude-proportional vectors.

  - GIFs are created using the `imageio` library in the `generate_gif.py` script.
    1. `magnitude_proportional.gif` - Wind vectors with magnitude proportional sizing.
    2. `same_sized_with_colour.gif` - Wind vectors with same sizing, colored by magnitude.

---

### Color Mapping

Color maps using various **color palettes** and **scales** are plotted using `matplotlib` and `cartopy` to interpret precipitation and temperature trends. The visualizations highlight the influence of color choices and scaling methods on readability and clarity of data representation.

Code for generating color maps is in the `color_mapping/` subfolder under `scivis/`.

#### Code Structure

- **Scripts**:
  - `color_mapping.py`: Main script for loading data, applying color maps, and generating visualizations.
- **Data Files**:
  - `pr_2021.nc` - Precipitation data.
  - `tmmx_2021.nc` - Temperature data.
  - `sample_dates.csv` - Specific dates of interest.

#### Downloading Data

To download the required NetCDF files for precipitation and temperature, follow these steps:

1. Go to the [Northwest Knowledge Network METDATA page](https://www.northwestknowledge.net/metdata/data/).

2. Scroll down to find the links to download specific data files.

3. Download the following files:
   - **Precipitation**: `pr_2021.nc`
   - **Maximum Temperature**: `tmmax_2021.nc`

#### Experiments with Color Palette and Scales

1. **Precipitation Visualization**:
   - **Color Palettes**: `YlGnBu`, `Viridis`, and `coolwarm`.
   - **Scale Types**: Discrete, logarithmic, and continuous scales were used to compare visualization clarity.
   - A discrete scale was effective for precipitation to highlight different intensity levels

2. **Temperature Visualization**:
   - **Color Palettes**: `hot`, `coolwarm`, and `RdYlBu`.
   - **Scale Types**: Continuous, logarithmic and discrete scales were tried
   - A continuous scale with `coolwarm` palette for its smooth transition from cool to warm colors was found effective as it aligns naturally with temperature data.

#### GIF and Image Generation
- For the dates in the `sample_dates.csv` color maps have been generated and are stored in `images/` folder.
- For both temperature and precipitation, GIFs were generated from the images using `imageio` and saved in the `gifs/` folder in the `color map/` folder.
- Generated GIFs: 
    1. `temperature.gif` 
    2. `precipitation.gif`

---

### Contour Mapping

Code for generating contour plots is in the `contour_mapping/` subfolder under `scivis/`.

#### Code Structure

- **Scripts**:
  - `contour_mapping.py`: Main script for loading data, applying contour maps.
- **Data Files**:
  - `vpd_2021.nc`: Contains VPD data.
  - `sph_2021.nc`: Contains Specific Humidity data.
  - `sample_dates_2.csv`: Contains sample dates.

#### Downloading Data Files

To download the required NetCDF files for specific humidity and vapour pressure deficit, follow these steps:

1. Go to the [Northwest Knowledge Network METDATA page](https://www.northwestknowledge.net/metdata/data/).

2. Scroll down to find the links to download specific data files.

3. Download the following files:
   - **Specific humidity**: `sph_2021.nc`
   - **Vapour pressure deficit**: `vpd_2021.nc`

#### Code Workflow
   - Reading the NetCDF files (`vpd_2021.nc` and `sph_2021.nc`) using the `netCDF4` library.
   - Defining the `plot_contour` function to generate filled or line contour plots based on VPD or Specific Humidity data.
   - `plot_contour` has parameters to choose the datafile, sample dates, line(marching square contour) or the filled contour, colourmap, level, max/min.

#### Experiments
  1. **Filled Contour**
  2. **Marching Square Contour**

#### Generated Plots
- **Vapor Pressure Deficit (VPD)**:
  - **Filled Contour Plots**: To visualize VPD data with filled color contours.
  - **Marching Square Contour Plots**: To display marching square contour lines for Specific Humidity

- **Specific Humidity (SPH)**:
  - **Filled Contour Plots**: To visualize Specific Humidity data using color-filled contours.
  - **Marching Square Contour Plots**: To display marching square contour lins for Specific Humidity.

#### GIF Creation
- GIFs are generated contour plots using the `imageio` library to visualize temporal changes over sample dates.
- The GIFs are generated and saved in the `gifs/` folder.
- - Generated GIFs: 
    1. `vph_filled_contour.gif` 
    2. `vph_marching_square_contour.gif` (marching square contour)
    3. `sph_filled_contour.gif` 
    4. `sph_marching_square_contour.gif` (marching square contour)

---

## Information Visualization
### Node Link Diagram
[ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) dataset is used for visualizing the nodelink diagram.

#### Requirements
- Python librarys: **networkx**
- **Gephi**: A visualization platform to display and explore large network

Code for generating node link graphs is in the `NodeLink/` subfolder under `infovis/`.

#### Code Structure

- **Scripts**:
  - `nodelink.py`: script for generating files to be uploaded to gephi.
- **Data Files**:
  - `facebook_combined.txt`: Contains the combined edges of the Facebook social network.
  - `ego_id.edges`: Contains the edges specific to a particular ego network (user).
  - `ego_id.circles`: Defines the community or circle information for a specific ego network.

#### Data Files


#### Generating .gefx files
- The Facebook network data is read from `facebook_combined.txt`, with nodes and edges being added to a NetworkX graph.
- Ego networks are identified by specific `ego_id` values, and for each ego network, edges and community information are loaded from corresponding `.edges` and `.circles` files.
- Each ego network's nodes are assigned a community (circle) based on the data in the `.circles` files.
- The community assignments are added as node attributes in the NetworkX graph, allowing for community detection and visualization.
- The processed graph is saved to a `.gexf` file, which can then be imported into Gephi for further analysis and visualization.

#### Task performed in Gephi
- Visualization of Node Groups Based on Ego Network.
- Visualization of Ego_348 Subgraph Based on Community.
- Visualization of Ego_348 Subgraph Based on Degree.

#### Graph Layout Algorithms used
- Force Atlas 2 Algorithm
- Yifan Hu Layout
- Fruchterman-Reingold Algorithm
- Radial Axis Layout

---


### Parallel Coordinates Plot

The code for the visualization is in the `infoviz/pcp` folder. The data used is `spotify_songs_without_duplicates.csv`.

#### Data Cleanup

The following steps were performed to clean the dataset (available at [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs?select=spotify_songs.csv)):

1. **Handling Missing Data**:  
   Songs with any missing values were dropped since they comprised a small percentage of the dataset.

2. **Duplicate Removal**:  
   - Duplicate song entries across multiple playlists were removed to ensure accurate analysis of unique songs and artists.
   - Duplicate combinations of `track_id` and `track_album_id` were dropped to ensure each track was counted only once per album. Only albums with more than one track were included.

The data cleanup script is available in `data_preparation.py`.

#### Visualization

- **HTML and JavaScript Files**:
  - `index.html` - HTML file for displaying the parallel coordinates plot.
  - `script.js` - Contains the JavaScript code for data loading and plot interactions.

- **Viewing the Plot**:
  To view the plot, start a simple HTTP server in the folder with the HTML and the Javascript file:

  ```bash
  python -m http.server 8000
  ```

  Open your browser and go to [http://localhost:8000](http://localhost:8000).

  Alternatively, you can use the VSCode Live Server extension.

- **User Interactions**:
  1. **Brushing**: Click and drag on any axis to filter the range of values shown for that attribute. Multiple axes can be brushed simultaneously.
  2. **Axis Reordering**: Drag and rearrange axes to compare specific attributes more effectively.

---

### Treemap

Code for generating Treemaps is in the `Treemap/` subfolder under `infovis/`.

Three different treemaps have been created to represent various aspects of the dataset:

1. **Top 10 Artists by Year**: Shows the most popular artists for each year, allowing users to identify trends in artist popularity over time.
2. **Top Artists by Genre and Track Count**: Displays the most prolific artists within each genre based on their track count, revealing trends in genre-specific popularity.
3. **Genre and Subgenre Popularity**: Highlights the popularity distribution across genres and their subgenres, showing which subgenres contribute most to each genre's overall popularity.

#### Dataset

The data used for these treemaps was derived from `spotify_songs.csv` after data processing to handle duplicates and remove incomplete entries. Processed data is saved in the following CSV files:
   - `top_artists_by_year.csv` - Used for the "Top 10 Artists by Year" treemap.
   - `top_artists_by_genre.csv` - Used for the "Top Artists by Genre and Track Count" treemap.
   - `genre_and_subgenre_popularity.csv` - Used for the "Genre and Subgenre Popularity" treemap.

Data preparation steps and CSV file generation are managed in the `generate_csv.py`.

#### Visualization Files

- **HTML and JavaScript Files**:
  - `index.html` - HTML file for rendering the treemaps.
  - `script.js` - Contains JavaScript code to load data and handle plot interactions.
  - `style.css` - CSS file for basic styling of the page.

#### Viewing the Treemaps

To view the treemaps, start a simple HTTP server in the folder containing the HTML and JavaScript files:

```bash
python -m http.server 8000
```

Then open the browser and go to [http://localhost:8000](http://localhost:8000).

The VSCode Live Server extension can also be used for easier preview.

#### User Interactions

The treemaps support the following interactions:

- **Hover**: Hovering over a block in the treemap reveals additional information, such as artist or subgenre popularity, rank, and track count (where applicable).
- **Click to Zoom**: Clicking on a block zooms into the selected genre or subgenre, allowing users to explore the dataset at different levels of detail.
- **Year Selection**: Users can select a specific year to view data for that period (available in the "Top 10 Artists by Year" treemap).
- **Colour palette selction**: Users can select their desired color palette from a list that is displayed to them.

This interactive functionality enables a more detailed exploration of trends in artist and genre popularity within the dataset.
