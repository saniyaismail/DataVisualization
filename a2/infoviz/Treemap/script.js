function updateVisualization() {
    const selectedTreemap = document.getElementById("treemapDropdown").value;
    const layoutType = document.getElementById("layoutDropdown").value;
    const colorScheme = document.getElementById("colorDropdown").value;

    document.getElementById("yearRange").classList.toggle("hidden", selectedTreemap !== "yearArtist");

    if (selectedTreemap === "genreSubgenre") {
        loadGenreSubgenreTreemap(layoutType, colorScheme);
    } else if (selectedTreemap === "yearArtist") {
        loadYearArtistTreemap(layoutType, colorScheme);
    } else if (selectedTreemap === "genreArtist") {
        loadGenreArtistTreemap(layoutType, colorScheme);
    }
}

function loadGenreSubgenreTreemap(layoutType, colorScheme) {
    d3.csv("genre_subgenre_data.csv").then(function(data) {
        const rootLabel = "All Genres";
        const genreSubgenreData = data.map(d => ({
            playlist_genre: d.playlist_genre,
            playlist_subgenre: d.playlist_subgenre,
            track_popularity: +d.track_popularity,
            rank: +d.rank
        }));

        const labels = [rootLabel];
        const parents = [""];
        const values = [0];
        const text = [""];
        const genreLabels = [...new Set(genreSubgenreData.map(d => d.playlist_genre))];

        genreLabels.forEach(genre => {
            labels.push(genre);
            parents.push(rootLabel);
            values.push(0);
            text.push(`Genre: ${genre}`);
        });

        genreSubgenreData.forEach(d => {
            labels.push(d.playlist_subgenre);
            parents.push(d.playlist_genre);
            values.push(d.track_popularity);
            text.push(`Subgenre: ${d.playlist_subgenre}<br>Popularity: ${d.track_popularity.toFixed(2)}<br>Rank: ${d.rank}`);
        });

        Plotly.newPlot('treemapContainer', [{
            type: "treemap",
            labels: labels,
            parents: parents,
            values: values,
            marker: {
                colors: values,
                colorscale: colorScheme,
                cmin: Math.min(...values),
                cmax: Math.max(...values),
                colorbar: { title: 'Popularity' }
            },
            text: text,
            hoverinfo: "text",
            textinfo: "label+value",
            tiling: { packing: layoutType }
        }], {
            title: 'Treemap of Genres and Subgenres by Popularity',
            width: 1000,
            height: 1000
        });
    });
}

function loadYearArtistTreemap(layoutType, colorScheme) {
    const startYear = parseInt(document.getElementById("startYear").value);
    const endYear = parseInt(document.getElementById("endYear").value);

    if (startYear > endYear) {
        alert("Start year should be less than or equal to end year.");
        return;
    }

    d3.csv("year_artist_data.csv").then(function(data) {
        const rootLabel = "All Years";
        const yearArtistData = data
            .filter(d => +d.year >= startYear && +d.year <= endYear)
            .map(d => ({
                year: d.year,
                track_artist: d.track_artist,
                track_popularity: +d.track_popularity,
                year_avg_popularity: +d.year_avg_popularity,
                rank: +d.rank
            }));

        const labels = [rootLabel];
        const parents = [""];
        const values = [0];
        const text = [""];
        const yearLabels = [...new Set(yearArtistData.map(d => d.year))];

        yearLabels.forEach(year => {
            labels.push(year);
            parents.push(rootLabel);
            values.push(Math.max(...yearArtistData.filter(d => d.year === year).map(d => d.year_avg_popularity)));
            text.push(`Year: ${year}<br>Average Popularity: ${values[values.length - 1].toFixed(2)}`);
        });

        yearArtistData.forEach(d => {
            labels.push(d.track_artist);
            parents.push(d.year);
            values.push(d.track_popularity);
            text.push(`Artist: ${d.track_artist}<br>Popularity: ${d.track_popularity.toFixed(2)}<br>Rank: ${d.rank}`);
        });

        Plotly.newPlot('treemapContainer', [{
            type: "treemap",
            labels: labels,
            parents: parents,
            values: values,
            marker: {
                colors: values,
                colorscale: colorScheme,
                cmin: Math.min(...values),
                cmax: Math.max(...values),
                colorbar: { title: 'Popularity' }
            },
            text: text,
            hoverinfo: "text",
            textinfo: "label+value",
            tiling: { packing: layoutType }
        }], {
            title: `Top 10 Artists by Year (${startYear}-${endYear})`,
            width: 1000,
            height: 1000
        });
    });
}

function loadGenreArtistTreemap(layoutType, colorScheme) {
d3.csv("genre_artist_popularity_trackcount.csv").then(function(data) {
    const rootLabel = "All Genres"; // Root node for the third treemap

    // Step 1: Calculate the total track count for each genre
    const genreTrackCountMap = {};
    data.forEach(d => {
        if (!genreTrackCountMap[d.playlist_genre]) {
            genreTrackCountMap[d.playlist_genre] = 0;
        }
        genreTrackCountMap[d.playlist_genre] += +d.track_count;
    });

    // Step 2: Calculate the total sum for the root node ("All Genres")
    const totalTrackCount = Object.values(genreTrackCountMap).reduce((acc, count) => acc + count, 0);

    // Step 3: Define labels, parents, values, text, and colorValues arrays
    const labels = [rootLabel];
    const parents = [""];
    const values = [totalTrackCount]; // Set the root node value to the total of all genres
    const text = [""]; // Tooltips
    const colorValues = [0]; // Use 0 or any default value for the root color

    // Add each genre as a child of the root, using the track count as its value
    Object.keys(genreTrackCountMap).forEach(genre => {
        labels.push(genre);
        parents.push(rootLabel);
        values.push(genreTrackCountMap[genre]);
        text.push(`Genre: ${genre}<br>Total Track Count: ${genreTrackCountMap[genre]}`);
        colorValues.push(0); // Placeholder for genre nodes; could be an average or another value if desired
    });

    // Add each artist with their popularity and track count as children of genres
    data.forEach(d => {
        labels.push(`${d.track_artist} (${d.playlist_genre})`);
        parents.push(d.playlist_genre);
        values.push(+d.track_count);
        text.push(`Artist: ${d.track_artist}<br>Genre: ${d.playlist_genre}<br>Popularity: ${(+d.track_popularity).toFixed(2)}<br>Track Count: ${d.track_count}`);
        colorValues.push(+d.track_popularity); // Use track popularity for coloring
    });

    const colorOptions = {
    "Viridis": "Viridis",
    "Cividis": "Cividis",
    "Plasma": "Plasma",
    "Turbo": "Turbo",
    "Bluered": "Bluered",
    "YlGnBu": "YlGnBu",
    "RdBu": "RdBu"
};

    // Function to update the treemap with a selected layout
    
    Plotly.newPlot('treemapContainer', [{
            type: "treemap",
            pathbar: { visible: true },
            labels: labels,
            parents: parents,
            values: values,
            marker: { 
                colors: colorValues, // Use colorValues for popularity-based coloring
                colorscale: colorScheme,
                cmin: Math.min(...colorValues),
                cmax: Math.max(...colorValues),
                colorbar: {title: 'Popularity'}
            },
            text: text,
            hoverinfo: "text",
            textinfo: "label+value",
            branchvalues: "total",
            tiling: { packing: layoutType },
        }], {

        
            title: "Top Artists by Genre, Popularity, and Track Count",
            width: 1000,
            height: 1000,
            coloraxis: { colorbar: { title: "Popularity" } }
        });

    });
}
document.getElementById("treemapDropdown").addEventListener("change", updateVisualization);

// Initialize with default treemap
updateVisualization();