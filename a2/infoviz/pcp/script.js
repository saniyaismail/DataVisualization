// Load the CSV data using D3.js
d3.csv('spotify_songs_without_duplicates.csv').then(function(rows) {
    // Function to unpack data
    function unpack(rows, key) {
        return rows.map(function(row) {
            return row[key];
        });
    }

    // Map genres to numerical IDs
    const genreMap = {};
    
    const colorScale = [
        [0.0, "#0000FF"], [1 / 6, "#0000FF"],
        [1 / 6, "#00FFFF"], [2 / 6, "#00FFFF"],
        [2 / 6, "#00FF00"], [3 / 6, "#00FF00"],
        [3 / 6, "#FFFF00"], [4 / 6, "#FFFF00"],
        [4 / 6, "#FF7F00"], [5 / 6, "#FF7F00"],
        [5 / 6, "#FF0000"], [1.0, "#FF0000"]
    ];
        

    let genres = [...new Set(unpack(rows, 'playlist_genre'))]; // Unique genres
    
    genres.forEach((genre, index) => {
        genreMap[genre] = index; // Map genre to index
    });

    // Assign genre IDs to rows
    rows.forEach(row => {
        row.genre_id = genreMap[row.playlist_genre];
    });

    // Define the data for the parallel coordinates plot
    var data = [{
        type: 'parcoords',
        pad: [80, 80, 80, 80],
        line: {
            color: rows.map(row => row.genre_id), // Use genre_id for coloring
            colorscale: colorScale, // Use the specified discrete color scale
            showscale: true, // Show color scale
            colorbar: {
                title: 'Playlist Genre',
                tickvals: Object.values(genreMap),
                ticktext: Object.keys(genreMap)
            }
        },
        dimensions: [
            {
                range: [0, 90], // track_popularity range
                label: 'Track Popularity',
                values: unpack(rows, 'track_popularity')
            },
            {
                range: [0, 5], // genre_id range
                label: 'Genre',
                values: unpack(rows, 'genre_id'),
                tickvals: [0, 1, 2, 3, 4, 5], // Specify tick positions
                ticktext: ['pop', 'rap', 'rock', 'latin', 'r&b', 'edm'] // Specify tick labels
            },
            {
                range: [0, 1], // danceability range
                label: 'Danceability',
                values: unpack(rows, 'danceability')
            },
            {
                range: [0, 1], // energy range
                label: 'Energy',
                values: unpack(rows, 'energy')
            },
            {
                range: [0, 11], // key range
                label: 'Key',
                values: unpack(rows, 'key')
            },
            {
                range: [-46.448, 11], // loudness range
                label: 'Loudness',
                values: unpack(rows, 'loudness')
            },
            {
                range: [0, 1], // mode range
                label: 'Mode',
                values: unpack(rows, 'mode'),
                tickvals: [0, 1]
            },
            {
                range: [0, 1], // speechiness range
                label: 'Speechiness',
                values: unpack(rows, 'speechiness')
            },
            {
                range: [0, 1], // acousticness range
                label: 'Acousticness',
                values: unpack(rows, 'acousticness')
            },
            {
                range: [0, 0.994], // instrumentalness range
                label: 'Instrumentalness',
                values: unpack(rows, 'instrumentalness')
            },
            {
                range: [0, 0.996], // liveness range
                label: 'Liveness',
                values: unpack(rows, 'liveness')
            },
            {
                range: [0, 1], // valence range
                label: 'Valence',
                values: unpack(rows, 'valence')
            },
            {
                range: [0, 240], // tempo range
                label: 'Tempo',
                values: unpack(rows, 'tempo')
            },
            {
                range: [4000, 517810], // duration_ms range
                label: 'Duration (ms)',
                values: unpack(rows, 'duration_ms')
            }
        ]
    }];

    // Define the layout for the plot
    var layout = {
        title: 'Parallel Coordinates Plot of Spotify Songs',
        width: 1200,
        height: 600, // Set a height for better visibility
    };

    // Render the plot
    Plotly.newPlot('myDiv', data, layout);
}).catch(function(error) {
    console.error('Error loading the CSV data:', error);
});
