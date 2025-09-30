import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import netCDF4
import pandas as pd
from netcdftime import num2date
import xarray as xr
import numpy as np
from netCDF4 import Dataset, date2num, num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata


# '''
# Simple and meshgrid sampling
# '''


# ds = xr.open_dataset('data/wind_from_direction20210910.nc')
# wind_dirn_df = ds.to_dataframe()

# ds = xr.open_dataset('data/wind_speed_20210910.nc')
# wind_speed_df = ds.to_dataframe()

# df = pd.merge(wind_speed_df, wind_dirn_df, on=['lat', 'lon']).reset_index()


# '''
# Simple sampling
# '''

# placeholder_value = df['wind_speed'].max() 
# df['wind_speed'] = df['wind_speed'].replace(placeholder_value, np.nan)
# df['wind_from_direction'] = df['wind_from_direction'].replace(placeholder_value, np.nan)

# # Drop rows with NaN values
# df.dropna(subset=['wind_speed', 'wind_from_direction'], inplace=True)

# # Convert wind direction from degrees to radians
# df['wind_from_direction_rad'] = np.deg2rad(df['wind_from_direction'])

# # Calculate u (eastward) and v (northward) wind components
# df['u'] = df['wind_speed'] * np.cos(df['wind_from_direction_rad'])
# df['v'] = df['wind_speed'] * np.sin(df['wind_from_direction_rad'])
# df['magnitude'] = np.sqrt(df['u']**2 + df['v']**2)

# # Define the sampling rate 
# sampling_rate = 500

# # Subsample the data by selecting every nth point based on sampling_rate
# df_subsampled = df.iloc[::sampling_rate]

# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# ax.coastlines()

# q = ax.quiver(df_subsampled['lon'], df_subsampled['lat'], df_subsampled['u'], df_subsampled['v'], transform=ccrs.PlateCarree(), scale=12, scale_units='width', cmap="viridis")

# plt.quiverkey(q, X=0.9, Y=1.05, U=1, label='1 m/s', labelpos='E')

# plt.title("Wind Vectors (Subsampled)")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.savefig('1.1.1.png')


# plt.close()

# '''
# Meshgrid sampling
# '''

# def produce_quiver_plots(df):
#     placeholder_value = df['wind_speed'].max() 
#     df['wind_speed'] = df['wind_speed'].replace(placeholder_value, np.nan)
#     df['wind_from_direction'] = df['wind_from_direction'].replace(placeholder_value, np.nan)

#     # Drop rows with NaN values
#     df.dropna(subset=['wind_speed', 'wind_from_direction'], inplace=True)

#     # Convert wind direction from degrees to radians
#     df['wind_from_direction_rad'] = np.deg2rad(df['wind_from_direction'])

#     # Calculate u (eastward) and v (northward) wind components
#     df['u'] = df['wind_speed'] * np.cos(df['wind_from_direction_rad'])
#     df['v'] = df['wind_speed'] * np.sin(df['wind_from_direction_rad'])

#     grid_sizes = [30, 40]
#     filenames = ['1.1.2.png', '1.1.2.png']
#     titles = ['Grid Size: 30', 'Grid Size: 40']

#     for grid_size, filename, title in zip(grid_sizes, filenames, titles):
#         fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
#         # Create a mesh grid for the wind vector field
#         lat_min, lat_max = df['lat'].min(), df['lat'].max()
#         lon_min, lon_max = df['lon'].min(), df['lon'].max()
        
#         # Define the grid size
#         grid_lat = np.linspace(lat_min, lat_max, num=grid_size)  # Adjust num for resolution
#         grid_lon = np.linspace(lon_min, lon_max, num=grid_size)  
#         lons, lats = np.meshgrid(grid_lon, grid_lat)

#         # Interpolate wind components onto the grid
#         u_grid = griddata((df['lon'], df['lat']), df['u'], (lons, lats), method='linear')
#         v_grid = griddata((df['lon'], df['lat']), df['v'], (lons, lats), method='linear')

#         q = ax.quiver(lons, lats, u_grid, v_grid, scale=12, scale_units='width', cmap="viridis")

#         magnitude = np.sqrt(u_grid**2 + v_grid**2)

#         plt.quiverkey(q, X=0.9, Y=1.05, U=1, label='1 m/s', labelpos='E')

#         ax.set_title(title)
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         ax.coastlines()

#         plt.savefig(filename)
#         plt.close(fig)  

# produce_quiver_plots(df)


'''
Producing all the plots with meshgrid sampling
(with same sized vectors and magnitude proportional vectors)
'''

list_of_dates = ['20210826', '20210829', '20210901', '20210904', '20210910']
list_of_dfs = []

for f in list_of_dates:
    ds = xr.open_dataset('data/wind_from_direction' + f + '.nc')
    wind_dirn_df = ds.to_dataframe()

    ds = xr.open_dataset('data/wind_speed_' + f + '.nc')
    wind_speed_df = ds.to_dataframe()

    df = pd.merge(wind_speed_df, wind_dirn_df, on=['lat', 'lon']).reset_index()
    list_of_dfs.append(df)

def quiver_plot_magnitude_proportional(df, label, filename):
    # Replace placeholder fill values with NaN (if necessary)
    placeholder_value = df['wind_speed'].max()
    df['wind_speed'] = df['wind_speed'].replace(placeholder_value, np.nan)
    df['wind_from_direction'] = df['wind_from_direction'].replace(placeholder_value, np.nan)

    # Drop rows with NaN values
    df.dropna(subset=['wind_speed', 'wind_from_direction'], inplace=True)

    # Convert wind direction from degrees to radians
    df['wind_from_direction_rad'] = np.deg2rad(df['wind_from_direction'])

    # Calculate u (eastward) and v (northward) wind components
    df['u'] = df['wind_speed'] * np.cos(df['wind_from_direction_rad'])
    df['v'] = df['wind_speed'] * np.sin(df['wind_from_direction_rad'])

    # Define grid for sampling
    grid_size = 30
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    grid_lat = np.linspace(lat_min, lat_max, num=grid_size)
    grid_lon = np.linspace(lon_min, lon_max, num=grid_size)
    lons, lats = np.meshgrid(grid_lon, grid_lat)

    # Interpolate wind components onto the grid
    u_grid = griddata((df['lon'], df['lat']), df['u'], (lons, lats), method='linear')
    v_grid = griddata((df['lon'], df['lat']), df['v'], (lons, lats), method='linear')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    
    # Plot the quiver with vectors proportional to magnitude
    q = ax.quiver(lons, lats, u_grid, v_grid, color='black', scale=20, scale_units='width')

    plt.quiverkey(q, X=0.9, Y=1.05, U=1, label='1 m/s', labelpos='E')

    # Add titles and labels
    ax.set_title(f'Wind vectors magnitude proportional - {label}')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.savefig(filename)
    plt.show()

# displays all the magnitude proportional quiver plots
quiver_plot_magnitude_proportional(list_of_dfs[0], '26-08-2021', '1.1.4.png')
quiver_plot_magnitude_proportional(list_of_dfs[1], '29-08-2021', '1.1.5.png')
quiver_plot_magnitude_proportional(list_of_dfs[2], '01-09-2021', '1.1.6.png')
quiver_plot_magnitude_proportional(list_of_dfs[3], '04-09-2021', '1.1.7.png')
quiver_plot_magnitude_proportional(list_of_dfs[4], '10-09-2021', '1.1.8.png')


def quiver_plot_uniform_size_colored(df, label, filename):
    placeholder_value = df['wind_speed'].max()
    df['wind_speed'] = df['wind_speed'].replace(placeholder_value, np.nan)
    df['wind_from_direction'] = df['wind_from_direction'].replace(placeholder_value, np.nan)

    # Drop rows with NaN values
    df.dropna(subset=['wind_speed', 'wind_from_direction'], inplace=True)

    # Convert wind direction from degrees to radians
    df['wind_from_direction_rad'] = np.deg2rad(df['wind_from_direction'])

    # Calculate u (eastward) and v (northward) wind components
    df['u'] = df['wind_speed'] * np.cos(df['wind_from_direction_rad'])
    df['v'] = df['wind_speed'] * np.sin(df['wind_from_direction_rad'])

    # Define grid for sampling
    grid_size = 30
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    grid_lat = np.linspace(lat_min, lat_max, num=grid_size)
    grid_lon = np.linspace(lon_min, lon_max, num=grid_size)
    lons, lats = np.meshgrid(grid_lon, grid_lat)

    # Interpolate wind components onto the grid
    u_grid = griddata((df['lon'], df['lat']), df['u'], (lons, lats), method='linear')
    v_grid = griddata((df['lon'], df['lat']), df['v'], (lons, lats), method='linear')

    # Calculate the magnitude of wind speed on the grid
    magnitude = np.sqrt(u_grid**2 + v_grid**2)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    
    # Plot the quiver with a fixed vector length but colors representing magnitude
    q = ax.quiver(lons, lats, u_grid/magnitude, v_grid/magnitude, magnitude, scale=50, scale_units='width', cmap="viridis")

    # Add a colorbar for the magnitude
    cbar = plt.colorbar(q, ax=ax, orientation='vertical', label='Wind Speed (m/s)', fraction=0.02, pad=0.05)

    ax.set_title(f'Wind Vectors colored by magnitude - {label}')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(filename)

    plt.show()

'''
Same sized coloured vectors
'''

quiver_plot_uniform_size_colored(list_of_dfs[0], '26-08-2021', '1.1.9.png')
quiver_plot_uniform_size_colored(list_of_dfs[1], '29-08-2021', '1.1.10.png')
quiver_plot_uniform_size_colored(list_of_dfs[2], '01-09-2021', '1.1.11.png')
quiver_plot_uniform_size_colored(list_of_dfs[3], '04-09-2021', '1.1.12.png')
quiver_plot_uniform_size_colored(list_of_dfs[4], '10-09-2021', '1.1.13.png')

