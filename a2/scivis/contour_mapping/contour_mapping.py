import netCDF4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cftime import num2date
import cartopy.crs as ccrs
import imageio
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

import netCDF4

try:
    vpd_nc = netCDF4.Dataset("./vpd_2021.nc", mode='r')
    print("File opened successfully!")
    print("Variables in the file:", vpd_nc.variables.keys())
except OSError as e:
    print("Error:", e)

sph_nc = netCDF4.Dataset("./sph_2021.nc", mode='r')
print("File opened successfully!")
print("Variables in the file:", sph_nc.variables.keys())

# Accessing Time Variable and Convert to Datetime for Each File
time_var_vpd = vpd_nc.variables['day']
time_var_sph = sph_nc.variables['day']

dates_pr = num2date(time_var_vpd[:], time_var_vpd.units)
dates_sph = num2date(time_var_sph[:], time_var_sph.units)

# Converting dates to YYYY-MM-DD format
formatted_dates_pr = [date.strftime('%Y-%m-%d') for date in dates_pr]
formatted_dates_sph = [date.strftime('%Y-%m-%d') for date in dates_sph]

# Reading Sample Dates from CSV
sample_df = pd.read_csv('./sample_dates_2.csv')
sample_dates = pd.to_datetime(sample_df['dates']).dt.strftime('%Y-%m-%d')

# Findng Indices of Sample Dates in NetCDF Dates
indices_vpd = [i for i, date in enumerate(formatted_dates_pr) if date in sample_dates.values]
indices_sph = [i for i, date in enumerate(formatted_dates_sph) if date in sample_dates.values]

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import os

def plot_contour(data_nc, indices, sample_dates, variable_name=None, plot_type='filled', cmap='YlOrRd', min_value=0, max_value=7, levels=15, gif_filename="output.gif"):
    """
    Plots contour maps, shows them, and generates a GIF of all sample dates.

    Parameters:
        data_nc (Dataset): NetCDF dataset containing the variables.
        indices (list): List of indices to plot over time.
        sample_dates (pd.Series): Series of sample dates corresponding to the indices.
        variable_name (str): Name of the variable to plot (either 'mean_vapor_pressure_deficit' or 'specific_humidity').
                             If None, it will attempt to detect based on available variables.
        plot_type (str): Type of contour plot ('filled' or 'line'). Default is 'filled'.
        cmap (str): Colormap to use for the contours. Default is 'YlOrRd'.
        min_value (float): Minimum value for the contour levels. Default is 0.
        max_value (float): Maximum value for the contour levels. Default is 7.
        levels (int): Number of contour levels to use. Default is 15.
        gif_filename (str): Filename for the generated GIF. Default is "output.gif".
    """

    if variable_name is None:
        if 'mean_vapor_pressure_deficit' in data_nc.variables:
            variable_name = 'mean_vapor_pressure_deficit'
        elif 'specific_humidity' in data_nc.variables:
            variable_name = 'specific_humidity'
        else:
            raise ValueError("Variable name must be specified if dataset does not contain 'mean_vapor_pressure_deficit' or 'specific_humidity'")

    # Setting contour levels
    contour_levels = np.linspace(min_value, max_value, levels)

    # Create directory to store images
    os.makedirs("frames", exist_ok=True)
    frame_files = []

    lon = data_nc.variables['lon'][:]
    lat = data_nc.variables['lat'][:]
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for idx in indices:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        data_slice = data_nc.variables[variable_name][idx, :, :]

        if plot_type == 'filled':
            filled_contour = ax.contourf(lon_grid, lat_grid, data_slice, levels=contour_levels, cmap=cmap, transform=ccrs.PlateCarree())
            contour_lines = ax.contour(lon_grid, lat_grid, data_slice, levels=contour_levels, colors='black', linewidths=0.1, transform=ccrs.PlateCarree())
            plt.colorbar(filled_contour, ax=ax, label=f'{variable_name.replace("_", " ").title()} (kPa)', fraction=0.02, pad=0.05)
        elif plot_type == 'line':
            line_contours = ax.contour(lon_grid, lat_grid, data_slice, levels=contour_levels, cmap=cmap, linewidths=1, transform=ccrs.PlateCarree())
            plt.colorbar(line_contours, ax=ax, label=f'{variable_name.replace("_", " ").title()} (kPa)', fraction=0.02, pad=0.05)

        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

        ax.grid(False)

        ax.set_xticks(np.arange(lon.min(), lon.max(), 5))
        ax.set_yticks(np.arange(lat.min(), lat.max(), 5))

        ax.set_xticklabels([f'{x:.1f}°' for x in np.arange(lon.min(), lon.max(), 5)])
        ax.set_yticklabels([f'{y:.1f}°' for y in np.arange(lat.min(), lat.max(), 5)])

        date_str = sample_dates.iloc[indices.index(idx)]
        plt.title(f'{variable_name.replace("_", " ").title()} {plot_type.capitalize()} Contour on {date_str}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Save the frame image
        frame_file = f"frames/{variable_name}_{date_str}_{plot_type}.png"
        plt.savefig(frame_file)
        frame_files.append(frame_file)

        # Display the plot
        plt.show()

        plt.close()

    # Generate GIF
    with imageio.get_writer(gif_filename, mode='I', duration=2, loop = 0) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)

    print(f"GIF saved as {gif_filename}")

vpd = vpd_nc.variables['mean_vapor_pressure_deficit'][:]
# Calculate global min and max VPD
global_min_vpd = vpd.min()
global_max_vpd = vpd.max()

print(f"Global Min VPD: {global_min_vpd}, Global Max VPD: {global_max_vpd}")

# Example usage:
plot_contour(
    data_nc=vpd_nc,
    indices=indices_vpd,
    sample_dates=sample_dates,
    variable_name='mean_vapor_pressure_deficit',
    plot_type='filled',
    cmap='YlOrRd',
    min_value=global_min_vpd,
    max_value=7,
    levels=15,
    gif_filename="vpd_filled_contour.gif"
)

plot_contour(
    data_nc=vpd_nc,
    indices=indices_vpd,
    sample_dates=sample_dates,
    variable_name='mean_vapor_pressure_deficit',
    plot_type='line',
    cmap='viridis',
    min_value=global_min_vpd,
    max_value=6,
    levels=15 ,
    gif_filename="vpd_marching_square_contour_.gif"
)

sph = sph_nc.variables['specific_humidity'][:]

global_min_sph = sph.min()
global_max_sph = sph.max()

print(f"Global Min VPD: {global_min_sph}, Global Max VPD: {global_max_sph}")

plot_contour(
    data_nc=sph_nc,
    indices=indices_sph,
    sample_dates=sample_dates,
    variable_name='specific_humidity',
    plot_type='filled',
    cmap='Blues',
    min_value=global_min_sph,
    max_value=global_max_sph,
    levels=15,
    gif_filename="sph_filled_contour.gif"
)

plot_contour(
    data_nc=sph_nc,
    indices=indices_sph,
    sample_dates=sample_dates,
    variable_name='specific_humidity',
    plot_type='line',
    cmap='Blues',
    min_value=global_min_sph,
    max_value=global_max_sph,
    levels=15 ,
    gif_filename="sph_marching_square_contour.gif"
)
