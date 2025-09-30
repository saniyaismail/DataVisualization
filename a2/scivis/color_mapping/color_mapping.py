import netCDF4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cftime import num2date
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import imageio

pr_nc_file = 'pr_2021.nc'
pr_nc = netCDF4.Dataset(pr_nc_file, mode='r')

tmmx_nc_file = 'tmmx_2021.nc'
tmmx_nc = netCDF4.Dataset(tmmx_nc_file, mode='r')

print(pr_nc.variables.keys())
print(tmmx_nc.variables.keys())

#Accessing Time Variable and Convert to Datetime
time_var = pr_nc.variables['day']
dates = num2date(time_var[:], time_var.units)

#Converting NetCDF Dates to YYYY-MM-DD Format
formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]
print(formatted_dates)

sample_dates_df = pd.read_csv('../sample_dates.csv')
sample_dates = sample_dates_df['dates']
print(dates)

sample_dates = pd.to_datetime(sample_dates_df['dates']).dt.strftime('%Y-%m-%d')
print(sample_dates)

indices = [i for i, date in enumerate(formatted_dates) if date in sample_dates.values]
print(indices)

precipitation = pr_nc.variables['precipitation_amount'][:]
temperature = tmmx_nc.variables['air_temperature'][:]
longitude = pr_nc.variables['lon'][:]
latitude = pr_nc.variables['lat'][:]
lon, lat = np.meshgrid(longitude, latitude)


def plot_color_map(data, lon, lat, title, cmap, scale='continuous', global_min=None, global_max=None, min_threshold=1e-3):
    data = np.ma.masked_invalid(data)

    if scale == 'log':
        data = np.ma.masked_where(data.mask, np.maximum(data, min_threshold))

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title(f"{title} - Color Map: {cmap}, Scale: {scale}")

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xticks(np.arange(lon.min(), lon.max() + 2, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat.min(), lat.max() + 2, 10), crs=ccrs.PlateCarree())

    if scale == 'log':
        norm = mcolors.LogNorm(vmin=min_threshold, vmax=global_max, clip=True)
        mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    elif scale == 'discrete':
        levels = np.linspace(global_min, global_max, 10)
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
        mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    else:
        mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=global_min, vmax=global_max, transform=ccrs.PlateCarree(), shading='auto')


    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', fraction = 0.0125, pad=0.05, aspect=30)
    cbar.set_label('Value')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()


sample_date_index = 243

sample_precipitation = precipitation[sample_date_index]
sample_temperature = temperature[sample_date_index]

global_min_precip = precipitation.min()
global_max_precip = precipitation.max()
global_min_temp = temperature.min()
global_max_temp = temperature.max()

print(f"Global Min Precip: {global_min_precip}, Global Max Precip: {global_max_precip}")
print(f"Global Min Temp: {global_min_temp}, Global Max Temp: {global_max_temp}")


for cmap in ['YlGnBu', 'viridis', 'coolwarm']:
    for scale in ['continuous', 'log', 'discrete']:
        plot_color_map(sample_precipitation, lon, lat, 'Precipitation', cmap, scale, global_min_precip, global_max_precip)

for cmap in ['hot', 'RdYlBu', 'coolwarm']:
    for scale in ['continuous', 'log', 'discrete']:
        plot_color_map(sample_temperature, lon, lat, 'Max Temperature', cmap, scale, global_min_temp, global_max_temp)


def visualize_sampled_dates(data, lon, lat, title, cmap, scale, global_min, global_max, dates):
    for date in dates:
        date_str = sample_dates.iloc[indices.index(date)]
        data_date = data[date]
        plot_color_map(data_date, lon, lat, f"{title} on {date_str}", cmap, scale, global_min, global_max)
        plt.savefig(f"{title}_{date}.png")
        plt.close()
        
        
print("Visualizing Precipitation")
visualize_sampled_dates(precipitation, lon, lat, "Precipitation", 'YlGnBu', 'discrete', global_min_precip, global_max_precip, indices)

print("Visualizing Max Temperature")
visualize_sampled_dates(temperature, lon, lat, "Max Temperature", 'coolwarm', 'continuous', global_min_temp, global_max_temp, indices)


#GIF generation from the sample date images
sample_indices = ['237', '240', '243', '246', '252']

precipitation_images = [imageio.v2.imread(f"Precipitation_{index}.png") for index in sample_indices]
imageio.mimsave("precipitation.gif", precipitation_images, duration=2, loop=0)

temperature_images = [imageio.v2.imread(f"Max Temperature_{index}.png") for index in sample_indices]
imageio.mimsave("temperature.gif", temperature_images, duration=2, loop=0)

