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



'''
Sampling from the vs_2021.c file
'''


# vs variable
# Load the NetCDF file
nc_file = Dataset('data/vs_2021.nc', 'r')

# Extract the variables
time_var = nc_file.variables['day']
latitudes = nc_file.variables['lat'][:]
longitudes = nc_file.variables['lon'][:]
wind_speed_var = nc_file.variables['wind_speed']

# Convert 'day' variable to dates for comparison
time_unit = time_var.units  # "days since 1900-01-01 00:00:00"
calendar = time_var.calendar if hasattr(time_var, 'calendar') else "gregorian"
time_dates = num2date(time_var[:], units=time_unit, calendar=calendar)

# Convert the input dates DataFrame to datetime objects
df = pd.read_csv('../sample_dates.csv')
df['dates'] = pd.to_datetime(df['dates'])

# Find indices of the matching dates
date_indices = [np.where(time_dates == date)[0][0] for date in df['dates'] if date in time_dates]

# Extract wind speed data for the matching dates
for idx, date_idx in zip(df['dates'], date_indices):
    wind_speed_data = wind_speed_var[date_idx, :, :]
    
    # Apply scale factor and offset if they exist
    if hasattr(wind_speed_var, 'scale_factor') and hasattr(wind_speed_var, 'add_offset'):
        wind_speed_data = wind_speed_data * wind_speed_var.scale_factor + wind_speed_var.add_offset
    
    # Save each date's data to a separate file
    output_filename = f"data/wind_speed_{idx.strftime('%Y%m%d')}.nc"
    with Dataset(output_filename, "w", format="NETCDF4") as nc_out:
        # Create dimensions
        nc_out.createDimension('lat', len(latitudes))
        nc_out.createDimension('lon', len(longitudes))
        
        lat = nc_out.createVariable('lat', 'f4', ('lat',))
        lon = nc_out.createVariable('lon', 'f4', ('lon',))
        wind_speed = nc_out.createVariable('wind_speed', 'f4', ('lat', 'lon',))
        
        lat.units = 'degrees_north'
        lon.units = 'degrees_east'
        wind_speed.units = wind_speed_var.units
        
        lat[:] = latitudes
        lon[:] = longitudes
        wind_speed[:, :] = wind_speed_data

        print(f"Saved data for {idx.strftime('%Y-%m-%d')} to {output_filename}")

nc_file.close()



'''
Sampling from the th_2021.c file
'''

# th file

nc_file = Dataset('data/th_2021.nc', 'r')

# Extract the variables
time_var = nc_file.variables['day']
latitudes = nc_file.variables['lat'][:]
longitudes = nc_file.variables['lon'][:]
wind_speed_var = nc_file.variables['wind_from_direction']

# Convert 'day' variable to dates for comparison
time_unit = time_var.units  # "days since 1900-01-01 00:00:00"
calendar = time_var.calendar if hasattr(time_var, 'calendar') else "gregorian"
time_dates = num2date(time_var[:], units=time_unit, calendar=calendar)

# Convert the input dates DataFrame to datetime objects
df = pd.read_csv('../sample_dates.csv')
df['dates'] = pd.to_datetime(df['dates'])

# Find indices of the matching dates
date_indices = [np.where(time_dates == date)[0][0] for date in df['dates'] if date in time_dates]

# Extract wind speed data for the matching dates
for idx, date_idx in zip(df['dates'], date_indices):
    wind_speed_data = wind_speed_var[date_idx, :, :]
    
    # Apply scale factor and offset if they exist
    if hasattr(wind_speed_var, 'scale_factor') and hasattr(wind_speed_var, 'add_offset'):
        wind_speed_data = wind_speed_data * wind_speed_var.scale_factor + wind_speed_var.add_offset
    
    # Save each date's data to a separate file
    output_filename = f"data/wind_from_direction{idx.strftime('%Y%m%d')}.nc"
    with Dataset(output_filename, "w", format="NETCDF4") as nc_out:
        # Create dimensions
        nc_out.createDimension('lat', len(latitudes))
        nc_out.createDimension('lon', len(longitudes))
        
        lat = nc_out.createVariable('lat', 'f4', ('lat',))
        lon = nc_out.createVariable('lon', 'f4', ('lon',))
        wind_speed = nc_out.createVariable('wind_from_direction', 'f4', ('lat', 'lon',))
        
        lat.units = 'degrees_north'
        lon.units = 'degrees_east'
        wind_speed.units = wind_speed_var.units
        
        lat[:] = latitudes
        lon[:] = longitudes
        wind_speed[:, :] = wind_speed_data

        print(f"Saved data for {idx.strftime('%Y-%m-%d')} to {output_filename}")

nc_file.close()

