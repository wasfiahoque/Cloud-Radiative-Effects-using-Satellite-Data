#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 2025

@author: wasfiahoque

Modified to loop through multiple granules with different dates and profile ranges
Updated to show correct seconds since start of granule on x-axis
"""
#!/usr/bin/env python
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define all granule parameters
years = [2010, 2010, 2012, 2012, 2013, 2013, 2015, 2015, 2016, 2016]
months = ['July 1st', 'December 1st', 'July 1st', 'December 1st', 'July 1st', 
          'December 1st', 'July 1st', 'December 1st', 'July 1st', 'December 1st']
granules = [22207, 24435, 32857, 35085, 38172, 40401, 48803, 51031, 54133, 56361]
start_profiles = [20338, 20331, 20338, 20338, 20338, 20331, 20338, 20338, 20338, 19963]
end_profiles = [21531, 21525, 21531, 21531, 21531, 21531, 21531, 21531, 21531, 21156]

# File names (matching your screenshot)
file_names = [
    '2010182065732_22207_CS_2B-GEOPROF_GRANULE_P1_R05_E03_F00.hdf',
    '2010335065029_24435_CS_2B-GEOPROF_GRANULE_P1_R05_E03_F00.hdf',
    '2012183064021_32857_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2012336063351_35085_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2013182060932_38172_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2013335074203_40401_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2015182064654_48803_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2015335063953_51031_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2016183065820_54133_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf',
    '2016336065203_56361_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F01.hdf'
]

# Time interval between profiles (seconds)
profile_interval = 0.16
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Force Aspect Ratio - This makes the figure box be wider than it is tall
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Loop through all granules
for i in range(len(granules)):
    year = years[i]
    month = months[i]
    granule = granules[i]
    start_profile = start_profiles[i]
    end_profile = end_profiles[i]
    file_name = file_names[i]
    
    print(f'\n=== Processing {year} {month} Granule {granule} ===')
    print(f'Start Profile: {start_profile}, End Profile: {end_profile}')
    
    # Calculate actual seconds since start of granule for this subset
    time_start = start_profile * profile_interval
    time_end = end_profile * profile_interval
    print(f'Time range: {time_start:.1f} to {time_end:.1f} seconds since granule start')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Get CPR_Cloud_mask sds
    file = SD(file_name, SDC.READ) # Reading the File
    file_info = file.info() # getting final info
    print(file_info) # number of sds and metadata
    sds_obj = file.select('CPR_Cloud_mask') # select sds
    data = sds_obj.get() # Getting the CPR Cloud Mask Data
    sds_info = sds_obj.info() # Getting the CPR Cloud Mask Info
    print('sds info: CPR_Cloud_mask')
    print(data.shape)
    print(sds_info)
    print(sds_info[0], sds_info[1])
    print('sds attributes')
    pprint.pprint(sds_obj.attributes())
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Finding and replacing the fill value with Nan
    fillvalue = (data < 0) # finds the index/location of the fill value data (-9)
    data = data.astype(float) # makes data floating
    data[fillvalue] = np.nan # Changes the fill values to Nans
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Finding the Maximum, Minimum (without Nans), and halfway for making a neat color bar
    max_CPR_value = np.nanmax(data) # Finds the Max Value in the array ignorning Nans
    min_CPR_value = np.nanmin(data) # Find the Min Value in the array igorning Nans
    number_of_bins = 2 # Setting the number color bar segments
    number_of_ticks = 4 # Setting the number of ticks (including the halfway and top)
    halfway_CPR_values = max_CPR_value/number_of_bins # Calculating the middle of the color bar
    quarter_CPR_values = max_CPR_value/number_of_ticks # Calculating the quarter of the color bar
    threequarter_CPR_values = quarter_CPR_values*3 # Calculating the quarter of the color bar
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Plot full granule
    f = plt.figure() # Plotting the figure
    ax = f.add_subplot(111) # One plot, one row, first row
    cmap = [(0.0,0.0,1.0)] + [(1.0, 1.0, 1.0)] # two specific colors Blue + White
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = [min_CPR_value,halfway_CPR_values,max_CPR_value] # Colorbar Boundaries
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # Here we're showing the data
    img = plt.imshow(data.T, extent=[0,35000,-5,25], aspect='auto',
                     cmap=cmap, norm=norm, interpolation='none')
    # Setting the Colorbar up
    cbar_bounds = [min_CPR_value,halfway_CPR_values,max_CPR_value]
    cbar_ticks = [quarter_CPR_values,threequarter_CPR_values]
    cbar_ticks_labels = ['Not Cloudy', 'Cloudy']
    # Add ColorBar to the Figure
    cbar = plt.colorbar(img, fraction=0.01, cmap=cmap, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks_labels, fontsize=8)
    plt.title(f'CloudSat CPR cloud mask - {month} {year} Granule {granule}', fontsize=8)
    plt.xlabel('Seconds since the start of the granule', fontsize=8)
    plt.ylabel('Height (km)', fontsize=8)
    forceAspect(ax,aspect=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(f"cloudsat_cpr_cloud_mask_granule{granule}_{year}_{month.replace(' ', '_')}.png", 
                bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Zooming into South Asia region and making a new plot
    data_SouthAsia = data[start_profile:end_profile,:] # Subsetting the data
    
    # Finding the Maximum, Minimum (without Nans), and halfway for making a neat color bar
    max_CPR_value_SouthAsia = np.nanmax(data_SouthAsia)
    min_CPR_value_SouthAsia = np.nanmin(data_SouthAsia)
    number_of_bins = 2
    number_of_ticks = 4
    halfway_CPR_values_SouthAsia = max_CPR_value_SouthAsia/number_of_bins
    quarter_CPR_values_SouthAsia = max_CPR_value_SouthAsia/number_of_ticks
    threequarter_CPR_values_SouthAsia = quarter_CPR_values_SouthAsia*3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Plot South Asia subset with CORRECT time axis
    f = plt.figure()
    ax = f.add_subplot(111)
    cmap = [(0.0,0.0,1.0)] + [(1.0, 1.0, 1.0)] # two specific colors Blue + White
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = [min_CPR_value_SouthAsia,halfway_CPR_values_SouthAsia,max_CPR_value_SouthAsia]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # USE ACTUAL SECONDS SINCE START OF GRANULE (not just subset size!)
    img = plt.imshow(data_SouthAsia.T, extent=[time_start, time_end, -5, 25], aspect='auto',
                     cmap=cmap, norm=norm, interpolation='none')
    
    # Setting the Colorbar up
    cbar_bounds = [min_CPR_value_SouthAsia,halfway_CPR_values_SouthAsia,max_CPR_value_SouthAsia]
    cbar_ticks = [quarter_CPR_values_SouthAsia,threequarter_CPR_values_SouthAsia]
    cbar_ticks_labels = ['Not Cloudy', 'Cloudy']
    # Add ColorBar to the Figure
    cbar = plt.colorbar(img, fraction=0.01, cmap=cmap, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks_labels, fontsize=8)
    plt.title(f'CloudSat CPR cloud mask - South Asia {month} {year} Granule {granule} Orbit 18', fontsize=8)
    plt.xlabel('Seconds since the start of the granule', fontsize=8)
    plt.ylabel('Height (km)', fontsize=8)
    forceAspect(ax,aspect=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(f"cloudsat_cpr_cloud_mask_SouthAsia_{month.replace(' ', '_')}_{year}.png", 
                bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    print(f'Completed processing {year} {month} Granule {granule}')

print('\n=== All granules processed successfully! ===')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# CREATE COMPOSITE PLOTS - ALL JULY AND ALL DECEMBER SOUTH ASIA FIGURES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print('\n=== Creating composite plots ===')

# Storage for all South Asia data
july_data_list = []
july_time_starts = []
july_time_ends = []
july_years = []
july_granules = []

december_data_list = []
december_time_starts = []
december_time_ends = []
december_years = []
december_granules = []

# Re-process to collect data for composite plots
for i in range(len(granules)):
    year = years[i]
    month = months[i]
    granule = granules[i]
    start_profile = start_profiles[i]
    end_profile = end_profiles[i]
    file_name = file_names[i]
    
    # Calculate time range
    time_start = start_profile * profile_interval
    time_end = end_profile * profile_interval
    
    # Load data
    file = SD(file_name, SDC.READ)
    sds_obj = file.select('CPR_Cloud_mask')
    data = sds_obj.get()
    
    # Process fill values
    fillvalue = (data < 0)
    data = data.astype(float)
    data[fillvalue] = np.nan
    
    # Subset to South Asia
    data_SouthAsia = data[start_profile:end_profile,:]
    
    # Separate into July and December
    if 'July' in month:
        july_data_list.append(data_SouthAsia)
        july_time_starts.append(time_start)
        july_time_ends.append(time_end)
        july_years.append(year)
        july_granules.append(granule)
    else:  # December
        december_data_list.append(data_SouthAsia)
        december_time_starts.append(time_start)
        december_time_ends.append(time_end)
        december_years.append(year)
        december_granules.append(granule)

# Create JULY composite plot
print('\nCreating July composite plot...')
from matplotlib.gridspec import GridSpec

fig_july = plt.figure(figsize=(20, 12))
fig_july.suptitle('CloudSat CPR Cloud Mask - All July South Asia', 
                  fontsize=16, fontweight='bold')

# Create custom grid: 2 plots in first two rows, 1 centered plot in last row
gs = GridSpec(3, 4, figure=fig_july)
subplot_specs = [
    gs[0, 0:2],  # Top left
    gs[0, 2:4],  # Top right
    gs[1, 0:2],  # Middle left
    gs[1, 2:4],  # Middle right
    gs[2, 1:3]   # Bottom center (spans middle 2 columns)
]

for idx, data_sa in enumerate(july_data_list):
    ax = fig_july.add_subplot(subplot_specs[idx])
    
    # Calculate colorbar parameters
    max_val = np.nanmax(data_sa)
    min_val = np.nanmin(data_sa)
    halfway_val = max_val / 2
    quarter_val = max_val / 4
    threequarter_val = quarter_val * 3
    
    # Create colormap
    cmap = [(0.0,0.0,1.0)] + [(1.0, 1.0, 1.0)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = [min_val, halfway_val, max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot
    img = plt.imshow(data_sa.T, extent=[july_time_starts[idx], july_time_ends[idx], -5, 25], 
                     aspect='auto', cmap=cmap, norm=norm, interpolation='none')
    
    # Colorbar
    cbar_bounds = [min_val, halfway_val, max_val]
    cbar_ticks = [quarter_val, threequarter_val]
    cbar_ticks_labels = ['Not Cloudy', 'Cloudy']
    cbar = plt.colorbar(img, fraction=0.01, cmap=cmap, norm=norm, 
                       boundaries=cbar_bounds, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks_labels, fontsize=7)
    
    ax.set_title(f'{july_years[idx]} - Granule {july_granules[idx]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Seconds since start of granule', fontsize=8)
    ax.set_ylabel('Height (km)', fontsize=8)
    ax.tick_params(labelsize=7)
    
    forceAspect(ax, aspect=3)

plt.tight_layout()
plt.savefig("cloudsat_july_composite_all_years.png", bbox_inches='tight', dpi=150)
plt.show()
plt.close()

# Create DECEMBER composite plot
print('Creating December composite plot...')

fig_dec = plt.figure(figsize=(20, 12))
fig_dec.suptitle('CloudSat CPR Cloud Mask - All December South Asia', 
                 fontsize=16, fontweight='bold')

# Create custom grid: 2 plots in first two rows, 1 centered plot in last row
gs = GridSpec(3, 4, figure=fig_dec)
subplot_specs = [
    gs[0, 0:2],  # Top left
    gs[0, 2:4],  # Top right
    gs[1, 0:2],  # Middle left
    gs[1, 2:4],  # Middle right
    gs[2, 1:3]   # Bottom center (spans middle 2 columns)
]

for idx, data_sa in enumerate(december_data_list):
    ax = fig_dec.add_subplot(subplot_specs[idx])
    
    # Calculate colorbar parameters
    max_val = np.nanmax(data_sa)
    min_val = np.nanmin(data_sa)
    halfway_val = max_val / 2
    quarter_val = max_val / 4
    threequarter_val = quarter_val * 3
    
    # Create colormap
    cmap = [(0.0,0.0,1.0)] + [(1.0, 1.0, 1.0)]
    cmap = mpl.colors.ListedColormap(cmap)
    bounds = [min_val, halfway_val, max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot
    img = plt.imshow(data_sa.T, extent=[december_time_starts[idx], december_time_ends[idx], -5, 25], 
                     aspect='auto', cmap=cmap, norm=norm, interpolation='none')
    
    # Colorbar
    cbar_bounds = [min_val, halfway_val, max_val]
    cbar_ticks = [quarter_val, threequarter_val]
    cbar_ticks_labels = ['Not Cloudy', 'Cloudy']
    cbar = plt.colorbar(img, fraction=0.01, cmap=cmap, norm=norm, 
                       boundaries=cbar_bounds, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks_labels, fontsize=7)
    
    ax.set_title(f'{december_years[idx]} - Granule {december_granules[idx]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Seconds since start of granule', fontsize=8)
    ax.set_ylabel('Height (km)', fontsize=8)
    ax.tick_params(labelsize=7)
    
    forceAspect(ax, aspect=3)

plt.tight_layout()
plt.savefig("cloudsat_december_composite_all_years.png", bbox_inches='tight', dpi=150)
plt.show()
plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# CALCULATE CLOUD COVER STATISTICS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

print('\n=== Cloud Cover Statistics ===')

# Calculate for July
print('\nJULY:')
for idx, data_sa in enumerate(july_data_list):
    # Cloud mask: 0 = not cloudy, values > 0 = cloudy
    # Count cloudy pixels (values > 0 and not NaN)
    total_pixels = np.sum(~np.isnan(data_sa))  # Total non-NaN pixels
    cloudy_pixels = np.sum(data_sa > 0)  # Pixels with cloud
    cloud_fraction = (cloudy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    print(f'  {july_years[idx]} Granule {july_granules[idx]}: {cloud_fraction:.2f}% cloud cover')

# Calculate for December
print('\nDECEMBER:')
for idx, data_sa in enumerate(december_data_list):
    # Cloud mask: 0 = not cloudy, values > 0 = cloudy
    total_pixels = np.sum(~np.isnan(data_sa))
    cloudy_pixels = np.sum(data_sa > 0)
    cloud_fraction = (cloudy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    print(f'  {december_years[idx]} Granule {december_granules[idx]}: {cloud_fraction:.2f}% cloud cover')

# Calculate average cloud cover for July and December
july_cloud_fractions = []
for data_sa in july_data_list:
    total_pixels = np.sum(~np.isnan(data_sa))
    cloudy_pixels = np.sum(data_sa > 0)
    cloud_fraction = (cloudy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    july_cloud_fractions.append(cloud_fraction)

december_cloud_fractions = []
for data_sa in december_data_list:
    total_pixels = np.sum(~np.isnan(data_sa))
    cloudy_pixels = np.sum(data_sa > 0)
    cloud_fraction = (cloudy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    december_cloud_fractions.append(cloud_fraction)

print(f'\n=== AVERAGE CLOUD COVER ===')
print(f'July Average: {np.mean(july_cloud_fractions):.2f}%')
print(f'December Average: {np.mean(december_cloud_fractions):.2f}%')

print('\n=== Composite plots and statistics completed! ===')