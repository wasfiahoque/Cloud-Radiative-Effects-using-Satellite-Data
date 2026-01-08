#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 17:54:11 2025

@author: wasfiahoque
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

nc_file = "CERES_EBAF-TOA_Ed4.2_Subset_200003-202407.nc"

# South Asia region 
lat_min = 5
lat_max = 35
lon_min = 65
lon_max = 100
# ==================================================

print("Loading data...")
ds = xr.open_dataset(nc_file)

### JULY LW ###
# Get July months only
print("Filtering for July months...")
july_all = ds['toa_lw_all_mon'].sel(time=ds['toa_lw_all_mon'].time.dt.month == 7)
july_clr = ds['toa_lw_clr_c_mon'].sel(time=ds['toa_lw_clr_c_mon'].time.dt.month == 7)

print(f"Found {len(july_all.time)} July months")

# Calculate averages
print("Computing averages...")
july_all_mean = july_all.mean(dim='time')
july_clr_mean = july_clr.mean(dim='time')
cre_july = july_clr_mean - july_all_mean

print(f"\nJuly Results:")
print(f"  All-sky mean:  {float(july_all_mean.mean()):.2f} W m⁻²")
print(f"  Clear-sky mean: {float(july_clr_mean.mean()):.2f} W m⁻²")
print(f"  CRE mean: {float(cre_july.mean()):.2f} W m⁻²")

# Create figure 1 for July
print("\nCreating July plots...")
fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle('JULY', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: All-sky
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im1 = ax1.pcolormesh(july_all_mean.lon, july_all_mean.lat, july_all_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax1.set_title('TOA Longwave Flux - All-Sky (July Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 2: Clear-sky
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im2 = ax2.pcolormesh(july_clr_mean.lon, july_clr_mean.lat, july_clr_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax2.set_title('TOA Longwave Flux - Clear-Sky (July Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 3: Cloud Radiative Effect (the difference)
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

# Center colormap at zero
abs_max = np.nanmax(np.abs(cre_july.values))
im3 = ax3.pcolormesh(cre_july.lon, cre_july.lat, cre_july,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('Longwave Cloud Radiative Effect (Clear-Sky - All-Sky) - South Asia', 
              fontsize=14, fontweight='bold')

plt.tight_layout()


### DECEMBER LW ###
# Get December months only
print("Filtering for December months...")
dec_all = ds['toa_lw_all_mon'].sel(time=ds['toa_lw_all_mon'].time.dt.month == 12)
dec_clr = ds['toa_lw_clr_c_mon'].sel(time=ds['toa_lw_clr_c_mon'].time.dt.month == 12)

print(f"Found {len(dec_all.time)} December months")

# Calculate averages
print("Computing averages...")
dec_all_mean = dec_all.mean(dim='time')
dec_clr_mean = dec_clr.mean(dim='time')
cre_dec = dec_clr_mean - dec_all_mean

print(f"\nDecember Results:")
print(f"  All-sky mean:  {float(dec_all_mean.mean()):.2f} W m⁻²")
print(f"  Clear-sky mean: {float(dec_clr_mean.mean()):.2f} W m⁻²")
print(f"  CRE mean: {float(cre_dec.mean()):.2f} W m⁻²")

# Create figure 2 for December
print("\nCreating December plots...")
fig2 = plt.figure(figsize=(18, 14))
fig2.suptitle('DECEMBER', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: All-sky
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im1 = ax1.pcolormesh(dec_all_mean.lon, dec_all_mean.lat, dec_all_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax1.set_title('TOA Longwave Flux - All-Sky (December Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 2: Clear-sky
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im2 = ax2.pcolormesh(dec_clr_mean.lon, dec_clr_mean.lat, dec_clr_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax2.set_title('TOA Longwave Flux - Clear-Sky (December Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 3: Cloud Radiative Effect (the difference)
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

# Center colormap at zero
abs_max = np.nanmax(np.abs(cre_dec.values))
im3 = ax3.pcolormesh(cre_dec.lon, cre_dec.lat, cre_dec,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('Longwave Cloud Radiative Effect (Clear-Sky - All-Sky) - South Asia', 
              fontsize=14, fontweight='bold')

plt.tight_layout()

# Show both figures at once
print("\nShowing both plots... (close windows to exit)")
plt.show()

### SW for July and December
# Get July months only
print("Filtering for July months...")
july_all = ds['toa_sw_all_mon'].sel(time=ds['toa_sw_all_mon'].time.dt.month == 7)
july_clr = ds['toa_sw_clr_c_mon'].sel(time=ds['toa_sw_clr_c_mon'].time.dt.month == 7)

print(f"Found {len(july_all.time)} July months")

# Calculate averages
print("Computing averages...")
july_all_mean = july_all.mean(dim='time')
july_clr_mean = july_clr.mean(dim='time')
cre_july = july_clr_mean - july_all_mean

print(f"\nJuly Results:")
print(f"  All-sky mean:  {float(july_all_mean.mean()):.2f} W m⁻²")
print(f"  Clear-sky mean: {float(july_clr_mean.mean()):.2f} W m⁻²")
print(f"  CRE mean: {float(cre_july.mean()):.2f} W m⁻²")

# Create figure 1 for July
print("\nCreating July plots...")
fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle('JULY', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: All-sky
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im1 = ax1.pcolormesh(july_all_mean.lon, july_all_mean.lat, july_all_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax1.set_title('TOA Shortwave Flux - All-Sky (July Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 2: Clear-sky
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im2 = ax2.pcolormesh(july_clr_mean.lon, july_clr_mean.lat, july_clr_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax2.set_title('TOA Shortwave Flux - Clear-Sky (July Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 3: Cloud Radiative Effect (the difference)
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

# Center colormap at zero
abs_max = np.nanmax(np.abs(cre_july.values))
im3 = ax3.pcolormesh(cre_july.lon, cre_july.lat, cre_july,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('Shortwave Cloud Radiative Effect (Clear-Sky - All-Sky) - South Asia', 
              fontsize=14, fontweight='bold')

plt.tight_layout()


### DECEMBER SW ###
# Get December months only
print("Filtering for December months...")
dec_all = ds['toa_sw_all_mon'].sel(time=ds['toa_sw_all_mon'].time.dt.month == 12)
dec_clr = ds['toa_sw_clr_c_mon'].sel(time=ds['toa_sw_clr_c_mon'].time.dt.month == 12)

print(f"Found {len(dec_all.time)} December months")

# Calculate averages
print("Computing averages...")
dec_all_mean = dec_all.mean(dim='time')
dec_clr_mean = dec_clr.mean(dim='time')
cre_dec = dec_clr_mean - dec_all_mean

print(f"\nDecember Results:")
print(f"  All-sky mean:  {float(dec_all_mean.mean()):.2f} W m⁻²")
print(f"  Clear-sky mean: {float(dec_clr_mean.mean()):.2f} W m⁻²")
print(f"  CRE mean: {float(cre_dec.mean()):.2f} W m⁻²")

# Create figure 2 for December
print("\nCreating December plots...")
fig2 = plt.figure(figsize=(18, 14))
fig2.suptitle('DECEMBER', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: All-sky
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im1 = ax1.pcolormesh(dec_all_mean.lon, dec_all_mean.lat, dec_all_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax1.set_title('TOA Shortwave Flux - All-Sky (December Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 2: Clear-sky
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

im2 = ax2.pcolormesh(dec_clr_mean.lon, dec_clr_mean.lat, dec_clr_mean,
                     transform=ccrs.PlateCarree(), cmap='YlOrRd', shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, label='W m⁻²')
ax2.set_title('TOA Shortwave Flux - Clear-Sky (December Average) - South Asia', fontsize=14, fontweight='bold')

# Plot 3: Cloud Radiative Effect (the difference)
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

# Center colormap at zero
abs_max = np.nanmax(np.abs(cre_dec.values))
im3 = ax3.pcolormesh(cre_dec.lon, cre_dec.lat, cre_dec,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('Shortwave Cloud Radiative Effect (Clear-Sky - All-Sky) - South Asia', 
              fontsize=14, fontweight='bold')

plt.tight_layout()

# Show both figures at once
print("\nShowing both plots... (close windows to exit)")
plt.show()



### Net

print("Loading data...")
ds = xr.open_dataset(nc_file)

### JULY ###
print("\n" + "="*60)
print("JULY ANALYSIS")
print("="*60)

# Get July months
print("Filtering for July months...")
july_lw_all = ds['toa_lw_all_mon'].sel(time=ds['toa_lw_all_mon'].time.dt.month == 7)
july_lw_clr = ds['toa_lw_clr_c_mon'].sel(time=ds['toa_lw_clr_c_mon'].time.dt.month == 7)
july_sw_all = ds['toa_sw_all_mon'].sel(time=ds['toa_sw_all_mon'].time.dt.month == 7)
july_sw_clr = ds['toa_sw_clr_c_mon'].sel(time=ds['toa_sw_clr_c_mon'].time.dt.month == 7)

print(f"Found {len(july_lw_all.time)} July months")

# Calculate averages
print("Computing averages...")
july_lw_all_mean = july_lw_all.mean(dim='time')
july_lw_clr_mean = july_lw_clr.mean(dim='time')
july_sw_all_mean = july_sw_all.mean(dim='time')
july_sw_clr_mean = july_sw_clr.mean(dim='time')

# Calculate CREs
lw_cre_july = july_lw_clr_mean - july_lw_all_mean
sw_cre_july = july_sw_clr_mean - july_sw_all_mean
net_cre_july = lw_cre_july + sw_cre_july

print(f"\nJuly Results:")
print(f"  LW CRE mean: {float(lw_cre_july.mean()):.2f} W m⁻²")
print(f"  SW CRE mean: {float(sw_cre_july.mean()):.2f} W m⁻²")
print(f"  NET CRE mean: {float(net_cre_july.mean()):.2f} W m⁻²")

# Create figure 1 for July
print("\nCreating July plots...")
fig1 = plt.figure(figsize=(18, 16))
fig1.suptitle('JULY - Cloud Radiative Effects', fontsize=18, fontweight='bold', y=0.995)

# Plot 1: LW CRE
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(lw_cre_july.values))
im1 = ax1.pcolormesh(lw_cre_july.lon, lw_cre_july.lat, lw_cre_july,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax1.set_title('Longwave CRE (Clear-Sky - All-Sky)', fontsize=14, fontweight='bold')

# Plot 2: SW CRE
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(sw_cre_july.values))
im2 = ax2.pcolormesh(sw_cre_july.lon, sw_cre_july.lat, sw_cre_july,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax2.set_title('Shortwave CRE (Clear-Sky - All-Sky)', fontsize=14, fontweight='bold')

# Plot 3: NET CRE
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(net_cre_july.values))
im3 = ax3.pcolormesh(net_cre_july.lon, net_cre_july.lat, net_cre_july,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('NET CRE (LW + SW) - Total cloud radiative effect',fontsize=14, fontweight='bold')

plt.tight_layout()


### DECEMBER ###
print("\n" + "="*60)
print("DECEMBER ANALYSIS")
print("="*60)

# Get December months
print("Filtering for December months...")
dec_lw_all = ds['toa_lw_all_mon'].sel(time=ds['toa_lw_all_mon'].time.dt.month == 12)
dec_lw_clr = ds['toa_lw_clr_c_mon'].sel(time=ds['toa_lw_clr_c_mon'].time.dt.month == 12)
dec_sw_all = ds['toa_sw_all_mon'].sel(time=ds['toa_sw_all_mon'].time.dt.month == 12)
dec_sw_clr = ds['toa_sw_clr_c_mon'].sel(time=ds['toa_sw_clr_c_mon'].time.dt.month == 12)

print(f"Found {len(dec_lw_all.time)} December months")

# Calculate averages
print("Computing averages...")
dec_lw_all_mean = dec_lw_all.mean(dim='time')
dec_lw_clr_mean = dec_lw_clr.mean(dim='time')
dec_sw_all_mean = dec_sw_all.mean(dim='time')
dec_sw_clr_mean = dec_sw_clr.mean(dim='time')

# Calculate CREs
lw_cre_dec = dec_lw_clr_mean - dec_lw_all_mean
sw_cre_dec = dec_sw_clr_mean - dec_sw_all_mean
net_cre_dec = lw_cre_dec + sw_cre_dec

print(f"\nDecember Results:")
print(f"  LW CRE mean: {float(lw_cre_dec.mean()):.2f} W m⁻²")
print(f"  SW CRE mean: {float(sw_cre_dec.mean()):.2f} W m⁻²")
print(f"  NET CRE mean: {float(net_cre_dec.mean()):.2f} W m⁻²")

# Create figure 2 for December
print("\nCreating December plots...")
fig2 = plt.figure(figsize=(18, 16))
fig2.suptitle('DECEMBER - Cloud Radiative Effects', fontsize=18, fontweight='bold', y=0.995)

# Plot 1: LW CRE
ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.coastlines(linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(lw_cre_dec.values))
im1 = ax1.pcolormesh(lw_cre_dec.lon, lw_cre_dec.lat, lw_cre_dec,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax1.set_title('Longwave CRE (Clear-Sky - All-Sky)', fontsize=14, fontweight='bold')

# Plot 2: SW CRE
ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.coastlines(linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(sw_cre_dec.values))
im2 = ax2.pcolormesh(sw_cre_dec.lon, sw_cre_dec.lat, sw_cre_dec,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax2.set_title('Shortwave CRE (Clear-Sky - All-Sky)', fontsize=14, fontweight='bold')

# Plot 3: NET CRE
ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.coastlines(linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

abs_max = np.nanmax(np.abs(net_cre_dec.values))
im3 = ax3.pcolormesh(net_cre_dec.lon, net_cre_dec.lat, net_cre_dec,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     vmin=-150, vmax=150, shading='gouraud')
plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7, 
             label='W m⁻²', extend='both')
ax3.set_title('NET CRE (LW + SW) - Total cloud radiative effect', 
              fontsize=14, fontweight='bold')


plt.tight_layout()

# Show both figures at once
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nJuly NET CRE:    {float(net_cre_july.mean()):>6.1f} W m⁻²")
print(f"December NET CRE: {float(net_cre_dec.mean()):>6.1f} W m⁻²")
print(f"\nNegative NET CRE = Clouds have net COOLING effect (SW reflection wins)")
print(f"Positive NET CRE = Clouds have net WARMING effect (LW trapping wins)")
print("\nShowing both plots... (close windows to exit)")
plt.show()

ds.close()

# Subset each CRE field to South Asia region
lw_cre_july_reg = lw_cre_july.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
sw_cre_july_reg = sw_cre_july.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
net_cre_july_reg = net_cre_july.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

lw_cre_dec_reg = lw_cre_dec.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
sw_cre_dec_reg = sw_cre_dec.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
net_cre_dec_reg = net_cre_dec.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

# Compute simple (unweighted) regional means
lw_july_mean = float(lw_cre_july_reg.mean())
sw_july_mean = float(sw_cre_july_reg.mean())
net_july_mean = float(net_cre_july_reg.mean())

lw_dec_mean = float(lw_cre_dec_reg.mean())
sw_dec_mean = float(sw_cre_dec_reg.mean())
net_dec_mean = float(net_cre_dec_reg.mean())

# Print the results neatly
print(f"\n-- JULY --")
print(f"  Longwave CRE (LW):  {lw_july_mean:6.2f} W/m²")
print(f"  Shortwave CRE (SW): {sw_july_mean:6.2f} W/m²")
print(f"  Net CRE (LW+SW):    {net_july_mean:6.2f} W/m²")

print(f"\n-- DECEMBER --")
print(f"  Longwave CRE (LW):  {lw_dec_mean:6.2f} W/m²")
print(f"  Shortwave CRE (SW): {sw_dec_mean:6.2f} W/m²")
print(f"  Net CRE (LW+SW):    {net_dec_mean:6.2f} W/m²")

print("\nInterpretation:")
print("  • Positive LW CRE = clouds trap heat (warming).")
print("  • Negative SW CRE = clouds reflect sunlight (cooling).")
print("  • NET CRE = LW + SW (overall: negative = cooling, positive = warming).")



























