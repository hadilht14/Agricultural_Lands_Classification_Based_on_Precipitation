import numpy as np
import pandas as pd
from scipy.interpolate import griddata


input_csv_path = 'Original_Data\CsvFormat_data\Dataset_12_2022.csv'  
df = pd.read_csv(input_csv_path)

df['Precipitation'] = df['Precipitation'].replace(-9999.9, np.nan)

latitudes = df['Latitude'].values
longitudes = df['Longitude'].values
precipitation = df['Precipitation'].values

valid_mask = ~np.isnan(precipitation)


valid_latitudes = latitudes[valid_mask]
valid_longitudes = longitudes[valid_mask]
valid_precipitation = precipitation[valid_mask]


lat_bin_size = 0.1  
lon_bin_size = 0.1  
lat_bins = np.arange(latitudes.min(), latitudes.max() + lat_bin_size, lat_bin_size)
lon_bins = np.arange(longitudes.min(), longitudes.max() + lon_bin_size, lon_bin_size)
latitude_grid, longitude_grid = np.meshgrid(lat_bins, lon_bins)


interpolated_precipitation = griddata(
    (valid_longitudes, valid_latitudes),
    valid_precipitation,
    (longitude_grid.flatten(), latitude_grid.flatten()),
    method='nearest'
)


interpolated_precipitation = interpolated_precipitation.reshape(latitude_grid.shape)


def interpolate_precipitation(row):
    if np.isnan(row['Precipitation']):
        lat_idx = np.abs(lat_bins - row['Latitude']).argmin()
        lon_idx = np.abs(lon_bins - row['Longitude']).argmin()
        return interpolated_precipitation[lon_idx, lat_idx]
    else:
        return row['Precipitation']


df['Final_Precipitation'] = df.apply(interpolate_precipitation, axis=1)


final_df = df[['Latitude', 'Longitude', 'Final_Precipitation']]


final_df = final_df.rename(columns={'Final_Precipitation': 'Precipitation'})


output_csv_path = 'cleaned_12_2022.csv'  # New output path
final_df.to_csv(output_csv_path, index=False)

print(f"Cleaned CSV file saved as {output_csv_path}")
