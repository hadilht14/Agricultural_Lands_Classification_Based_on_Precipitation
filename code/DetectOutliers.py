import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


input_csv_path = 'Cleaned_CsvFormat_data/cleaned_02_2023.csv'  
df = pd.read_csv(input_csv_path)

print(df.head())


def detect_local_outliers(df, lat_col, lon_col, value_col, radius=0.5):
    """
    Detect local outliers based on geographic neighborhoods.
    """
    coords = df[[lat_col, lon_col]].dropna().to_numpy()  
    values = df[value_col].dropna().to_numpy()           

    tree = cKDTree(coords)
    
    outliers_mask = np.zeros(len(df), dtype=bool)
    
    for idx, (lat, lon, val) in enumerate(zip(df[lat_col], df[lon_col], df[value_col])):
        if np.isnan(val):  
            continue
        
        neighbor_indices = tree.query_ball_point([lat, lon], radius)
        
        neighbor_values = values[neighbor_indices]
        
        Q1 = np.percentile(neighbor_values, 25)
        Q3 = np.percentile(neighbor_values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if val < lower_bound or val > upper_bound:
            outliers_mask[idx] = True

    return outliers_mask


outliers_mask = detect_local_outliers(df, 'Latitude', 'Longitude', 'Precipitation')


num_outliers = outliers_mask.sum()
outliers_range = (df.loc[outliers_mask, 'Precipitation'].min(), df.loc[outliers_mask, 'Precipitation'].max())
print(f"Number of local outliers: {num_outliers}")
print(f"Range of local outliers: {outliers_range}")


df.loc[outliers_mask, 'Precipitation'] = np.nan


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


output_csv_path = 'PreProcessedData/preprocessed_03_2023_outliers_handled.csv'  # New output path
final_df.to_csv(output_csv_path, index=False)

print(f"Cleaned CSV file with outliers handled saved as {output_csv_path}")


outliers_mask_final = detect_local_outliers(final_df, 'Latitude', 'Longitude', 'Precipitation')


num_outliers_final = outliers_mask_final.sum()
outliers_range_final = (final_df.loc[outliers_mask_final, 'Precipitation'].min(), final_df.loc[outliers_mask_final, 'Precipitation'].max())
print(f"Number of local outliers in final data: {num_outliers_final}")
print(f"Range of local outliers in final data: {outliers_range_final}")


# plt.figure(figsize=(10, 6))
# sns.boxplot(data=final_df, x='Precipitation', color='lightblue')
# plt.title('Box Plot of Precipitation (Outliers Handled)', fontsize=14)
# plt.xlabel('Precipitation (mm)', fontsize=12)
# plt.show()