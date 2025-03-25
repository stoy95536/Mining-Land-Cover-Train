import os
import numpy as np
import rasterio

def reclassify_raster(input_folder, output_folder):
    # Define the reclassification mapping: original values 0-9 to new values 1-10
    reclass_mapping = {i: i + 1 for i in range(10)}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Walk through all subdirectories and files in the input folder
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(('.tif', '.tiff')):
                input_path = os.path.join(root, file_name)
                
                # Define output path to save all files in the output folder without subfolders
                output_path = os.path.join(output_folder, file_name)
                
                with rasterio.open(input_path) as src:
                    # Copy the profile and set the number of bands to 1 for single-band output
                    profile = src.profile
                    profile.update(count=1)

                    # Read the first band of data
                    data = src.read(1)
                    
                    # Initialize the reclassified data array
                    reclassified_data = np.copy(data)
                    
                    # Apply reclassification mapping
                    for old_value, new_value in reclass_mapping.items():
                        reclassified_data[data == old_value] = new_value
                    
                    # Write the reclassified raster as a single band
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(reclassified_data, 1)
                        
                print(f"Reclassified raster saved as: {output_path}")


# Specify your folders here
input_folder = r"E:\Bojana\Training\multiband_tiffs_results"
output_folder = r"E:\Bojana\Training\reclass_rasters"

# Call the function
reclassify_raster(input_folder, output_folder)