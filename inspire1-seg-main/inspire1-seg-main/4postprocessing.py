import numpy as np
import os
import geopandas as gpd
import pandas as pd

"Change the path to the data folder:"
data_path = r''  #######################################################################################################


for folder in os.listdir(data_path):
    shapefile_path = os.path.join(data_path, folder, 'training', folder + '_results', 'ClassifiedVector.shp')
    if not os.path.isfile(shapefile_path):
        print(shapefile_path)
        continue
    shapefile = gpd.read_file(shapefile_path)
    shapefile = shapefile[shapefile['meanB0'] > 1]
    shapefile.to_file(os.path.join(data_path, folder, 'training', folder + '_results', 'ClassifiedVector_fixed.shp'))

    shapefile['area'] = shapefile['geometry'].area
    class_area = pd.DataFrame(columns=['area', 'class'], index=shapefile['class'].unique())
    class_area.sort_index(inplace=True)
    for cl in shapefile['class'].unique():
        class_area.loc[cl, 'class'] = cl
        class_area.loc[cl, 'area'] = shapefile[shapefile['class'] == cl]['area'].sum()

    class_area['percent'] = (class_area['area'] / class_area['area'].sum()) * 100
    class_area.to_csv(os.path.join(data_path, folder, 'training', folder + '_results', 'class_area_percentage.csv'), index=False)

    folder_path = os.path.join(data_path, folder, 'training')
    if not os.path.isdir(folder_path) or not any(os.scandir(folder_path)):
        print(folder_path)
        continue
    training_vectors = []
    validation_vectors = []
    for file in os.listdir(folder_path):
        if file.__contains__('.shp') and file.lower().__contains__('train'):
            training_vectors.append(os.path.join(folder_path, file))
        if file.__contains__('.shp') and file.lower().__contains__('val'):
            validation_vectors.append(os.path.join(folder_path, file))

    if not training_vectors or not validation_vectors:
        continue

    training_polygons = gpd.read_file(training_vectors[0])
    validation_polygons = gpd.read_file(validation_vectors[0])
    if len(training_vectors) > 1:
        for i in range(1, len(training_vectors)):
            training_polygons = training_polygons.append(gpd.read_file(training_vectors[i]))
    if len(validation_vectors) > 1:
        for i in range(1, len(validation_vectors)):
            validation_polygons = validation_polygons.append(gpd.read_file(validation_vectors[i]))

    polygon_number = pd.DataFrame(columns=['class', 'training_polygons_number', 'validation_polygons_number'], index=training_polygons['class'].unique())
    polygon_number.sort_index(inplace=True)
    for cl in training_polygons['class'].unique():
        polygon_number.loc[cl, 'class'] = cl
        polygon_number.loc[cl, 'training_polygons_number'] = np.count_nonzero(training_polygons['class'] == cl)
        polygon_number.loc[cl, 'validation_polygons_number'] = np.count_nonzero(validation_polygons['class'] == cl)

    polygon_number.to_csv(os.path.join(folder_path, folder + '_results', 'number_of_polygons.csv'), index=False)