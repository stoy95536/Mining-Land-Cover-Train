import otbApplication as otb
import os

"Change the path to the data folder:"
data_path = r''  #######################################################################################################


def vectorization(folder_path, input_image_name, range_radius):
    app1 = otb.Registry.CreateApplication("LSMSSegmentation")
    app1.SetParameterString("in", os.path.join(folder_path, input_image_name))
    app1.SetParameterString("out", os.path.join(folder_path, 'Segmented_r' + str(range_radius) + '.tif'))
    app1.SetParameterFloat("ranger", range_radius)  # 5, 8
    app1.ExecuteAndWriteOutput()
    app2 = otb.Registry.CreateApplication("LSMSSmallRegionsMerging")
    app2.SetParameterString("in", os.path.join(folder_path, input_image_name))
    app2.SetParameterString("inseg", os.path.join(folder_path, 'Segmented_r' + str(range_radius) + '.tif'))
    app2.SetParameterString("out", os.path.join(folder_path, 'Small_regions_merged_r' + str(range_radius) + '.tif'))
    app2.ExecuteAndWriteOutput()
    app3 = otb.Registry.CreateApplication("LSMSVectorization")
    app3.SetParameterString("in", os.path.join(folder_path, input_image_name))
    app3.SetParameterString("inseg", os.path.join(folder_path, 'Small_regions_merged_r' + str(range_radius) + '.tif'))
    app3.SetParameterString("out", os.path.join(folder_path, 'Vectorized_r' + str(range_radius) + '.shp'))
    app3.ExecuteAndWriteOutput()


for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder, folder + '_vektorizacija')
    for file in os.listdir(folder_path):
        if file.lower().__contains__('multiband'):
            input_image_name = file
    if 'input_image_name' not in locals():
        print(folder)
        continue
    for range_radius in [3]:
        vectorization(folder_path, input_image_name, range_radius)
    del input_image_name