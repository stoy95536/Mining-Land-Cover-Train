import otbApplication as otb
import os

"Change the path to the data folder:"
data_path = r''  #######################################################################################################


def classification(vector_data, model, output_classification):
    app = otb.Registry.CreateApplication("VectorClassifier")
    app.SetParameterString("in", vector_data)
    app.SetParameterString("model", model)
    app.SetParameterString("out", output_classification)
    app.SetParameterStringList("feat", ["meanB0", "meanB1", "meanB2", "meanB3", "meanB4", "varB0", "varB1", "varB2", "varB3", "varB4"])
    app.SetParameterString("cfield", "class")
    app.SetParameterString("confmap", 'True')
    app.ExecuteAndWriteOutput()


for folder in os.listdir(data_path):
    vector_data = os.path.join(data_path, folder, folder + '_vektorizacija', 'Vectorized_r3.shp')
    model = os.path.join(data_path, folder, 'training', folder + '_results', 'RandomForest.txt')
    if not os.path.isfile(vector_data) or not os.path.isfile(model):
        print(folder)
        continue
    output_classification = os.path.join(data_path, folder, 'training', folder + '_results', 'ClassifiedVector.shp')
    classification(vector_data, model, output_classification)