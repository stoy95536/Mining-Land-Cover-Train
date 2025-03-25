import otbApplication as otb
import os

"Change the path to the data folder:"
data_path = r''  #######################################################################################################


def training(folder_path, training_vectors, validation_vectors):
    output_folder_name = os.path.dirname(folder_path).split('\\')[-1] + '_results'
    output_folder_path = os.path.join(folder_path, output_folder_name)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    app = otb.Registry.CreateApplication("TrainVectorClassifier")
    app.SetParameterStringList("io.vd", training_vectors)
    app.SetParameterString("io.out", os.path.join(output_folder_path, "RandomForest.txt"))
    app.SetParameterStringList("feat", ["meanB0", "meanB1", "meanB2", "meanB3", "meanB4", "varB0", "varB1", "varB2", "varB3", "varB4"])
    app.SetParameterStringList("valid.vd", validation_vectors)
    app.SetParameterString("cfield", "class")
    app.SetParameterString("classifier", "rf")
    app.SetParameterInt("classifier.rf.max", 15)
    app.SetParameterInt("classifier.rf.min", 5)
    app.SetParameterString("io.confmatout", os.path.join(output_folder_path, 'confusion_matrix.csv'))
    app.ExecuteAndWriteOutput()


for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder, 'training')
    if not os.path.isdir(folder_path):
        print(folder)
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
    print('Training:', training_vectors)
    print('Validaion:', validation_vectors)
    training(folder_path, training_vectors, validation_vectors)