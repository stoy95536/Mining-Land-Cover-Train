import os
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class IntegratedSegmentor:
    def __init__(self, image_folder, geojson_folder) -> None:
        self.image_folder = image_folder
        self.geojson_folder = geojson_folder
        self.data = []
        self.img_size = (26, 26)  # Consistent patch size for CNN if needed

    def extract_and_sum_pixel_values(self, json_data, image, class_name, feature_num, image_filename):
        feature_num = int(feature_num)

        if feature_num < len(json_data['features']):
            shape = json_data['features'][feature_num]

            # Get geographic coordinates from the GeoJSON
            coordinates = np.array(shape['geometry']['coordinates'][0], dtype=np.float32)

            # Use rasterio to transform geographic coordinates to pixel coordinates
            with rasterio.open(image_filename) as src:
                pixel_coords = [rowcol(src.transform, x, y) for x, y in coordinates]

            pixel_coords = np.array(pixel_coords)

            x_min, y_min = max(0, int(pixel_coords[:, 1].min())), max(0, int(pixel_coords[:, 0].min()))
            x_max, y_max = min(image.shape[1], int(pixel_coords[:, 1].max())), min(image.shape[0], int(pixel_coords[:, 0].max()))

            if x_max > x_min and y_max > y_min:
                patch = image[y_min:y_max, x_min:x_max]

                if patch.size > 0:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pixel_coords], 255)

                    masked_image = cv2.bitwise_and(image, image, mask=mask)

                    total_sum = masked_image.sum()
                    pixel_mean = masked_image[mask == 255].mean()
                    pixel_max = masked_image[mask == 255].max()
                    pixel_std = masked_image[mask == 255].std()

                    # Additional features
                    mean_r, mean_g, mean_b = masked_image[mask == 255].mean(axis=0)
                    std_r, std_g, std_b = masked_image[mask == 255].std(axis=0)

                    width = x_max - x_min
                    height = y_max - y_min
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0

                    self.data.append([class_name, total_sum, pixel_mean, pixel_max, pixel_std, mean_r, mean_g, mean_b, std_r, std_g, std_b, area, aspect_ratio])

    def load_data(self):
        for file in os.listdir(self.image_folder):
            if file.endswith('.tif'):
                filename_parts = file.split('_')
                class_name = filename_parts[2]
                feature_num = filename_parts[-1].replace('f', '').split('.')[0]

                # Construct the GeoJSON filename based on the TIFF image filename pattern
                geojson_filename = f"bb_points_Ds_{'_'.join(filename_parts[0:2])}.geojson"

                image_filename = os.path.join(self.image_folder, file)
                geojson_filepath = os.path.join(self.geojson_folder, geojson_filename)

                if os.path.exists(image_filename) and os.path.exists(geojson_filepath):
                    with open(geojson_filepath, 'r') as json_file:
                        json_data = json.load(json_file)

                    with rasterio.open(image_filename) as src:
                        image = src.read()

                        # Stack the first 3 bands (assuming RGB) for pixel calculations
                        if image.shape[0] >= 3:
                            image = np.stack([image[0], image[1], image[2]], axis=-1)
                        else:
                            image = np.expand_dims(image, axis=-1)

                        # Extract patches and calculate pixel statistics
                        self.extract_and_sum_pixel_values(json_data, image, class_name, feature_num, image_filename)

    def csv_maker(self):
        temp = os.path.join(os.path.dirname(self.image_folder), "results")
        os.makedirs(temp, exist_ok=True)
        output_file = os.path.join(temp, 'train.csv')
        df = pd.DataFrame(self.data, columns=['ClassName', 'PixelSum', 'Pixel Mean', 'Pixel Max', 'Pixel-std', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'area', 'aspect_ratio'])
        df.to_csv(output_file, index=False)

    def random_forest_classifier(self):
        df = pd.read_csv(os.path.join(os.path.dirname(self.image_folder), "results", "train.csv"))
        X = df[['PixelSum', 'Pixel Mean', 'Pixel Max', 'Pixel-std', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'area']]
        y = df['ClassName']

        # Calculate class weights dynamically
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=200, class_weight=class_weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)

        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)

        results_folder = os.path.join(os.path.dirname(self.image_folder), "results")

        # Save the trained RandomForest model
        model_file_path = os.path.join(results_folder, "rf_model.pkl")
        joblib.dump(model, model_file_path)
        print(f"RandomForest model saved to {model_file_path}")

        # Plot and save confusion matrix
        self.plot_confusion_matrix(cm, classes, os.path.join(results_folder, "confusion_matrix.png"))

        # Save classification report
        self.save_classification_report(report, os.path.join(results_folder, "classification_report.csv"))

    def plot_confusion_matrix(self, cm, classes, output_path):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(output_path)
        plt.close()

    def save_classification_report(self, report_dict, output_path_csv):
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.to_csv(output_path_csv, index=True)

if __name__ == "__main__":
    input_folder = r"E:\Bojana\Training\clipped_1x1"
    geojson_folder = r"E:\Bojana\Training\geojsons_remap"
    checker = IntegratedSegmentor(input_folder, geojson_folder)
    checker.load_data()
    checker.csv_maker()
    checker.random_forest_classifier()
