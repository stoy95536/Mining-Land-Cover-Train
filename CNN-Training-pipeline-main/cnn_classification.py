import rasterio
import os
import numpy as np
import cv2
import pickle
from keras.models import load_model
from rasterio.enums import Resampling

class Orthomosaic_Classifier:
    def __init__(self, model_path, label_encoder_path, img_size=(26, 26), batch_size=32):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.batch_size = batch_size

        # Load the label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Define color map using provided color codes
        self.class_color_map = {
            'No Data': [0, 0, 0],
            'forests': [36, 71, 17],
            'shrubs': [178, 222, 137],
            'meadows': [230, 255, 0],
            'roads': [114, 74, 26],
            'water': [80, 198, 230],
            'agricultural fields': [255, 127, 0],
            'rocks': [125, 139, 143],
            'plantations': [36, 143, 50],
            'human-made objects': [255, 29, 0]
        }
        # Map labels to colors using label encoder classes
        self.class_color_map_encoded = {
            i: self.class_color_map[label] for i, label in enumerate(self.label_encoder.classes_)
        }

    def preprocess_patch(self, image_patch):
        patch_resized = cv2.resize(image_patch, self.img_size)
        patch_resized = patch_resized / 255.0
        return patch_resized

    def classify_orthomosaic(self, orthomosaic_filename, patch_size=26):
        with rasterio.open(orthomosaic_filename) as src:
            orthomosaic = src.read()  # Read all 5 bands
            orthomosaic = np.moveaxis(orthomosaic, 0, -1)  # Move bands to the last dimension
            transform = src.transform
            profile = src.profile

        height, width, _ = orthomosaic.shape
        classified_image = np.zeros((height, width), dtype=np.uint8)

        patches = []
        coords = []

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = orthomosaic[y:min(y+patch_size, height), x:min(x+patch_size, width)]
                patches.append(self.preprocess_patch(patch))
                coords.append((y, x))

                if len(patches) == self.batch_size:
                    self.process_batch(patches, coords, classified_image, patch_size, height, width)
                    patches = []  # Reset the batch
                    coords = []  # Reset the coordinates

        if patches:
            self.process_batch(patches, coords, classified_image, patch_size, height, width)

        return classified_image, transform, profile

    def process_batch(self, patches, coords, classified_image, patch_size, height, width):
        patches = np.array(patches)  # Convert to numpy array
        predictions = self.model.predict(patches)
        predicted_classes = np.argmax(predictions, axis=1)

        for idx, (y, x) in enumerate(coords):
            classified_image[y:min(y+patch_size, height), x:min(x+patch_size, width)] = predicted_classes[idx]

    def generate_colormap(self, classified_image):
        colored_image = np.zeros((classified_image.shape[0], classified_image.shape[1], 3), dtype=np.uint8)

        for label, color in self.class_color_map_encoded.items():
            colored_image[classified_image == label] = color

        return colored_image

    def save_classified_image(self, classified_image, colormap_image, output_folder, tiff_name, transform, profile):
        # Save PNG
        png_path = os.path.join(output_folder, f"{tiff_name}.png")
        cv2.imwrite(png_path, colormap_image)

        # Save GeoTIFF
        tiff_path = os.path.join(output_folder, f"{tiff_name}.tif")
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )

        with rasterio.open(tiff_path, 'w', **profile) as dst:
            dst.write(classified_image, 1)

        print(f"Results saved to {output_folder}")

    def process_folder(self, tiff_folder, result_folder, patch_size=26):
        for tiff_file in os.listdir(tiff_folder):
            if tiff_file.endswith('.tif'):
                tiff_path = os.path.join(tiff_folder, tiff_file)
                tiff_name = os.path.splitext(tiff_file)[0]
                
                # Create a result folder for each TIFF
                output_folder = os.path.join(result_folder, tiff_name)
                os.makedirs(output_folder, exist_ok=True)

                print(f"Processing {tiff_file}...")

                # Classify the orthomosaic
                classified_img, transform, profile = self.classify_orthomosaic(tiff_path, patch_size)

                # Generate color-mapped image
                colored_img = self.generate_colormap(classified_img)

                # Save the output in the result folder
                self.save_classified_image(classified_img, colored_img, output_folder, tiff_name, transform, profile)

if __name__ == "__main__":
    classifier = Orthomosaic_Classifier(
        r"E:\Bojana\Training\Final-Site-sorted-results\Backi_Monostor\cnn_model.h5",
        r"E:\Bojana\Training\Final-Site-sorted-results\Backi_Monostor\label_encoder.pkl"
    )

    tiff_folder = r"E:\Bojana\Training\multiband_tiffs"
    result_folder = r"E:\Bojana\Training\multiband_tiffs_results"
    os.makedirs(result_folder, exist_ok=True)

    classifier.process_folder(tiff_folder, result_folder)
