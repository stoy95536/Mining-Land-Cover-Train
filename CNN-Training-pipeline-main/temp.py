import os
import numpy as np
import cv2
import pickle
import rasterio
import warnings

warnings.filterwarnings("ignore")
from keras.models import load_model

class Orthomosaic_Classifier:
    def __init__(self, img_size=(26, 26), batch_size=64, no_data_value=0):
        self.img_size = img_size
        self.batch_size = batch_size
        self.no_data_value = no_data_value
        self.class_color_map = {
            'forests': [36, 113, 31],
            'shrubs': [178, 222, 137],
            'meadows': [230, 255, 0],
            'roads': [114, 74, 26],
            'water': [80, 198, 230],
            'agricultural fields': [255, 127, 0],
            'rocks': [125, 139, 143],
            'plantations': [36, 143, 50],
            'human-made objects': [255, 29, 0]
        }

    def load_site_model_and_encoder(self, site_folder):
        model_path = os.path.join(site_folder, 'cnn_model.h5')
        encoder_path = os.path.join(site_folder, 'label_encoder.pkl')
        model = load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        class_color_map_encoded = {
            i: self.class_color_map[label] for i, label in enumerate(label_encoder.classes_)
        }
        return model, label_encoder, class_color_map_encoded

    def preprocess_patch(self, image_patch):
        if np.all(image_patch == self.no_data_value):
            return None
        patch_resized = cv2.resize(image_patch, self.img_size)
        patch_resized = patch_resized / 255.0
        return patch_resized

    def classify_orthomosaic(self, model, orthomosaic_filename, patch_size=26):
        with rasterio.open(orthomosaic_filename) as src:
            orthomosaic = src.read()  # Read all 5 bands
            orthomosaic = np.moveaxis(orthomosaic, 0, -1)  # Move bands to the last dimension
            transform = src.transform
            profile = src.profile

        height, width, _ = orthomosaic.shape
        classified_image = np.full((height, width), self.no_data_value, dtype=np.uint8)

        patches = []
        coords = []

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = orthomosaic[y:min(y+patch_size, height), x:min(x+patch_size, width)]
                processed_patch = self.preprocess_patch(patch)
                
                if processed_patch is not None:
                    patches.append(processed_patch)
                    coords.append((y, x))

                if len(patches) == self.batch_size:
                    self.process_batch(model, patches, coords, classified_image, patch_size, height, width)
                    patches = []  # Reset the batch
                    coords = []  # Reset the coordinates

        if patches:
            self.process_batch(model, patches, coords, classified_image, patch_size, height, width)

        return classified_image, transform, profile

    def process_batch(self, model, patches, coords, classified_image, patch_size, height, width):
        patches = np.array(patches)
        predictions = model.predict(patches)
        predicted_classes = np.argmax(predictions, axis=1)

        for idx, (y, x) in enumerate(coords):
            classified_image[y:min(y+patch_size, height), x:min(x+patch_size, width)] = predicted_classes[idx]

    def generate_colormap(self, classified_image, class_color_map_encoded):
        # Initialize the color map with white for no-data values
        colored_image = np.full((classified_image.shape[0], classified_image.shape[1], 3), [255, 255, 255], dtype=np.uint8)

        for label, color in class_color_map_encoded.items():
            colored_image[classified_image == label] = color

        return colored_image

    def save_classified_image(self, classified_image, colormap_image, output_folder, tiff_name, transform, profile):
        png_path = os.path.join(output_folder, f"{tiff_name}.png")
        cv2.imwrite(png_path, colormap_image)

        tiff_path = os.path.join(output_folder, f"{tiff_name}.tif")
        

        # Set no-data value in profile if required
        profile['nodata'] = self.no_data_value

        with rasterio.open(tiff_path, 'w', **profile) as dst:
            dst.write(classified_image, 1)

        print(f"Results saved to {output_folder}")

    def process_folder(self, tiff_folder, site_model_folder, result_folder, patch_size=26):
        for tiff_file in os.listdir(tiff_folder):
            if tiff_file.endswith('_multiband.tiff') or tiff_file.endswith('_multiband.tif'):
                tiff_path = os.path.join(tiff_folder, tiff_file)
                tiff_name = os.path.splitext(tiff_file)[0].replace('_multiband', '')

                site_folder = os.path.join(site_model_folder, tiff_name)
                if not os.path.isdir(site_folder):
                    print(f"Warning: No model folder found for {tiff_file}. Skipping...")
                    continue

                print(f"Processing {tiff_file} using model and encoder from {site_folder}...")

                model, label_encoder, class_color_map_encoded = self.load_site_model_and_encoder(site_folder)

                classified_img, transform, profile = self.classify_orthomosaic(model, tiff_path, patch_size)
                colored_img = self.generate_colormap(classified_img, class_color_map_encoded)

                output_folder = os.path.join(result_folder, tiff_name)
                os.makedirs(output_folder, exist_ok=True)
                self.save_classified_image(classified_img, colored_img, output_folder, tiff_name, transform, profile)
          

if __name__ == "__main__":
    classifier = Orthomosaic_Classifier()

    tiff_folder = r"E:\Bojana\Training\multiband_tiffs"
    site_model_folder = r"E:\Bojana\Training\Final-Site-sorted-results-CNN"
    result_folder = r"E:\Bojana\Training\multiband_tiffs_results"
    os.makedirs(result_folder, exist_ok=True)

    classifier.process_folder(tiff_folder, site_model_folder, result_folder)
