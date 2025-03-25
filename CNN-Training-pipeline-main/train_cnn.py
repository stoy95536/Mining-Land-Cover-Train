import pandas as pd
import os
from keras.layers import UpSampling2D
import numpy as np
from sklearn.metrics import cohen_kappa_score
from keras import backend as K
import json
import pickle
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import rasterio
from rasterio.transform import rowcol
from keras.layers import Input, Conv2DTranspose, concatenate, Cropping2D
from keras.models import Model


class CNN_Segmentor:
    def __init__(self, image_folder, geojson_folder, class_list) -> None:
        self.image_folder = image_folder
        self.geojson_folder = geojson_folder
        # self.result_folder = os.path.join(os.path.dirname(self.image_folder), "results_multi-band_cnn")
        # os.makedirs(self.result_folder, exist_ok=True)
        self.image_patches = []
        self.labels = []
        self.class_list = class_list
        self.img_size = (26, 26)  # You can adjust this size as needed.

    def save_label_encoder(self, label_encoder):
        """Save label encoder to the result folder"""
        with open(os.path.join(self.result_folder, "label_encoder.pkl"), "wb") as file:
            pickle.dump(label_encoder, file)
    
    def check_missing_classes(self, y, label_encoder):
        """Check for missing classes in the processed data"""
        present_classes = set(np.argmax(y, axis=1))  # Classes that are present in y
        all_classes = set(range(len(label_encoder.classes_)))
        missing_classes = all_classes - present_classes
        
        if missing_classes:
            missing_class_names = [label_encoder.classes_[i] for i in missing_classes]
            print(f"Missing classes in this site: {missing_class_names}")
            with open(os.path.join(self.result_folder, "missing_classes.txt"), "w") as file:
                file.write(f"Missing classes: {', '.join(missing_class_names)}\n")
        else:
            print("All classes are present in this site.")
    
    def set_result_folder(self, site_folder, result_folder):
        """Set result folder based on the site folder"""
        self.result_folder = os.path.join(result_folder, os.path.basename(site_folder))
        os.makedirs(self.result_folder, exist_ok=True)

    def extract_patches(self, json_data, image, class_name, feature_num, image_filename):
        feature_num = int(feature_num)

        if feature_num < len(json_data['features']):
            shape = json_data['features'][feature_num]
            coordinates = np.array(shape['geometry']['coordinates'][0], dtype=np.float32)

            # Use the raster transform to map geographic coordinates to pixel coordinates
            with rasterio.open(image_filename) as src:
                pixel_coords = [rowcol(src.transform, x, y) for x, y in coordinates]
            
            pixel_coords = np.array(pixel_coords)
            x_min, y_min = max(0, int(pixel_coords[:, 1].min())), max(0, int(pixel_coords[:, 0].min()))
            x_max, y_max = min(image.shape[1], int(pixel_coords[:, 1].max())), min(image.shape[0], int(pixel_coords[:, 0].max()))

            if x_max > x_min and y_max > y_min:
                patch = image[y_min:y_max, x_min:x_max]
                if patch.size > 0:
                    patch_resized = cv2.resize(patch, self.img_size)
                    self.image_patches.append(patch_resized)
                    self.labels.append(class_name)

    def load_data(self):
        for file in os.listdir(self.image_folder):
            if file.endswith('.tif'):
                filename_parts = file.split('_')
                class_name = None
                class_index = -1  # To store the index where the class name is found

                # Loop through filename parts to find a class name in the list
                for idx, part in enumerate(filename_parts):
                    if part in self.class_list:
                        class_name = part
                        class_index = idx  # Store the index of the class name
                        break
              
                if class_name is None:
                    print(f"Warning: No class found in {file}. Skipping...")
                    continue

                # Handle feature number extraction, which is the last part of the filename
                feature_num = filename_parts[-1].replace('feature', '').split('.')[0]

                # Create geojson filename by using only the parts before the class index
                # Assuming the geojson filename is formed by joining the parts before the class name
                geojson_filename = f"{'_'.join(filename_parts[0:class_index])}.geojson"

                image_filename = os.path.join(self.image_folder, file)
                geojson_filepath = os.path.join(self.geojson_folder, geojson_filename)

                if os.path.exists(image_filename) and os.path.exists(geojson_filepath):
                    with open(geojson_filepath, 'r') as json_file:
                        json_data = json.load(json_file)

                    with rasterio.open(image_filename) as src:
                        image = src.read()

                        # Now handle 5-band images
                        if image.shape[0] == 5:
                            image = np.stack([image[0], image[1], image[2], image[3], image[4]], axis=-1)
                        else:
                            print(f"Warning: The image {image_filename} does not have 5 bands.")

                        # Call the method to extract patches
                        self.extract_patches(json_data, image, class_name, feature_num, image_filename)

    def preprocess_data(self):
        X = np.array(self.image_patches) / 255.0  # Normalize pixel values
        X = X.reshape(-1, self.img_size[0], self.img_size[1], 5)  # Adjust for 5-band images
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.labels)
        y = to_categorical(y)  # One-hot encode labels
        return X, y, label_encoder

    def augment_data(self, X, y, label_encoder):
        class_counts = np.bincount(np.argmax(y, axis=1))
        max_count = class_counts.max()
        augmented_images = []
        augmented_labels = []

        datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0.2,
                                     zoom_range=0.1, horizontal_flip=True)

        for class_idx in range(len(class_counts)):
            class_mask = np.argmax(y, axis=1) == class_idx
            class_images = X[class_mask]
            class_labels = y[class_mask]

            if class_counts[class_idx] < max_count:
                augment_size = max_count - class_counts[class_idx]
                i = 0
                for batch in datagen.flow(class_images, class_labels, batch_size=1):
                    augmented_images.append(batch[0].reshape(self.img_size[0], self.img_size[1], 5))
                    augmented_labels.append(batch[1].reshape(y.shape[1]))  # Reshape augmented labels to match y's dimensions
                    i += 1
                    if i >= augment_size:
                        break

        if augmented_images:
            X_augmented = np.array(augmented_images)
            y_augmented = np.array(augmented_labels)
            X = np.concatenate((X, X_augmented), axis=0)
            y = np.concatenate((y, y_augmented), axis=0)

        return X, y




    def build_cnn(self):
        """Build a simple CNN model for 5-band images"""
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 5)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second convolutional block
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third convolutional block
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the output from convolutional layers
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        
        # Output layer (softmax for multi-class classification)
        model.add(Dense(len(np.unique(self.labels)), activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train_cnn(self, model, X_train, y_train, X_test, y_test):
        history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
        model.save(os.path.join(self.result_folder, "cnn_model.h5"))
        return history

    def mean_iou(self, y_true, y_pred):
        # Convert the predictions to binary labels based on the maximum probability
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.argmax(y_true, axis=-1)
        
        # Calculate IoU for each class and take the mean IoU
        iou_list = []
        for i in range(len(np.unique(y_true))):
            true_labels = K.equal(y_true, i)
            pred_labels = K.equal(y_pred, i)
            intersection = K.sum(K.cast(true_labels & pred_labels, 'float32'))
            union = K.sum(K.cast(true_labels | pred_labels, 'float32'))
            iou = intersection / (union + K.epsilon())
            iou_list.append(iou)
        
        return K.mean(K.stack(iou_list))

    def evaluate_cnn(self, model, X_test, y_test, label_encoder):
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true)
        print(f"Accuracy: {accuracy}")
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred_classes)
        report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
        
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Kappa index calculation
        kappa = cohen_kappa_score(y_true, y_pred_classes)
        print(f"Kappa Index: {kappa}")
        
        # mIoU calculation
        iou = self.mean_iou(y_test, y_pred).numpy()  # Convert tensor to numpy
        print(f"Mean IoU: {iou}")
        
        # Save results
        with open(os.path.join(self.result_folder, "cnn_evaluation_results.txt"), 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Kappa Index: {kappa}\n")
            f.write(f"Mean IoU: {iou}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.result_folder, "cnn_confusion_matrix.png"))
        plt.close()

        # Plot classification report
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0, 0.5, report, fontsize=12, va='center')
        ax.axis('off')
        plt.savefig(os.path.join(self.result_folder,  "cnn_classification_report.png"))
        plt.close()
        
    def plot_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(os.path.join(self.result_folder, "cnn_training_history.png"))
        plt.close()


if __name__ == "__main__":
    input_folder = r"E:\Bojana\Training\Clipped_v2"
    geojson_folder = r"E:\Bojana\Training\Geojsons"
    results_folder = r"E:\Bojana\Training\Final-Site-sorted-results-CNN"
    os.makedirs(results_folder, exist_ok=True)
    class_list = ['plantations', 'shrubs', 'meadows', 'roads', 'agricultural fields', 'water', 'human-made objects', 'forests', 'rocks']

    all_patches = []
    all_labels = []

    for site_folder in os.listdir(input_folder):
        full_site_folder = os.path.join(input_folder, site_folder)
        
        if os.path.isdir(full_site_folder):
            print(f"Processing site: {site_folder}")

            checker = CNN_Segmentor(full_site_folder, geojson_folder, class_list)
            
            checker.set_result_folder(full_site_folder, results_folder)

            checker.load_data()
            X, y, label_encoder = checker.preprocess_data()
            X, y = checker.augment_data(X, y, label_encoder)

            # Save the label encoder for reference
            checker.save_label_encoder(label_encoder)

            # Check for missing classes
            checker.check_missing_classes(y, label_encoder)

            all_patches.extend(X)
            all_labels.extend(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            site_model = checker.build_cnn()
            history_site = checker.train_cnn(site_model, X_train, y_train, X_test, y_test)

            checker.evaluate_cnn(site_model, X_test, y_test, label_encoder)
            checker.plot_history(history_site)

            print(f"Completed processing for site: {site_folder}")

    # print("Training on the full dataset...")
    # all_patches = np.array(all_patches)
    # all_labels = np.array(all_labels)

    # X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(all_patches, all_labels, test_size=0.2, random_state=42)

    # overall_checker = CNN_Segmentor(input_folder, geojson_folder, class_list)

    # full_model = overall_checker.build_cnn()
    # history_full = overall_checker.train_cnn(full_model, X_train_full, y_train_full, X_test_full, y_test_full)

    # overall_checker.set_result_folder(input_folder, results_folder)
    # overall_checker.evaluate_cnn(full_model, X_test_full, y_test_full, label_encoder)
    # overall_checker.plot_history(history_full)

    # print("Training on the full dataset completed.")

