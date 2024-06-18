import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.transform import resize
import nibabel as nib
from tensorflow.keras.models import load_model

TARGET_SHAPE = (64, 64, 64)

class Model:
    def __init__(self, model_path, model_type):
        if model_path.endswith('.h5'):
            self.model = load_model(model_path)
            self.model_type = 'keras'
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            self.model_type = 'sklearn'
        self.label_encoder_gender = LabelEncoder()
        self.label_encoder_hand = LabelEncoder()
        self.scaler = StandardScaler()

    def preprocess_image(self, image_path):
        img = nib.load(image_path)
        data = img.get_fdata()
        data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        if data_normalized.shape != TARGET_SHAPE:
            data_resized = resize(data_normalized, TARGET_SHAPE, mode='constant', anti_aliasing=True)
        else:
            data_resized = data_normalized
        return data_resized.astype(np.float32)

    def preprocess_clinical_data(self, clinical_data):
        data = pd.DataFrame([clinical_data])
        data.columns = data.columns.astype(str)
        data['M/F'] = self.label_encoder_gender.fit_transform(data['M/F'])
        data['Hand'] = self.label_encoder_hand.fit_transform(data['Hand'])
        
        if 'CDR' in data.columns:
            cdr_one_hot = pd.get_dummies(data['CDR'], prefix='CDR', dtype=float)
            data = pd.concat([data.drop(columns=['CDR']), cdr_one_hot], axis=1)

        columns_to_scale = data.drop(columns=['ID', 'M/F', 'Hand', 'Delay'])
        scaled_features = self.scaler.fit_transform(columns_to_scale)

        scaled_data = pd.DataFrame(scaled_features, columns=columns_to_scale.columns)
        scaled_data.insert(0, 'Hand', data['Hand'].values)
        scaled_data.insert(0, 'M/F', data['M/F'].values)
        if 'CDR' in locals():
            scaled_data = pd.concat([scaled_data, cdr_one_hot], axis=1)
        scaled_data.insert(0, 'ID', data['ID'].values)

        return scaled_data.values

    def combine_data(self, image_data, clinical_data):
        # Combinar datos de imagen y clínicos
        if self.model_type == 'keras':
            combined_data = [image_data.reshape(1, *TARGET_SHAPE, 1), clinical_data]
        else:
            combined_data = np.hstack([image_data.flatten().reshape(1, -1), clinical_data])
        return combined_data

    def predict(self, image_path, clinical_data):
        processed_image = self.preprocess_image(image_path)
        processed_clinical_data = self.preprocess_clinical_data(clinical_data)
        combined_data = self.combine_data(processed_image, processed_clinical_data)
        prediction = self.model.predict(combined_data)
        return prediction
