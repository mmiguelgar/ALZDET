from django.shortcuts import render
from .forms import UploadForm
from .model import Model
import os
import pandas as pd
from django.core.files.storage import FileSystemStorage

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_paths = {
    'cnn_model_gfc': os.path.join(BASE_DIR, 'app/models/cnn_model_gfc.h5'),
    'cnn_model_masked_gfc': os.path.join(BASE_DIR, 'app/models/cnn_model_masked_gfc.h5'),
    'gb_model': os.path.join(BASE_DIR, 'app/models/gb_model.pkl'),
    'rf_model': os.path.join(BASE_DIR, 'app/models/rf_model.pkl'),
    'svm_model': os.path.join(BASE_DIR, 'app/models/svm_model.pkl'),
}

def classify_image(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['mri_image']
            model_choice = form.cleaned_data['model_choice']
            clinical_data_file = request.FILES['clinical_data']

            # Guardar el archivo subido temporalmente
            fs = FileSystemStorage()
            image_filename = fs.save(image.name, image)
            uploaded_image_path = fs.path(image_filename)

            clinical_data_filename = fs.save(clinical_data_file.name, clinical_data_file)
            uploaded_clinical_data_path = fs.path(clinical_data_filename)

            # Leer los datos clínicos del archivo .xlsx
            clinical_data_df = pd.read_excel(uploaded_clinical_data_path)
            clinical_data = clinical_data_df.to_dict('records')[0]

            model_path = model_paths[model_choice]
            model = Model(model_path, model_choice)

            # Pasar la ruta del archivo guardado al modelo
            prediction = model.predict(uploaded_image_path, clinical_data)

            # Borrar los archivos temporales después de usarlos
            fs.delete(image_filename)
            fs.delete(clinical_data_filename)

            return render(request, 'result.html', {'prediction': prediction})

    else:
        form = UploadForm()
    return render(request, 'upload.html', {'form': form})
