from django import forms

class UploadForm(forms.Form):
    MRI_CHOICES = [
        ('cnn_model_gfc', 'CNN Model GFC (.h5)'),
        ('cnn_model_masked_gfc', 'CNN Model Masked GFC (.h5)'),
        ('gb_model', 'Gradient Boosting Model (.pkl)'),
        ('rf_model', 'Random Forest Model (.pkl)'),
        ('svm_model', 'SVM Model (.pkl)'),
    ]

    mri_image = forms.FileField()
    model_choice = forms.ChoiceField(choices=MRI_CHOICES)
    clinical_data = forms.FileField(help_text="Upload the clinical data file (.xlsx) here.")
