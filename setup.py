import os
import requests
from setuptools import setup

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

def load_models():
    binary_model_url = 'https://drive.google.com/uc?export=download&id=14igHOLLg74WiM-eTHPVA9sKs_hAmiuVr'
    binary_model_path = os.path.join('Setophaga', 'models', 'binary.h5')
    download_model(binary_model_url, binary_model_path)

    warbler_model_url = 'https://drive.google.com/uc?export=download&id=1cFwNVpCaMacM9fDv_2qIEOB70XkwKfKs'
    warbler_model_path = os.path.join('Setophaga', 'models', 'warbler.h5')
    download_model(warbler_model_url, warbler_model_path)

# Download and load models
load_models()

# Define other package details
setup(
    name='Setophaga',
    version='1.0',
    packages=['Setophaga'],
    # Include package data (e.g., model files)
    include_package_data=True,
    # Add other package metadata
    author='Oliver Burrus',
    description='A Warbler Nocturnal Flight Call Monitoring Project',
    install_requires=[
        'sounddevice',
        'numpy',
        'tensorflow',
        'scipy',
        'matplotlib',
        'pandas',
        'wave',
        'pylab',
        'pydub'
    ]
)
