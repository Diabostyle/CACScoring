import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import random
import csv
from statistics import mean, median, mode, variance
import nibabel as nib
import pandas as pd
import sys
import xml.etree.ElementTree as et
import plistlib
import seaborn as sns
import pickle
import os
from PIL import Image, ImageDraw

from scipy import ndimage
import shutil


#########################################################################
"""
This script processes 2D coronary CT scan data by calculating Agatston and volume scores from DICOM files
and corresponding mask images. It organizes mask files by patient, converts pixel intensities to Hounsfield
Units (HU), and calculates total scores for each patient, which are saved in an Excel and CSV file.

Main functionalities:
- `organize_files_by_patient`: Groups mask files by patient ID.
- `transform_to_hu`: Converts DICOM pixel values to Hounsfield Units (HU) based on metadata.
- `calculate_score`: Calculates Agatston and volume scores for a specific CT image and mask pair.
- `calculate_total_scores`: Computes the total Agatston and volume scores for each patient across all slices.
- `get_corresponding_ct_path`: Identifies the DICOM file corresponding to a specific mask file.
- Data export: Aggregates results into a DataFrame and exports it to Excel and CSV.

The script expects a structured directory for DICOM and mask files, processes each patient, and saves the
scoring results in an organized format.
"""
#########################################################################



dataset_name ='Dataset004_COCA2Dv3'
base_dir = "/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2"
xml_path = base_dir + '/Gated_release_final/calcium_xml'
dicom_path = base_dir + '/Gated_release_final/patient'
nifti_path = base_dir + "/nn-Unetdataset/nnUNet_raw/"+ dataset_name



def organize_files_by_patient(directory):
    # Dictionnaire pour stocker les fichiers par ID de patient
    patients_files = {}

    # Liste tous les fichiers dans le répertoire donné
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            # Extraire l'ID du patient à partir du nom de fichier
            patient_id = filename.split('_')[1][:3]

            # Ajouter le fichier à la liste correspondante dans le dictionnaire
            if patient_id not in patients_files:
                patients_files[patient_id] = []
            patients_files[patient_id].append(filename)

    return patients_files


files_by_patient = organize_files_by_patient(nifti_path+'/labelsTs')

# for patient_id, files in files_by_patient.items():
#     print(f"Patient ID: {patient_id}")
#     for file in files:
#         print(f"  - {file}")




def transform_to_hu(dicom_path):
    # Charger l'image médicale DICOM à partir du chemin fourni
    medical_image = dcmread(dicom_path)

    # Extraire l'image (tableau de pixels) de l'objet DICOM
    image = medical_image.pixel_array

    # Accéder aux métadonnées pour la transformation en HU
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope

    # Calculer les unités Hounsfield
    hu_image = image * slope + intercept

    return hu_image


def load_nifti_file(file_path):
    nifti_data = nib.load(file_path)
    return nifti_data.get_fdata()

    
def calculate_score(ct_dicom_path, mask_image_path):
    # Charger les données DICOM en HU
    dicom_data = pydicom.dcmread(ct_dicom_path)
    hu_image = dicom_data.pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
    
    # Charger le fichier de masque (supposé NIFTI)
    mask = load_nifti_file(mask_image_path)
    mask = mask > 0  # Assurez-vous que le masque est booléen

    # Appliquer le masque pour isoler les zones calcifiées
    calcified_areas = hu_image * mask

    # Identifier les composants connectés
    labeled_array, num_features = ndimage.label(mask)

    # Récupérer les tailles de pixel pour le calcul de l'aire
    pixel_spacing = dicom_data.PixelSpacing

    pixel_area = pixel_spacing[0] * pixel_spacing[1]  # Taille d'un pixel en mm²
    pixel_area = pixel_area 
    
    ascore = 0
    vscore = 0
    # Calculer le score pour chaque région calcifiée
    for region_index in range(1, num_features + 1):
        region_mask = (labeled_array == region_index)
        region_hu_values = hu_image[region_mask]
        # Multiplier le nombre de pixels par la taille d'un pixel en mm² pour obtenir l'aire
        area_mm = np.sum(region_mask) * pixel_area  
        max_hu = np.max(region_hu_values)

        # Déterminer le poids basé sur l'intensité maximale de HU
        if max_hu >= 400:
            weight = 4
        elif max_hu >= 300:
            weight = 3
        elif max_hu >= 200:
            weight = 2
        elif max_hu >= 130:
            weight = 1
        else:
            weight = 0

        ascore += area_mm * weight

        vscore=area_mm * 3

    return(ascore,vscore)

# CT_image ='/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/0/Pro_Gated_CS_3.0_I30f_3_70%/IM-6130-0023.dcm'
# Mask_image = '/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset003_COCA2Dv2/labelsTr/CT_00034.nii.gz'
# print(calculate_agatston_score(CT_image,Mask_image))




def calculate_total_scores(nifti_mask_path, files_by_patient):
    # Dictionnaire pour stocker les scores totaux par patient
    patient_scores = {}
    patient_volume_scores = {}

    # Parcourir chaque patient et ses fichiers de masque
    for patient_id, mask_files in files_by_patient.items():
        total_ascore = 0 
        total_vscore = 0
        # Calculer le score pour chaque fichier de masque
        for mask_file in mask_files:
            # Construire les chemins complets pour les fichiers CT et masque
            mask_path = os.path.join(nifti_mask_path,mask_file)
            ct_path = get_corresponding_ct_path(mask_file)

            # Calculer le score d'Agatston pour cette paire d'images
            ascore , vscore = calculate_score(ct_path, mask_path)
            total_ascore += ascore
            total_vscore += vscore
            

        # Stocker le score total pour ce patient
        patient_scores[patient_id] = total_ascore
        patient_volume_scores[patient_id] = total_vscore
        
    return (patient_scores,patient_volume_scores)


def get_corresponding_ct_path(mask_file):
    pid = int(mask_file[3:6])   # Extraire les trois premiers chiffres après 'CT_'
    layer = int(mask_file[6:8]) # Extraire les trois chiffres suivants

    dicom_patient_firstpath = '%s/%s'      %(dicom_path, pid)
    fichier_bizarre = os.listdir(dicom_patient_firstpath)[0]
    dicom_patient_path = '%s/%s/%s'      %(dicom_path, pid , fichier_bizarre)
    dicom_patient = [f for f in os.listdir(dicom_patient_path) if f.endswith('.dcm')]
    dicom_patient.sort()
    nslice = len(dicom_patient)
    dicom_file_path = '%s/%s'      %(dicom_patient_path  , dicom_patient[nslice-layer-1])

    return dicom_file_path

# Utilisation de la fonction

patient_scores , patient_volume_scores = calculate_total_scores(nifti_path +'/nnUNet_predict', files_by_patient)
gived_patient_scores , gived_patient_volume_scores = calculate_total_scores(nifti_path +'/labelsTs', files_by_patient)


# Créer un DataFrame avec les résultats
results = []
for patient_id in sorted(patient_scores.keys()):
    results.append({
        "patient_name": patient_id,
        "given_agatston": float(gived_patient_scores.get(patient_id, 0)),
        "computed_agatston": float(patient_scores.get(patient_id, 0)),
        "given_vol": float(gived_patient_volume_scores.get(patient_id, 0)),
        "computed_vol": float(patient_volume_scores.get(patient_id, 0))

    })

results_df = pd.DataFrame(results)

# Enregistrer les résultats dans un fichier Excel
results_df.to_excel('Result_'+ dataset_name + '.xlsx', index=False)
results_df.to_csv('Result_'+ dataset_name + '.csv', index=False) 

print("Les fichiers ont été créés avec succès.")