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


#################################################################
"""
This script processes 3D coronary CT scan data, calculating Agatston and volume scores from DICOM files 
and corresponding NIfTI mask images. It organizes mask files by patient, converts pixel intensities to 
Hounsfield Units (HU), and computes total scores for each patient, saving the results into Excel and CSV files.

Main functionalities:
- `transform_to_hu`: Converts DICOM pixel values to Hounsfield Units (HU) using slope and intercept.
- `calculate_score`: Computes Agatston and volume scores for a specific CT slice and mask pair.
- `calculate_total_scores`: Aggregates total Agatston and volume scores for each patient by processing 
   all slices in a NIfTI mask file.
- `get_corresponding_ct_path`: Finds the correct DICOM file path corresponding to a given patient ID 
   and slice index.
- Data export: Creates a DataFrame with results and exports it to Excel and CSV.

This script requires a structured directory for DICOM images and NIfTI masks and calculates scores for 
each patient, facilitating further analysis of coronary calcium and volumetric data.
"""
#################################################################

dataset_name ='Dataset002_COCA3D'
base_dir = "/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2"
xml_path = base_dir + '/Gated_release_final/calcium_xml'
dicom_path = base_dir + '/Gated_release_final/patient'
nifti_path = base_dir + "/nn-Unetdataset/nnUNet_raw/"+ dataset_name



def transform_to_hu(dicom_path):
    medical_image = pydicom.dcmread(dicom_path)
    image = medical_image.pixel_array
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def calculate_score(ct_dicom_path, mask):
    # Charger les données DICOM en HU
    dicom_data = pydicom.dcmread(ct_dicom_path)
    hu_image = dicom_data.pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
    
    # Assurez-vous que le masque est booléen
    mask = mask > 0

    # Appliquer le masque pour isoler les zones calcifiées
    calcified_areas = hu_image * mask

    # Identifier les composants connectés
    labeled_array, num_features = ndimage.label(mask)

    # Récupérer les tailles de pixel pour le calcul de l'aire
    pixel_spacing = dicom_data.PixelSpacing
    pixel_area = pixel_spacing[0] * pixel_spacing[1]  # Taille d'un pixel en mm²

    ascore = 0
    vscore = 0
    # Calculer le score pour chaque région calcifiée
    for region_index in range(1, num_features + 1):
        region_mask = (labeled_array == region_index)
        region_hu_values = hu_image[region_mask]
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
        vscore = area_mm * 3

    return ascore, vscore

def calculate_total_scores(directory):
    # Dictionnaire pour stocker les scores totaux par patient
    patient_scores = {}
    patient_volume_scores = {}

    for filename in os.listdir(directory):
            if filename.endswith('.nii.gz'):
                total_ascore = 0 
                total_vscore = 0

                # Extraire l'ID du patient à partir du nom de fichier
                pid = int(filename.split('_')[1][:3])

                nifti_file = nib.load(directory + '/' + filename)
                niftitab = nifti_file.get_fdata()

                for i in range(niftitab.shape[2]):
                    sclice = niftitab[:, :, i]
                    ct_dicom_path=get_corresponding_ct_path(pid,i)


                    ascore,vscore = calculate_score(ct_dicom_path, sclice)
                    total_ascore += ascore
                    total_vscore += vscore

                # Stocker le score total pour ce patient
                patient_scores[pid] = total_ascore
                patient_volume_scores[pid] = total_vscore
    return (patient_scores,patient_volume_scores)



def get_corresponding_ct_path(pid,layer):

    dicom_patient_firstpath = '%s/%s'      %(dicom_path, pid)
    fichier_bizarre = os.listdir(dicom_patient_firstpath)[0]
    dicom_patient_path = '%s/%s/%s'      %(dicom_path, pid , fichier_bizarre)
    dicom_patient = [f for f in os.listdir(dicom_patient_path) if f.endswith('.dcm')]
    dicom_patient.sort()
    nslice = len(dicom_patient)
    dicom_file_path = '%s/%s'      %(dicom_patient_path  , dicom_patient[nslice-layer-1])

    return dicom_file_path


patient_scores , patient_volume_scores = calculate_total_scores(nifti_path +'/nnUNet_predict')
gived_patient_scores , gived_patient_volume_scores = calculate_total_scores(nifti_path +'/labelsTs')



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




























