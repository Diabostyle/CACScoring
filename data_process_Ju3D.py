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
import shutil


###########################################################################
"""
This script processes and organizes 3D coronary CT scan datasets for deep learning applications. It handles
the conversion of DICOM images to NIfTI format, generates 3D mask volumes based on XML annotations, and
organizes the data into structured directories for training and testing. 

Main functionalities:
- `process_xml`: Extracts pixel coordinates of coronary artery regions from XML files to generate masks.
- `dicom_to_nifti_3D`: Converts a series of DICOM files into a single 3D NIfTI image.
- `create_mask_image` and `create_3d_mask_array`: Generate 2D mask images from pixel coordinates and combine
   them into a 3D mask volume for each patient.
- `save_mask_as_nifti_3D`: Saves the 3D mask volume as a NIfTI file.
- `Initialisation_3D`: Processes all patient data, converts DICOMs to 3D NIfTI images, and saves associated
   3D masks in structured folders.
- `split_data_by_pid`: Splits data into training and test sets based on specified patient IDs (PIDs).

To use this script, specify the directory paths for DICOM images, XML annotations, and the output folder for
NIfTI files. Run `Initialisation_3D` to process the entire dataset, and use `split_data_by_pid` to partition
the data based on a predefined list of PIDs for model testing.
"""
###########################################################################

# set the random seed to create the same train/val/test split
np.random.seed(10015321)


plt.close("all")



pid = 6  # Remplacer par l'ID réel du patient
base_dir = "/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2"
xml_path = base_dir + '/Gated_release_final/calcium_xml'
dicom_path = base_dir + '/Gated_release_final/patient'
nifti_path = base_dir + "/nn-Unetdataset/nnUNet_raw/Dataset002_COCA3D"

# cornary name to id
cornary_name_2_id = {"Right Coronary Artery": 0,
                     "Left Anterior Descending Artery": 1,
                     "Left Coronary Artery": 2,
                     "Left Circumflex Artery":3}

def get_pix_coords(pxy):
    x, y = eval(pxy)
    x = int(x+0.99)
    y = int(y+0.99)
    assert x > 0 and x < 512, f"Invalid {x} value for pixel coordinate"
    assert y > 0 and y < 512, f"Invalid {y} value for pixel coordinate"
    return (x, y)

def process_xml(f):
    # input XML file
    # output - directory containing various meta data
    with open(f, 'rb') as fin:
        pl = plistlib.load(fin)
    # extract needed info from XML
    data = {}
    for image in pl["Images"]:
        iidx = image["ImageIndex"]
        num_rois = image["NumberOfROIs"]
        assert num_rois == len(image["ROIs"]), f"{num_rois} ROIs but not all specified in {f}"
        for roi in image["ROIs"]:
            if (len(roi['Point_px']) > 0):
                if iidx not in data:
                    data[iidx] = []
                data[iidx].append({"cid" : cornary_name_2_id[roi['Name']]})
                assert len(roi['Point_px']) == roi['NumberOfPoints'], f"Number of ROI points does not match with given length for {f}"
                data[iidx][-1]['pixels'] = [get_pix_coords(pxy) for pxy in roi['Point_px']]
            else:
                print (f"Warning: ROI without pixels specified for {iidx} in {f}")

    return data

#data=process_xml(f'C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml/{pid}.xml')






import numpy as np
import nibabel as nib
import pydicom
import os

def dicom_to_nifti_3D(pid, dicom_folder, output_folder):

    # Liste des fichiers DICOM pour le patient spécifié
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # Assurez-vous que les fichiers sont triés pour maintenir l'ordre correct des coupes
    dicom_files.reverse()

    # Vérifier si des fichiers DICOM ont été trouvés
    if not dicom_files:
        print("Aucun fichier DICOM trouvé pour ce patient.")
        return

    # Lire le premier fichier pour obtenir les dimensions de l'image
    ref_dicom = pydicom.read_file(dicom_files[0])
    rows, cols = int(ref_dicom.Rows), int(ref_dicom.Columns)
    slices = len(dicom_files)

    # Créer un array 3D pour stocker toutes les données des images DICOM
    image_array = np.zeros((rows, cols, slices), dtype=ref_dicom.pixel_array.dtype)

    # Remplir l'array 3D avec les données de chaque fichier DICOM
    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.read_file(dicom_file)
        image_array[:, :, i] = ds.pixel_array

    # Créer l'objet NIfTI
    affine_matrix = np.eye(4)  # Vous pouvez ajuster ceci si nécessaire
    nifti_image = nib.Nifti1Image(image_array, affine_matrix)

    # Enregistrer le fichier NIfTI
    nifti_filename = f"PID_{pid:03}_0000.nii.gz"
    nifti_filepath = os.path.join(output_folder, nifti_filename)
    nib.save(nifti_image, nifti_filepath)

    print(f"Nifti files save as : {nifti_filepath}")

# Utilisation de la fonction
# dicom_folder = 'C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/6/Pro_Gated_CS_3.0_I30f_3_70%'
# output_folder = 'C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset002_COCA3D'

#dicom_to_nifti_3D(pid, dicom_folder, output_folder)


def create_mask_image(data, layer_index, image_size=(512, 512)):
    # Créer une image noire de taille donnée
    mask_image = np.zeros(image_size, dtype=np.uint8)

    # Vérifier si l'indice de la couche est dans les données
    if layer_index in data:
        layer_data = data[layer_index]
        for item in layer_data:
            # Récupe les coordonnées des pixels pour chaque contour
            pixels = np.array(item['pixels'], dtype=np.int32)
            cv2.fillPoly(mask_image, [pixels], 1)

    else:
        print(f"Layer index {layer_index} not found in data.")
        return None


    return mask_image


def create_3d_mask_array(data, nslices, image_size=(512, 512)):
 
    # Initialiser l'array 3D avec des zéros (toutes les tranches noires par défaut)
    mask_volume = np.zeros((image_size[0], image_size[1], nslices), dtype=np.uint8)

    # Remplir l'array avec les masques pour chaque couche où des données existent
    for layer_index in range(nslices):
        if layer_index in data:
            mask_image = create_mask_image(data, layer_index, image_size)
            if mask_image is not None:
                mask_volume[:, :, layer_index] = mask_image

    return mask_volume


def save_mask_as_nifti_3D(mask_volume, dest_path, pid):
    # Créer l'objet NIfTI
    nifti_img = nib.Nifti1Image(mask_volume, affine=np.eye(4))

    # Sauvegarder le fichier NIfTI
    nifti_filename = f"PID_{pid:03}.nii.gz"
    file_path = os.path.join(dest_path, nifti_filename)
    
    nib.save(nifti_img, file_path)
    print(f"Nifti mask 3D saved as : {file_path}")



def Initialisation_3D():
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        pid = int(xml_file[:-4])
        xml_file_path = os.path.join(xml_path, xml_file)
        data=process_xml(xml_file_path)

        dicom_patient_firstpath = '%s/%s'      %(dicom_path, pid)
        fichier_bizarre = os.listdir(dicom_patient_firstpath)[0]
        dicom_patient_path = '%s/%s/%s'      %(dicom_path, pid , fichier_bizarre)
        dicom_patient = [f for f in os.listdir(dicom_patient_path) if f.endswith('.dcm')]
        nslice = len(dicom_patient)

        dicom_to_nifti_3D(pid,dicom_patient_path,nifti_path+'/imagesTr')

        mask_volume = create_3d_mask_array(data, nslice, image_size=(512, 512))
        save_mask_as_nifti_3D(mask_volume, nifti_path + '/labelsTr', pid)


        print('\n')
        print(pid)
    return()
        
Initialisation_3D()

#Création du même test set que Mo 
numbers = [ 144, 267, 184, 240, 393, 223, 387, 1, 319, 107, 111, 19, 253, 231, 64, 15, 206, 42, 430, 274, 243, 6, 416, 9, 122,
    396, 263, 86, 66, 449, 276, 4, 148, 34, 151, 62, 309, 140, 439, 248, 245, 71, 117, 157, 287, 286, 194, 373, 425, 249,
    25, 405, 196, 323, 95, 109, 415, 55, 226, 247, 171, 318, 402, 314, 103, 93, 43, 431, 394, 50, 132, 450, 161, 283, 200,
    285, 442, 225, 244, 362, 51, 173, 156, 429, 17, 96, 349 ]

def split_data_by_pid(base_dir, numbers):

    nifti_path = os.path.join(base_dir, "nn-Unetdataset/nnUNet_raw/Dataset002_COCA3D")
    os.makedirs(os.path.join(nifti_path, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(nifti_path, "labelsTs"), exist_ok=True)

    # Liste tous les fichiers dans le dossier imagesTr
    all_images = [f for f in os.listdir(os.path.join(nifti_path, "imagesTr")) if f.endswith('.nii.gz')]

    # Filtrer les images dont le PID est dans la liste numbers
    filtered_images = [img for img in all_images if int(img.split('_')[1]) in numbers]

    # Déplace les images et les labels sélectionnés
    for image_name in filtered_images:
        shutil.move(os.path.join(nifti_path+ "/imagesTr", image_name), os.path.join(nifti_path+ "/imagesTs", image_name))
        print(image_name)
        label_name = image_name[:-12] + ".nii.gz"
        shutil.move(os.path.join(nifti_path+ "/labelsTr", label_name), os.path.join(nifti_path+"/labelsTs", label_name))

    print(f"Moved {len(filtered_images)} images and labels to the test set based on specific PIDs.")


#split_data_by_pid(base_dir,numbers)