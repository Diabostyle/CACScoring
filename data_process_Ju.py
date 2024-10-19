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

# set the random seed to create the same train/val/test split
np.random.seed(10015321)


plt.close("all")

#Path management
base_dir = "C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2"
pid =10





# Function to read and display DICOM images
def display_images(image_folder):
    # Get list of files in the folder
    files = [f for f in os.listdir(image_folder) if f.endswith('.dcm')]
    n_image=len(files)

    # Check if there are any DICOM files in the folder
    if not files:
        print("No DICOM files found in the folder.")
        return

    # Read and display each DICOM image
    for file in files:
        file_path = os.path.join(image_folder, file)
        ds = pydicom.dcmread(file_path)

        plt.figure(figsize=(5, 5))
        plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
        plt.title(file)
        plt.axis('off')
        plt.show()


# Example usage of the display_images function
# Update this path to the folder containing your DICOM files
image_folder = "C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/1/Pro_Gated_CS_3.0_I30f_3_70%/"
#display_images(image_folder)





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

#data=process_xml('C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml/1.xml')
#print(data)



# #Plot only the contour of the xml data
def plot_layer(layer_data):
    for item in layer_data:
        pixels = item['pixels']
        category_id = item['cid']
        x = [p[0] for p in pixels]
        y = [p[1] for p in pixels]
        plt.scatter(x, y, label=f'Category {category_id}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Pixel Coordinates for Layer {layer}')
    plt.legend()
    plt.show()

# for layer, layer_data in data.items():
#      plot_layer(layer_data)


#Plot les deux superposer
def display_imagesandcontour(image_folder, data):
    files = [f for f in os.listdir(image_folder) if f.endswith('.dcm')]
    files.sort()  # Ensure the files are sorted by name


    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 10))  # Configure grid size and padding
    image_index=len(files)
    for ax, file in zip(grid, files):
        image_index-=1
        file_path = os.path.join(image_folder, file)
        ds = pydicom.dcmread(file_path)

        ax.imshow(ds.pixel_array, cmap=plt.cm.gray)  # Display the image in grayscale
        #ax.set_title(file)
        ax.axis('off')


        if image_index in data:
            layer_data = data[image_index]
            for item in layer_data:
                pixels = item['pixels']
                x = [p[0] for p in pixels]
                y = [p[1] for p in pixels]
                ax.scatter(x, y, color='red', s=10)  # Plot points, adjust size as needed

    plt.show()

# Assuming data is already loaded and processed from XML
image_folder = '%s/%s/%s/%s/%s'      %(base_dir , 'Gated_release_final' , 'patient', pid , 'Pro_Gated_CS_3.0_I30f_3_70%')
#display_imagesandcontour(image_folder, data)




def dicom_to_nifti(dicom_file_path, nifti_dest, pid, layer):
    # Check if the specified DICOM file exists
    if not os.path.exists(dicom_file_path):
        print("The specified DICOM file does not exist.")
        return

    # Read the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Get pixel array from DICOM object
    image_data = ds.pixel_array

    # Create a NIfTI image (Assuming the default affine identity matrix)
    nifti_img = nib.Nifti1Image(image_data, np.eye(4))

    # Construct the filename for the NIfTI file based on the DICOM file name
    nifti_filename = f"CT_{pid:03}{layer:02}_0000.nii.gz"
    nifti_full_path = os.path.join(nifti_dest, nifti_filename)

    # Save the NIfTI file
    nib.save(nifti_img, nifti_full_path)
    print(f'NIfTI file saved as {nifti_full_path}')

# Example usage
dicom_file_path = "C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/1/Pro_Gated_CS_3.0_I30f_3_70%/IM-6112-0001.dcm"  # Path to the DICOM file
nifti_dest = "C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset001_COCA/imagesTr"  # Destination folder for the NIfTI file
# dicom_to_nifti(dicom_file_path, nifti_dest)


# def print_dicom_image_size(dicom_file_path):
#     ds = pydicom.dcmread(dicom_file_path)
#     rows = ds.Rows
#     columns = ds.Columns
#     try:
#         pixel_spacing = ds.PixelSpacing
#         print(f"Dicom Image Size: {rows} x {columns} pixels, Pixel Spacing: {pixel_spacing[0]} mm x {pixel_spacing[1]} mm")
#     except AttributeError:
#         print(f"Dicom Image Size: {rows} x {columns} pixels, Pixel Spacing: not specified")

# print_dicom_image_size(dicom_file_path)

# def print_nifti_image_size(nifti_file_path):
#     img = nib.load(nifti_file_path)
#     header = img.header
#     dims = header.get_data_shape()
#     voxel_sizes = header.get_zooms()
#     print(f"Nifti Image Size: {dims}, Voxel Sizes: {voxel_sizes} mm")

# nifti_file_path = "C:/Users/julie/Desktop/ROTTERDAM/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset001_COCA/imagesTr/IM-6112-0001.nii"
# print_nifti_image_size(nifti_file_path)



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

# Exemple d'utilisation
#  layer_index = 40
#  mask_image = create_mask_image(data, layer_index)

#  cv2.imshow('Mask Image', mask_image)
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()


def save_mask_as_nifti(mask, dest_path, pid ,layer):
    # Créer un objet NIfTI avec le masque NumPy
    nifti_image = nib.Nifti1Image(mask, affine=np.eye(4))

    nifti_filename = f"CT_{pid:03}{layer:02}.nii.gz"
    file_path = os.path.join(dest_path, nifti_filename)
    
    nib.save(nifti_image, file_path)
    print(f"Saved NIfTI image to {nifti_filename}")


base_dir = "/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2"
xml_path = base_dir + '/Gated_release_final/calcium_xml'
dicom_path = base_dir + '/Gated_release_final/patient'
nifti_path = base_dir + "/nn-Unetdataset/nnUNet_raw/Dataset004_COCA2Dv3"

#Création de nos fichier avec les images en nifti bien organisée
def Initialisation():
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        pid = int(xml_file[:-4])
        xml_file_path = os.path.join(xml_path, xml_file)
        data=process_xml(xml_file_path)

        dicom_patient_firstpath = '%s/%s'      %(dicom_path, pid)
        fichier_bizarre = os.listdir(dicom_patient_firstpath)[0]
        dicom_patient_path = '%s/%s/%s'      %(dicom_path, pid , fichier_bizarre)
        dicom_patient = [f for f in os.listdir(dicom_patient_path) if f.endswith('.dcm')]
        dicom_patient.sort()
        nslice = len(dicom_patient)

        for layer in data.keys():
            dicom_file_path = '%s/%s'      %(dicom_patient_path  , dicom_patient[nslice-layer-1])
            dicom_to_nifti(dicom_file_path , nifti_path +'/imagesTr', pid, layer)


            mask_array = create_mask_image(data, layer)
            save_mask_as_nifti(mask_array, nifti_path + '/labelsTr', pid , layer)

        print('\n')
        print(pid)
        
#Initialisation()

print(nifti_path +"/imagesTr")





#Création du Test set
def split_data(base_dir, percentage=10):

    # Crée les dossiers s'ils n'existent pas déjà
    os.makedirs(nifti_path +"/imagesTs", exist_ok=True)
    os.makedirs(nifti_path +"/labelsTs", exist_ok=True)

    # Liste tous les fichiers dans le dossier imagesTr
    
    all_images = [f for f in os.listdir(nifti_path +"/imagesTr") if f.endswith('.nii.gz')]

    # Sélectionne aléatoirement un pourcentage des images
    selected_images = np.random.choice(all_images, size=int(len(all_images) * (percentage / 100)), replace=False)

    # Déplace les images et les labels sélectionnés
    for image_name in selected_images:
        shutil.move(os.path.join(nifti_path +"/imagesTr", image_name), os.path.join(nifti_path +"/imagesTs", image_name))
        print(image_name)
        label_name = image_name[:-12]+".nii.gz"
        shutil.move(os.path.join(nifti_path +"/labelsTr", label_name), os.path.join(nifti_path +"/labelsTs", label_name))

    print(f"Moved {len(selected_images)} images and labels to the test set.")


#split_data(base_dir)


def split_data_by_pid(nifti_path, numbers):

    # Convertit les numéros en chaînes de caractères formatées sur trois chiffres
    formatted_numbers = [f"{num:03}" for num in numbers]

    # Crée les dossiers s'ils n'existent pas déjà
    os.makedirs(nifti_path +"/imagesTs", exist_ok=True)
    os.makedirs(nifti_path +"/labelsTs", exist_ok=True)

    # Liste tous les fichiers dans le dossier imagesTr
    all_images = [f for f in os.listdir(nifti_path +"/imagesTr") if f.endswith('.nii.gz')]

    # Filtrer les images dont le PID figure dans la liste `formatted_numbers`
    selected_images = [img for img in all_images if img.split('_')[1][:3] in formatted_numbers]

    # Déplace les images et les labels sélectionnés
    for image_name in selected_images:
        shutil.move(os.path.join(nifti_path + "/imagesTr", image_name), os.path.join(nifti_path + "/imagesTs", image_name))
        print(f"Moved {image_name} to test set.")

        # Génération du nom du fichier de label associé
        label_name = image_name[:-12]+".nii.gz"
        shutil.move(os.path.join(nifti_path + "/labelsTr", label_name), os.path.join(nifti_path + "/labelsTs", label_name))
        print(f"Moved {label_name} to test set.")

    print(f"Moved {len(selected_images)} images and labels to the test set.")

# Liste des PIDs pour le test set, par exemple
number_list = [144, 267, 184, 240, 393, 223, 387, 1, 319, 107, 111, 19, 253, 231, 64, 15, 206, 42, 430, 274, 243, 6, 416, 9, 122, 396, 263, 86, 66, 449, 276, 4, 148, 34, 151, 62, 309, 140, 439, 248, 245, 71, 117, 157, 287, 286, 194, 373, 425, 249, 25, 405, 196, 323, 95, 109, 415, 55, 226, 247, 171, 318, 402, 314, 103, 93, 43, 431, 394, 50, 132, 450, 161, 283, 200, 285, 442, 225, 244, 362, 51, 173, 156, 429, 17, 96, 349]

split_data_by_pid(nifti_path , number_list)


