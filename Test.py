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

base_dir = "/trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2"
xml_path = base_dir + '/Gated_release_final/calcium_xml'
dicom_path = base_dir + '/Gated_release_final/patient'
nifti_path = base_dir + "/nn-Unetdataset/nnUNet_raw/Dataset001_COCA"
pid = 1

dicom_patient_firstpath = '%s/%s'      %(dicom_path, pid)
fichier_bizarre = os.listdir(dicom_patient_firstpath)[0]
dicom_patient_path = '%s/%s/%s'      %(dicom_path, pid , fichier_bizarre)
dicom_patient = [f for f in os.listdir(dicom_patient_path) if f.endswith('.dcm')]
dicom_patient.sort()
nslice = len(dicom_patient)
print(dicom_patient)
print("test")

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

data=process_xml(xml_path+'/1.xml')
#print(data)



for layer in data.keys():
    print(nslice-layer)

print('\n')

for layer in data.keys():
    print(layer)
