'''
Making dataset for DR Grading from yolo result and ZhSeg1k dataset
'''
import urllib.request
import argparse
import pandas as pd
import os
import numpy as np
import cv2 as cv
import imageio
import csv

# Read Arguments
parser = argparse.ArgumentParser()
''' Input CSV file'''
parser.add_argument('--in_csv_path', type=str, default='./isbi_tr_vl.csv')
''' Input Label directory ( Yolo result label directory )'''
parser.add_argument('--in_label_dir', type=str, default='./labels_isbi')

arg = parser.parse_args()
''' Process all the input arguments '''

in_csv_path = arg.in_csv_path
if not os.path.exists(in_csv_path):
    raise EOFError('Input CSV file (' + str(in_csv_path) + ') not found')
in_label_dir = arg.in_label_dir

main_db = pd.read_csv(in_csv_path, keep_default_na=False)
# ''' Read the CSV file '''
# image_urls = main_db['image_url']

''' Opening the csv file in 'w' mode '''
file = open('arash_yolo_isbi.csv', 'w', newline='')
# with file:
# identifying header
file.header = ['img_name', 'dr', 'level', 'bl_num', 'bl_size', 'he_num', 'he_size', 'laser_num', 'laser_size']
file.writer = csv.DictWriter(file, fieldnames=file.header)
file.writer.writeheader()

for i in range(len(main_db['image_path'])):
    name = main_db['image_id'][i]
    l_name = name + '.txt'
    path = os.path.join(in_label_dir, l_name)

    bl_co = 0
    he_co = 0
    bl_size = 0
    he_size = 0
    laser_co=0
    laser_size = 0
    if os.path.isfile(path):

        label_file = open(path, 'r')
        for line in label_file:
            sline = line.split(sep=' ')
            label = int(sline[0])
            w = float(sline[3])
            h = float(sline[4])
            bl_co = bl_co + 1 if label == 0 else bl_co
            bl_size = bl_size + (w * h) if label == 0 else bl_size
            he_co = he_co + 1 if label == 3 else he_co
            he_size = he_size + (w * h) if label == 3 else he_size
            laser_co = laser_co + 1 if label == 5 else laser_co
            laser_size = laser_size + (w * h) if label == 5 else laser_size

        print(bl_co)
        print(bl_size * 50)
        quality = int(main_db['Overall quality'][i])
        left = main_db['left_eye_DR_Level'][i]
        right = main_db['right_eye_DR_Level'][i]
        '''check eye side (left or right)'''
        level = 0
        if left in (None, ''):
            level = str(right)
        else:
            level = str(left)

        dr = 'FALSE' if int(level) <= 1 else 'TRUE'

        if quality == 1:
            file.writer.writerow(
                {'img_name': name, 'dr': dr, 'level': level, 'bl_num': bl_co, 'bl_size': bl_size, 'he_num': he_co,
                 'he_size': he_size, 'laser_num': laser_co, 'laser_size': laser_size})

    # print(path)
