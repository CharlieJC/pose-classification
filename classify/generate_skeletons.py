import sys
sys.path.append('/workspace/yolov7')

import numpy as np
from utils.classification import plot_pose_skeleton,normalize_kpts
import csv
import ast
import os
import cv2


categories = {}
categories['kneeling_keypoints.csv'] = 'kneeling'
categories['lying_keypoints.csv'] = 'lying'
categories['standing_keypoints.csv'] = 'standing'

#skel_img = np.zeros((256,256,3), dtype = np.uint8);
#normalize_kpts(temp_kpts, 3)
#plot_pose_skeleton(skel_img, temp_kpts , 3)
#skeleton_images.append(skel_img)
for file_name, category in categories.items():
    count = 1
    with open('kpdata/' + file_name, 'r') as file:
        if not os.path.exists(category):
            os.makedirs(category)
        # initialize csv reader
        csv_reader = csv.reader(file)
        for line in csv_reader:
            line_data = []

                # for each cell in the line
            for cell in line:
                    # parse the tuple and add it to the line's data
                    # ast.literal_eval safely parses the string as a Python literal
                pair = ast.literal_eval(cell)
                line_data.append(pair[0])
                line_data.append(pair[1])
                line_data.append(1)
            if max(line_data) == min(line_data):
                continue
                    # print out this line's data
            skel_img = np.zeros((256,256,3), dtype = np.uint8);
            normalize_kpts(line_data, 3)
            plot_pose_skeleton(skel_img, line_data , 3)
            cv2.imwrite(category + "/" + str(count) + '.jpg', skel_img)
            count += 1
        