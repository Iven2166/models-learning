import random

import numpy as np
import cv2
import os
import csv
import pickle
from operator import itemgetter


def read_img_2_list(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
    return [_[0] for _ in img.tolist()]


def get_random_char_vectors(img_dir):
    img_path = os.listdir(img_dir)[random.randint(1, len(os.listdir(img_dir)) - 1)]
    img_vector_dict = {}
    img_vector_dict[img_path[0]] = read_img_2_list(img_path=os.path.join(img_dir, img_path))
    return img_vector_dict


def get_all_char_vectors(img_dir):
    img_paths = [_ for _ in os.listdir(img_dir) if _.endswith("png")]
    img_vector_dict = {}
    for img_path in img_paths:
        img_vector_dict[img_path[0]] = read_img_2_list(img_path=os.path.join(img_dir, img_path))
    return img_vector_dict


# dict_path = '../../../../dataset/charsCNpng/dict_pkl.pickle'
# img_vector_dict = get_random_char_vectors('../../../../dataset/charsCNpng')
img_vector_dict_all = get_all_char_vectors('../../../../dataset/charsCNpng')
# print(list(img_vector_dict.items())[0:10])
# print(list(img_vector_dict.items())[0][0])
# print(len(list(img_vector_dict.items())[0][1]))
# print(list(img_vector_dict.values())[0])


with open('charCnVector.csv', 'w', newline='') as csvfile:
    fieldnames = ['charCn', 'vector']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k, v in img_vector_dict_all.items():
        writer.writerow({'charCn': k,
                         'vector': ' '.join([str(i) for i in v])
                         })

# with open('demo.txt', 'a') as f:
#     f.write('#'.join([str(i) for i in list(img_vector_dict.values())[0]]))
