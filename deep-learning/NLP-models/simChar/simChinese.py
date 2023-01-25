# 形近字

# import pygame
# pygame.init()
#
# with open("./charsCN3500.txt", "r") as f:
#     chars = f.read().strip()
#
# print(chars.__len__())  # 3753
# print(pygame.font.get_fonts()[0:10])
#
# # 转化为黑白图片
# for i, char in enumerate(chars):
#     font = pygame.font.SysFont('songti', 100)
#     rtext = font.render(char, True, (0,0,0), (255,255,255))
#     pygame.image.save(rtext, "../../../../dataset/charsCNpng/{}_{}.png".format(char, i))
#

import numpy as np
import cv2
import os
import pickle
from operator import itemgetter

def read_img_2_list(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
    return [_[0] for _ in img.tolist()]


def get_all_char_vectors(img_dir):
    img_paths = [_ for _ in os.listdir(img_dir) if _.endswith("png")]
    img_vector_dict = {}
    for img_path in img_paths:
        img_vector_dict[img_path[0]] = read_img_2_list(img_path=os.path.join(img_dir,img_path))
    return img_vector_dict


# 计算两个向量之间的余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5)*(normB**0.5))


if __name__ == '__main__':
    dict_path = '../../../../dataset/charsCNpng/dict_pkl.pickle'
    if not os.path.isfile(dict_path):
        img_vector_dict = get_all_char_vectors('../../../../dataset/charsCNpng')
        pickle.dump(img_vector_dict, open(dict_path, "wb"))
    else:
        img_vector_dict = pickle.load(open(dict_path), "rb")
    # 获取最接近的汉字
    similarity_dict = {}
    # print(img_vector_dict.keys())
    while True:
        match_char = input("输入汉字:")
        match_vector = img_vector_dict[match_char]
        for char, vector in img_vector_dict.items():
            cosine_similar = cosine_similarity(match_vector, vector)
            similarity_dict[char] = cosine_similar
        # 按相似度排序，取前10个
        sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
        print([(char, round(similarity, 4)) for char, similarity in sorted_similarity[:10]])