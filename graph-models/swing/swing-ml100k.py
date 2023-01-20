import pandas as pd
from itertools import combinations
import json
import os

alpha = 0.5
top_k = 20


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])
    test_data = pd.read_csv(test_path, sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])

    print(train_data.head(5))
    print(test_data.head(5))
    return train_data, test_data


def get_uitems_iusers(train):
    u_items = dict()
    i_users = dict()
    for index, row in train.iterrows():
        u_items.setdefault(row["userid"], set())
        i_users.setdefault(row["movieid"], set())

        u_items[row["userid"]].add(row["movieid"])
        i_users[row["movieid"]].add(row["userid"])
    print("使用的用户个数为：{}".format(len(u_items)))
    print("使用的item个数为：{}".format(len(i_users)))
    return u_items, i_users


def cal_similarity(u_items, i_users):
    item_pairs = list(combinations(i_users.keys(), 2))
    print("item pairs length：{}".format(len(item_pairs))) # 1410360
    item_sim_dict = dict()
    cnt = 0
    for (i, j) in item_pairs:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        user_pairs = list(combinations(i_users[i] & i_users[j], 2))
        result = 0.0
        for (u, v) in user_pairs:
            result += 1 / (alpha + list(u_items[u] & u_items[v]).__len__())

        item_sim_dict.setdefault(i, dict())
        item_sim_dict[i][j] = result
        # print(item_sim_dict[i][j])

    return item_sim_dict


def save_item_sims(item_sim_dict, path):
    new_item_sim_dict = dict()
    for item, sim_items in item_sim_dict.items():
        new_item_sim_dict.setdefault(item, dict())
        new_item_sim_dict[item] = dict(sorted(sim_items.items(), key = lambda k:k[1], reverse=True)[:top_k])
    json.dump(new_item_sim_dict, open(path, "w"))
    print("item 相似 item（{}）保存成功！".format(top_k))
    return new_item_sim_dict


if __name__ == "__main__":
    train_data_path = "./ua.base"
    test_data_path = "./ua.test"
    item_sim_save_path = "../../../dataset/item_sim_dict.json"

    train, test = load_data(train_data_path, test_data_path)
    if not os.path.exists(item_sim_save_path):
        u_items, i_users = get_uitems_iusers(train)
        item_sim_dict = cal_similarity(u_items, i_users)

        new_item_sim_dict = save_item_sims(item_sim_dict, item_sim_save_path)
    else:
        new_item_sim_dict = json.load(open(item_sim_save_path, "r"))