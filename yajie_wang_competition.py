# encoding=utf8
import csv
import json
import math
import sys
import time


import numpy as np
import pandas as pd
import pyspark
from pyspark.mllib.recommendation import ALS,Rating
from sklearn.linear_model import LinearRegression
from scipy.stats import mode

import csv
from sklearn import linear_model



def read_data(train_data):
    lines = sc.textFile(train_data)
    lines = lines.mapPartitions(lambda _: csv.reader(_))
    header = lines.first()
    lines = lines.filter(lambda _: _ != header)
    return lines

def num_stan(star_pred):
    if star_pred<1:
        return 1
    elif star_pred <=5:
        return float(star_pred)
    elif star_pred >5:
        return 5.0





def get_dict(train_lines, test_lines):
    # construct user name dict
    user_name_lines = train_lines.union(test_lines).map(
        lambda _: _[0]).distinct(10)
    user_name_id_lines = user_name_lines.zipWithUniqueId()
    user_name_id_lines = user_name_id_lines.collect()
    user_name_id_dict = dict(user_name_id_lines)
    id_user_name_dict = dict([(v, k) for k, v in user_name_id_lines])

    # construct business name dict
    business_name_lines = train_lines.union(test_lines).map(
        lambda _: _[1]).distinct(10)
    business_name_id_lines = business_name_lines.zipWithUniqueId()
    business_name_id_lines = business_name_id_lines.collect()
    business_name_id_dict = dict(business_name_id_lines)
    id_business_name_dict = dict([(v, k) for k, v in business_name_id_lines])

    # broadcast dict
    user_name_id_dict = sc.broadcast(user_name_id_dict)
    business_name_id_dict = sc.broadcast(business_name_id_dict)
    id_business_name_dict = sc.broadcast(id_business_name_dict)
    id_user_name_dict = sc.broadcast(id_user_name_dict)
    return user_name_id_dict, business_name_id_dict, id_business_name_dict, id_user_name_dict


def save_res(test_lines, src_pred_lines, id_user_name_dict,
    id_business_name_dict, output_path):
    src_pred_lines = src_pred_lines.persist(
        pyspark.StorageLevel.MEMORY_AND_DISK)

    out_file = open(output_path, "w")
    # id to name
    pred_lines = src_pred_lines.map(lambda _: (
        id_user_name_dict.value[_[0]], id_business_name_dict.value[_[1]], _[2]))
    prediction_list = sorted(pred_lines.collect())
    return prediction_list




def model_based_cf(train_data, test_data, output_path):
    train_lines = read_data(train_data)
    train_lines = train_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    test_lines = read_data(test_data)
    test_lines = test_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    user_name_id_dict, business_name_id_dict, id_business_name_dict, id_user_name_dict = get_dict(
        train_lines, test_lines)
    # name to id
    train_lines = train_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]], _[2]))
    test_lines = test_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]], _[2]))

    model = ALS.train(train_lines, rank=5, iterations=20, lambda_=0.1)
    src_pred_lines = model.predictAll(
        test_lines.map(lambda _: (_[0], _[1]))).map(
        lambda _: (_[0], _[1], _[2]))
    a=save_res(test_lines, src_pred_lines, id_user_name_dict,
             id_business_name_dict, output_path)
    return a


def predict_by_business(cur_user, cur_business, similar_business,
    bid_uidstarlist_dict, uidbid_star_dict):
    if cur_business not in bid_uidstarlist_dict:
        return 0.0
    cur_business_data = bid_uidstarlist_dict[cur_business]
    a_temp = [x[1] for x in cur_business_data]
    a_mean = sum(a_temp) / len(a_temp)

    if len(similar_business) == 0:
        return a_mean

    num = 0
    den = 0
    for similar_business in similar_business:
        other_business = similar_business[1]

        key = (cur_user, other_business)
        if key in uidbid_star_dict:
            o_rating = uidbid_star_dict[key]

            other_data = bid_uidstarlist_dict[other_business]
            o_temp = [x[1] for x in other_data if
                      other_data[0] != cur_user]

            o_mean = sum(o_temp) / len(o_temp)
            den = den + abs(similar_business[0])
            num = num + similar_business[0] * (o_rating - o_mean)

    if den == 0:
        return a_mean
    else:
        return a_mean + num / den


def predict_by_user(cur_user, cur_business, similar_users, uid_bidstarlist_dict,
    uidbid_star_dict):
    if cur_user not in uid_bidstarlist_dict:
        return 0
    cur_user_data = uid_bidstarlist_dict[cur_user]
    a_temp = [x[1] for x in cur_user_data]
    a_mean = sum(a_temp) / len(a_temp)

    if len(similar_users) == 0:
        return a_mean

    num = 0
    den = 0
    for similar_user in similar_users:
        other_user = similar_user[1]

        key = (other_user, cur_business)
        if key in uidbid_star_dict:
            # check condition
            o_rating = uidbid_star_dict[key]

            other_data = uid_bidstarlist_dict[other_user]
            o_temp = [x[1] for x in other_data if
                      other_data[0] != cur_business]

            o_mean = sum(o_temp) / len(o_temp)
            den = den + abs(similar_user[0])
            num = num + similar_user[0] * (o_rating - o_mean)

    if den == 0:
        return a_mean
    else:
        return a_mean + num / den


def user_correlation(cur_user_data, other_user_data):
    cur_user_data = set([_[0] for _ in cur_user_data])
    other_user_data = set([_[0] for _ in other_user_data])
    return 1.0 * len(cur_user_data & other_user_data) / len(
        cur_user_data | other_user_data)


def get_similar_users(cur_user, cur_business, bid_uidlist_dict,
    uid_bidstar_dict):
    similar_users = list()

    if cur_business not in bid_uidlist_dict or cur_user not in uid_bidstar_dict:
        # item cold start
        similar_users.append((0, cur_user))
        return similar_users

    cur_user_data = uid_bidstar_dict[cur_user]
    other_users = bid_uidlist_dict[cur_business]

    for uid in other_users:
        if cur_user != uid:
            other_user_data = uid_bidstar_dict[uid]
            similarity = user_correlation(cur_user_data, other_user_data)
            if similarity > 0:
                similar_users.append((similarity, uid))

    return similar_users


def business_correlation(cur_business_data, other_business_data):
    cur_business_data = set([_[0] for _ in cur_business_data])
    other_business_data = set([_[0] for _ in other_business_data])
    return 1.0 * len(cur_business_data & other_business_data) / len(
        cur_business_data | other_business_data)


def get_similar_business(cur_user, cur_business, uid_bidlist_dict,
    bid_uidstarlist_dict):
    similar_business = list()

    if cur_user not in uid_bidlist_dict or cur_business not in bid_uidstarlist_dict:
        similar_business.append((0, cur_business))
        return similar_business

    cur_business_data = bid_uidstarlist_dict[cur_business]
    other_business = uid_bidlist_dict[cur_user]

    for bid in other_business:
        other_business_data = bid_uidstarlist_dict[bid]
        similarity = business_correlation(cur_business_data,
                                          other_business_data)
        if similarity > 0:
            similar_business.append((similarity, bid))
    return similar_business


def get_similar_business_with_lsh(cur_business, bid_uidstarlist_dict,
    similar_pairs1, similar_pairs2):
    similar_business = list()

    topSimilarMovies = list()
    if (cur_business not in similar_pairs1) and (
        cur_business not in similar_pairs2):
        # assign average rating of user
        topSimilarMovies.append((1, cur_business))
        return topSimilarMovies

    if cur_business not in bid_uidstarlist_dict:
        # item never rated by any user (item cold start)
        topSimilarMovies.append((1, cur_business))
        return topSimilarMovies

    other_business = list()
    if cur_business in similar_pairs1:
        other_business.extend(similar_pairs1[cur_business])
    if cur_business in similar_pairs2:
        other_business.extend(similar_pairs2[cur_business])

    cur_business_data = bid_uidstarlist_dict[cur_business]
    other_business = list(set(other_business))

    for bid in other_business:
        other_business_data = bid_uidstarlist_dict[bid]
        similarity = business_correlation(cur_business_data,
                                          other_business_data)
        if similarity > 0.001:
            similar_business.append((similarity, bid))
    return similar_business


def user_based_cf(train_data, test_data, output_path):
    train_lines = read_data(train_data)
    train_lines = train_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    test_lines = read_data(test_data)
    test_lines = test_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    user_name_id_dict, business_name_id_dict, id_business_name_dict, id_user_name_dict = get_dict(
        train_lines, test_lines)

    # name to id
    train_lines = train_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]], _[2]))

    src_test_lines = test_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]],
        _[2])).cache()

    # (user_id, business_id)
    test_lines = src_test_lines.map(lambda x: (x[0], x[1]))

    # ((user_id, business_id), star)
    uidbid_star_dict = train_lines.map(
        lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    uidbid_star_dict = sc.broadcast(uidbid_star_dict)

    # (user_id, [(business_id, star), (business_id, star)])
    uid_bidstar_dict = train_lines.map(
        lambda _: (_[0], (_[1], _[2]))).groupByKey(10).mapValues(
        list).collectAsMap()
    uid_bidstar_dict = sc.broadcast(uid_bidstar_dict)

    # (business_id, [user_id, user_id])
    bid_uidlist_dict = train_lines.map(
        lambda x: (x[1], x[0])).groupByKey(10).mapValues(
        list).collectAsMap()
    bid_uidlist_dict = sc.broadcast(bid_uidlist_dict)

    similar_user_lines = test_lines.map(lambda x: (x[0], x[1],
                                                   get_similar_users(x[0], x[1],
                                                                     bid_uidlist_dict.value,
                                                                     uid_bidstar_dict.value)))
    predictions = similar_user_lines.map(lambda x: (x[0], x[1],
                                                    predict_by_user(x[0], x[1],
                                                                    x[2],
                                                                    uid_bidstar_dict.value,
                                                                    uidbid_star_dict.value)))

    a=save_res(src_test_lines, predictions, id_user_name_dict,
             id_business_name_dict, output_path)
    return a


def item_based_cf(train_data, test_data, output_path):
    train_lines = read_data(train_data)
    train_lines = train_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    test_lines = read_data(test_data)
    test_lines = test_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    user_name_id_dict, business_name_id_dict, id_business_name_dict, id_user_name_dict = get_dict(
        train_lines, test_lines)

    # name to id
    train_lines = train_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]], _[2]))

    src_test_lines = test_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]],
        _[2])).cache()

    # train_lines = train_lines.map( lambda _: ("u_%s" % _[0], "b_%s" % _[1], _[2]))
    # src_test_lines = src_test_lines.map( lambda _: ("u_%s" % _[0], "b_%s" % _[1],_[2]))

    # (user_id, business_id)
    test_lines = src_test_lines.map(lambda x: (x[0], x[1]))

    # ((user_id, business_id), star)
    uidbid_star_dict = train_lines.map(
        lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    uidbid_star_dict = sc.broadcast(uidbid_star_dict)

    # (business_id, [(user_id, star), (user_id, star)])
    bid_uidstarlist_dict = train_lines.map(
        lambda _: (_[1], (_[0], _[2]))).groupByKey(10).mapValues(
        list).collectAsMap()
    bid_uidstarlist_dict = sc.broadcast(bid_uidstarlist_dict)

    # (user_id, [business_id, business_id])
    uid_bidlist_dict = train_lines.map(
        lambda x: (x[0], x[1])).groupByKey(10).mapValues(
        list).collectAsMap()
    uid_bidlist_dict = sc.broadcast(uid_bidlist_dict)

    similar_business_lines = test_lines.map(lambda x: (x[0], x[1],
                                                       get_similar_business(
                                                           x[0], x[1],
                                                           uid_bidlist_dict.value,
                                                           bid_uidstarlist_dict.value)))
    predictions = similar_business_lines.map(lambda x: (x[0], x[1],
                                                        predict_by_business(
                                                            x[0], x[1], x[2],
                                                            bid_uidstarlist_dict.value,
                                                            uidbid_star_dict.value)))

    a=save_res(src_test_lines, predictions, id_user_name_dict,
             id_business_name_dict, output_path)
    return a


hashes = [[913, 901, 24593], [14, 23, 769], [11, 101, 193], [11, 91, 1543],
          [387, 552, 98317], [1, 37, 3079], [2, 63, 97], [41, 67, 6151],
          [91, 29, 12289], [3, 79, 53], [73, 803, 49157], [7, 119, 389],
          [13, 83, 197], [29, 521, 4937], [797, 809, 49157],
          [3203, 3209, 98317],[797,809,49157],[1543,3209,98317],[271,449,3079],[587, 857,12289],
          [913, 901, 24593], [13, 73, 98317], [63, 521, 193], [67, 91, 1543],
          [3, 521, 98317], [37, 7, 3079],[913, 901, 98317], [67, 23, 769], [3, 101, 193], [11, 91, 1543],
          [387, 552, 98317], [3, 37, 3079], [11, 63, 97], [91, 67, 98317],
          [91, 29, 12289], [3, 79, 53], [73, 803, 49157], [7, 119, 389],
          [13, 83, 98317],[73, 803, 49157]]


def f(x, has, users_number):
    a = has[0]
    b = has[1]
    p = has[2]
    return min([((a * e + b) % p) % users_number for e in x[1]])


def sig(x, b, r):
    # for e in x:
    res = []
    for i in range(b):
        res.append(((i, tuple(x[1][i * r:(i + 1) * r])), [x[0]]))
    return res


def pairs(x):
    res = []
    length = len(x[1])
    whole = list(x[1])
    whole.sort()
    for i in range(length):
        for j in range(i + 1, length):
            res.append(((whole[i], whole[j]), 1))
    return res


def jaccard(x, business_name_id_dict, bid_uidlist):
    a = set(bid_uidlist[business_name_id_dict[x[0]]][1])
    b = set(bid_uidlist[business_name_id_dict[x[1]]][1])
    inter = a & b
    union = a | b
    jacc = len(inter) / len(union)
    return (x[0], x[1], jacc)


def getSimilarPairs(lines):
    buss = lines.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x).map(
        lambda x: x[0]).collect()
    business_name_id_dict = {}
    for i, e in enumerate(buss):
        business_name_id_dict[e] = i
    business_name_id_dict = sc.broadcast(business_name_id_dict)

    user_lines = lines.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x).map(
        lambda x: x[0])
    users = list(user_lines.collect())
    user_name_id_dict = {}
    for i, e in enumerate(users):
        user_name_id_dict[e] = i
    user_name_id_dict = sc.broadcast(user_name_id_dict)

    # (business_id, [user_id, user_id])
    bid_uidlist_lines = lines.map(
        lambda x: (x[1], [user_name_id_dict.value[x[0]]])).reduceByKey(
        lambda x, y: x + y).cache()
    # (business_id, [user_id, user_id,... ])
    bid_uidlist = bid_uidlist_lines.collect()
    bid_uidlist = sc.broadcast(bid_uidlist)
    # print(matrix)
    users_number = len(users)
    # (business_id, (hash_val, hash_val, ...))
    signatures = bid_uidlist_lines.map(
        lambda x: (x[0], [f(x, has, users_number) for has in hashes]))

    hash_fun_num = len(hashes)  # the size of the signature column
    b = 20
    r = int(hash_fun_num / b)

    cand_lines = signatures.flatMap(lambda _: sig(_, b, r)).reduceByKey(
        lambda x, y: x + y).filter(
        lambda x: len(x[1]) > 1).flatMap(pairs).reduceByKey(lambda x, y: x).map(
        lambda x: x[0])

    result_lines = cand_lines.map(
        lambda _: jaccard(_, business_name_id_dict.value,
                          bid_uidlist.value)).filter(lambda x: x[2] >= 0.5)
    result_lines = result_lines.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    similar_pairs1 = result_lines.map(lambda _: (_[0], _[1])).collect()
    similar_pairs2 = result_lines.map(lambda _: (_[1], _[0])).collect()
    return similar_pairs1, similar_pairs2


def item_based_lsh_cf(train_data, test_data, output_path):
    train_lines = sc.textFile(train_data)
    similar_pairs1, similar_pairs2 = getSimilarPairs(train_lines)

    train_lines = read_data(train_data)
    train_lines = train_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    test_lines = read_data(test_data)
    test_lines = test_lines.map(lambda _: (_[0], _[1], float(_[2]))).cache()

    user_name_id_dict, business_name_id_dict, id_business_name_dict, id_user_name_dict = get_dict(
        train_lines, test_lines)

    # name to id
    train_lines = train_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]], _[2]))

    src_test_lines = test_lines.map(lambda _: (
        user_name_id_dict.value[_[0]], business_name_id_dict.value[_[1]],
        _[2])).cache()

    # (user_id, business_id)
    test_lines = src_test_lines.map(lambda x: (x[0], x[1]))

    # ((user_id, business_id), star)
    uidbid_star_dict = train_lines.map(
        lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    uidbid_star_dict = sc.broadcast(uidbid_star_dict)

    # (business_id, [(user_id, star), (user_id, star)])
    bid_uidstarlist_dict = train_lines.map(
        lambda _: (_[1], (_[0], _[2]))).groupByKey(10).mapValues(
        list).collectAsMap()
    bid_uidstarlist_dict = sc.broadcast(bid_uidstarlist_dict)

    similar_business_lines = test_lines.map(lambda x: (x[0], x[1],
                                                       get_similar_business_with_lsh(
                                                           x[1],
                                                           bid_uidstarlist_dict.value,
                                                           similar_pairs1,
                                                           similar_pairs2)))
    predictions = similar_business_lines.map(lambda x: (x[0], x[1],
                                                        predict_by_business(
                                                            x[0], x[1], x[2],
                                                            bid_uidstarlist_dict.value,
                                                            uidbid_star_dict.value)))

    a=save_res(src_test_lines, predictions, id_user_name_dict, id_business_name_dict, output_path)
    return a


if __name__ == "__main__":
    train_data = sys.argv[1]+'yelp_train.csv'
    test_data = sys.argv[2]
    #train_data = '/Users/yajiewang/Downloads/553/553hw3/数据/yelp_train.csv'
    #test_data = '/Users/yajiewang/Downloads/553/553hw3/数据/yelp_val.csv'
    case_id ='1'
    output_path = sys.argv[3]
    global sc
    sc = pyspark.SparkContext('local[*]', 'task2')
    #print(sys.argv)
    train_lines1 = read_data(train_data)
    test_lines1 = read_data(test_data)
    start_time = time.time()

    #if case_id == '1':
        #pred_1=model_based_cf(train_data, test_data, output_path)
    if case_id == '1':

       pred_1=item_based_lsh_cf(train_data, test_data, output_path)
       #print('*************1')

    if case_id == '1':
        pred_2=user_based_cf(train_data, test_data, output_path)
        u_avg = train_lines1.map(lambda s: (s[0], [float(s[2])])).reduceByKey(lambda x, y: x + y) \
            .map(lambda s: (s[0], sum(s[1]) / len(s[1]))).collectAsMap()
        #print('*************2')
    if case_id == '1':
        pred_3=item_based_cf(train_data, test_data, output_path)
        b_avg = train_lines1.map(lambda s: (s[1], [float(s[2])])).reduceByKey(lambda x, y: x + y) \
            .map(lambda s: (s[0], sum(s[1]) / len(s[1]))).collectAsMap()
        #print('*************3')






        user_file = sys.argv[1]+'user.json'
        user_file_data = sc.textFile(user_file).map(lambda s: (
        json.loads(s)['user_id'], json.loads(s)['review_count'], json.loads(s)['useful'],
        json.loads(s)['average_stars'])).collect()
        business_file = sys.argv[1]+'business.json'
        business_file_data = sc.textFile(business_file).map(
            lambda s: (json.loads(s)['business_id'], json.loads(s)['stars'], json.loads(s)['review_count'])).collect()
        user_info_dic = {}
        for item in user_file_data:
            info = []
            info = info + [item[1]] + [item[2]] + [item[3]]
            user_info_dic[item[0]] = info

        # print(user_info_dic)
        busi_info_dic = {}
        for item in business_file_data:
            info = []
            info = info + [item[1]] + [item[2]]
            busi_info_dic[item[0]] = info
        input_file = train_data#路径
        test_file = test_data#路径
        train_rdd = sc.textFile(input_file)
        header = 'user_id, business_id, stars'
        l_train_data = train_rdd.filter(lambda s: s != header).map(lambda s: s.split(',')).collect()
        l_test_data = sc.textFile(test_file).filter(lambda s: s != header).map(lambda s: s.split(',')).collect()
        new_train_data = []
        for a_data in l_train_data:
            one_data = []
            one_data = one_data + user_info_dic[a_data[0]] + busi_info_dic[a_data[1]] + [float(a_data[2])]
            new_train_data.append(one_data)
        new_train_data = pd.DataFrame(new_train_data)
        new_test_data = []
        for a_data in l_test_data:
            one_data = []
            one_data = one_data + user_info_dic[a_data[0]] + busi_info_dic[a_data[1]] + [float(a_data[2])]
            new_test_data.append(one_data)
        new_test_data = pd.DataFrame(new_test_data)
        new_train_only_data = new_train_data.iloc[:, 0:5]
        new_train_label = new_train_data.iloc[:, 5]
        new_test_only_data = new_test_data.iloc[:, 0:5]
        new_test_label = new_test_data.iloc[:, 5]
        # print('#####################')
        # print(new_test_label)
        # print('#####################')
        # print(new_test_only_data)
        from sklearn.linear_model import LinearRegression

        linreg = LinearRegression()

        linreg.fit(new_train_only_data, new_train_label)
        y_pre = linreg.predict(new_test_only_data)
        result = []
        for i in range(len(l_test_data)):
            one_data = [l_test_data[i][0]] + [l_test_data[i][1]] + [float(y_pre[i])]
            result.append(one_data)
        #print('using linear')


        #pred_3 = pred_3


    result_1 = sc.parallelize(pred_1)
    result_2 = sc.parallelize(pred_2)
    result_3 = sc.parallelize(pred_3)
    result_4 = sc.parallelize(result)


    #print(result_1)

    y = test_lines1.map(lambda s: ((s[0], s[1]), s[2])).sortBy(lambda s: (s[0][0], s[0][1])).map(lambda s: s[1]).collect()
    X = result_1.union(result_2).union(result_3).union(result_4).map(lambda s: ((s[0], s[1]), [s[2]])) \
        .reduceByKey(lambda x, y: x + y).sortBy(lambda s: (s[0][0], s[0][1])).map(lambda s: s[1]+[u_avg[s[0][0]]]).collect()


    X = np.array(X)
    #X = np.column_stack((X, pred_4))
    y = np.array(y)


    reg = LinearRegression().fit(X, y)
    #print(reg)
    result = reg.predict(X)

    index = test_lines1 .map(lambda s: (s[0], s[1])).sortBy(lambda s: (s[0], s[1])).collect()
    #print(result)


    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['user_id', 'business_id', 'prediction'])
        for i in range(len(y)):
            writer.writerow((index[i][0], index[i][1], result[i]))





    end_time = time.time()
    #print("Elasped time %.2f seconds." % (end_time - start_time))
