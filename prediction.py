import pandas as pd
import numpy as np
import os
import time
import xgboost
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from preprocessing import Nielsen

# 데이터 있는곳 위치
load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))),'nielsen_pickle_data(180110)')
final_file_name = 'final_data_171215'

# seed
split_seed = 0
state_seed = 0

# devide_points
devide_points_list = [5.986000e+03, 1.439980e+04, 4.039190e+04]

def split_dataset(datatable, split_point_list):
    return np.split(datatable.sample(frac = 1, random_state = split_seed), split_point_list)

def print_duration(start):
    print('{}s 걸림'.format(time.time() - start))

def make_result_table(datatable, model):
    sub_table = datatable.copy()[['일자','프로그램명','프로그램편성시작시간','목표']]
    sub_table['예측']=model.predict(datatable.values[:,4:])
    return sub_table

def make_no_channel_data(data):
    no_channel_col_list = list()
    for col in data.columns:
        if '채널' not in col:
            no_channel_col_list.append(col)
    data_no_channel = data[no_channel_col_list]
    li = list()
    for col in data_no_channel.columns:
        if '2_' not in col:
            if '가중치' not in col:
                li.append(col)
    data_no_channel = data_no_channel[li]
    return data_no_channel

def result_table_error_rate(data, rate = 0.2):
    data['차이정도'] = data.apply(lambda x: np.abs((x['목표'] - x['예측']) / x['목표']), axis=1)
    return len(data[data['차이정도'] < rate]) / len(data)

def change_list_to_tuple(li):
    if li[0]!=0:
        li.insert(0, 0)
    li.append(100000000000000)
    tmp = list()
    for i in range(len(li)-1):
        tmp.append((li[i], li[i+1]))
    return tmp

def change2Class(x, tup):
    result = int()
    for idx, (a,b) in enumerate(tup):
        if a < x <= b:
            result = idx
    return result

def train_regression_model(string, model, dictionary):
    print('{}(regression)'.format(string))
    model.fit(dictionary['train'][0], dictionary['train'][1])
    for key in dictionary.keys():
        print("{}      :  {:,}".format(key, mean_squared_error(dictionary[key][1], model.predict(dictionary[key][0]))))
    return model


def train_classification_model(string, model, dictionary):
    print('{}(classification)'.format(string))
    model.fit(dictionary['train'][0], dictionary['train'][1])
    for key in dictionary.keys():
        print("{}      :  {:,}".format(key, accuracy_score(dictionary[key][1], model.predict(dictionary[key][0]))))

    return model

def regression(data, cut_rate=0.2):
    no_channel_data, d3_train, d3_validate, d3_test, d3_dict = data_preprocessing(data, bool_classification = False)

    ## RandomForest
    # regr = RandomForestRegressor(n_estimators=20, random_state=state_seed)
    # regr = train_regression_model('RandomForest', regr, dict)

    # xgboost
    xgb = xgboost.XGBRegressor(max_depth=10, random_state = state_seed)
    xgb_result = train_regression_model('XgBoost', xgb, d3_dict)

    # result table
    result_table = make_result_table(d3_test, xgb_result)

    print('\n목표값과 +- rate가 {}인 경우, Accuracy: {}'.format(cut_rate, result_table_error_rate(result_table, cut_rate)))
    return xgb_result, result_table


def classification(data):
    # 그냥 지상파인지 여부만 체크

    no_channel_data, d3_train, d3_validate, d3_test, d3_dict = data_preprocessing(data, bool_classification = True)

    # classification 인 경우,
    ## RandomForest
    # randomForest = RandomForestClassifier(n_estimators=10, random_state = state_seed)
    # randomForest_result = train_classification_model('RandomForest', randomForest, data_dictionary)
    # randomForest_result.predict_proba(data_dictionary['test'][0])

    ## AdaBoost
    # adaboost = AdaBoostClassifier(n_estimators=20, random_state = state_seed)
    # adaboost_result = train_classification_model('Adaboost', adaboost, data_dictionary)
    # adaboost_result.predict_proba(data_dictionary['test'][0])

    ## XgBoost
    xbg = xgboost.XGBClassifier(max_depth=10, random_state = state_seed)
    xbg_result = train_classification_model('XgBoost', xbg, d3_dict)
    # xbg_result.predict_proba(data_dictionary['test'][0])

    ##Voting 하는 경우,
    # voting = VotingClassifier(estimators=[
    #     ('random', randomForest)
    #     , ('ada', adaboost)
    #     , (' xgboost', xbg)
    # ]
    #     , voting='soft')
    # voting_result = train_classification_model('Voting', voting, data_dictionary)
    # voting_result.predict_proba(data_dictionary['test'][0])


    return xbg_result

def data_preprocessing(datatable, bool_classification=False):
    data = datatable.copy()
    data['지상파'] = data.apply(lambda x: 1 if x['채널_1'] + x['채널_2'] + x['채널_3'] + x['채널_4'] + x['채널_5'] == 1 else 0,
                             axis=1)
    # target 을 class 데이터로 바꿔줌
    if bool_classification:
        li = change_list_to_tuple(devide_points_list)
        data['목표'] = data['목표'].apply(lambda x: change2Class(x, li))

    no_channel_data = make_no_channel_data(data)

    # Data
    ## - train, validate, test 세개로 나눈 경우
    d3_train, d3_validate, d3_test = split_dataset(no_channel_data,
                                                   [int(.6 * len(no_channel_data)), int(.8 * len(no_channel_data))])

    d3_train_x, d3_train_y = d3_train.iloc[:, 4:].values, d3_train.iloc[:, 3].values
    d3_validate_x, d3_validate_y = d3_validate.iloc[:, 4:].values, d3_validate.iloc[:, 3].values
    d3_test_x, d3_test_y = d3_test.iloc[:, 4:].values, d3_test.iloc[:, 3].values

    d3_dict = dict(
        {'train': (d3_train_x, d3_train_y), 'validate': (d3_validate_x, d3_validate_y), 'test': (d3_test_x, d3_test_y)})

    return no_channel_data, d3_train, d3_validate, d3_test, d3_dict

def change_features(prog_from, prog_to, data_no_channel):
    prog_from_copy= prog_from.copy()
    prog_to_copy= prog_to.copy()
    change_col_list = list()
    for idx, col in enumerate(data_no_channel.columns):
        if ('가중치' in col) or ('성별' in col) or ('연령' in col):
            change_col_list.append(idx)
    for idx in change_col_list:
        prog_to_copy[idx] = prog_from_copy[idx]
    return prog_to_copy

def main():
    # data load
    print('0. data 로드')
    data = pd.read_csv(os.path.join(load_path, '{}.csv'.format(final_file_name)), engine='python', encoding='cp949')
    print('1. regression')
    reg_result, reg_result_table = regression(data, 0.2)
    print('///////////////////')
    print('2. classification')
    cl_result = classification(data)


    print('end')

if __name__ == "__main__":
    main()
