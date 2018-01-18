# coding: utf-8

## OS에 따라서 split하는 방식을 바꿔줘야함.
# windows
# date = int(pickle_.split('\\')[-1][:8])

# ubuntu
# date = int(pickle_.split('/')[-1][:8])



import pickle
import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta
import datetime
import time

# 데이터 있는곳 위치
load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))),'nielsen_data')
save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))),'nielsen_pickle_data(180110)')

# 파일구조를 저장하기 위한 dictionary
dict_struc_channel = dict()
dict_struc_genre = dict()
dict_struc_etc = dict()

def print_duration(start):
    print('{}s 걸림'.format(time.time() - start))

def make_dictionary(path):
    # 장르
    struc_genre = pd.read_excel(os.path.join(path, 'SBS요청_파일구조.xlsx'), encoding='cp949', sheetname=2
                                , skiprows=2, names=['no', 'code', '1st', '2nd', '3rd', 'des'])
    struc_genre = struc_genre.drop(['no', 'des'], 1)
    struc_genre['code'] = struc_genre['code'].apply(lambda x: '{:09d}'.format(x))
    struc_genre['total'] = struc_genre.apply(lambda x: x[1] + '&' + x[2] + '&' + x[3], axis=1)
    # 채널
    struc_channel = pd.read_excel(os.path.join(path, 'SBS요청_파일구조.xlsx'), encoding='cp949', sheetname=3,
                                  names=['code', 'des'])
    # 기타
    struc_etc = pd.read_excel(os.path.join(path, 'SBS요청_파일구조.xlsx'), encoding='cp949', sheetname=1).fillna(
        method='ffill')
    # 연령
    dict_struc_age = {i: str(i * 10) + "대" for i in range(10)}

    dict_struc_genre = dict()
    for row in struc_genre.iterrows():
        dict_struc_genre[str(row[1]['code'])] = row[1]['total']
    dict_struc_genre['nan'] = 'nan'
    dict_struc_channel = struc_channel.set_index('code').to_dict('des')['des']

    dict_struc_etc = dict()
    for ele in struc_etc['구분'].drop_duplicates():
        sub_table = struc_etc[struc_etc['구분'] == ele]
        dict_struc_stc_sub = dict()
        for row in sub_table.iterrows():
            dict_struc_stc_sub[str(row[1]['코드'])] = row[1]['설명']
        dict_struc_etc[ele] = dict_struc_stc_sub

    dict_struc_etc['연령'] = dict_struc_age
    return dict_struc_channel, dict_struc_genre, dict_struc_etc


class Nielsen():
    def __init__(self, date, datatable, dict_struc_etc):
        self.date = date
        self.datatable = datatable
        self.dict_struc_etc = dict_struc_etc
        self.program_table = self.make_program_table(self.datatable)
        self.person_table = self.make_person_table(self.datatable)
        self.program_list = self.program_table.to_dict('records')
        self.person_list = self.person_table.to_dict('records')
        self.program_person_dict = self.make_program_person_dict(self.program_list, self.datatable)
        self.add_value_on_program_person_dict()
        self.add_previous_next_program_on_program_person_dict()
        self.add_same_time_program_on_program_person_dict()

    def make_program_table(self, datatable):
        # 채널과 프로그램편성시작시간으로 규정할 수 있음.
        table = datatable[['일자', '채널', '프로그램명', '프로그램편성시작시간', '프로그램편성종료시간', '프로그램장르', '가중치']]
        grouped_table = table.groupby(['일자', '채널', '프로그램명', '프로그램편성시작시간'
                                          , '프로그램편성종료시간', '프로그램장르'], as_index=False).sum()
        grouped_table = grouped_table.rename(columns={'가중치': '가중치(sum)'})
        return grouped_table.sort_values(by=['채널', '프로그램편성시작시간'])

    def make_person_table(self, datatable):
        # 채널과 프로그램편성시작시간으로 규정할 수 있음.
        table = datatable[['패널가구ID', '개인ID', '성별', '연령', '직업', '학력', '소득']].drop_duplicates()
        return table

    def make_program_person_dict(self, program_list, datatable):
        program_person_dict = dict()
        for program in program_list:
            tmp = datatable[(datatable['채널'] == program['채널']) & (datatable['프로그램편성시작시간'] == program['프로그램편성시작시간'])]
            program_person_dict[(program['채널'], program['프로그램편성시작시간'])] = dict({'persons': tmp})
        return program_person_dict

    def find_program(self, string):
        l = list()
        for idx, program in enumerate(self.program_list):
            if string in program['프로그램명']:
                l.append((idx, program))
        return l

    def find_person(self, panel_id, person_id):
        for idx, person in enumerate(self.person_list):
            if (person['패널가구ID'] == panel_id) & (person['개인ID'] == person_id):
                result = person
        return result

    def add_value_on_program_person_dict(self):
        for program in self.program_list:
            (a, b) = (program['채널'], program['프로그램편성시작시간'])
            table = self.program_person_dict[(a, b)]['persons']

            #             total_factor = pd.DataFrame.sum(table['가중치'])
            #             program_time = program['프로그램편성종료시간'] - program['프로그램편성시작시간']

            tmp_dict = dict()
            tmp_dict2 = dict()

            for cat in self.dict_struc_etc.keys():
                tmp_dict[cat] = pd.DataFrame.sum(table.pivot(columns=cat, values='가중치')).to_dict()
                for key in self.dict_struc_etc[cat].keys():
                    try:
                        tmp = tmp_dict[cat][key]
                    except KeyError:
                        tmp = 0
                    except ValueError:
                        tmp = 0
                    tmp_dict2[(cat, key)] = tmp
            self.program_person_dict[(a, b)]['values'] = tmp_dict2

    def add_previous_next_program_on_program_person_dict(self):
        length = len(self.program_list)
        for idx, program in enumerate(self.program_list):
            (a, b) = (program['채널'], program['프로그램편성시작시간'])
            channel = program['채널']
            if idx == 0:
                previous_program = None
            else:
                if channel != self.program_list[idx - 1]['채널']:
                    previous_program = None
                else:
                    previous_program = self.program_list[idx - 1]
            if idx == length - 1:
                next_program = None
            else:
                if channel != self.program_list[idx + 1]['채널']:
                    next_program = None
                else:
                    next_program = self.program_list[idx + 1]
            self.program_person_dict[(a, b)]['previous_program'] = previous_program
            self.program_person_dict[(a, b)]['next_program'] = next_program

    def add_same_time_program_on_program_person_dict(self):
        for idx, row in self.program_table.iterrows():
            (a, b) = (row['채널'], row['프로그램편성시작시간'])
            same_time_programs = self.program_table[(self.program_table['프로그램편성시작시간'] < row['프로그램편성종료시간'])
                                                    & (self.program_table['프로그램편성종료시간'] > row['프로그램편성시작시간'])
                                                    & (self.program_table['채널'] != row['채널'])]
            self.program_person_dict[(a, b)]['same_time_programs'] = same_time_programs.sort_values('가중치(sum)',
                                                                                                    ascending=False)


def time_to_value(time):
    time = '{:08.1f}'.format(time)
    return int(time[0:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:6])


def modify_day(datatable):
    def time_to_value(time):
        time = '{:08.1f}'.format(time)
        return int(time[0:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:6])

    time_col_list = list(['시청시작시간', '시청종료시간', '프로그램편성시작시간', '프로그램편성종료시간'])
    for ele in time_col_list:
        datatable[ele] = datatable[ele].apply(lambda x: time_to_value(x))
    return datatable


def change_starttime(dataframe):
    if dataframe['시청시작시간'] < dataframe['프로그램편성시작시간']:
        return dataframe['프로그램편성시작시간']
    else:
        return dataframe['시청시작시간']


def change_endtime(dataframe):
    if dataframe['시청종료시간'] > dataframe['프로그램편성종료시간']:
        return dataframe['프로그램편성종료시간']
    else:
        return dataframe['시청종료시간']


def change_txt_to_nielsen(load_path, save_path, start, end, cut_rate=0.5):
    file_list = list()
    dict_struc_channel, dict_struc_genre, dict_struc_etc = make_dictionary(load_path)
    for root, dirs, files in os.walk(load_path):
        for file in files:
            if file.endswith(".txt"):
                date = file[:-4]
                if start <= int(date) <= end:
                    data = pd.read_table(os.path.join(root, file), sep='^', encoding='cp949'
                                         , names=['일자', '패널가구ID', '개인ID', '가중치', '성별', '연령', '직업', '학력'
                            , '소득', '채널', '시청시작시간', '시청종료시간', '프로그램시청시간', '프로그램명'
                            , '프로그램편성시작시간', '프로그램편성종료시간', '프로그램장르']
                                         , dtype={'일자': np.str, '성별': np.str, '소득': np.str, '프로그램장르': np.str},
                                         engine='python')
                    # 전처리
                    data = data[~data.isnull().any(axis=1)]  # null 데이터 변환
                    data = modify_day(data)  # 일자를 숫자로 변경
                    data['시청시작시간'] = data.apply(lambda x: change_starttime(x), axis=1)  # 시청시작시간 수정
                    data['시청종료시간'] = data.apply(lambda x: change_endtime(x), axis=1)  # 시청종료시간 수정
                    data['연령'] = data['연령'].apply(lambda x: x // 10)  # 연령 10대, 20대로 변경
                    data = data[data.apply(lambda x: x['프로그램시청시간'] > (x['프로그램편성종료시간'] - x['프로그램편성시작시간']) * cut_rate,
                                           axis=1)]  # 시청시간이 편성시간의 cut_rate만큼 안되면 제거
                    data = data[data['프로그램명'].apply(lambda x: '<재>' not in x)]
                    # channel_list = [1, 2, 3, 4, 5, 11, 13, 15, 18, 21, 29, 33, 69, 76, 96, 97, 98, 103, 106, 107, 127,
                    #                 128, 130, 143, 154, 158, 159, 160, 174, 188, 216, 222, 230, 250, 253, 273, 294, 299,
                    #                 345, 363, 745, 770, 771, 772, 773, 774, 789, 794, 857, 16119]
                    # data = data[data['채널'].apply(lambda x: x in channel_list)]

                    # 클래스로 변환해서 pickle로 변환
                    nielsen = Nielsen(date, data, dict_struc_etc)
                    with open(os.path.join(save_path, '{}.pickle'.format(date)), 'wb') as file_:
                        pickle.dump(nielsen, file_, -1)
                    print(file)


def make_pickle_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        start = time.time()
        for file in files:
            if file.endswith(".pickle"):
                file_list.append(os.path.join(root, file))
    return file_list


def load_pickle_files(path, st, end):
    pickle_list = make_pickle_list(path)
    pickle_dict = dict()
    for pickle_ in pickle_list:
        # date = int(pickle_.split('\\')[-1][:8])
        date = int(pickle_.split('/')[-1][:8])
        if st <= date <= end:
            with open(pickle_, "rb", -1) as file_:
                pickle_dict[date] = pickle.load(file_)
    return pickle_dict

def get_weekday(string):
    ans = datetime.date(int(string[:4]), int(string[4:6]), int(string[6:8]))
    return ans.weekday()

def make_inter_data(nielsen_dict, bool_save = False):
    row = list()
    for idx, date in enumerate(nielsen_dict.keys()):
        if idx % 30 == 0:
            print(date)
        nielsen_ele = nielsen_dict[date]
        for program in nielsen_ele.program_list:
            element = list()
            index_1 = program['채널']
            index_2 = program['프로그램편성시작시간']

            for key in program.keys():
                element.append(program[key])

            for kind in dict_struc_etc.keys():
                if kind in ['성별', '연령']:
                    for key in dict_struc_etc[kind].keys():
                        element.append(nielsen_ele.program_person_dict[(index_1, index_2)]['values'][(kind, key)])

            # 앞에 프로그램 가중치
            if nielsen_ele.program_person_dict[(index_1, index_2)]['previous_program']:
                element.append(nielsen_ele.program_person_dict[(index_1, index_2)]['previous_program']['가중치(sum)'])
            else:
                element.append(0)
            # 뒤에 프로그램 가중치
            if nielsen_ele.program_person_dict[(index_1, index_2)]['next_program']:
                element.append(nielsen_ele.program_person_dict[(index_1, index_2)]['next_program']['가중치(sum)'])
            else:
                element.append(0)
            a = nielsen_ele.program_person_dict[(index_1, index_2)]['same_time_programs'].sort_values('가중치(sum)',
                                                                                                      ascending=False)[
                '가중치(sum)']
            for idx, ele in enumerate(list(a)[:5]):
                element.append(ele)
            row.append(element)

    column_names = list()
    for key in program.keys():
        column_names.append(key)
    for kind in dict_struc_etc.keys():
        if kind in ['성별', '연령']:
            for key in dict_struc_etc[kind].keys():
                column_names.append(kind + str(key))
    column_names.append('이전')
    column_names.append('이후')
    for idx, ele in enumerate(list(a)[:5]):
        column_names.append('동시' + str(idx))
    data = pd.DataFrame(row, columns=column_names)

    data['요일'] = data['일자'].apply(lambda x: get_weekday(x))
    data['프로그램장르'] = data['프로그램장르'].apply(lambda x: dict_struc_genre[str(x)].split('&')[0])

    if bool_save:
        data.to_csv(
            os.path.join(save_path, 'inter_data_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))),
            index=False)
    return data

# 기타 정보


# 채널, 프로그램명, 프로그램장르 순으로 distinct한 프로그램 리스트를 구하고,
def make_final_data(inter_data, bool_save = False, no = 3):
    total_list = list()
    once_for_collist = True
    distinct_table = inter_data[['채널', '프로그램명', '프로그램장르']].drop_duplicates()
    print('전체: {}개'.format(distinct_table.shape[0]))
    for idx, (_, distinct_row) in enumerate(distinct_table.iterrows()):
        # 구한 프로그램 리스트 별로, 위에 순차적으로 내려오면서 데이터를 만듬
        if idx % 500 == 0:
            print(idx)
        filtered_rows = inter_data[(inter_data['채널'] == distinct_row['채널'])
                                   & (inter_data['프로그램명'] == distinct_row['프로그램명'])
                                   & (inter_data['프로그램장르'] == distinct_row['프로그램장르'])].sort_values('일자', ascending=False)
        for _, row in filtered_rows.iterrows():
            exact_row = filtered_rows[(np.abs(filtered_rows['프로그램편성시작시간'] - row['프로그램편성시작시간']) < 2000)
                                      & (filtered_rows['일자'] < row['일자'])].head(no)
            if exact_row.shape[0] >= no:
                drop_col_list = ['일자', '채널', '프로그램명', '프로그램편성시작시간', '프로그램편성종료시간', '프로그램장르', '성별2', '연령9', '요일']
                exact_row = exact_row.drop(drop_col_list, axis=1)

                if once_for_collist:
                    col_list = list()
                    for i in range(no):
                        col_list += list(str(i) + '_' + exact_row.columns)
                    onece_for_collist = False

                tmp = pd.DataFrame(exact_row.values.reshape(1, -1))

                tmp['일자'] = row['일자']
                tmp['채널'] = row['채널']
                tmp['프로그램명'] = row['프로그램명']
                tmp['프로그램편성시작시간'] = row['프로그램편성시작시간']
                tmp['프로그램편성시작시간'] = row['프로그램편성종료시간']
                tmp['프로그램장르'] = row['프로그램장르']
                tmp['요일'] = row['요일']
                tmp['목표'] = row['가중치(sum)']

                total_list.append(tmp)

    final_data = pd.concat(total_list, ignore_index=True)
    final_data.columns = col_list + list(final_data.columns[len(col_list):])
    final = final_data.copy()
    dummy_channel = pd.get_dummies(final['채널'], prefix='채널')
    dummy_day = pd.get_dummies(final['요일'], prefix='요일')
    dummy_genre = pd.get_dummies(final['프로그램장르'], prefix='장르')
    final = final.drop(['채널', '요일', '프로그램장르'], axis=1)
    final = pd.concat([final, dummy_channel, dummy_day, dummy_genre], axis=1)
    final = pd.concat([final[['일자', '프로그램명', '프로그램편성시작시간', '목표']], final.drop(['일자', '프로그램명', '프로그램편성시작시간', '목표'], axis=1)],
                      axis=1)
    if bool_save:
        final.to_csv(os.path.join(save_path,'final_data_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))), index=False)

def main():
    start = 20160701
    end = 20171231

    global dict_struc_channel, dict_struc_genre, dict_struc_etc
    dict_struc_channel, dict_struc_genre, dict_struc_etc = make_dictionary(load_path)

    # 0. 폴더가 있는지 확인
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print('0. 폴더 생성')

    # 1. 로그데이터를 nielsen class 형식으로 변경해줌
    change_txt_to_nielsen(load_path, save_path, start, end)
    print('1. change 완료')


    # 2. nielsen class 데이터 load
    nielsen_dict = load_pickle_files(save_path, start, end)
    print('2. load 완료')

    # 3. 프로그램별 시청률 정보를 나타낸 dataset 만들기 / 파일로 저장하기 위해서는 bool_save = True 로
    inter_data = make_inter_data(nielsen_dict)
    print('3. inter_data 생성 완료')

    # 4. 몇회전 프로그램의 시청률 정보까지 고려해서 final dataset 만들기 / 파일로 저장하기 위해서는 bool_save = True 로
    final_data = make_final_data(inter_data)
    print('4. final_data 생성 완료')

    print('end')

if __name__ == "__main__":
    # execute only if run as a script
    main()