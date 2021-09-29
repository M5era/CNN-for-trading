import numpy as np
import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from collections import Counter

def create_labels(self, df, col_name, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2
    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
              size = total-(window_size)+1
    """

    self.log("creating label with original paper strategy")
    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    print("Calculating labels")
    pbar = tqdm(total=total_rows)

    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = (window_begin + window_end) / 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = 0
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 2

        row_counter = row_counter + 1
        pbar.update(1)

    pbar.close()
    return labels



if __name__ == "__main__":
    print("go")

    df = pd.read_csv("data/XLE_daily.csv")
    print(create_labels(df=df, col_name="Open"))

    list_features = list(df.loc[:, 'open':'eom_26'].columns)
    print('Total number of features', len(list_features))
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'open':'eom_26'].values, df['labels'].values,
                                                        train_size=0.8,
                                                        test_size=0.2, random_state=2, shuffle=True,
                                                        stratify=df['labels'].values)

    if 0.7 * x_train.shape[0] < 2500:
        train_split = 0.8
    else:
        train_split = 0.7

    print('train_split =', train_split)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=train_split, test_size=1 - train_split,
                                                    random_state=2, shuffle=True, stratify=y_train)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
    x_train = mm_scaler.fit_transform(x_train)
    x_cv = mm_scaler.transform(x_cv)
    x_test = mm_scaler.transform(x_test)

    print("Shape of x, y train/cv/test {} {} {} {} {} {}".format(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape,
                                                                 x_test.shape, y_test.shape))

    """
    # --------------------------------------------------------------------------------------------
    num_features = 225  # should be a perfect square
    selection_method = 'all'
    topk = 320 if selection_method == 'all' else num_features

    if selection_method == 'anova' or selection_method == 'all':
        select_k_best = SelectKBest(f_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_train, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_train, y_train)

        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        print(selected_features_anova)
        print(select_k_best.get_support(indices=True))
        print("****************************************")

    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_train, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_train, y_train)

        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        print(len(selected_features_mic), selected_features_mic)
        print(select_k_best.get_support(indices=True))

    if selection_method == 'all':
        common = list(set(selected_features_anova).intersection(selected_features_mic))
        print("common selected featues", len(common), common)
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topk variable"'.format(
                    len(common), num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        print(feat_idx)  # x_train[:, feat_idx] will give you training data with desired features
        
    """