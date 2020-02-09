import pandas as pd
import numpy as np
import os.path
from sklearn.linear_model import LogisticRegression

train_file = 'source/train.csv'
test_file = 'source/test.csv'
result_file = 'source/result.csv'


# load train data

def train_based_on_gender_and_class(features, test, target):
    classifier = LogisticRegression()
    classifier_ = classifier.fit(features, target)
    return classifier_.predict(test)


def fix_gender_to_int(_gender):
    # fix gender
    _gender[_gender == 'male'] = 1
    _gender[_gender == 'female'] = 0
    return _gender.astype(np.float)


def export_csv(result):
    pd.DataFrame(result).to_csv(result_file)


if os.path.exists(train_file):
    if os.path.exists(test_file):
        # load data
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        # remove cabin
        train = train.drop(['Cabin'], axis=1)
        test = test.drop(['Cabin'], axis=1)

        # drop NULL value to avoid learning prob
        data = pd.DataFrame(train)
        data_no_null = data.dropna()
        # convert pandas to numpy
        data_numpy = data_no_null.values

        data_test = test.values
        # survived
        survived = data_numpy[:, 1]
        survived = survived.astype(np.float)

        # pClass
        pClass = data_numpy[:, 2]
        pClass = pClass.astype(np.float)

        test_pClass = data_test[:, 1]
        test_pClass = test_pClass.astype(np.float)

        # gender
        gender = data_numpy[:, 4]
        gender = fix_gender_to_int(gender)

        test_gender = data_test[:, 3]
        test_gender = fix_gender_to_int(test_gender)

        train_result = train_based_on_gender_and_class(np.column_stack((pClass, gender)),
                                                       np.column_stack((test_pClass, test_gender)), survived)

        train_result_combine = np.column_stack((data_test[:, 0], train_result))
        export_csv(train_result_combine)
