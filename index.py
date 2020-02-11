import pandas as pd
import numpy as np
import os.path
from sklearn.linear_model import LogisticRegression

train_file = 'source/train.csv'
test_file = 'source/test.csv'
result_file = 'source/result.csv'


# load train data

def train_based_on_gender_and_class(features, result, target):
    classifier = LogisticRegression()
    _classifier = classifier.fit(features, target)
    return _classifier.predict(result)


def map_gender_to_int(_gender):
    # fix gender
    _gender = _gender.map({'male': 1, 'female': 2})
    return _gender.astype(np.float)


def fix_age_with_mean(_input):
    # not recommend, bad result
    # fix gender by plotting average to null value
    _input.fillna({'Age': 0})
    print(_input['Age'].mean())
    return _input.fillna({'Age': _input['Age'].mean()})


def fix_age_with_class(_input):
    _input = _input.fillna({'Age': 0})
    # get pClass average
    _result_after_pClass_and_age_drop = pd.DataFrame()
    _number_of_pclass = _input.drop_duplicates(subset='Pclass')['Pclass']

    for i in _number_of_pclass:
        _pClass_and_age_drop = _input.drop(_input[_input['Pclass'] != i].index)
        _pClass_and_age_drop['Age'] = _pClass_and_age_drop['Age'].replace(0, _pClass_and_age_drop['Age'].mean())
        _result_after_pClass_and_age_drop = _result_after_pClass_and_age_drop.append(_pClass_and_age_drop)

    return _result_after_pClass_and_age_drop


def export_csv(result):
    pd.DataFrame(result).to_csv(result_file, index=False)


if os.path.exists(train_file):
    if os.path.exists(test_file):
        # load data
        train_csv = pd.read_csv(train_file)
        test_csv = pd.read_csv(test_file)

        # remove cabin as it too much empty
        train = train_csv.drop(['Cabin'], axis=1)
        test = test_csv.drop(['Cabin'], axis=1)

        # fix data
        # drop NULL value to avoid learning prob
        # train_no_null = pd.DataFrame(train).dropna()
        train_no_null = fix_age_with_class(train)
        train_no_null['Sex'] = map_gender_to_int(train_no_null['Sex'])
        train_numpy = train_no_null.values

        # test
        test_no_null = fix_age_with_class(test)
        test_no_null['Sex'] = map_gender_to_int(test_no_null['Sex'])
        test_numpy = test_no_null.values

        # survived
        survived = train_numpy[:, 1]
        survived = survived.astype(np.float)

        # pClass
        train_pClass = train_numpy[:, 2]
        train_pClass = train_pClass.astype(np.float)

        test_pClass = test_numpy[:, 1]
        test_pClass = test_pClass.astype(np.float)

        # gender
        train_gender = train_numpy[:, 4]
        test_gender = test_numpy[:, 3]

        # age
        train_age = train_numpy[:, 5]
        test_age = test_numpy[:, 4]

        train_result = train_based_on_gender_and_class(np.column_stack((train_pClass, train_gender, train_age)),
                                                       np.column_stack((test_pClass, test_gender, test_age)), survived)

        train_result_combine = np.column_stack((test_numpy[:, 0], train_result))
        train_result_combine_column = pd.DataFrame(train_result_combine, columns=['PassengerId', 'Survived'])
        train_result_combine_column['Survived'] = train_result_combine_column['Survived'].astype(int)
        print(train_result_combine_column)
        export_csv(train_result_combine_column)
