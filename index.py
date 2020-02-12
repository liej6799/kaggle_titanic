import pandas as pd
import numpy as np
import os.path
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

train_file = 'source/train.csv'
test_file = 'source/test.csv'
result_file = 'source/result.csv'

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# load train data

def train_logistic_regression(features, result, target):
    classifier = LogisticRegression()
    _classifier = classifier.fit(features, target)
    print(_classifier.score(features, target))
    return _classifier.predict(result)


def train_gradient_boost(features, result, target):
    classifier = GradientBoostingClassifier()
    _classifier = classifier.fit(features, target)
    print(_classifier.score(features, target))
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


def sum_sib_and_parch(_input):
    _input['sumSibSpParch'] = _input['SibSp'] + _input['Parch']
    return _input

def fix_fare(_input):
    _input.fillna({'Fare': 0})
    return _input.fillna({'Fare': _input['Fare'].mean()})

def extract_title_from_name(_input):
    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    _input['Title'] = _input['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    _input['Title'] = _input.Title.map(title_dictionary)

    titles_dummies = pd.get_dummies(_input['Title'], prefix='Title')
    _input = pd.concat([_input, titles_dummies], axis=1)
    _input.drop('Name', axis=1, inplace=True)
    _input.drop('Title', axis=1, inplace=True)
    return _input


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
        train_no_null = extract_title_from_name(train)
        train_no_null = fix_age_with_class(train_no_null)
        train_no_null = sum_sib_and_parch(train_no_null)
        train_no_null['Sex'] = map_gender_to_int(train_no_null['Sex'])
        train_no_null.drop('Ticket', axis=1, inplace=True)
        train_no_null.drop('Embarked', axis=1, inplace=True)

        survived = train_no_null['Survived']
        train_no_null.drop('Survived', axis=1, inplace=True)
        train_no_null.drop('PassengerId', axis=1, inplace=True)
        train_no_null.drop('Title_Royalty', axis=1, inplace=True)

        # test
        test_no_null = extract_title_from_name(test)
        test_no_null = fix_age_with_class(test_no_null)
        test_no_null = sum_sib_and_parch(test_no_null)
        test_no_null = fix_fare(test_no_null)
        test_no_null['Sex'] = map_gender_to_int(test['Sex'])
        test_no_null.drop('Ticket', axis=1, inplace=True)

        test_passenger = test_no_null['PassengerId']

        test_no_null.drop('PassengerId', axis=1, inplace=True)
        test_no_null.drop('Embarked', axis=1, inplace=True)


        clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
        clf = clf.fit(train_no_null, survived)

        featu = pd.DataFrame()
        featu['feature'] = train_no_null.columns
        featu['importance'] = clf.feature_importances_
        featu.sort_values(by=['importance'], ascending=True, inplace=True)
        featu.set_index('feature', inplace=True)

        featu.plot(kind='barh', figsize=(25, 25))

        model = SelectFromModel(clf, prefit=True)
        train_reduced = model.transform(train_no_null)
        test_reduced = model.transform(test_no_null)

        train_result = train_gradient_boost(train_reduced, test_reduced, survived)


        train_result_combine = np.column_stack((test_passenger, train_result))
        train_result_combine_column = pd.DataFrame(train_result_combine, columns=['PassengerId', 'Survived'])
        train_result_combine_column['Survived'] = train_result_combine_column['Survived'].astype(int)
        print(train_result_combine_column)
        export_csv(train_result_combine_column)
