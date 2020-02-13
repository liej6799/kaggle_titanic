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


def map_gender_to_int(_input):
    # fix gender
    _input['Sex'] = _input['Sex'].map({'male': 1, 'female': 2})
    _input['Sex'] = _input['Sex'].astype(np.int)
    return _input


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


def fix_embarked(_input):
    _input['Embarked'] = _input['Embarked'].map({'S': 1, 'Q': 2, 'C': 3})
    return _input


def fix_embarked_with_fare(_input_train):
    _input_train = fix_embarked(_input_train)

    _embarked_train = _input_train.copy()
    _embarked_test = _input_train.copy()
    _embarked_train = _embarked_train.dropna(subset=['Embarked'])
    _embarked_test = _embarked_test[_embarked_test.isnull().any(1)]
    _embarked_valid_data = _embarked_train['Embarked']

    # remove ticket and before train
    _embarked_train.drop('Embarked', axis=1, inplace=True)
    _embarked_test.drop('Embarked', axis=1, inplace=True)

    # train the result and store the null value.
    train_result = train_gradient_boost(_embarked_train, _embarked_test, _embarked_valid_data)
    _embarked_test['Embarked'] = train_result
    _embarked_train['Embarked'] = _embarked_valid_data
    _result = _embarked_train.append(_embarked_test)

    # print(_embarked_test.shape)
    return _result


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

        train = train.drop(['Ticket'], axis=1)
        test = test.drop(['Ticket'], axis=1)

        # fix data
        train_no_null = extract_title_from_name(train)
        train_no_null = fix_age_with_class(train_no_null)
        train_no_null = sum_sib_and_parch(train_no_null)
        train_no_null = map_gender_to_int(train_no_null)
        train_no_null = fix_embarked_with_fare(train_no_null)

        # remove ticket

        test_no_null = extract_title_from_name(test)
        test_no_null = fix_age_with_class(test_no_null)
        test_no_null = sum_sib_and_parch(test_no_null)
        test_no_null = map_gender_to_int(test_no_null)
        test_no_null = fix_fare(test_no_null)
        test_no_null = fix_embarked(test_no_null)

        survived = train_no_null['Survived']
        test_passenger = test_no_null['PassengerId']

        train_no_null = train_no_null.drop(['Survived'], axis=1)
        train_no_null = train_no_null.drop(['Title_Royalty'], axis=1)

        test_no_null = test_no_null.drop(['PassengerId'], axis=1)
        train_no_null = train_no_null.drop(['PassengerId'], axis=1)

        print(train_no_null.shape)
        print(test_no_null.shape)
        print(survived.shape)

        clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
        clf = clf.fit(train_no_null, survived)

        feature_classification = pd.DataFrame()
        feature_classification['feature'] = train_no_null.columns
        feature_classification['importance'] = clf.feature_importances_
        feature_classification.sort_values(by=['importance'], ascending=True, inplace=True)
        feature_classification.set_index('feature', inplace=True)

        feature_classification.plot(kind='barh', figsize=(25, 25))
        plt.show()

        model = SelectFromModel(clf, prefit=True)
        train_reduced = model.transform(train_no_null)
        test_reduced = model.transform(test_no_null)

        print(train_reduced.shape)
        print(test_reduced.shape)
        print(survived.shape)

        train_result = train_gradient_boost(train_reduced, test_reduced, survived)

        train_result_combine = np.column_stack((test_passenger, train_result))
        train_result_combine_column = pd.DataFrame(train_result_combine, columns=['PassengerId', 'Survived'])
        train_result_combine_column['Survived'] = train_result_combine_column['Survived'].astype(int)
        print(train_result_combine_column)
        export_csv(train_result_combine_column)
