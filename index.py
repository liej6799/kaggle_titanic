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

def train_random_forest(features, result, target):
    classifier = RandomForestClassifier()
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
    return _input.fillna({'Fare': _input['Fare'].median()})




def fix_embarked_with_fare(_input):

    _input['Embarked'] = _input['Embarked'].fillna('S')
    embarked_dummies = pd.get_dummies(_input['Embarked'], prefix='Embarked')
    _input = pd.concat([_input, embarked_dummies], axis=1)
    _input.drop('Embarked', inplace=True, axis=1)
    return _input


def extract_title_from_name(_input):
    title_mapping = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    _input['Title'] = _input['Name'].str.extract(',\s(.+?)\.', expand=True)[0]
    _input['Title'] = _input['Title'].map(title_mapping)

    titles_dummies = pd.get_dummies(_input['Title'], prefix='Title')
    _input = pd.concat([_input, titles_dummies], axis=1)
    _input.drop('Name', axis=1, inplace=True)
    _input.drop('Title', axis=1, inplace=True)
    return _input


def clean_ticket(ticket):

    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(), ticket)
    ticket = list(filter(lambda t: not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'X'


def extract_ticket(_input):
    _input['Ticket'] = _input['Ticket'].map(clean_ticket)
    ticket_dummies = pd.get_dummies(_input['Ticket'], prefix='Ticket')
    _input = pd.concat([_input, ticket_dummies], axis=1)
    _input.drop('Ticket', inplace=True, axis=1)
    return _input

def process_family(_input):
    _input['FamilySize'] = _input['Parch'] + _input['SibSp'] + 1

    _input['Singleton'] = 0
    _input.loc[_input['FamilySize'] == 1, 'Singleton'] = 1

    _input['SmallFamily'] = 0
    _input.loc[_input['FamilySize'].between(2, 4), 'SmallFamily'] = 1

    _input['LargeFamily'] = 0
    _input.loc[_input['FamilySize'] > 4, 'LargeFamily'] = 1

    _input.drop('Parch', inplace=True, axis=1)
    _input.drop('SibSp', inplace=True, axis=1)
    return _input

def process_cabin(_input):
    _input['Cabin'] = _input['Cabin'].fillna('U')
    _input['Cabin'] = _input['Cabin'].map(lambda c: c[0])
    cabin_dummies = pd.get_dummies(_input['Cabin'], prefix='Cabin')
    _input = pd.concat([_input, cabin_dummies], axis=1)
    _input.drop('Cabin', inplace=True, axis=1)
    return _input

def process_pclass(_input):
    pclass_dummies = pd.get_dummies(_input['Pclass'], prefix='Pclass')
    _input = pd.concat([_input, pclass_dummies], axis=1)
    _input.drop('Pclass', inplace=True, axis=1)
    return _input


def export_csv(result):
    pd.DataFrame(result).to_csv(result_file, index=False)


if os.path.exists(train_file):
    if os.path.exists(test_file):
        # load data
        train_csv = pd.read_csv(train_file)
        test_csv = pd.read_csv(test_file)

        train_survived = train_csv['Survived']
        test_passenger = test_csv['PassengerId']

        train_csv = train_csv.drop(['Survived'], axis=1)

        # Combine trains and test data
        input_combined = train_csv.append(test_csv)

        # remove PassengerId
        input_combined = input_combined.drop(['PassengerId'], axis=1)

        # extract Title from Name
        input_combined = extract_title_from_name(input_combined)

        input_combined = process_pclass(input_combined)

        input_combined = extract_ticket(input_combined)

        input_combined = fix_age_with_mean(input_combined)

        input_combined = process_family(input_combined)

        input_combined = fix_fare(input_combined)

        input_combined = process_cabin(input_combined)

        input_combined = fix_embarked_with_fare(input_combined)

        input_combined = map_gender_to_int(input_combined)

        _input_final_train = input_combined[:891]
        _input_final_test = input_combined[891:]


        clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
        clf = clf.fit(_input_final_train, train_survived)

        feature_classification = pd.DataFrame()
        feature_classification['feature'] = _input_final_train.columns
        feature_classification['importance'] = clf.feature_importances_
        feature_classification.sort_values(by=['importance'], ascending=True, inplace=True)
        feature_classification.set_index('feature', inplace=True)

        feature_classification.plot(kind='barh', figsize=(25, 25))
        plt.show()

        model = SelectFromModel(clf, prefit=True)
        train_reduced = model.transform(_input_final_train)
        test_reduced = model.transform(_input_final_test)

        print(train_reduced.shape)
        print(test_reduced.shape)
        print(test_passenger.shape)

        train_result = train_random_forest(train_reduced, test_reduced, train_survived)

        train_result_combine = np.column_stack((test_passenger, train_result))
        train_result_combine_column = pd.DataFrame(train_result_combine, columns=['PassengerId', 'Survived'])
        train_result_combine_column['Survived'] = train_result_combine_column['Survived'].astype(int)
        #print(train_result_combine_column)
        export_csv(train_result_combine_column)

'''
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
'''
