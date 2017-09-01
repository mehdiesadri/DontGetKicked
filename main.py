import os
import warnings
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modeling import evaluate_results, compare_models
from preprocess import preprocess_df, summarize_df, split_df_train_test


def warn(*args, **kwargs):
    pass


base_path = "./data"
validation_size = 0.2  # 80-20 ratio
seed = 42
k = 10  # kFold cross validation
class_column_name = 'IsBadBuy'
accept_patterns = ['yes', 'y', '1']

# models to compare
models = {'LR': LogisticRegression(), 'LDA': LinearDiscriminantAnalysis(), 'KNN': KNeighborsClassifier(),
          'CART': DecisionTreeClassifier(), 'NB': GaussianNB(),
          'RF': RandomForestClassifier(), 'SVM': SVC()}

warnings.warn = warn

print('# loading the training and test data sets...')
df_train = pd.read_csv(os.path.join(base_path, "training.csv"), index_col=0)
df_test = pd.read_csv(os.path.join(base_path, "test.csv"), index_col=0)

summary_choice = raw_input("Do you want to check summary of the training dataset (before preprocessing)? ")
if summary_choice or str(summary_choice).lower() in accept_patterns:
    summarize_df(df_train, class_column_name)

print('# preprocessing the training & test data sets (with downsampling)...')
df_train = preprocess_df(df_train, class_column_name, True)
df_test = preprocess_df(df_test, class_column_name, False)
df_test[class_column_name] = 0

summary_choice = raw_input("Do you want to check summary of the training dataset (after preprocessong)? ")
if summary_choice or str(summary_choice).lower() in accept_patterns:
    summarize_df(df_train, class_column_name)

print('# splitting the training data for offline evaluation with the validation size = {}.'.format(validation_size))
train_X, test_X, train_Y, test_Y = split_df_train_test(df_train, class_column_name, validation_size, seed)

print('# available classification models: ')
for m in models.keys():
    print(" %s: %s" % (m, models[m].__class__.__name__))
compare_choice = raw_input("Do you wan to compare models based on F1 measure? ")
if compare_choice or str(compare_choice).lower() in accept_patterns:
    # in order to include all the training data for cross validation analysis
    tmp_X = df_train.drop(class_column_name, axis=1)
    tmp_Y = pd.DataFrame(df_train, columns=[class_column_name])
    parameters = {'k': k, 'seed': seed, 'train_X': tmp_X, 'train_Y': tmp_Y, 'scoring': 'f1_macro'}
    print('## Models and the {}Fold cross validation: '.format(parameters['k']))
    compare_models(models, parameters)

model_choice = ''
while model_choice not in models.keys():
    model_choice = raw_input('Please select a model from the above list?')
print('# training a {} model on the dataset.'.format(model_choice))
selected_model = models[str(model_choice).strip()]
selected_model.fit(train_X, train_Y)
predicted = selected_model.predict(test_X)

print('# evaluating the model...')
overall_stats, precision, recall, fscore, support = evaluate_results(test_Y, predicted)
print('\n## Overall Statistics(precision, recall, fscore, support) : {}'.format(overall_stats))
print('## Class Level Statistics:')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

final_choice = raw_input('Do you want to predict on the original test set and save the predictions?')
if final_choice or str(final_choice).lower() in accept_patterns:
    final_X = pd.DataFrame(df_test, columns=df_test.columns[1:])
    final_Y = pd.DataFrame(df_test, columns=[class_column_name])
    final_prediction = selected_model.predict(final_X)
    df_test[class_column_name] = final_prediction
    final_entry = pd.DataFrame(df_test, columns=[class_column_name])
    final_entry_path = os.path.join(base_path, "final_entry.csv")
    final_entry.to_csv(final_entry_path, sep=',')
    print('# done saving the results at {}.'.format(os.path.abspath(final_entry_path)))
