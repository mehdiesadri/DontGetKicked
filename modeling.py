from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support


def compare_models(models, parameters):
    results = []
    names = []
    for name in models:
        model = models[name]
        kfold = model_selection.KFold(n_splits=parameters['k'], random_state=parameters['seed'])
        cv_results = model_selection.cross_val_score(model, parameters['train_X'], parameters['train_Y'], cv=kfold,
                                                     scoring=parameters['scoring'])
        results.append(cv_results)
        names.append(name)
        msg = "## %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def evaluate_results(expected, predicted):
    # overall statistics
    overall_stats = precision_recall_fscore_support(expected, predicted, average='macro')
    precision, recall, fscore, support = precision_recall_fscore_support(expected, predicted)
    return overall_stats, precision, recall, fscore, support

