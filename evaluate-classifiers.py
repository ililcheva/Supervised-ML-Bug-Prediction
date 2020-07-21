from train_classifiers import *
from sklearn.model_selection import cross_validate
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

algorithm = str(sys.argv[1])

classifiers_dict = {
    'clf': 'DecisionTreeClassifier',
    'gnb': 'GaussianNB',
    'lsvc': 'LinearSVC',
    'mlpc': 'MLPClassifier',
    'rfc': 'RandomForestClassifier',
    'biased': 'BiasedClassifier'
}

evoke_classifiers = {
    'clf': clf,
    'gnb': gnb,
    'lsvc': lsvc,
    'mlpc': mlpc,
    'rfc': rfc,
    'biased': biased
}

scoring = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}

algorithm_scores = {
    'f1': [],
    'precision': [],
    'recall': []
}

algorithm_statistics = {
    'metric':['f1', 'precision', 'recall'],
    'mean': [],
    'std': [],
}

for i in range(20):
    algorithm_score = cross_validate(evoke_classifiers[algorithm], X, y, scoring=scoring,
                               cv=5, return_train_score=False, return_estimator=False)
    algorithm_scores['f1'].append(algorithm_score['test_f1'])
    algorithm_scores['precision'].append(algorithm_score['test_precision'])
    algorithm_scores['recall'].append(algorithm_score['test_recall'])

algorithm_scores['f1'] = np.array(algorithm_scores['f1']).flatten()
algorithm_scores['precision'] = np.array(algorithm_scores['precision']).flatten()
algorithm_scores['recall'] = np.array(algorithm_scores['recall']).flatten()
algorithm_statistics['mean'].append(np.mean(algorithm_scores['f1']))
algorithm_statistics['std'].append(np.std(algorithm_scores['f1']))
algorithm_statistics['mean'].append(np.mean(algorithm_scores['precision']))
algorithm_statistics['std'].append(np.std(algorithm_scores['precision']))
algorithm_statistics['mean'].append(np.mean(algorithm_scores['recall']))
algorithm_statistics['std'].append(np.std(algorithm_scores['recall']))
algorithm_df = pd.DataFrame(algorithm_scores)
statistics_df = pd.DataFrame(algorithm_statistics)
algorithm_df.to_csv("./classifier_evaluations/scores-{0}.csv".format(classifiers_dict[algorithm]), sep=',', index=False)
statistics_df.to_csv("./classifier_evaluations/statistics-{0}.csv".format(classifiers_dict[algorithm]), sep=',', index=False)
algorithm_df.boxplot().get_figure().savefig('./classifier_evaluations/{0}.png'.format(classifiers_dict[algorithm]))

