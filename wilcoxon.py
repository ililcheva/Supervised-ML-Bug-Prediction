from scipy.stats import mannwhitneyu
import pandas as pd

clf_df = pd.read_csv('./classifier_evaluations/scores-DecisionTreeClassifier.csv', sep=',', header=0)
gnb_df = pd.read_csv('./classifier_evaluations/scores-GaussianNB.csv', sep=',', header=0)
lsvc_df = pd.read_csv('./classifier_evaluations/scores-LinearSVC.csv', sep=',', header=0)
mlpc_df = pd.read_csv('./classifier_evaluations/scores-MLPClassifier.csv', sep=',', header=0)
rfc_df = pd.read_csv('./classifier_evaluations/scores-RandomForestClassifier.csv', sep=',', header=0)
biased_df = pd.read_csv('./classifier_evaluations/scores-BiasedClassifier.csv', sep=',', header=0)

''' Comparing DecisionTreeClassifier and RandomForestClassifier '''
_, p_clf_rfc_f1 = mannwhitneyu(clf_df['f1'].values, rfc_df['f1'].values)
_, p_clf_rfc_precision = mannwhitneyu(clf_df['precision'].values, rfc_df['precision'].values)
_, p_clf_rfc_recall = mannwhitneyu(clf_df['recall'].values, rfc_df['recall'].values)

''' Comparing GaussianNB and LinearSVC '''
_, p_gnb_lsvc_f1 = mannwhitneyu(gnb_df['f1'].values, lsvc_df['f1'].values)
_, p_gnb_lsvc_precision = mannwhitneyu(gnb_df['precision'].values, lsvc_df['precision'].values)
_, p_gnb_lsvc_recall = mannwhitneyu(gnb_df['recall'].values, lsvc_df['recall'].values)

''' Comparing MLPClassifier and LinearSVC '''
_, p_mlpc_lsvc_f1 = mannwhitneyu(mlpc_df['f1'].values, lsvc_df['f1'].values)
_, p_mlpc_lsvc_precision = mannwhitneyu(mlpc_df['precision'].values, lsvc_df['precision'].values)
_, p_mlpc_lsvc_recall = mannwhitneyu(mlpc_df['recall'].values, lsvc_df['recall'].values)

''' Comparing GaussianNB and RandomForestClassifier '''
_, p_gnb_rfc_f1 = mannwhitneyu(gnb_df['f1'].values, rfc_df['f1'].values)
_, p_gnb_rfc_precision = mannwhitneyu(gnb_df['precision'].values, rfc_df['precision'].values)
_, p_gnb_rfc_recall = mannwhitneyu(gnb_df['recall'].values, rfc_df['recall'].values)

''' Comparing DecisionTreeClassifier and MLPClassifier '''
_, p_clf_mlpc_f1 = mannwhitneyu(clf_df['f1'].values, mlpc_df['f1'].values)
_, p_clf_mlpc_precision = mannwhitneyu(clf_df['precision'].values, mlpc_df['precision'].values)
_, p_clf_mlpc_recall = mannwhitneyu(clf_df['recall'].values, mlpc_df['recall'].values)

with open('./statistical_tests/wilcoxon-comparisons.csv', 'a') as f:
    pd.DataFrame([['DecisionTreeClassifier - RandomForestClassifier', p_clf_rfc_f1, p_clf_rfc_precision, p_clf_rfc_recall]], columns=['pair', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['GaussianNB - LinearSVC', p_gnb_lsvc_f1, p_gnb_lsvc_precision, p_gnb_lsvc_recall]], columns=['pair', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['MLPClassifier - LinearSVC', p_mlpc_lsvc_f1, p_mlpc_lsvc_precision, p_mlpc_lsvc_recall]], columns=['pair', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['GaussianNB - RandomForestClassifier', p_gnb_rfc_f1, p_gnb_rfc_precision, p_gnb_rfc_recall]], columns=['pair', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['DecisionTreeClassifier - MLPClassifier', p_clf_mlpc_f1, p_clf_mlpc_precision, p_clf_mlpc_recall]], columns=['pair', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)


# Comparing each algorithm with the biased classifier
''' DecisionTreeClassifier and BiasedClassifier '''
_, p_clf_biased_f1 = mannwhitneyu(clf_df['f1'].values, biased_df['f1'].values)
_, p_clf_biased_precision = mannwhitneyu(clf_df['precision'].values, biased_df['precision'].values)
_, p_clf_biased_recall = mannwhitneyu(clf_df['recall'].values, biased_df['recall'].values)

''' GaussianNB and BiasedClassifier '''
_, p_gnb_biased_f1 = mannwhitneyu(gnb_df['f1'].values, biased_df['f1'].values)
_, p_gnb_biased_precision = mannwhitneyu(gnb_df['precision'].values, biased_df['precision'].values)
_, p_gnb_biased_recall = mannwhitneyu(gnb_df['recall'].values, biased_df['recall'].values)

''' LinearSVC and BiasedClassifier '''
_, p_lsvc_biased_f1 = mannwhitneyu(lsvc_df['f1'].values, biased_df['f1'].values)
_, p_lsvc_biased_precision = mannwhitneyu(lsvc_df['precision'].values, biased_df['precision'].values)
_, p_lsvc_biased_recall = mannwhitneyu(lsvc_df['recall'].values, biased_df['recall'].values)

''' MLPClassifier and BiasedClassifier '''
_, p_mlpc_biased_f1 = mannwhitneyu(mlpc_df['f1'].values, biased_df['f1'].values)
_, p_mlpc_biased_precision = mannwhitneyu(mlpc_df['precision'].values, biased_df['precision'].values)
_, p_mlpc_biased_recall = mannwhitneyu(mlpc_df['recall'].values, biased_df['recall'].values)

''' RandomForestClassifier and BiasedClassifier '''
_, p_rfc_biased_f1 = mannwhitneyu(rfc_df['f1'].values, biased_df['f1'].values)
_, p_rfc_biased_precision = mannwhitneyu(rfc_df['precision'].values, biased_df['precision'].values)
_, p_rfc_biased_recall = mannwhitneyu(rfc_df['recall'].values, biased_df['recall'].values)

with open('./statistical_tests/biased-comparisons.csv', 'a') as f:
    pd.DataFrame([['DecisionTreeClassifier', p_clf_biased_f1, p_clf_biased_precision, p_clf_biased_recall]], columns=['classifier', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['GaussianNB', p_gnb_biased_f1, p_gnb_biased_precision, p_gnb_biased_recall]], columns=['classifier', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['LinearSVC', p_lsvc_biased_f1, p_lsvc_biased_precision, p_lsvc_biased_recall]], columns=['classifier', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['MLPClassifier', p_mlpc_biased_f1, p_mlpc_biased_precision, p_mlpc_biased_recall]], columns=['classifier', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)
    pd.DataFrame([['RandomForestClassifier', p_rfc_biased_f1, p_rfc_biased_precision, p_rfc_biased_recall]], columns=['classifier', 'f1', 'precision', 'recall']).to_csv(f, sep=',', index=False, header=f.tell()==0)