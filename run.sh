#!/bin/bash
echo 'Extracting the feature vectors...'
python3 extract_feature_vectors.py
echo 'Finished!'
echo 'Training the classifiers...'
python3 train_classifiers.py
echo 'Finished!'
echo 'Computing classifier evaluations...'
sh ./evaluate-classifiers.sh
echo 'Finished!'
echo 'Performing Wilcoxon tests...'
sh ./wilcoxon.sh
echo 'Finished!'