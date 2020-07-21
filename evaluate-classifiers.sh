mkdir -p classifier_evaluations
for algo in 'clf' 'gnb' 'lsvc' 'mlpc' 'rfc' 'biased'
do
        python3 evaluate-classifiers.py $algo
done