from sklearn.metrics import cohen_kappa_score
import pandas as pd 

# 두 평가자의 예측 결과
y1 = pd.read_csv('human1_test_dataset.csv')
y2 = pd.read_csv('human2_test_dataset.csv')

# Kappa Score 계산
kappa = cohen_kappa_score(y1, y2)
print("Kappa Score:", kappa)
