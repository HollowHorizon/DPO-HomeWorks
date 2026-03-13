# HW06 - Report

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: 18000 строк, 39 столбцов
- Целевая переменная: `target`, доли классов {0: 0.7374, 1: 0.2626}
- Признаки: {'float64': 37}

## 2. Protocol

- Разбиение: train/test = 80%/20%, `random_state=42`, `stratify=y`
- Подбор: `GridSearchCV` с 3-fold `StratifiedKFold` только на train
- Метрики: `accuracy`, `f1`, `roc_auc`; для выбранного бинарного датасета ключевой метрикой выбрал `roc_auc`, потому что она лучше отражает качество ранжирования на нелинейной задаче с умеренным дисбалансом

## 3. Models

Сравнивались модели:

- `DummyClassifier(strategy="most_frequent")`
- `Pipeline(StandardScaler + LogisticRegression)`
- `DecisionTreeClassifier` с подбором `max_depth`, `min_samples_leaf`, `class_weight`
- `RandomForestClassifier` с подбором `max_depth`, `min_samples_leaf`, `max_features`
- `HistGradientBoostingClassifier` с подбором `learning_rate`, `max_depth`, `min_samples_leaf`, `l2_regularization`

## 4. Results

| model | accuracy | f1 | roc_auc | cv_roc_auc |
| --- | --- | --- | --- | --- |
| HistGradientBoostingClassifier | 0.9033 | 0.7974 | 0.9307 | 0.9272 |
| RandomForestClassifier | 0.8908 | 0.7579 | 0.9281 | 0.9267 |
| DecisionTreeClassifier | 0.7872 | 0.6562 | 0.8443 | 0.834 |
| LogisticRegression | 0.8119 | 0.5607 | 0.7977 | nan |
| DummyClassifier | 0.7375 | 0.0 | 0.5 | nan |

Победитель: **HistGradientBoostingClassifier** с ROC-AUC = **0.9307**. Он лучше baseline-моделей и дерева, что соответствует ожиданию для датасета с нелинейными взаимодействиями признаков.

## 5. Analysis

- Устойчивость по 5 разным `random_state` для лучшей модели:

| seed | roc_auc | f1 |
| --- | --- | --- |
| 7 | 0.9297 | 0.8058 |
| 13 | 0.9252 | 0.8 |
| 21 | 0.9305 | 0.8 |
| 42 | 0.9307 | 0.7974 |
| 77 | 0.933 | 0.7967 |

- Confusion matrix сохранена в `./artifacts/figures/confusion_matrix.png`
- Permutation importance показала, что сильнее всего влияют: f16, f01, f07, f08, f30
- По ROC-кривым видно, что ансамбли дают более качественное разделение классов, чем линейный baseline

## 6. Conclusion

- Одинокое дерево улучшает baseline, но быстро упирается в bias/variance trade-off.
- Лес и бустинг выигрывают на нелинейной задаче, потому что лучше собирают взаимодействия признаков.
- `HistGradientBoostingClassifier` оказался лучшим компромиссом между качеством и устойчивостью.
- Фиксированный train/test split и отдельный CV на train позволяют не подгонять гиперпараметры под test.
- Даже при умеренном дисбалансе `roc_auc` даёт более полезную картину, чем одна только accuracy.
