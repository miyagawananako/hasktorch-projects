traingとvalidをデータをshuffleしてない
-
| ***1回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 34 | 29 |
| dead(actual) | 20 | 95 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.7247191 | 0.6296296 | 0.53968257 | 0.58119655
---
| ***2回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 29 | 34 |
| dead(actual) | 18 | 97 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.7078652 | 0.61702126 | 0.46031746 | 0.52727276
---
| ***3回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 31 | 32 |
| dead(actual) | 19 | 96 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.71348315　| 0.62 | 0.4920635 | 0.54867256

trainingDataとvalidData全体をshuffleした-> recallとf1scoreが低い
-
| ***1回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 23 | 44 |
| dead(actual) | 6 | 105 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.71910113　| 0.79310346 | 0.3432836 | 0.47916666
---
| ***2回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 24 | 47 |
| dead(actual) | 15 | 92 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.6516854 | 0.61538464 | 0.33802816 | 0.43636364

---
| ***3回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 24 | 48 |
| dead(actual) | 21 | 83 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.6123595 | 0.5531915 | 0.35135135 | 0.42975208

---
|　***4回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 0 | 56 |
| dead(actual) | 1 | 121 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.6797753 | 0.0 | 0.0 | 0.0

---
|　***5回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 27 | 32 |
| dead(actual) | 16 | 103 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.7303371 | 0.627907 | 0.45762712 | 0.52941173

複数回実行して平均と分散を出す