<!-- traingとvalidをデータをshuffleしてない
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
| 0.71348315　| 0.62 | 0.4920635 | 0.54867256 -->

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

<!-- 
lossを各データにした→学習が進まないことを確認した（学習率が1e-5だったからだと判明）
- 
|　***1回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 0 | 63 |
| dead(actual) | 0 | 115 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.64606744 | 1.0 | 0.0 | 0.0

|　***2回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 0 | 79 |
| dead(actual) | 0 | 99 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.55617976 | 1.0 | 0.0 | 0.0
 -->

lossを各データにして、学習率を1e-2にあげた
-
|　***1回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 31 | 34 |
| dead(actual) | 21 | 92 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.69101125 | 0.59615386 | 0.47692308 | 0.5299145

|　***2回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 37 | 37 |
| dead(actual) | 23 | 81 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.66292137 | 0.6166667 | 0.5 | 0.5522388

|　***3回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 31 | 33 |
| dead(actual) | 25 | 89 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.6741573 | 0.5535714 | 0.484375 | 0.5166666

|　***4回目*** | survived(predcited) | dead(predicted) |
| - | - | - |
| survived(actual) | 26 | 38 |
| dead(actual) | 16 | 98 |

| accuracy | precision | recall | f1score |
| - | - | - | - |
| 0.6966292 | 0.61904764 | 0.40625 | 0.49056602

<!-- 複数回実行して平均と分散を出すこと -->