# eccDNA Orchestrated Experiment Runner Summary Report

## Summary Table

```
==========================================================================================
Model & Feature Subset                         Accuracy    Recall@1    Recall@5      mAP
------------------------------------------------------------------------------------------
Logistic Regression - Baseline (189 feats)     0.4448      0.3125      0.7535      0.2769
Logistic Regression - SHAP Top 20              0.4058      0.2960      0.7444      0.2583
Logistic Regression - Info-Theoretic (124 feats) 0.4281      0.3142      0.7542      0.2699
------------------------------------------------------------------------------------------
MLP Siamese Model - Baseline (189 feats)       0.4430      0.3369      0.7631      0.2960
MLP Siamese Model - SHAP Top 20                0.2572      0.2841      0.7233      0.2228
MLP Siamese Model - Info-Theoretic (124 feats) 0.4193      0.3163      0.7499      0.2801
------------------------------------------------------------------------------------------
1D-CNN Siamese Model - Baseline (189 feats)    0.4178      0.3260      0.7485      0.2777
1D-CNN Siamese Model - SHAP Top 20             0.2794      0.2634      0.7166      0.2121
1D-CNN Siamese Model - Info-Theoretic (124 feats) 0.4037      0.3107      0.7411      0.2643
==========================================================================================
```

## Selected Top 20 SHAP Features

1. **sequence_length**
2. **kmer_3_TCC**
3. **kmer_3_AAA**
4. **MI_Resolved_T_2**
5. **kmer_3_CTC**
6. **MI_tau_10**
7. **kmer_3_TTT**
8. **kmer_3_CCC**
9. **MI_tau_15**
10. **MI_Resolved_C_2**
11. **MI_Resolved_A_5**
12. **MI_tau_16**
13. **kmer_3_CTT**
14. **MI_Resolved_G_2**
15. **kmer_3_TCT**
16. **MI_tau_6**
17. **kmer_3_CAG**
18. **MI_Resolved_C_4**
19. **MI_tau_30**
20. **MI_Resolved_A_3**

## Detailed Classification Reports

### Linear_baseline
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.46      0.58      0.52      1494
                Chronic Kidney Disease       0.40      0.01      0.02       211
              Fetal Growth Restriction       0.58      0.22      0.32       168
                  Histiocytic Lymphoma       0.43      0.50      0.46      1496
Hypopharyngeal Squamous Cell Carcinoma       0.49      0.45      0.47       804
                        Ovarian Cancer       0.47      0.56      0.51      1451
                       Prostate Cancer       0.34      0.21      0.26      1480

                              accuracy                           0.44      7104
                             macro avg       0.45      0.36      0.37      7104
                          weighted avg       0.44      0.44      0.43      7104
```

### Linear_shap_selected
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.42      0.61      0.50      1494
                Chronic Kidney Disease       0.00      0.00      0.00       211
              Fetal Growth Restriction       0.44      0.10      0.16       168
                  Histiocytic Lymphoma       0.37      0.45      0.40      1496
Hypopharyngeal Squamous Cell Carcinoma       0.48      0.37      0.42       804
                        Ovarian Cancer       0.45      0.55      0.50      1451
                       Prostate Cancer       0.27      0.13      0.18      1480

                              accuracy                           0.41      7104
                             macro avg       0.35      0.31      0.31      7104
                          weighted avg       0.38      0.41      0.38      7104
```

### Linear_info_theoretic
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.45      0.58      0.51      1494
                Chronic Kidney Disease       0.00      0.00      0.00       211
              Fetal Growth Restriction       0.59      0.14      0.23       168
                  Histiocytic Lymphoma       0.41      0.50      0.45      1496
Hypopharyngeal Squamous Cell Carcinoma       0.46      0.33      0.38       804
                        Ovarian Cancer       0.45      0.60      0.51      1451
                       Prostate Cancer       0.33      0.18      0.23      1480

                              accuracy                           0.43      7104
                             macro avg       0.38      0.33      0.33      7104
                          weighted avg       0.41      0.43      0.40      7104
```

### MLP_baseline
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.48      0.50      0.49      1494
                Chronic Kidney Disease       0.20      0.17      0.18       211
              Fetal Growth Restriction       0.46      0.55      0.50       168
                  Histiocytic Lymphoma       0.41      0.53      0.46      1496
Hypopharyngeal Squamous Cell Carcinoma       0.51      0.53      0.52       804
                        Ovarian Cancer       0.48      0.58      0.52      1451
                       Prostate Cancer       0.34      0.14      0.19      1480

                              accuracy                           0.44      7104
                             macro avg       0.41      0.43      0.41      7104
                          weighted avg       0.43      0.44      0.42      7104
```

### MLP_shap_selected
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.34      0.34      0.34      1494
                Chronic Kidney Disease       0.05      0.19      0.08       211
              Fetal Growth Restriction       0.18      0.39      0.25       168
                  Histiocytic Lymphoma       0.30      0.31      0.31      1496
Hypopharyngeal Squamous Cell Carcinoma       0.16      0.14      0.15       804
                        Ovarian Cancer       0.34      0.32      0.33      1451
                       Prostate Cancer       0.22      0.12      0.16      1480

                              accuracy                           0.26      7104
                             macro avg       0.23      0.26      0.23      7104
                          weighted avg       0.27      0.26      0.26      7104
```

### MLP_info_theoretic
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.47      0.49      0.48      1494
                Chronic Kidney Disease       0.12      0.20      0.15       211
              Fetal Growth Restriction       0.39      0.51      0.44       168
                  Histiocytic Lymphoma       0.43      0.48      0.46      1496
Hypopharyngeal Squamous Cell Carcinoma       0.41      0.39      0.40       804
                        Ovarian Cancer       0.45      0.61      0.52      1451
                       Prostate Cancer       0.36      0.13      0.19      1480

                              accuracy                           0.42      7104
                             macro avg       0.37      0.40      0.38      7104
                          weighted avg       0.41      0.42      0.40      7104
```

### CNN_baseline
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.44      0.54      0.49      1494
                Chronic Kidney Disease       0.23      0.13      0.17       211
              Fetal Growth Restriction       0.49      0.55      0.52       168
                  Histiocytic Lymphoma       0.40      0.43      0.42      1496
Hypopharyngeal Squamous Cell Carcinoma       0.47      0.47      0.47       804
                        Ovarian Cancer       0.48      0.47      0.47      1451
                       Prostate Cancer       0.29      0.23      0.26      1480

                              accuracy                           0.42      7104
                             macro avg       0.40      0.40      0.40      7104
                          weighted avg       0.41      0.42      0.41      7104
```

### CNN_shap_selected
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.36      0.34      0.35      1494
                Chronic Kidney Disease       0.02      0.01      0.01       211
              Fetal Growth Restriction       0.28      0.21      0.24       168
                  Histiocytic Lymphoma       0.30      0.28      0.29      1496
Hypopharyngeal Squamous Cell Carcinoma       0.19      0.23      0.21       804
                        Ovarian Cancer       0.33      0.40      0.36      1451
                       Prostate Cancer       0.20      0.17      0.19      1480

                              accuracy                           0.28      7104
                             macro avg       0.24      0.23      0.23      7104
                          weighted avg       0.27      0.28      0.28      7104
```

### CNN_info_theoretic
```
                                        precision    recall  f1-score   support

               Cervical Adenocarcinoma       0.46      0.47      0.46      1494
                Chronic Kidney Disease       0.17      0.08      0.11       211
              Fetal Growth Restriction       0.52      0.39      0.45       168
                  Histiocytic Lymphoma       0.38      0.50      0.43      1496
Hypopharyngeal Squamous Cell Carcinoma       0.43      0.39      0.41       804
                        Ovarian Cancer       0.47      0.44      0.45      1451
                       Prostate Cancer       0.31      0.26      0.28      1480

                              accuracy                           0.40      7104
                             macro avg       0.39      0.36      0.37      7104
                          weighted avg       0.40      0.40      0.40      7104
```

