# eccDNA Orchestrated Experiment Runner Summary Report

## Summary Table

```
==========================================================================================
Model & Feature Subset                         Accuracy    Recall@1    Recall@5      mAP
------------------------------------------------------------------------------------------
Logistic Regression - Baseline (189 feats)     0.4448      0.3125      0.7535      0.2769
Logistic Regression - SHAP Top 20              0.3984      0.2876      0.7304      0.2442
Logistic Regression - Info-Theoretic (124 feats) 0.4281      0.3142      0.7542      0.2699
------------------------------------------------------------------------------------------
MLP Siamese Model - Baseline (189 feats)       0.4430      0.3369      0.7631      0.2960
MLP Siamese Model - SHAP Top 20                0.3956      0.3159      0.7490      0.2810
MLP Siamese Model - Info-Theoretic (124 feats) 0.4193      0.3163      0.7499      0.2801
------------------------------------------------------------------------------------------
1D-CNN Siamese Model - Baseline (189 feats)    0.4178      0.3260      0.7485      0.2777
1D-CNN Siamese Model - SHAP Top 20             0.4147      0.3021      0.7466      0.2661
1D-CNN Siamese Model - Info-Theoretic (124 feats) 0.4037      0.3107      0.7411      0.2643
==========================================================================================
```

## Selected Top 20 SHAP Features

1. **kmer_3_GGA**
2. **kmer_3_GGG**
3. **kmer_3_CCG**
4. **kmer_3_GAG**
5. **kmer_3_AAA**
6. **kmer_3_CGC**
7. **kmer_3_GGT**
8. **kmer_3_AGG**
9. **kmer_3_GGC**
10. **kmer_3_GTG**
11. **kmer_3_GCC**
12. **kmer_3_TTT**
13. **MI_tau_31**
14. **kmer_3_CCA**
15. **MI_Resolved_T_3**
16. **GC_pct**
17. **kmer_3_CAG**
18. **MI_Resolved_C_3**
19. **MI_tau_30**
20. **kmer_3_AGC**

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

               Cervical Adenocarcinoma       0.40      0.55      0.46      1494
                Chronic Kidney Disease       0.00      0.00      0.00       211
              Fetal Growth Restriction       0.50      0.11      0.18       168
                  Histiocytic Lymphoma       0.37      0.45      0.41      1496
Hypopharyngeal Squamous Cell Carcinoma       0.48      0.33      0.39       804
                        Ovarian Cancer       0.45      0.50      0.47      1451
                       Prostate Cancer       0.31      0.21      0.25      1480

                              accuracy                           0.40      7104
                             macro avg       0.36      0.31      0.31      7104
                          weighted avg       0.38      0.40      0.38      7104
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

               Cervical Adenocarcinoma       0.46      0.42      0.44      1494
                Chronic Kidney Disease       0.10      0.34      0.16       211
              Fetal Growth Restriction       0.31      0.65      0.42       168
                  Histiocytic Lymphoma       0.39      0.47      0.43      1496
Hypopharyngeal Squamous Cell Carcinoma       0.44      0.42      0.43       804
                        Ovarian Cancer       0.48      0.55      0.51      1451
                       Prostate Cancer       0.36      0.11      0.17      1480

                              accuracy                           0.40      7104
                             macro avg       0.36      0.42      0.36      7104
                          weighted avg       0.41      0.40      0.39      7104
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

               Cervical Adenocarcinoma       0.46      0.47      0.46      1494
                Chronic Kidney Disease       0.19      0.15      0.17       211
              Fetal Growth Restriction       0.49      0.42      0.45       168
                  Histiocytic Lymphoma       0.38      0.50      0.43      1496
Hypopharyngeal Squamous Cell Carcinoma       0.40      0.54      0.46       804
                        Ovarian Cancer       0.47      0.53      0.50      1451
                       Prostate Cancer       0.33      0.12      0.17      1480

                              accuracy                           0.41      7104
                             macro avg       0.39      0.39      0.38      7104
                          weighted avg       0.40      0.41      0.40      7104
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

