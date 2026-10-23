# OpsInsight model results

## Final models (test period = newest 20% of the data, never used for choosing)

| Prediction | Model | Test AUC | Problem rate (all) | Problem rate (top 10% flagged) | Lift | Test rows |
| --- | --- | --- | --- | --- | --- | --- |
| Support tickets: SLA breach at opening | LightGBM + logistic regression (average of both risks) | 0.791 | 22.5% | 61.3% | 2.73x | 4,929 |
| Sales orders: late delivery at approval | LightGBM + logistic regression (average of both risks) | 0.732 | 5.3% | 10.6% | 1.99x | 19,204 |

Lift = how many times more problems a manager finds by checking the model's top 10% instead of a random 10%.

## Model comparison: Support tickets: SLA breach at opening

| model | validation_auc | test_auc | test_top10_precision | picked |
| --- | --- | --- | --- | --- |
| LightGBM, default settings | 0.762 | 0.777 | 0.521 | yes |
| Logistic regression | 0.759 | 0.788 | 0.602 |  |
| LightGBM, bigger trees | 0.758 | 0.768 | 0.515 |  |
| LightGBM, cautious (current) | 0.753 | 0.771 | 0.517 |  |
| LightGBM, slower + more trees | 0.751 | 0.764 | 0.489 |  |
| LightGBM, very small trees | 0.742 | 0.761 | 0.489 |  |

## Model comparison: Sales orders: late delivery at approval

| model | validation_auc | test_auc | test_top10_precision | picked |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.744 | 0.737 | 0.11 | yes |
| LightGBM, very small trees | 0.7 | 0.679 | 0.097 |  |
| LightGBM, cautious (current) | 0.695 | 0.715 | 0.103 |  |
| LightGBM, slower + more trees | 0.694 | 0.687 | 0.097 |  |
| LightGBM, bigger trees | 0.688 | 0.72 | 0.109 |  |
| LightGBM, default settings | 0.668 | 0.684 | 0.116 |  |

`picked` = best single model on validation AUC (newest 20% of the training period). The final model averages LightGBM and logistic regression, which beat both alone on incident validation data (0.783).
