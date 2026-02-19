import miseval
from miseval import evaluate

def get_eval_metrices_outcome(y_true=None,y_pred=None,num_class=2):

    metrices= [
        "TP",                # True Positive
        "TN",                # True Negative
        "FP",                # False Positive
        "FN",                # False Negative
        "DSC",               # Dice Similarity Coefficient
        "IoU",               # Intersection over Union (Jaccard)
        "ACC",               # Accuracy
        "BACC",              # Balanced Accuracy
        "ARI",               # Adjusted Rand Index
        "AUC",               # Area Under Curve
        "KAP",               # Cohen's Kappa
        "Sensitivity",       # Recall / TPR
        "Specificity",       # TNR
        "PREC",              # Precision
        "VS",                # Volumetric Similarity
        "MCC",               # Matthews Correlation Coefficient
        "nMCC",              # Normalized MCC
        "aMCC"               # Absolute MCC
    ]
    results={m:evaluate(truth=y_true,pred=y_pred,metric=m,multi_class=True,n_classes=num_class) for m in metrices}
    return results