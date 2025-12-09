"""
Utilities: train-test split, metrics (all built from scratch).
"""
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    return np.array([[tn, fp],[fn, tp]])

def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def roc_auc_score_simple(y_true, y_score, n_bins=100):
    # approximate AUC by computing TPR/FPR at thresholds
    thresholds = np.linspace(0, 1, n_bins)
    tpr = []
    fpr = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0]
        fn, tp = cm[1]
        tpr.append(tp / (tp + fn) if (tp + fn)>0 else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn)>0 else 0.0)
    # approximate integrate using trapezoid rule
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    order = np.argsort(fpr)
    return np.trapz(tpr[order], fpr[order])
