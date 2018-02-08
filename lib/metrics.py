import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, f1_score, precision_score, recall_score


def calc_metrics(y_true, y_hat, max_steps=1000):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    metrics = {}
    metrics['Logloss'] = float(log_loss(y_true, y_hat))
    metrics['AUC'] = roc_auc_score(y_true, y_hat)
    metrics['F1'] = []
    metrics['Precision'] = []
    metrics['Recall'] = []
    for i in range(1, max_steps):
        threshold = float(i) / max_steps
        y_tmp = y_hat > threshold
        metrics['F1'].append(f1_score(y_true, y_tmp))
        metrics['Precision'].append(precision_score(y_true, y_tmp))
        metrics['Recall'].append(recall_score(y_true, y_tmp))
    max_idx = np.argmax(metrics['F1'])
    metrics['F1'] = metrics['F1'][max_idx]
    metrics['Precision'] = metrics['Precision'][max_idx]
    metrics['Recall'] = metrics['Recall'][max_idx]
    metrics['Threshold'] = float(max_idx + 1) / max_steps
    return metrics


def get_metrics(y_true, y_pred, target_labels):
    metrics = {}
    for i, label in enumerate(target_labels):
        metrics[label] = calc_metrics(np.array(y_true)[:, i], y_pred[:, i])
    keys = metrics[target_labels[0]].keys()
    metrics['Avg'] = {key: np.mean([metric[key] for label, metric in metrics.items()]) for key in keys}
    return metrics


def print_metrics(metrics):
    result_str = []
    for label, metric in metrics.items():
        metric_str = []
        for metric_name, value in metric.items():
            metric_str.append('{} = {}'.format(metric_name, value))
        metric_str = '\n\t'.join(metric_str)
        result_str.append('{}\n\t{}'.format(label, metric_str))
    return '\n'.join(result_str)
