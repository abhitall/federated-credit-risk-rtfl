import matplotlib.pyplot as plt
import numpy as np

def _sanitize_metric(metric, rounds, name="Metric"):
    # Convert None to np.nan, pad or truncate to match rounds
    metric = list(metric)
    if len(metric) < len(rounds):
        metric = metric + [np.nan] * (len(rounds) - len(metric))
    elif len(metric) > len(rounds):
        metric = metric[:len(rounds)]
    metric = [np.nan if v is None else v for v in metric]
    if len(metric) != len(rounds):
        print(f"[WARNING] {name} length {len(metric)} does not match rounds {len(rounds)}")
    return metric

def plot_global_metrics(rounds, accuracies, f1_scores, aucs):
    accuracies = _sanitize_metric(accuracies, rounds, "Accuracy")
    f1_scores = _sanitize_metric(f1_scores, rounds, "F1 Score")
    aucs = _sanitize_metric(aucs, rounds, "AUC")
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', label='Accuracy')
    plt.plot(rounds, f1_scores, marker='s', label='F1 Score')
    plt.plot(rounds, aucs, marker='^', label='AUC')
    plt.xlabel('Round')
    plt.ylabel('Metric Value')
    plt.title('Global Model Metrics per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_ebcd_stats(rounds, variances, kurtoses, skewnesses):
    variances = _sanitize_metric(variances, rounds, "Variance")
    kurtoses = _sanitize_metric(kurtoses, rounds, "Kurtosis")
    skewnesses = _sanitize_metric(skewnesses, rounds, "Skewness")
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, variances, label='Variance', marker='o')
    plt.plot(rounds, kurtoses, label='Kurtosis', marker='s')
    plt.plot(rounds, skewnesses, label='Skewness', marker='^')
    plt.xlabel('Round')
    plt.ylabel('EBCD Statistic')
    plt.title('EBCD Stats per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_server_status(rounds, statuses, coordinator_ids):
    statuses = _sanitize_metric(statuses, rounds, "Server Status")
    coordinator_ids = _sanitize_metric(coordinator_ids, rounds, "Coordinator ID")
    plt.figure(figsize=(10, 4))
    plt.step(rounds, statuses, where='mid', marker='o', label='Server Status')
    plt.scatter(rounds, coordinator_ids, marker='x', color='red', label='Coordinator ID')
    plt.xlabel('Round')
    plt.ylabel('Status/Coordinator')
    plt.title('Server Status and Coordinator per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_dp_noise_scale(rounds, dp_noise_scales):
    dp_noise_scales = _sanitize_metric(dp_noise_scales, rounds, "DP Noise Scale")
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, dp_noise_scales, marker='o', linestyle='-', color='tab:blue')
    plt.xlabel('Round')
    plt.ylabel('DP Noise Stddev (mean per round)')
    plt.title('DP Noise Scale per Round')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_agg_client_counts(rounds, agg_client_counts):
    agg_client_counts = _sanitize_metric(agg_client_counts, rounds, "Aggregated Clients")
    plt.figure(figsize=(10, 4))
    plt.bar(rounds, agg_client_counts, color='tab:green', alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('Aggregated Clients')
    plt.title('Number of Clients Aggregated per Round')
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()

def plot_zkip_failures(rounds, zkip_failures):
    zkip_failures = _sanitize_metric(zkip_failures, rounds, "ZKIP Proof Failures")
    plt.figure(figsize=(10, 4))
    plt.bar(rounds, zkip_failures, color='red', alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('ZKIP Proof Failures')
    plt.title('ZKIP Proof Failures per Round')
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()

def plot_delta_norm(rounds, delta_norms):
    delta_norms = _sanitize_metric(delta_norms, rounds, "Delta Norm")
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, delta_norms, marker='o', color='purple', linestyle='-')
    plt.xlabel('Round')
    plt.ylabel('Delta Norm (L2)')
    plt.title('Delta Norm per Round (DSS)')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_ebcd_alerts(rounds, ebcd_alerts):
    ebcd_alerts = _sanitize_metric(ebcd_alerts, rounds, "EBCD Alerts")
    plt.figure(figsize=(10, 4))
    plt.step(rounds, ebcd_alerts, where='mid', marker='o', color='orange')
    plt.xlabel('Round')
    plt.ylabel('EBCD Alerts (1=Alert, 0=No Alert)')
    plt.title('EBCD Alerts per Round')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_tcm_state_count(rounds, tcm_counts):
    tcm_counts = _sanitize_metric(tcm_counts, rounds, "TCM State Count")
    plt.figure(figsize=(10, 4))
    plt.bar(rounds, tcm_counts, color='green', alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('TCM State Count')
    plt.title('TCM State Count per Round')
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()

def plot_per_client_update_norms(rounds, client_update_norms, client_ids):
    plt.figure(figsize=(12, 6))
    for idx, client_id in enumerate(client_ids):
        y = [client_update_norms[r][idx] if r < len(client_update_norms) and idx < len(client_update_norms[r]) else np.nan for r in range(len(rounds))]
        y = _sanitize_metric(y, rounds, f"Update Norm {client_id}")
        plt.plot(rounds, y, marker='o', label=f'Client {client_id}')
    plt.xlabel('Round')
    plt.ylabel('Update Norm (L2)')
    plt.title('Per-Client Local Update Norms per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_per_client_ebcd_stats(rounds, client_ebcd_stats, client_ids, stat_name):
    plt.figure(figsize=(12, 6))
    # client_ebcd_stats: [client][round] shape
    for idx, client_id in enumerate(client_ids):
        y = client_ebcd_stats[idx]
        if len(y) < len(rounds):
            y = list(y) + [np.nan]*(len(rounds)-len(y))
        y = [np.nan if v is None else v for v in y]
        plt.plot(rounds, y, marker='o', label=f'Client {client_id}')
    plt.xlabel('Round')
    plt.ylabel(stat_name)
    plt.title(f'Per-Client {stat_name} per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_per_client_zkip_status(rounds, client_zkip_status, client_ids):
    plt.figure(figsize=(12, 6))
    for idx, client_id in enumerate(client_ids):
        y = [client_zkip_status[r][idx] if r < len(client_zkip_status) and idx < len(client_zkip_status[r]) else np.nan for r in range(len(rounds))]
        y = _sanitize_metric(y, rounds, f"ZKIP Status {client_id}")
        plt.step(rounds, y, where='mid', marker='o', label=f'Client {client_id}')
    plt.xlabel('Round')
    plt.ylabel('ZKIP Proof Status (1=OK, 0=Fail)')
    plt.title('Per-Client ZKIP Proof Status per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_early_stopping_metric(rounds, best_metrics, metric_name="Best Validation Metric", ylabel=None):
    best_metrics = _sanitize_metric(best_metrics, rounds, metric_name)
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, best_metrics, marker='o', color='teal')
    plt.xlabel('Round')
    plt.ylabel(ylabel or metric_name)
    plt.title(f'Early Stopping: {metric_name} per Round')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
