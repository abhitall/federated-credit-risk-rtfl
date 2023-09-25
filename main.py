import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from data_utils import load_and_preprocess_data, split_data_for_clients
from fl_client import FLClient
from fl_server import FLServer
import charting as charts
from sklearn.model_selection import train_test_split

NUM_CLIENTS = 5
NUM_ROUNDS = 10
CLIENT_EPOCHS = 3
BASE_LEARNING_RATE = 0.01
DP_EPSILON = 1.0
DP_DELTA = 1e-5
DP_L2_NORM_CLIP = 1.0
EARLYSTOP_PATIENCE = 3

def main():
    print("Starting RTFL Simulation with Differential Privacy...")
    print(f"DP Parameters: Epsilon={DP_EPSILON}, Delta={DP_DELTA}, L2_Norm_Clip={DP_L2_NORM_CLIP}")
    X_train_full, y_train_full, X_test, y_test, feature_names = load_and_preprocess_data()
    if X_train_full is None:
        print("Failed to load data. Exiting.")
        return
    num_features = X_train_full.shape[1]
    print(f"Data loaded: {X_train_full.shape[0]} train samples, {X_test.shape[0]} test samples. Num features: {num_features}")
    client_datasets = split_data_for_clients(X_train_full, y_train_full, NUM_CLIENTS)
    clients = []
    client_ids = [f"client_{i}" for i in range(NUM_CLIENTS)]
    # Split each client's data into train/val
    client_val_sets = []
    for i in range(NUM_CLIENTS):
        X_c, y_c = client_datasets[i]
        if X_c.shape[0] > 5:
            X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(X_c, y_c, test_size=0.2, random_state=i)
        else:
            X_c_train, y_c_train = X_c, y_c
            X_c_val, y_c_val = None, None
        clients.append(FLClient(client_ids[i], X_c_train, y_c_train, num_features, 
                                learning_rate=BASE_LEARNING_RATE,
                                dp_epsilon=DP_EPSILON, 
                                dp_delta=DP_DELTA, 
                                dp_l2_norm_clip=DP_L2_NORM_CLIP,
                                random_state=i,
                                X_val=X_c_val, y_val=y_c_val, earlystop_patience=EARLYSTOP_PATIENCE))
        client_val_sets.append((X_c_val, y_c_val))
    # For server validation, concatenate all client val sets
    X_val_server = np.concatenate([v[0] for v in client_val_sets if v[0] is not None]) if any(v[0] is not None for v in client_val_sets) else None
    y_val_server = np.concatenate([v[1] for v in client_val_sets if v[1] is not None]) if any(v[1] is not None for v in client_val_sets) else None
    server_id = "main_server"
    server = FLServer(server_id, client_ids, num_features, X_val=X_val_server, y_val=y_val_server, earlystop_patience=EARLYSTOP_PATIENCE)
    initial_params_for_ebcd = [client.model_parameters() for client in clients if client.X_train.shape[0] > 0]
    if initial_params_for_ebcd:
        server.ebcd.establish_baseline(initial_params_for_ebcd)

    # --- Metrics storage ---
    rounds = []
    accuracies = []
    f1_scores = []
    aucs = []
    ebcd_variances = []
    ebcd_kurtoses = []
    ebcd_skewnesses = []
    server_statuses = []
    coordinator_ids = []
    dp_noise_scales = []
    agg_client_counts = []
    zkip_failures = []
    delta_norms = []
    ebcd_alerts = []
    tcm_counts = []
    # Per-client metrics: [round][client]
    per_client_update_norms = []
    per_client_ebcd_stats = []
    per_client_zkip_status = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
        start_time = time.time()
        global_params_for_round = server.get_global_model_parameters_for_clients()
        if global_params_for_round is None:
            print(f"Round {round_num}: Critical - Could not get model from coordinator. Attempting TCM recovery.")
            latest_tcm_entry = server.tcm.get_latest_state_info()
            if latest_tcm_entry:
                recovered_params, _ = server.tcm.recover_state_by_round(latest_tcm_entry[1]) 
                if recovered_params:
                    server.global_model_parameters = recovered_params
                    global_params_for_round = server.get_global_model_parameters_for_clients() 
                    print(f"Round {round_num}: Recovered model from TCM (Round {latest_tcm_entry[1]}).")
                else:
                    print(f"Round {round_num}: TCM recovery failed. Skipping round.")
                    continue
            else:
                 print(f"Round {round_num}: No TCM entries to recover from. Skipping round.")
                 continue
        client_deltas_with_proofs = []
        client_data_sizes_for_agg = []
        active_clients_this_round_ids = []
        # --- DP noise scale for this round (all clients, average) ---
        round_noise_scales = []
        round_zkip_failures = 0
        round_delta_norm = 0.0
        round_ebcd_alert = 0
        round_client_update_norms = []
        round_client_ebcd_stats = []
        round_client_zkip_status = []
        for client in clients:
            if client.simulate_failure(probability=0.15): 
                round_client_update_norms.append(None)
                round_client_ebcd_stats.append((None, None, None))
                round_client_zkip_status.append(None)
                continue 
            active_clients_this_round_ids.append(client.client_id)
            client.set_global_model_parameters(global_params_for_round)
            delta_weights, proof = client.train(epochs=CLIENT_EPOCHS)
            # ZKIP proof check (simulate server-side check)
            if delta_weights is not None and proof is not None:
                from zkip import ZeroKnowledgeIntegrityProofs
                zkip = ZeroKnowledgeIntegrityProofs()
                zkip_status = zkip.verify_proof(delta_weights, proof)
                round_client_zkip_status.append(zkip_status)
                # Delta norm (L2)
                norm = 0.0
                for v in delta_weights.values():
                    norm += np.linalg.norm(v.flatten())**2
                update_norm = np.sqrt(norm)
                round_client_update_norms.append(update_norm)
                # Per-client EBCD stats (variance, kurtosis, skewness) for delta_weights['coef_']
                if 'coef_' in delta_weights and hasattr(delta_weights['coef_'], 'flatten'):
                    flat = delta_weights['coef_'].flatten()
                    v = np.var(flat)
                    k = kurtosis(flat, fisher=True)
                    s = skew(flat)
                    round_client_ebcd_stats.append((v, k, s))
                else:
                    round_client_ebcd_stats.append((None, None, None))
                if not zkip_status:
                    round_zkip_failures += 1
                round_delta_norm += update_norm
                client_deltas_with_proofs.append((delta_weights, proof, client.client_id))
                client_data_sizes_for_agg.append(len(client.y_train))
                # DP noise scale: (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
                if client.dp_epsilon > 0:
                    noise_stddev = (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
                else:
                    noise_stddev = 0.0
                round_noise_scales.append(noise_stddev)
            else:
                round_client_update_norms.append(None)
                round_client_ebcd_stats.append((None, None, None))
                round_client_zkip_status.append(False)
        # Average delta norm for the round
        if len([n for n in round_client_update_norms if n is not None]) > 0:
            round_delta_norm = round_delta_norm / len([n for n in round_client_update_norms if n is not None])
        else:
            round_delta_norm = 0.0
        per_client_update_norms.append(round_client_update_norms)
        per_client_ebcd_stats.append(round_client_ebcd_stats)
        per_client_zkip_status.append(round_client_zkip_status)
        server.arrp.update_active_clients(active_clients_this_round_ids)
        aggregation_success, aggregated_from_clients = server.aggregate_model_deltas(client_deltas_with_proofs, client_data_sizes_for_agg)
        # EBCD alert (after aggregation)
        ebcd_alert = 1 if server.ebcd.check_for_corruption(server.global_model_parameters) else 0
        round_ebcd_alert = ebcd_alert
        server_state_details = {
            'arrp_status': server.arrp.status.name,
            'current_coordinator': server.arrp.get_current_coordinator_id(),
            'aggregation_successful': aggregation_success,
            'aggregated_from_clients_count': len(aggregated_from_clients),
            'dp_epsilon': DP_EPSILON,
            'dp_l2_norm_clip': DP_L2_NORM_CLIP
        }
        client_updates_summary = {cid: "OK_DP" for cid in aggregated_from_clients} 
        for cid in active_clients_this_round_ids:
            if cid not in client_updates_summary: client_updates_summary[cid] = "NO_UPDATE_OR_FAULTY"
        server.tcm.record_state(round_num, server.global_model_parameters, server_state_details, client_updates_summary)
        metrics = server.evaluate_global_model(X_test, y_test, round_num)
        print(f"Round {round_num} Eval (DP): Acc={metrics.get('accuracy',0):.3f}, F1={metrics.get('f1_score',0):.3f}, AUC={metrics.get('auc_roc',0):.3f}")
        round_duration = time.time() - start_time
        print(f"Round {round_num} duration: {round_duration:.2f}s. Coordinator: {server.arrp.get_current_coordinator_id()}")
        # --- Metrics collection ---
        rounds.append(round_num)
        accuracies.append(metrics.get('accuracy', 0))
        f1_scores.append(metrics.get('f1_score', 0))
        aucs.append(metrics.get('auc_roc', 0))
        agg_client_counts.append(len(aggregated_from_clients))
        server_statuses.append(server.arrp.status.name)
        coordinator_ids.append(server.arrp.get_current_coordinator_id())
        dp_noise_scales.append(np.mean(round_noise_scales) if round_noise_scales else 0.0)
        zkip_failures.append(round_zkip_failures)
        delta_norms.append(round_delta_norm)
        ebcd_alerts.append(round_ebcd_alert)
        tcm_counts.append(len(server.tcm.manifold_log))
        # EBCD stats (variance, kurtosis, skewness) for global model coef_
        coef = server.global_model_parameters['coef_']
        if coef is not None and hasattr(coef, 'flatten'):
            flat = coef.flatten()
            ebcd_variances.append(np.var(flat))
            ebcd_kurtoses.append(kurtosis(flat, fisher=True))
            ebcd_skewnesses.append(skew(flat))
        else:
            ebcd_variances.append(0)
            ebcd_kurtoses.append(0)
            ebcd_skewnesses.append(0)
    print("\n--- RTFL Simulation with DP Complete ---")
    print(f"Total states recorded by TCM: {len(server.tcm.manifold_log)}")
    if NUM_ROUNDS >= 3:
        target_recovery_round = NUM_ROUNDS // 2
        print(f"\nAttempting to recover model state from TCM for round {target_recovery_round}...")
        recovered_model_params, rec_info = server.tcm.recover_state_by_round(target_recovery_round)
        if recovered_model_params:
            print(f"Successfully recovered (noisy) model parameters for round {target_recovery_round}.")
        else:
            print(f"Failed to recover model for round {target_recovery_round}.")
    # --- Save per-client metrics as .npy for research ---
    np.save('per_client_update_norms.npy', np.array(per_client_update_norms, dtype=object))
    np.save('per_client_ebcd_stats.npy', np.array(per_client_ebcd_stats, dtype=object))
    np.save('per_client_zkip_status.npy', np.array(per_client_zkip_status, dtype=object))
    # --- Save plots for research ---
    charts.plot_global_metrics(rounds, accuracies, f1_scores, aucs)
    plt.savefig('global_metrics.png')
    plt.close()
    charts.plot_ebcd_stats(rounds, ebcd_variances, ebcd_kurtoses, ebcd_skewnesses)
    plt.savefig('ebcd_stats.png')
    plt.close()
    # For server status, encode status as int for plotting
    status_map = {s: i for i, s in enumerate(sorted(set(server_statuses)))}
    status_ints = [status_map[s] for s in server_statuses]
    charts.plot_server_status(rounds, status_ints, coordinator_ids)
    plt.savefig('server_status.png')
    plt.close()
    charts.plot_dp_noise_scale(rounds, dp_noise_scales)
    plt.savefig('dp_noise_scale.png')
    plt.close()
    charts.plot_agg_client_counts(rounds, agg_client_counts)
    plt.savefig('agg_client_counts.png')
    plt.close()
    charts.plot_zkip_failures(rounds, zkip_failures)
    plt.savefig('zkip_failures.png')
    plt.close()
    charts.plot_delta_norm(rounds, delta_norms)
    plt.savefig('delta_norm.png')
    plt.close()
    charts.plot_ebcd_alerts(rounds, ebcd_alerts)
    plt.savefig('ebcd_alerts.png')
    plt.close()
    charts.plot_tcm_state_count(rounds, tcm_counts)
    plt.savefig('tcm_state_count.png')
    plt.close()
    # --- Early stopping chart for server ---
    if hasattr(server, 'earlystop') and hasattr(server.earlystop, 'best_metric'):
        best_accs = []
        for _ in rounds:
            best_accs.append(server.earlystop.best_metric if server.earlystop.best_metric != -float('inf') else np.nan)
        charts.plot_early_stopping_metric(rounds, best_accs, metric_name="Best Validation Accuracy", ylabel="Accuracy")
        plt.savefig('earlystop_server_best_val_acc.png')
        plt.close()
    # --- Per-client plots ---
    charts.plot_per_client_update_norms(rounds, per_client_update_norms, client_ids)
    plt.savefig('per_client_update_norms.png')
    plt.close()
    # Per-client EBCD stats: variance, kurtosis, skewness
    for stat_idx, stat_name in enumerate(['Variance', 'Kurtosis', 'Skewness']):
        # Build [client][round] shape for plotting
        stat_data = [[None for _ in range(len(rounds))] for _ in range(len(client_ids))]
        for r in range(len(rounds)):
            for c in range(len(client_ids)):
                try:
                    val = per_client_ebcd_stats[r][c][stat_idx] if per_client_ebcd_stats[r][c] is not None else None
                except (IndexError, TypeError):
                    val = None
                stat_data[c][r] = val
        charts.plot_per_client_ebcd_stats(rounds, stat_data, client_ids, stat_name)
        plt.savefig(f'per_client_ebcd_{stat_name.lower()}.png')
        plt.close()
    charts.plot_per_client_zkip_status(rounds, per_client_zkip_status, client_ids)
    plt.savefig('per_client_zkip_status.png')
    plt.close()

if __name__ == "__main__":
    main()
