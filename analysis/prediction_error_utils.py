# analysis/prediction_error_utils.py
import pandas as pd
import numpy as np
import os

def calculate_and_add_prediction_errors(results_df, experiment_dir):
    """
    Calculates prediction error metrics (MAE, MSE, RMSE) for TT and NPS
    for each run and adds them to the results DataFrame.
    It now loads both Case_DB and the event log to get actual values.

    Args:
        results_df (pd.DataFrame): DataFrame of experiment results, must include a 'RUN' column.
        experiment_dir (str): Path to the main experiment directory.

    Returns:
        pd.DataFrame: The results_df DataFrame augmented with error metric columns.
    """
    error_metrics = [
        'mae_TT_pred_error', 'mse_TT_pred_error', 'rmse_TT_pred_error',
        'mae_NPS_pred_error', 'mse_NPS_pred_error', 'rmse_NPS_pred_error'
    ]
    for metric in error_metrics:
        if metric not in results_df.columns:
            results_df[metric] = np.nan

    for index, row in results_df.iterrows():
        run_id = row['RUN']
        case_db_filename = f"{run_id}_case_DB.csv"
        log_filename = f"{run_id}_log.csv"
        case_db_path = os.path.join(experiment_dir, str(run_id), case_db_filename)
        log_path = os.path.join(experiment_dir, str(run_id), log_filename)

        if not os.path.exists(case_db_path):
            print(f"Warning: Case_DB file not found for RUN {run_id} at {case_db_path}. Skipping error calculation.")
            continue
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found for RUN {run_id} at {log_path}. Skipping error calculation.")
            continue

        try:
            case_db_df = pd.read_csv(case_db_path)
            evlog_df = pd.read_csv(log_path)

            # --- Consolidate actuals from evlog to case_db_df ---
            # Get the final state for closed cases from the event log, which contains actuals
            closed_case_actuals = evlog_df[evlog_df['case_status'] == 'closed'][
                ['case_id', 'simulated_throughput_time', 'simulated_NPS']
            ].rename(columns={
                'simulated_throughput_time': 'actual_throughput_time',
                'simulated_NPS': 'actual_simulated_NPS'
            })
            
            # Merge these actuals into the case_db_df. Case_DB uses 'theta_idx' as its unique case identifier apparently.
            # The event log uses 'case_id'. We need to ensure these can be mapped.
            # Assuming 'theta_idx' in Case_DB corresponds to 'case_id' in evlog.
            # If Case_DB doesn't have a direct 'case_id' that matches evlog, this merge will fail or be incorrect.
            # Let's assume Case_DB's index or a specific column is the case_id.
            # For now, let's assume case_db_df.index can be used if 'case_id' or 'theta_idx' is the ID.
            # If Case_DB's 'theta_idx' IS the case_id:
            if 'theta_idx' in case_db_df.columns:
                merged_df = pd.merge(case_db_df, closed_case_actuals, left_on='theta_idx', right_on='case_id', how='left')
            else:
                 print(f"Warning: 'theta_idx' not in Case_DB for RUN {run_id}. Cannot map actuals from event log. Skipping error calculation.")
                 continue           

            # --- TT Prediction Error ---
            valid_tt_cases = merged_df[merged_df['actual_throughput_time'].notna() & merged_df['est_throughputtime'].notna()].copy()
            if not valid_tt_cases.empty:
                valid_tt_cases['TT_Pred_Error'] = valid_tt_cases['actual_throughput_time'] - valid_tt_cases['est_throughputtime']
                mae_tt = valid_tt_cases['TT_Pred_Error'].abs().mean()
                mse_tt = (valid_tt_cases['TT_Pred_Error']**2).mean()
                rmse_tt = np.sqrt(mse_tt)
                results_df.loc[index, 'mae_TT_pred_error'] = mae_tt
                results_df.loc[index, 'mse_TT_pred_error'] = mse_tt
                results_df.loc[index, 'rmse_TT_pred_error'] = rmse_tt
            
            # --- NPS Prediction Error ---
            valid_nps_cases = merged_df[merged_df['actual_simulated_NPS'].notna() & merged_df['est_NPS'].notna()].copy()
            if not valid_nps_cases.empty:
                valid_nps_cases['NPS_Pred_Error'] = valid_nps_cases['actual_simulated_NPS'] - valid_nps_cases['est_NPS']
                mae_nps = valid_nps_cases['NPS_Pred_Error'].abs().mean()
                mse_nps = (valid_nps_cases['NPS_Pred_Error']**2).mean()
                rmse_nps = np.sqrt(mse_nps)
                results_df.loc[index, 'mae_NPS_pred_error'] = mae_nps
                results_df.loc[index, 'mse_NPS_pred_error'] = mse_nps
                results_df.loc[index, 'rmse_NPS_pred_error'] = rmse_nps

        except pd.errors.EmptyDataError as e:
            print(f"Warning: File for RUN {run_id} ({e.filename}) is empty. Skipping error calculation.")
            continue
        except Exception as e:
            print(f"Error processing files for RUN {run_id}: {e}. Skipping error calculation for this run.")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue
            
    return results_df 