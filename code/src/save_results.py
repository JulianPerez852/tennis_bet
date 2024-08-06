import pandas as pd
import os
import pickle

def save_model(model, path_models, model_name):

    last_version = 0.0
    last_model = ''

    list_files = os.listdir(path_models)
    list_files = [file for file in list_files if model_name in file]
    

    for file in list_files:
        file = file.replace('.pkl', '')
        file = file.replace('.h5', '')
        try:
            file_parts = file.split('-')
            if len(file_parts) > 1:
                saved_model = file.split('-')[0]
                saved_version = float(file_parts[1])
            else:
                raise ValueError("Filename format is incorrect, should contain a version number")
        except ValueError as e:
            print(f"Error processing file {file}: {e}")

        if saved_version > last_version:
            last_version = saved_version
            last_model = saved_model

    if last_model == model_name:
        last_version = float(round(last_version + 0.1,1))
    else:
        last_version = float(round(last_version + 1))

    path_model = os.path.join(
    path_models,
    model_name +
    '-' +
    str(last_version) +
    ".pkl")
    pickle.dump(model, open(path_model, 'wb'))

    return last_version

def save_winner_metrics(winner_metrics, results_path, stage):
    if stage == 'classification':
        name_file = 'winner_metrics_class.xlsx'
    elif stage == 'regression':
        name_file = 'winner_metrics_pred.xlsx'
        
    winner_metrics_temp = pd.read_excel(os.path.join(results_path, name_file))
    winner_metrics_temp = pd.concat([winner_metrics_temp, pd.DataFrame(winner_metrics)], axis = 0)
    winner_metrics_temp.to_excel(os.path.join(results_path, name_file), index=False)