import os

class Parameters(object):
    # Rooths
    root_path                   = os.path.abspath(os.path.join("../"+os.path.dirname('__file__')))
    input_path                  = os.path.join(root_path, 'data', 'input')
    scraped_path                = os.path.join(root_path, 'data', 'input', 'scraped')
    train_path                  = os.path.join(root_path, 'data', 'input', 'train')
    utils_path                  = os.path.join(root_path, 'data', 'input', 'utils')
    daily_dump_path             = os.path.join(root_path, 'data', 'input', 'daily_dump')
    output_path                 = os.path.join(root_path, 'data', 'output')
    results_path                = os.path.join(root_path, 'data', 'output', 'results')
    models_path                 = os.path.join(root_path, 'data', 'output', 'models')
    plots_path                  = os.path.join(root_path, 'data', 'output', 'plots')
    
    # Files Train
    nombre_archivo_raw          = "tennis_data.csv"
    nombre_archivo_gold         = "tennis_data_cleaned.xlsx"
    # Files Scrapping
    req_headers =               {
                                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                                'accept-encoding': 'gzip, deflate, br',
                                'accept-language': 'en-US,en;q=0.8',
                                'upgrade-insecure-requests': '1',
                                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
                                }
    file_paises                 = 'paises.csv'
    file_match_result           = 'df_match_results.csv'
    file_fields_desc            = 'df_fields.csv'
    file_players_desc           = 'df_players_desc.csv'
    file_games                  = 'df_games.csv'
    # file_data_scrapped          = 'df_games_202203_03.xlsx'
    # file_data_scrapped_cleaned  = 'data_cleaned_202203.xlsx'
    
    # Files & Data Risk Analysis
    name_monte_carlo_dist       = 'monte_carlo_distribution'
    name_ev_comparation         = 'EV_comparison'
    name_sharpe_ratio_comparison= 'sharpe_ratio_comparison'
    file_betting_analysis       = 'betting_analysis'
    
    # Parámetro de nivel de riesgo (0 = muy conservador, 1 = muy agresivo)
    risk_tolerance              = 0.8       # Puedes ajustar este valor entre 0 y 1
    risk_free_rate              = 1.01      # Supongamos una tasa libre de riesgo del 1%
    # Configura los parámetros para la simulación
    num_simulations             = 1000      # Número de simulaciones
    num_bets                    = 1         # Número de apuestas por simulación
    total_money                 = 100
    
    