# Libraries
# =================
import os
import pandas as pd

# Parameters
# =================
from src.parameters import Parameters

# Data
# =================
from src.data_clean import get_data
from src.data_ranking import atp_ranking_last_date
from src.data_scrapped_clean import scrapped_data_organized

# Scrapping
# =================
from src.scrapping import scrapping_tennis_data


# Classification
# =================
from src.class_preprocessing import data_to_class
from src.class_classification import train_classification, fn_classification

# Risk
# =================
from src.risk_analysis import risk_analysis_montecarlo

# Optimization
# =================
from src.optimization import Optimization

# Warnings
# =================
import warnings
warnings.filterwarnings("ignore")

def process(train, start_date, end_date):
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    list_dates = pd.date_range(start=start_date, end=end_date).tolist()
    
    total_money_daily = Parameters.total_money / len(list_dates)
    

    for f, date in enumerate(list_dates):

        year, month, day= str(list_dates[f].year), \
                        str(list_dates[f].month).zfill(2), \
                        str(list_dates[f].day).zfill(2)
    
        n_date = f'{year}-{month}-{day}'
        ndate = f'{year}{month}{day}'
        
        print("="*17)               
        print(f'{year}-{month}-{day}')
        
        if train == True:
            # Train
            # =================
            df_raw = pd.read_csv(os.path.join(Parameters.train_path, Parameters.nombre_archivo_raw))
            df_raw = df_raw.rename(columns={'AvgW':'pl1_bet','AvgL':'pl2_bet'})
            df_bronze, df_silver, df_gold = get_data(df_raw, 
                                                    True, 
                                                    Parameters.train_path, 
                                                    Parameters.nombre_archivo_gold)

            X, y, preprocessor = data_to_class(df_gold)
            train_classification(X, y, preprocessor, Parameters.results_path, Parameters.models_path)
        
        else:
            df_games_of_day, df_games_acum = scrapping_tennis_data(year,month,day,
                                                        Parameters.req_headers,
                                                        Parameters.scraped_path, 
                                                        Parameters.file_match_result, 
                                                        Parameters.file_fields_desc,
                                                        Parameters.file_players_desc,
                                                        Parameters.file_games, 
                                                        Parameters.daily_dump_path)
            # Pred
            # =================

            atp_all = atp_ranking_last_date(Parameters.utils_path, "atp_mens_tour", n_date)
            df_fply, df_fply_bronze, df_fply_silver, df_fply_gold = scrapped_data_organized(df_games_of_day,
                                                                                            Parameters.utils_path, 
                                                                                            Parameters.daily_dump_path,
                                                                                            Parameters.file_paises,
                                                                                            f'df_games_cleaned_{ndate}.csv',
                                                                                            atp_all)

            # df_to_pred                              = pd.read_csv(os.path.join(results_path, file_data_scrapped_cleaned))
            df_to_pred                              = df_fply_gold.copy()
            X, y, preprocessor                      = data_to_class(df_to_pred)
            df_class, \
            df_class_consolidate, \
            reporte_clasificacion                   = fn_classification(df_to_pred, X, n_date, Parameters.results_path)
        
        # Risk
        # =================
        df_risk          = risk_analysis_montecarlo(df_class,
                                                    Parameters.risk_free_rate, 
                                                    Parameters.risk_tolerance, 
                                                    total_money_daily, 
                                                    Parameters.num_simulations, 
                                                    Parameters.num_bets,
                                                    Parameters.plots_path,
                                                    Parameters.results_path,
                                                    f'{Parameters.name_monte_carlo_dist}_{ndate}.png',
                                                    f'{Parameters.name_ev_comparation}_{ndate}.png',
                                                    f'{Parameters.name_sharpe_ratio_comparison}_{ndate}.png',
                                                    f'{Parameters.file_betting_analysis}_{ndate}.xlsx',
                                                    False)
        
        # Optimization
        # =================
        optimizer = Optimization(df_risk, total_money_daily, Parameters.max_loss_percentage, Parameters.min_percentage)
        df_risk_optimized = optimizer.optimize()
        df_risk_optimized = risk_analysis_montecarlo(df_risk_optimized,
                                                    Parameters.risk_free_rate, 
                                                    Parameters.risk_tolerance, 
                                                    total_money_daily, 
                                                    Parameters.num_simulations, 
                                                    Parameters.num_bets,
                                                    Parameters.plots_path,
                                                    Parameters.results_path,
                                                    f'{Parameters.name_monte_carlo_dist}_{ndate}__Optimized.png',
                                                    f'{Parameters.name_ev_comparation}_{ndate}_Optimized.png',
                                                    f'{Parameters.name_sharpe_ratio_comparison}_{ndate}_Optimized.png',
                                                    f'{Parameters.file_betting_analysis}_{ndate}_Optimized.xlsx',
                                                    True)
    
    return df_games_of_day, df_games_acum, df_fply_gold, df_class, df_class_consolidate, df_risk, df_risk_optimized