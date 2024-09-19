# Libraries
# =================
import os
import pandas as pd
import numpy as np

# Parameters
# =================
from src.parameters import Parameters

# Data
# =================
from src.data_clean import get_data, GetData
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
    df_dates_fail = []
    total_money_daily = Parameters.total_money / len(list_dates)
    df_class_all = pd.DataFrame()
    
    for f, date in enumerate(list_dates):
        try:
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
                # df_raw = pd.read_csv(os.path.join(Parameters.train_path, Parameters.nombre_archivo_raw))
                # df_raw = df_raw.rename(columns={'AvgW':'pl1_bet','AvgL':'pl2_bet'})
                # df_bronze, df_silver, df_gold = get_data(df_raw, 
                #                                         True, 
                #                                         Parameters.train_path, 
                #                                         Parameters.nombre_archivo_gold)
                
                list_files_scraped = [x for x in os.listdir(Parameters.daily_dump_path) if 'cleaned' in x]
                df_gold = pd.DataFrame()
                for file_scraped in list_files_scraped:
                    df_gold = pd.concat([df_gold, pd.read_csv(os.path.join(Parameters.daily_dump_path, file_scraped), sep = '|')])
                
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
                                                                                                atp_all
                                                                                                )
                
                df_fply_gold = df_fply_gold[(df_fply_gold["Result"] == 0)]

                # df_to_pred                              = pd.read_csv(os.path.join(results_path, file_data_scrapped_cleaned))
                df_to_pred                              = df_fply_gold.copy()
                X, y, preprocessor                      = data_to_class(df_to_pred)
                df_class, \
                df_class_consolidate, \
                reporte_clasificacion                   = fn_classification(df_to_pred, X, n_date, Parameters.results_path)

                df_class['date'] = ndate
            df_class_all = pd.concat([df_class_all, df_class], axis = 0)
            
            print(f'{date} runs correctly')
        except:
            df_dates_fail.append(date)
            print(f'{date} fail')
            
            
        if df_dates_fail:
            failed_dates_path = os.path.join(Parameters.results_path, "failed_dates.csv")
            df_failed_dates_all = pd.read_csv(failed_dates_path)
            df_failed_dates = pd.DataFrame(df_dates_fail, columns=['Failed_Date'])
            df_failed_dates_all = pd.concat([df_failed_dates_all, df_failed_dates], axis = 0).drop_duplicates()
            df_failed_dates.to_csv(failed_dates_path, index=False)
        
    syear, smonth, sday= str(start_date.year), \
            str(start_date.month).zfill(2), \
            str(start_date.day).zfill(2)
                
    s_date = f'{syear}-{smonth}-{sday}'
    sdate =  f'{syear}{smonth}{sday}'
                    
    eyear, emonth, eday= str(end_date.year), \
                    str(end_date.month).zfill(2), \
                    str(end_date.day).zfill(2)
    e_date =  f'{eyear}-{emonth}-{eday}'
    edate = f'{eyear}{emonth}{eday}'
    
    # Risk
    # =================
    df_risk = risk_analysis_montecarlo(df_class_all,
                                Parameters.risk_free_rate, 
                                Parameters.risk_tolerance, 
                                Parameters.total_money, 
                                Parameters.num_simulations, 
                                Parameters.num_bets,
                                Parameters.plots_path,
                                Parameters.results_path,
                                f'{Parameters.name_monte_carlo_dist}_{sdate}_{edate}.png',
                                f'{Parameters.name_ev_comparation}_{sdate}_{edate}.png',
                                f'{Parameters.name_sharpe_ratio_comparison}_{sdate}_{edate}.png',
                                f'{Parameters.file_betting_analysis}_{sdate}_{edate}.xlsx',
                                False)
    
    # Optimization
    # =================
    optimizer = Optimization(df_risk, Parameters.total_money, Parameters.max_loss_percentage, Parameters.min_percentage)
    df_risk_optimized = optimizer.optimize()
    df_risk_optimized = risk_analysis_montecarlo(df_risk_optimized,
                                                Parameters.risk_free_rate, 
                                                Parameters.risk_tolerance, 
                                                Parameters.total_money, 
                                                Parameters.num_simulations, 
                                                Parameters.num_bets,
                                                Parameters.plots_path,
                                                Parameters.results_path,
                                                f'{Parameters.name_monte_carlo_dist}_{sdate}_{edate}__Optimized.png',
                                                f'{Parameters.name_ev_comparation}_{sdate}_{edate}_Optimized.png',
                                                f'{Parameters.name_sharpe_ratio_comparison}_{sdate}_{edate}_Optimized.png',
                                                f'{Parameters.file_betting_analysis}_{sdate}_{edate}_Optimized.xlsx',
                                                True)
    
    
    df_all_optimized = df_risk_optimized.copy()
    num_apuestas = len(df_all_optimized)
    df_all_optimized['Money_earned'] = df_all_optimized['Money_to_Bet'] * df_all_optimized['Payout']
    df_winner = df_all_optimized[(df_all_optimized['Result'] == df_all_optimized['Class'])]
    df_winner.reset_index(drop=True,  inplace=True)
    alpha = df_all_optimized.groupby(['date'])[['Money_earned']].sum().rename(columns={'Money_earned':'Possible_Money_earned'})
    betha = df_all_optimized.groupby(['date'])[['match']].count().rename(columns={'match':'Num_bets'})
    gamma = df_winner.groupby(['date'])[['Money_earned']].sum()
    delta = df_winner.groupby(['date'])[['match']].count().rename(columns={'match':'Num_winner_bets'})

    iota = betha.join(delta).fillna(0)
    iota['Num_winner_bets'] = iota['Num_winner_bets'].astype(int)
    iota['perc_winner_bets'] = np.where(iota['Num_winner_bets'] == 0, 0,  iota['Num_winner_bets'] / iota['Num_bets'])
    iota['perc_winner_bets'] = round(iota['perc_winner_bets']*100,2).astype(str) + '%'

    iota = iota.join(alpha).join(gamma).fillna(0)
    iota['perc_Money_earned'] = np.where(iota['Money_earned'] == 0, 0,  iota['Money_earned'] / iota['Possible_Money_earned'])
    iota['perc_Money_earned'] = (iota['perc_Money_earned']*100).astype(int).astype(str) + '%'

    # iota['Money_earned_without_initial'] = iota['Money_earned'] - Parameters

    Money_earned = iota['Money_earned'].sum()
    num_apuestas_victoriosas = iota['Num_winner_bets'].sum()
    dinero_ganado_perdido = Parameters.total_money - Money_earned
    if dinero_ganado_perdido < 0:
        str_dinero_ganado_perdido = f'ganaste {dinero_ganado_perdido*-1}'
        str_patrimonio = f'Aumentando tu patrimonio un {(Parameters.total_money - Money_earned)/Parameters.total_money*-1}'
        
    else:
        str_dinero_ganado_perdido = f'perdiste {dinero_ganado_perdido}'
        str_patrimonio = f'Destruyendo tu patrimonio un {(Parameters.total_money - Money_earned)/Parameters.total_money}'
        
    print(f"Durante el {start_date.strftime('%Y-%m-%d')} y el {end_date.strftime('%Y-%m-%d')}, hiciste {num_apuestas} apuestas\n\
    de las cuales ganaste en {num_apuestas_victoriosas},\n\
    es decir, ganaste un {round((num_apuestas_victoriosas/num_apuestas)*100,1)}% de las veces \n\
    Obtuviste una ganancia {int(Money_earned)}\n\
    Tu inversión eran {Parameters.total_money}\n\
    Por ende, {str_dinero_ganado_perdido}\n\
    {str_patrimonio}\n\
    Tu patrimonio quedó en {Money_earned}\n\
        ")
    return iota
    # return df_games_of_day, df_games_acum, df_fply_gold, df_class, df_class_consolidate, df_risk, df_risk_optimized