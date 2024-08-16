import os
import pandas as pd

from src.data_clean import GetData

import warnings
warnings.filterwarnings("ignore")

def scrapped_data_organized(df_scrapped,
                            utils_path, 
                            path_to_save,
                            file_paises, 
                            file_data_scrapped_cleaned, 
                            atp_all):
# def scrapped_data_organized(utils_path, file_paises, results_path, file_scrapped, file_data_scrapped_cleaned, atp_all):

    # df_scrapped = pd.read_excel(os.path.join(results_path, file_scrapped)).drop(columns=['Info'])
    
    df_fply = df_scrapped.merge(atp_all, left_on=['Winner'], right_on=['Ply'], how = 'left')
    df_fply.rename(columns={'Rank':'WRank', 'Pts':'WPts'}, inplace=True)
    df_fply = df_fply.merge(atp_all, left_on=['Loser'], right_on=['Ply'], how = 'left')
    df_fply.rename(columns={'Rank':'LRank', 'Pts':'LPts'}, inplace=True)
    df_fply.drop(columns=['Ply_x', 'Ply_y'], inplace=True)
    iso3 = pd.read_csv(os.path.join(utils_path, file_paises))[['ENGLISH','ISO3']]
    iso3['ENGLISH_NoSpace'] = iso3['ENGLISH'].str.replace(" ","")
    iso3 = pd.concat([iso3[['ENGLISH','ISO3']],
                    iso3[['ENGLISH_NoSpace','ISO3']].rename(columns={'ENGLISH_NoSpace':'ENGLISH'})],
                    axis = 0).drop_duplicates()
    df_fply = df_fply.merge(iso3, left_on=['pl1_flag'], right_on=['ENGLISH'], how='left').drop(columns=['ENGLISH','pl1_flag']).rename(columns={'ISO3':'pl1_flag'})
    df_fply = df_fply.merge(iso3, left_on=['pl2_flag'], right_on=['ENGLISH'], how='left').drop(columns=['ENGLISH','pl2_flag']).rename(columns={'ISO3':'pl2_flag'})
    df_fply['pl1_hand'] = df_fply['pl1_hand'].str.capitalize() + "-Handed"
    df_fply['pl2_hand'] = df_fply['pl2_hand'].str.capitalize() + "-Handed"
    
    df_fply_bronze = GetData().data_raw(df_fply)
    df_fply_silver = GetData().data_without_atipics(df_fply_bronze)
    df_fply_gold = GetData().data_p1vsp2(False, df_fply_silver)
    df_fply_gold = GetData().save_data_cleaned(path_to_save, file_data_scrapped_cleaned, df_fply_gold)
    
    return df_fply, df_fply_bronze, df_fply_silver, df_fply_gold