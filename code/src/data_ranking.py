import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def atp_ranking_last_date(utils_path, atp_mens_tour_folder, date_limit):
    atp_mens_tour = pd.DataFrame()
    for file_name in os.listdir(os.path.join(utils_path, atp_mens_tour_folder)):
        atp_mens_tour = pd.concat([atp_mens_tour, 
                        pd.read_excel(os.path.join(utils_path, atp_mens_tour_folder, file_name))])
    atp_mens_tour['Date'] = atp_mens_tour['Date'].dt.strftime('%Y-%m-%d')

    atp_w = atp_mens_tour[['Date','Winner', 'WRank', 'WPts']].rename(columns={'Winner':'Ply','WRank':'Rank', 'WPts':'Pts'})
    atp_l = atp_mens_tour[['Date','Loser', 'LRank', 'LPts']].rename(columns={'Loser':'Ply','LRank':'Rank', 'LPts':'Pts'})
    atp_a = pd.concat([atp_w, atp_l], axis=0).drop_duplicates()

    # labm = Last appearance before match (or in match)
    # faam = First appearance after match
    atp_labm = atp_a[(atp_a['Date'] <= date_limit)]
    atp_faam = atp_a[(atp_a['Date'] > date_limit)]

    atp_labm = atp_labm.groupby(['Ply'])[['Date']].max().reset_index().merge(atp_labm, on = ['Date','Ply'], how = 'left').drop(columns='Date')
    atp_faam = atp_faam.groupby(['Ply'])[['Date']].min().reset_index().merge(atp_faam, on = ['Date','Ply'], how = 'left').drop(columns='Date')
    atp_faam = atp_faam[~(atp_faam['Ply'].isin(atp_labm['Ply'].unique().tolist()))]

    atp_all  = pd.concat([atp_labm, atp_faam], axis=0)
    
    return atp_all