import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np

import os

import warnings
warnings.filterwarnings("ignore")

class ScrappingTennis():
    def __init__(self,year,month,day, req_headers, scraped_path):
        
        self.year = year
        self.month = month
        self.day = day
        self.n_date = f'{year}{month}{day}'
        self.req_headers = req_headers
        self.scraped_path = scraped_path
        
    def scrappy_match_results(self, file_match_result):
        df_atp_filled_full = pd.read_csv(f'{self.scraped_path}\\{file_match_result}', sep='|')
        
        list_by_row = []
        # for year in list_year:
        #     for month in list_month:
        #         for day in list_day:
        urls = f'https://www.tennisexplorer.com/results/?type=atp-single&year={self.year}&month={self.month}&day={self.day}'
        r1 = requests.get(urls, headers = self.req_headers, verify=False)
        print(r1)
        r1 = BeautifulSoup(r1.content, 'lxml')
        r1 = r1.findAll('tr')
        
        for i in range(len(r1)):
            try:
                td_element = r1[i].findAll('td')
                
                if td_element[0].get('class') == ['t-name']:
                    td_element = td_element[0]
                    dict_temp = {}
                    url_match = np.nan
                    pl1_bet = np.nan
                    pl2_bet = np.nan
                    a_tag = td_element.find('a')
                    a_text = a_tag.get_text(strip=True)
                    a_href = a_tag.get('href')
                    if 'player' in a_href:
                        dict_temp.update({'Date':self.n_date,
                                        'Location':np.nan,
                                        'Info':url_match,
                                        'Player':a_text,
                                        'url':a_href,
                                        'pl1_bet':pl1_bet,
                                        'pl2_bet':pl2_bet})
                    else:
                        dict_temp.update({'Date':self.n_date,
                                        'Location':a_text,
                                        'Info':url_match,
                                        'Player':'Field',
                                        'url':a_href,
                                        'pl1_bet':pl1_bet,
                                        'pl2_bet':pl2_bet})
                        
                    list_by_row.append(dict_temp)
                        
                else:
                    url_match = td_element[11].find('a').get('href')
                    pl1_bet = td_element[8].get_text(strip=True)
                    pl2_bet = td_element[9].get_text(strip=True)
                    td_element = td_element[1]
                    dict_temp = {}
                    a_tag = td_element.find('a')
                    a_text = a_tag.get_text(strip=True)
                    a_href = a_tag.get('href')
                    
                    if 'player' in a_href:
                        dict_temp.update({'Date':self.n_date,
                                        'Location':np.nan,
                                        'Info':url_match,
                                        'Player':a_text,
                                        'url':a_href,
                                        'pl1_bet':pl1_bet,
                                        'pl2_bet':pl2_bet})
                    else:
                        dict_temp.update({'Date':self.n_date,
                                        'Location':a_text,
                                        'Info':url_match,
                                        'Player':'Field',
                                        'url':a_href,
                                        'pl1_bet':pl1_bet,
                                        'pl2_bet':pl2_bet})
                        
                    list_by_row.append(dict_temp)
            except:
                continue
        
        df_atp = pd.DataFrame(list_by_row)
        df_atp['pl1_bet'] =        df_atp['pl1_bet'].replace("",np.nan)
        df_atp['pl2_bet'] =        df_atp['pl2_bet'].replace("",np.nan)
        
        df_atp['ply'] = np.where(df_atp['Player']=='Field', "Field", np.where(df_atp['Info'].isna(),'p2','p1'))
        df_atp_filled =  df_atp.fillna(method='ffill')

        not_link = df_atp_filled[df_atp_filled['ply']=='Field'].groupby(['Info'])[['Info']].count()
        not_link = not_link[(not_link['Info']>1)]
        not_link = not_link.index.unique().tolist()

        df_atp_filled = df_atp_filled[~(df_atp_filled['Info'].isin(not_link))]
        
        df_fields = df_atp_filled[(df_atp_filled['Player'] == 'Field')]
        df_fields['url'] = df_fields["url"].str.split("/",expand=True)[1]
        df_fields['url'] = 'https://www.tennisexplorer.com/'+ df_fields['url']
        
        df_games = df_atp_filled[~(df_atp_filled['Player'] == 'Field')]
        df_games['url'] = 'https://www.tennisexplorer.com'+ df_games['url']
        
        df_atp_filled_full = pd.concat([df_atp_filled_full, df_atp_filled], axis = 0)
        df_atp_filled_full.to_csv(f'{self.scraped_path}\\{file_match_result}', index=False, sep='|')
        
        return df_atp_filled, df_atp_filled_full, df_fields, df_games
    
    def scrappy_fields_info(self, df_fields, file_fields_desc):

        df_fields = df_fields[['Location','url']].drop_duplicates()
        
        df_field_surface_full = pd.read_csv(f'{self.scraped_path}\\{file_fields_desc}', sep='|')
        df_fields = df_fields[~(df_fields['url'].isin(df_field_surface_full['url'].unique().tolist()))]
        
        df_field_surface = pd.DataFrame()
        for url_fld in  df_fields['url'].unique().tolist():
            df_field_temp = df_fields[(df_fields['url'] == url_fld)]
            f1 = requests.get(url_fld, headers = self.req_headers, verify=False)
            f1 = BeautifulSoup(f1.content, 'lxml')
            f1 = f1.findAll('div', class_='box boxBasic lGray')[1].get_text()
            f1 = f1.split(",")[-2].replace(" ","")
            df_field_temp['Surface'] = f1
            df_field_surface = pd.concat([df_field_surface, df_field_temp], axis = 0)
            
        df_field_surface_full = pd.concat([df_field_surface_full, df_field_surface], axis = 0)
        df_field_surface_full.to_csv(f'{self.scraped_path}\\{file_fields_desc}', index=False, sep='|')
        
        return df_field_surface_full
    
    def scrappy_players_description(self, df_games, file_players_desc):

        df_players = df_games[['Player','url']].drop_duplicates()

        df_players_desc_full = pd.read_csv(f'{self.scraped_path}\\{file_players_desc}', sep='|')
        df_players = df_players[~(df_players['url'].isin(df_players_desc_full['url'].unique().tolist()))]
        
        df_players_desc = pd.DataFrame()
        # for ply_url in [df_players['url'].unique().tolist()[0]]:
        for ply_url in df_players['url'].unique().tolist():
            df_players_temp = df_players[(df_players['url'] == ply_url)]
            p0 = requests.get(ply_url, headers = self.req_headers, verify=False)
            p0 = BeautifulSoup(p0.content, 'lxml')
            p1 = p0.findAll('div', class_='box boxBasic lGray')[1]
            
            for i in range(len(p1.findAll('div', class_="date"))):
                # print(p1.findAll('div', class_="date")[i].get_text())
                if 'Country' in p1.findAll('div', class_="date")[i].get_text():
                    df_players_temp['flag'] = p1.findAll('div', class_="date")[i].get_text().split(":")[-1].replace(" ","")
                elif 'Age' in p1.findAll('div', class_="date")[i].get_text():
                    df_players_temp['age'] = p1.findAll('div', class_="date")[i].get_text().split(":")[-1].replace(" ","").split("(")[0].replace(')','')
                elif 'Plays' in p1.findAll('div', class_="date")[i].get_text():
                    df_players_temp['hand'] = p1.findAll('div', class_="date")[i].get_text().split(":")[-1].replace(" ","")
                elif 'Weight' in p1.findAll('div', class_="date")[i].get_text():
                    try:
                        df_players_temp['height'] = p1.findAll('div', class_="date")[i].get_text().split(":")[-1].replace(" ","").split("/")[0].replace('cm','')
                        df_players_temp['weight'] = p1.findAll('div', class_="date")[1].get_text().split(":")[-1].replace(" ","").split("/")[1].replace('kg','')
                    except:
                        df_players_temp['height'] = np.NAN
                        df_players_temp['weight'] = np.NAN
                        
            p2 = p0.findAll('div', class_='box lGray')[2]
            df_players_temp['year_pro'] = min([p2.findAll("td", class_ = 'year')[i].get_text() for i in range(len(p2.findAll("td", class_ = 'year')))])
            df_players_desc = pd.concat([df_players_desc, df_players_temp], axis = 0)
            
        df_players_desc_full = pd.concat([df_players_desc_full, df_players_desc], axis = 0)
        df_players_desc_full.to_csv(f'{self.scraped_path}\\{file_players_desc}', index=False, sep='|')
        
        return df_players_desc_full
    
    def scrappy_games(self, 
                      df_games, 
                      df_players_desc, 
                      df_field_surface, 
                      file_games, 
                      daily_dump_path):
        
        df_games_acum = pd.read_csv(f'{self.scraped_path}\\{file_games}', sep='|')
        df_games = df_games.merge(df_players_desc, on = ['Player','url'], how = 'left')
        
        df_games_of_day = pd.DataFrame()
        for info in df_games['Info'].unique().tolist():
            df_games_temp = df_games[(df_games['Info'] == info)]
            df_games_hz = df_games_temp[['Location','Info','pl1_bet','pl2_bet']].drop_duplicates()
            df_pl1_temp = df_games_temp[(df_games_temp['ply'] == 'p1')]
            df_pl2_temp = df_games_temp[(df_games_temp['ply'] == 'p2')]
            df_games_hz['Winner'] = df_pl1_temp['Player'].iloc[0]
            df_games_hz['pl1_flag'] = df_pl1_temp['flag'].iloc[0]
            df_games_hz['pl1_year_pro'] = df_pl1_temp['year_pro'].iloc[0]
            df_games_hz['pl1_weight'] = df_pl1_temp['weight'].iloc[0]
            df_games_hz['pl1_height'] = df_pl1_temp['height'].iloc[0]
            df_games_hz['pl1_hand'] = df_pl1_temp['hand'].iloc[0]
            df_games_hz['pl1_age'] = df_pl1_temp['age'].iloc[0]
            df_games_hz['Loser'] = df_pl2_temp['Player'].iloc[0]
            df_games_hz['pl2_flag'] = df_pl2_temp['flag'].iloc[0]
            df_games_hz['pl2_year_pro'] = df_pl2_temp['year_pro'].iloc[0]
            df_games_hz['pl2_weight'] = df_pl2_temp['weight'].iloc[0]
            df_games_hz['pl2_height'] = df_pl2_temp['height'].iloc[0]
            df_games_hz['pl2_hand'] = df_pl2_temp['hand'].iloc[0]
            df_games_hz['pl2_age'] = df_pl2_temp['age'].iloc[0]
            df_games_of_day = pd.concat([df_games_of_day, df_games_hz], axis = 0)
            
        df_games_of_day = df_games_of_day.merge(df_field_surface[['Location','Surface']], on = 'Location', how = 'left')
        df_games_of_day.drop(columns='Info', inplace=True)
        df_games_of_day['Date'] = self.n_date
        df_games_of_day.to_csv(os.path.join(daily_dump_path, f'df_games_{self.n_date}.csv'), index=False, sep='|')
        
        df_games_acum = pd.concat([df_games_acum, df_games_of_day], axis = 0)
        df_games_acum.to_csv(f'{self.scraped_path}\\{file_games}', index=False, sep='|')
        
        return df_games_of_day, df_games_acum
    
def scrapping_tennis_data(year,month,day,
                          req_headers,
                          scraped_path, 
                          file_match_result, 
                          file_fields_desc,
                          file_players_desc,
                          file_games, 
                          daily_dump_path):
    
    scrapping_tennis = ScrappingTennis(year,
                                       month,
                                       day, 
                                       req_headers,
                                       scraped_path)
    
    df_atp_filled, df_atp_filled_full, df_fields, df_games = scrapping_tennis.scrappy_match_results(file_match_result)
    
    df_field_surface_full = scrapping_tennis.scrappy_fields_info(df_fields, file_fields_desc)
    
    df_players_desc_full = scrapping_tennis.scrappy_players_description(df_games, file_players_desc)
    
    df_games_of_day, df_games_acum = scrapping_tennis.scrappy_games(df_games, 
                                                                    df_players_desc_full, 
                                                                    df_field_surface_full, 
                                                                    file_games, 
                                                                    daily_dump_path)
    
    return df_games_of_day, df_games_acum