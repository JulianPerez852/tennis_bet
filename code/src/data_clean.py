import os
import pandas as pd
from ydata_profiling import ProfileReport

class GetData():
    def data_raw(self, df):
        
        # df = pd.read_csv(os.path.join(input_path, nombre_archivo_raw))
        
        # Cargando el dataset y generando un reporte de los datos automático con YData profiling
        # profile = ProfileReport(df, title="Profiling Report")
        # profile.to_file("your_report.html")
        
        columns_wihtbet = ["Date"
            , "Surface"
            , "Winner"
            , "WPts"
            , "LPts"
            , "pl1_flag"
            , "pl1_year_pro"
            , "pl1_weight"
            , "pl1_height"
            , "pl1_hand"
            , "pl1_bet"
            , "Loser"
            , "pl2_flag"
            , "pl2_year_pro"
            , "pl2_weight"
            , "pl2_height"
            , "pl2_hand"
            , "pl2_bet"
            ]
        
        df = df[columns_wihtbet]

        df.dropna(inplace=True)
        
        df['Result'] = 0
        
        df = df.rename(columns={'Winner':'pl1',
                                'Loser':'pl2',
                                'WPts':'pl1_pts', 
                                'LPts':'pl2_pts',})

        return df

    def data_without_atipics(self, df):
        ### quitando alturas atípicas
        df = df[~((df["pl1_height"] > 220) |  (df["pl1_height"] < 150) | (df["pl2_height"] > 220) |  (df["pl2_height"] < 150))]

        ### Quitando pesos atípicos
        df = df[~((df["pl1_weight"] > 120) |  (df["pl1_weight"] < 50) | (df["pl2_weight"] > 120) |  (df["pl2_weight"] < 50))]

        ### Quitando las fechas de pro atipicas
        df = df[~((df["pl1_year_pro"] > 2023) |  (df["pl1_year_pro"] < 1980) | (df["pl2_year_pro"] > 2023) |  (df["pl2_year_pro"] < 1980))]
        
        return df

    def data_p1vsp2(self, train, df):
        if train == True:
            df_p1 = df.copy()

            df_p2 = pd.DataFrame()

            df_p2["Date"] = df["Date"]
            df_p2["Surface"] = df["Surface"]
            df_p2["pl1"] = df["pl2"]
            df_p2["pl2"] = df["pl1"]
            df_p2["pl1_pts"] = df["pl2_pts"]
            df_p2["pl2_pts"] = df["pl1_pts"]
            df_p2["pl1_flag"] = df["pl2_flag"]
            df_p2["pl1_bet"] = df["pl2_bet"]
            df_p2["pl1_year_pro"] = df["pl2_year_pro"]
            df_p2["pl1_weight"] = df["pl2_weight"]
            df_p2["pl1_height"] = df["pl2_height"]
            df_p2["pl1_hand"] = df["pl2_hand"]
            df_p2["pl2_flag"] = df["pl1_flag"]
            df_p2["pl2_year_pro"] = df["pl1_year_pro"]
            df_p2["pl2_weight"] = df["pl1_weight"]
            df_p2["pl2_height"] = df["pl1_height"]
            df_p2["pl2_hand"] = df["pl1_hand"]
            df_p2["pl2_bet"] = df["pl1_bet"]
            df_p2["Result"] = 1

            df_all = pd.concat([df_p1, df_p2], axis = 0 )
        
        else:
            df_all = df.copy()
        
        ### Obtener columna que indica cuantos años lleva siendo profesional el tenista
        df_all["pl1_professional_time"] = pd.to_datetime(df_all["Date"]).dt.year - df_all["pl1_year_pro"]
        df_all["pl2_professional_time"] = pd.to_datetime(df_all["Date"]).dt.year - df_all["pl2_year_pro"]

        df_all['DiffProffTime'] = df_all["pl1_professional_time"] - df_all['pl2_professional_time']
        df_all['DiffPts'] = df_all["pl1_pts"] - df_all['pl2_pts']
        df_all['DiffWeight'] = df_all["pl1_weight"] - df_all['pl2_weight']
        df_all['DiffHeight'] = df_all["pl1_height"] - df_all['pl2_height']

        df_all = df_all.drop(columns=['Date','pl1_pts','pl2_pts','pl1_year_pro','pl2_year_pro','pl1_professional_time','pl2_professional_time',
                            'pl1_weight','pl1_height','pl2_weight','pl2_height'] )
        
        df_all = df_all.sample(frac=1).reset_index(drop=True)
        
        return df_all
    
    def save_data_cleaned(self, path_to_save, nombre_archivo, df_all):
        
        df_all.to_csv(os.path.join(path_to_save, nombre_archivo), index = False, sep='|')

        return df_all
            
        
def get_data(df_raw, train, path, nombre_archivo):
    
    df_bronze = GetData().data_raw(df_raw)
    df_silver = GetData().data_without_atipics(df_bronze)
    df_gold = GetData().data_p1vsp2(train, df_silver)
    df_gold = GetData().save_data_cleaned(path, nombre_archivo, train, df_gold)
    
    return df_bronze, df_silver, df_gold