import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from src.parameters import Parameters

def data_to_class(df_variables):
    # label_encoder = LabelEncoder()
    # for col in df_variables.select_dtypes(exclude=['number']).columns:
    #     df_variables[col] = label_encoder.fit_transform(df_variables[col])
    # label_encoder = LabelEncoder()
    # features_raw = df_variables.drop(['Class'], axis = 1)
    # features_final = pd.get_dummies(features_raw)
    
    # # Codificar las columnas categóricas
    # df_variables['pl1_flag'] = label_encoder.fit_transform(df_variables['pl1_flag'])
    # df_variables['pl2_flag'] = label_encoder.fit_transform(df_variables['pl2_flag'])
    # df_variables['pl1_hand'] = label_encoder.fit_transform(df_variables['pl1_hand'])
    # df_variables['pl2_hand'] = label_encoder.fit_transform(df_variables['pl2_hand'])
    
    
    for col in df_variables.select_dtypes(exclude=['number']).columns:
        df_variables[col] = df_variables[col].astype('category')
    class_raw = df_variables['Result']
    features_raw = df_variables.drop(['Result'], axis = 1)
    features_final = pd.get_dummies(features_raw)
    return features_final, class_raw
