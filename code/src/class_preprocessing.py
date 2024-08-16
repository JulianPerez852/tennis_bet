from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def data_to_class(df_variables):
    # for col in df_variables.select_dtypes(exclude=['number']).columns:
    #     df_variables[col] = df_variables[col].astype('category')
    
    df_variables = df_variables.drop(['pl1','pl2'], axis = 1)
    categorical_cols = df_variables.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_variables.select_dtypes(exclude=['object']).columns.tolist()
    numerical_cols.remove('Result')

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ],
        remainder='passthrough'
    )

    class_raw = df_variables['Result']
    features_raw = df_variables.drop(['Result'], axis = 1)
    # features_final = pd.get_dummies(features_raw)
    
    return features_raw, class_raw, preprocessor