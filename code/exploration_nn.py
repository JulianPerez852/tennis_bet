import pandas as pd

df = pd.read_csv('../data/input/scraped/df_games.csv', delimiter='|')

import numpy as np

pl1_cols = [col for col in df.columns if col.startswith('pl1_') or col == 'Winner']
pl2_cols = [col for col in df.columns if col.startswith('pl2_') or col == 'Loser']

# Crear una columna de indicador para el intercambio
df['swap'] = np.random.rand(len(df)) < 0.5

# Intercambiar las columnas donde swap es True
df.loc[df['swap'] == True, pl1_cols], df.loc[df['swap'] == True, pl2_cols] = \
    df.loc[df['swap'] == True, pl2_cols].values, df.loc[df['swap'] == True, pl1_cols].values

# Crear la columna 'winner_match'
df['winner_match'] = 'pl1'
df.loc[df['swap'] == True, 'winner_match'] = 'pl2'

# Eliminar la columna swap ya que no es necesaria
df = df.drop(columns=['swap'])

df.rename(columns={'Winner': 'pl1_name',
                    'Loser': 'pl2_name'}, inplace=True)

from sklearn.linear_model import LinearRegression

# Supongamos que df es tu dataframe original

# Paso 1: Crear un dataframe de jugadores con sus características

# Extraer las columnas de pl1 y pl2 y renombrarlas
pl1_cols = ['pl1_name', 'pl1_flag', 'pl1_year_pro', 'pl1_weight', 'pl1_height', 'pl1_hand', 'pl1_age']
pl2_cols = ['pl2_name', 'pl2_flag', 'pl2_year_pro', 'pl2_weight', 'pl2_height', 'pl2_hand', 'pl2_age']

pl1_df = df[pl1_cols].copy()
pl2_df = df[pl2_cols].copy()

pl1_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']
pl2_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']

# Combinar los dataframes de pl1 y pl2
players_df = pd.concat([pl1_df, pl2_df], ignore_index=True)

# Paso 2: Agrupar por jugador y agregar datos
player_data = players_df.groupby('name').agg({
    'flag': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'year_pro': 'mean',
    'weight': 'mean',
    'height': 'mean',
    'hand': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'age': 'mean'
}).reset_index()

# Paso 3: Identificar jugadores con valores faltantes
missing_players = player_data[player_data.isnull().any(axis=1)]

# Paso 4: Relaciones entre altura, peso y edad para llenar valores faltantes

# Filtrar jugadores con datos completos para entrenar los modelos
complete_players = player_data.dropna(subset=['height', 'weight', 'age'])

# Modelos de regresión lineal
model_weight = LinearRegression()
model_height = LinearRegression()
model_age = LinearRegression()

# Preparar datos para el modelo de peso
weight_data = complete_players[['height', 'age', 'weight']].dropna()
X_weight = weight_data[['height', 'age']]
y_weight = weight_data['weight']
model_weight.fit(X_weight, y_weight)

# Preparar datos para el modelo de altura
height_data = complete_players[['weight', 'age', 'height']].dropna()
X_height = height_data[['weight', 'age']]
y_height = height_data['height']
model_height.fit(X_height, y_height)

# Preparar datos para el modelo de edad
age_data = complete_players[['height', 'weight', 'age']].dropna()
X_age = age_data[['height', 'weight']]
y_age = age_data['age']
model_age.fit(X_age, y_age)

# Paso 5: Llenar los valores faltantes utilizando los modelos

# Función para predecir valores faltantes
def predict_missing_values(df, model, target, features):
    mask = df[target].isnull() & df[features].notnull().all(axis=1)
    if mask.any():
        df.loc[mask, target] = model.predict(df.loc[mask, features])

# Predecir peso faltante
predict_missing_values(player_data, model_weight, 'weight', ['height', 'age'])

# Predecir altura faltante
predict_missing_values(player_data, model_height, 'height', ['weight', 'age'])

# Predecir edad faltante
predict_missing_values(player_data, model_age, 'age', ['height', 'weight'])

# Paso 6: Calcular la media de la diferencia entre 'age' y 'year_pro'
# Esta diferencia representa cuántos años después del nacimiento se hicieron profesionales
age_year_pro_diff = player_data.dropna(subset=['age', 'year_pro'])
mean_age_year_pro_diff = (age_year_pro_diff['age'] - (pd.to_datetime('today').year - age_year_pro_diff['year_pro'])).mean()

# Función para calcular el 'year_pro' basándose en la edad
def calculate_year_pro(row, current_year):
    if pd.isnull(row['year_pro']) and not pd.isnull(row['age']):
        return current_year - (row['age'] - mean_age_year_pro_diff)
    return row['year_pro']

# Aplicar la regla para llenar los valores faltantes de 'year_pro'
current_year = pd.to_datetime('today').year
player_data['year_pro'] = player_data.apply(calculate_year_pro, axis=1, current_year=current_year)

# Paso 7: Llenar valores faltantes restantes con la media general solo en las columnas numéricas
numeric_cols = ['year_pro', 'weight', 'height', 'age']  # Definir las columnas numéricas

# Calcular la media de las columnas numéricas
mean_values = player_data[numeric_cols].mean()

# Llenar los valores faltantes con la media
player_data['weight'].fillna(mean_values['weight'], inplace=True)
player_data['height'].fillna(mean_values['height'], inplace=True)
player_data['age'].fillna(mean_values['age'], inplace=True)
player_data['year_pro'].fillna(mean_values['year_pro'], inplace=True)

# Paso 8: Asignar valores a 'hand' basado en porcentajes existentes
hand_counts = player_data['hand'].value_counts(normalize=True)
prob_right = hand_counts.get('right', 0)
prob_left = hand_counts.get('left', 0)

# Asignar valores aleatorios a 'hand' faltantes
mask_hand = player_data['hand'].isnull()
player_data.loc[mask_hand, 'hand'] = np.random.choice(
    ['right', 'left'],
    size=mask_hand.sum(),
    p=[prob_right, prob_left]
)

# Paso 9: Reemplazar los datos en el dataframe original

# Crear diccionarios de mapeo para cada característica
mapping_cols = ['flag', 'year_pro', 'weight', 'height', 'hand', 'age']
player_mappings = {col: player_data.set_index('name')[col].to_dict() for col in mapping_cols}

# Actualizar las columnas pl1 en df
for col in mapping_cols:
    df[f'pl1_{col}'] = df['pl1_name'].map(player_mappings[col])

# Actualizar las columnas pl2 en df
for col in mapping_cols:
    df[f'pl2_{col}'] = df['pl2_name'].map(player_mappings[col])

# Ahora, df tiene los valores faltantes llenados

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OneHotEncoder

# Copiar el dataframe para no modificar el original
data = df.copy()

# Listar las columnas categóricas
categorical_cols = ['Location', 'Surface', 'pl1_flag', 'pl1_hand', 'pl2_flag', 'pl2_hand']

# Aplicar OneHotEncoding a las variables categóricas
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Codificar la variable objetivo
label_encoder = LabelEncoder()
data['winner_match'] = label_encoder.fit_transform(data['winner_match'])  # 'pl1' -> 1, 'pl2' -> 0

# Definir X y y
X = data.drop(['winner_match', 'Date', 'pl1_name', 'pl2_name'], axis=1)
y = data['winner_match']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()

# Ajustar el escalador solo en los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo
logreg = LogisticRegression(max_iter=1000)

# Definir el grid de hiperparámetros
param_grid_logreg = {
    'C': [0.1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga']
}

# Configurar GridSearchCV
grid_logreg = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid_logreg,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Entrenar el modelo
print("Entrenando Regresión Logística...")
grid_logreg.fit(X_train_scaled, y_train)

# Resultados
print("Mejores hiperparámetros Regresión Logística:", grid_logreg.best_params_)
print("Accuracy en entrenamiento:", grid_logreg.best_score_)

# Definir el modelo
svm = SVC(max_iter=500)

# Definir el grid de hiperparámetros
param_grid_svm = {
    'C': [0.1, 10],
    'kernel': ['rbf'],
    'gamma': ['auto']
}

# Configurar GridSearchCV
grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Entrenar el modelo
print("Entrenando SVM...")
grid_svm.fit(X_train_scaled, y_train)

# Resultados
print("Mejores hiperparámetros SVM:", grid_svm.best_params_)
print("Accuracy en entrenamiento:", grid_svm.best_score_)

# Definir el modelo
rf = RandomForestClassifier(random_state=42)

# Definir el grid de hiperparámetros
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [5],
}

# Configurar GridSearchCV
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Entrenar el modelo
print("Entrenando Random Forest...")
grid_rf.fit(X_train, y_train)

# Resultados
print("Mejores hiperparámetros Random Forest:", grid_rf.best_params_)
print("Accuracy en entrenamiento:", grid_rf.best_score_)

# Definir el modelo
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Definir el grid de hiperparámetros
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 7],
    'learning_rate': [0.01, 0.1],
}

# Configurar GridSearchCV
grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Entrenar el modelo
print("Entrenando XGBoost...")
grid_xgb.fit(X_train, y_train)

# Resultados
print("Mejores hiperparámetros XGBoost:", grid_xgb.best_params_)
print("Accuracy en entrenamiento:", grid_xgb.best_score_)

# Definir el modelo
mlp = MLPClassifier(max_iter=500, random_state=42)

# Definir el grid de hiperparámetros
param_grid_mlp = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.001],
    'learning_rate': ['adaptive']
}

# Configurar GridSearchCV
grid_mlp = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid_mlp,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Entrenar el modelo
print("Entrenando MLPClassifier...")
grid_mlp.fit(X_train_scaled, y_train)

# Resultados
print("Mejores hiperparámetros MLPClassifier:", grid_mlp.best_params_)
print("Accuracy en entrenamiento:", grid_mlp.best_score_)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy en prueba para {model_name}: {accuracy:.4f}")
    print(f"Reporte de clasificación para {model_name}:\n", classification_report(y_test, y_pred))


# Regresión Logística
evaluate_model(grid_logreg.best_estimator_, X_test_scaled, y_test, 'Regresión Logística')

# SVM
evaluate_model(grid_svm.best_estimator_, X_test_scaled, y_test, 'SVM')

# Random Forest
evaluate_model(grid_rf.best_estimator_, X_test, y_test, 'Random Forest')

# XGBoost
evaluate_model(grid_xgb.best_estimator_, X_test, y_test, 'XGBoost')

# MLPClassifier
evaluate_model(grid_mlp.best_estimator_, X_test_scaled, y_test, 'MLPClassifier')

import pickle

# Guardar los modelos
with open('../models/logreg_model.pkl', 'wb') as file:
    pickle.dump(grid_logreg.best_estimator_, file)
    
with open('../models/svm_model.pkl', 'wb') as file:
    pickle.dump(grid_svm.best_estimator_, file)
    
with open('../models/rf_model.pkl', 'wb') as file:
    pickle.dump(grid_rf.best_estimator_, file)
    
with open('../models/xgb_model.pkl', 'wb') as file:
    pickle.dump(grid_xgb.best_estimator_, file)
    
with open('../models/mlp_model.pkl', 'wb') as file:
    pickle.dump(grid_mlp.best_estimator_, file)
    
# Guardar el escalador
with open('../models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
# Guardar el codificador de etiquetas
with open('../models/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
    
# Guardar el mapeo de columnas

# Crear un diccionario con los mapeos

mapping_dict = {
    'flag': player_mappings['flag'],
    'year_pro': player_mappings['year_pro'],
    'weight': player_mappings['weight'],
    'height': player_mappings['height'],
    'hand': player_mappings['hand'],
    'age': player_mappings['age']
}

# Guardar el diccionario
with open('../models/mapping_dict.pkl', 'wb') as file:
    pickle.dump(mapping_dict, file)
    
# Guardar el codificador OneHot
with open('../models/onehot_encoder.pkl', 'wb') as file:
    pickle.dump(pd.get_dummies, file)

