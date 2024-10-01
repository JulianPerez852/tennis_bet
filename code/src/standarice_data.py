import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


class TenisModelHandler:
    """
    Preprocesamiento y modelado de datos de tenis.

    Args:
        df (pd.DataFrame): DataFrame a preprocesar.
        models_path (str): Ruta donde se guardarán los modelos entrenados.
    """

    def __init__(self, df, models_path='models'):
        self.df = df
        self.models_path = models_path

    def preprocessing_train(self):
        """
        Preprocesamiento de los datos para entrenamiento.
        """
        self._swap_data()
        self.df.rename(columns={'Winner': 'pl1_name', 'Loser': 'pl2_name'}, inplace=True)
        self._fill_nas(is_training=True)

    def preprocessing_predict(self):
        """
        Preprocesamiento de los datos para predicción.
        """
        self.df.rename(columns={'Winner': 'pl1_name', 'Loser': 'pl2_name'}, inplace=True)
        self._fill_nas(is_training=False)

    def train_models(self, models_list: list):
        """
        Entrena los modelos especificados y los guarda en la ruta definida.

        Args:
            models_list (list): Lista de nombres de modelos a entrenar.
        """
        data = self.df.copy()

        # Codificar la variable objetivo
        label_encoder = LabelEncoder()
        data['winner_match'] = label_encoder.fit_transform(data['winner_match'])  # 'pl1' -> 1, 'pl2' -> 0

        # Guardar el LabelEncoder
        self._save_object(label_encoder, f'{self.models_path}/label_encoder.pkl')

        # Definir X y y
        X = data.drop(['winner_match', 'Date', 'pl1_name', 'pl2_name'], axis=1)
        y = data['winner_match']

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        categorical_cols = ['Location', 'Surface', 'pl1_flag', 'pl1_hand', 'pl2_flag', 'pl2_hand']
        numeric_cols = [col for col in X.columns if col not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        # Guardar el preprocesador
        self._save_object(preprocessor, f'{self.models_path}/preprocessor.pkl')

        for model_name in models_list:
            if model_name == 'logistic_regression':
                model_instance = LogisticRegression(max_iter=1000)
                param_grid = {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l2'],
                    'model__solver': ['lbfgs', 'saga']
                }
            elif model_name == 'svm':
                model_instance = SVC(max_iter=500)
                param_grid = {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf'],
                    'model__gamma': ['scale', 'auto']
                }
            elif model_name == 'random_forest':
                model_instance = RandomForestClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, None],
                    'model__min_samples_split': [2, 5],
                }
            elif model_name == 'xgboost':
                model_instance = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 7],
                    'model__learning_rate': [0.01, 0.1],
                }
            elif model_name == 'neural_network':
                model_instance = MLPClassifier(max_iter=500, random_state=42)
                param_grid = {
                    'model__hidden_layer_sizes': [(100,), (100, 50)],
                    'model__activation': ['relu', 'tanh'],
                    'model__solver': ['adam', 'sgd'],
                    'model__alpha': [0.0001, 0.001],
                    'model__learning_rate': ['constant', 'adaptive']
                }
            else:
                print(f"Modelo {model_name} no reconocido, se omitirá.")
                continue

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model_instance)
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            print(f"Entrenando {model_name}...")
            grid.fit(X_train, y_train)

            # Resultados
            print(f"Mejores hiperparámetros {model_name}:", grid.best_params_)
            print(f"Accuracy en entrenamiento {model_name}: {grid.best_score_:.4f}")

            self._evaluate_model(grid.best_estimator_, X_test, y_test, model_name)

            self._save_object(grid.best_estimator_, f'{self.models_path}/{model_name}.pkl')

    def predict_values(self, model_name):
        """
        Genera predicciones utilizando el modelo especificado y devuelve el DataFrame original con las predicciones.

        Args:
            model_name (str): Nombre del modelo a utilizar para predicciones.

        Returns:
            pd.DataFrame: DataFrame original con una columna adicional de predicciones.
        """
        # Cargar el modelo
        model_path = f'{self.models_path}/{model_name}.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Cargar el LabelEncoder
        label_encoder_path = f'{self.models_path}/label_encoder.pkl'
        with open(label_encoder_path, 'rb') as file:
            label_encoder = pickle.load(file)

        # Preparar los datos
        self.preprocessing_predict()

        data = self.df.copy()

        # Asegurarse de que las columnas necesarias estén presentes
        required_columns = model.named_steps['preprocessor'].transformers_[1][2] + \
                           model.named_steps['preprocessor'].transformers_[0][2]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Las siguientes columnas faltan en los datos: {missing_columns}")

        # Seleccionar las columnas necesarias
        X = data[required_columns]

        # Generar predicciones
        predictions = model.predict(X)

        predict_proba = model.predict_proba(X)

        # Mapear predicciones a etiquetas originales
        predictions_labels = label_encoder.inverse_transform(predictions)

        # Añadir predicciones al DataFrame
        data['winner_prediction'] = predictions_labels
        data['probability'] = predict_proba[np.arange(len(predictions)), predictions]

        return data

    def _evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy en prueba para {model_name}: {accuracy:.4f}")
        print(f"Reporte de clasificación para {model_name}:\n{classification_report(y_test, y_pred)}")

    def _swap_data(self):
        pl1_cols = [col for col in self.df.columns if col.startswith('pl1_') or col == 'Winner']
        pl2_cols = [col for col in self.df.columns if col.startswith('pl2_') or col == 'Loser']

        self.df['swap'] = np.random.rand(len(self.df)) < 0.5

        # Intercambiar datos
        swapped_indices = self.df['swap'] == True
        self.df.loc[swapped_indices, pl1_cols], self.df.loc[swapped_indices, pl2_cols] = \
            self.df.loc[swapped_indices, pl2_cols].values, self.df.loc[swapped_indices, pl1_cols].values

        self.df['winner_match'] = 'pl1'
        self.df.loc[swapped_indices, 'winner_match'] = 'pl2'

        self.df = self.df.drop(columns=['swap'])

    def _predict_missing_values(self, df, model, target, features):
        mask = df[target].isnull() & df[features].notnull().all(axis=1)
        if mask.any():
            df.loc[mask, target] = model.predict(df.loc[mask, features])

    def _save_object(self, obj, path_to_save):
        with open(path_to_save, 'wb') as file:
            pickle.dump(obj, file)

    def _load_object(self, path_to_load):
        with open(path_to_load, 'rb') as file:
            return pickle.load(file)

    def _calculate_year_pro(self, row, current_year, mean_age_year_pro_diff):
        if pd.isnull(row['year_pro']) and not pd.isnull(row['age']):
            return current_year - (row['age'] - mean_age_year_pro_diff)
        return row['year_pro']

    def _fill_nas(self, is_training=True):
        pl1_cols = ['pl1_name', 'pl1_flag', 'pl1_year_pro', 'pl1_weight', 'pl1_height', 'pl1_hand', 'pl1_age']
        pl2_cols = ['pl2_name', 'pl2_flag', 'pl2_year_pro', 'pl2_weight', 'pl2_height', 'pl2_hand', 'pl2_age']

        if is_training:
            # Proceso de entrenamiento
            pl1_df = self.df[pl1_cols].copy()
            pl2_df = self.df[pl2_cols].copy()

            pl1_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']
            pl2_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']

            players_df = pd.concat([pl1_df, pl2_df], ignore_index=True)

            player_data = players_df.groupby('name').agg({
                'flag': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                'year_pro': 'mean',
                'weight': 'mean',
                'height': 'mean',
                'hand': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                'age': 'mean'
            }).reset_index()

            complete_players = player_data.dropna(subset=['height', 'weight', 'age'])

            # Modelos de regresión lineal
            model_weight = LinearRegression()
            model_height = LinearRegression()
            model_age = LinearRegression()

            weight_data = complete_players[['height', 'age', 'weight']].dropna()
            X_weight = weight_data[['height', 'age']]
            y_weight = weight_data['weight']
            model_weight.fit(X_weight, y_weight)

            height_data = complete_players[['weight', 'age', 'height']].dropna()
            X_height = height_data[['weight', 'age']]
            y_height = height_data['height']
            model_height.fit(X_height, y_height)

            age_data = complete_players[['height', 'weight', 'age']].dropna()
            X_age = age_data[['height', 'weight']]
            y_age = age_data['age']
            model_age.fit(X_age, y_age)

            # Guardar modelos de regresión
            self._save_object(model_weight, f'{self.models_path}/model_weight.pkl')
            self._save_object(model_height, f'{self.models_path}/model_height.pkl')
            self._save_object(model_age, f'{self.models_path}/model_age.pkl')

            # Predecir valores faltantes
            self._predict_missing_values(player_data, model_weight, 'weight', ['height', 'age'])
            self._predict_missing_values(player_data, model_height, 'height', ['weight', 'age'])
            self._predict_missing_values(player_data, model_age, 'age', ['height', 'weight'])

            # Calcular diferencia media entre edad y año profesional
            age_year_pro_diff = player_data.dropna(subset=['age', 'year_pro'])
            mean_age_year_pro_diff = (age_year_pro_diff['age'] - (pd.to_datetime('today').year - age_year_pro_diff['year_pro'])).mean()
            self.mean_age_year_pro_diff = mean_age_year_pro_diff

            # Guardar diferencia media
            self._save_object(mean_age_year_pro_diff, f'{self.models_path}/mean_age_year_pro_diff.pkl')

            current_year = pd.to_datetime('today').year
            player_data['year_pro'] = player_data.apply(self._calculate_year_pro, axis=1, current_year=current_year, mean_age_year_pro_diff=mean_age_year_pro_diff)

            numeric_cols = ['year_pro', 'weight', 'height', 'age']
            mean_values = player_data[numeric_cols].mean()

            # Guardar valores medios
            self._save_object(mean_values, f'{self.models_path}/mean_values.pkl')

            player_data['weight'].fillna(mean_values['weight'], inplace=True)
            player_data['height'].fillna(mean_values['height'], inplace=True)
            player_data['age'].fillna(mean_values['age'], inplace=True)
            player_data['year_pro'].fillna(mean_values['year_pro'], inplace=True)

            # Probabilidades de 'hand'
            hand_counts = player_data['hand'].value_counts(normalize=True)
            hand_probs = hand_counts.to_dict()
            self._save_object(hand_probs, f'{self.models_path}/hand_probs.pkl')

            mask_hand = player_data['hand'].isnull()
            if mask_hand.any():
                player_data.loc[mask_hand, 'hand'] = np.random.choice(
                    hand_counts.index.tolist(),
                    size=mask_hand.sum(),
                    p=hand_counts.values
                )

            mapping_cols = ['flag', 'year_pro', 'weight', 'height', 'hand', 'age']
            player_mappings = {col: player_data.set_index('name')[col].to_dict() for col in mapping_cols}
            self._save_object(player_mappings, f'{self.models_path}/player_mappings.pkl')

            # Mapear datos a jugadores
            for col in mapping_cols:
                self.df[f'pl1_{col}'] = self.df['pl1_name'].map(player_mappings[col])

            for col in mapping_cols:
                self.df[f'pl2_{col}'] = self.df['pl2_name'].map(player_mappings[col])

            self.df.dropna(inplace=True)
        else:
            # Proceso de predicción
            # Cargar objetos guardados
            model_weight = self._load_object(f'{self.models_path}/model_weight.pkl')
            model_height = self._load_object(f'{self.models_path}/model_height.pkl')
            model_age = self._load_object(f'{self.models_path}/model_age.pkl')
            mean_age_year_pro_diff = self._load_object(f'{self.models_path}/mean_age_year_pro_diff.pkl')
            mean_values = self._load_object(f'{self.models_path}/mean_values.pkl')
            hand_probs = self._load_object(f'{self.models_path}/hand_probs.pkl')
            player_mappings = self._load_object(f'{self.models_path}/player_mappings.pkl')

            pl1_df = self.df[pl1_cols].copy()
            pl2_df = self.df[pl2_cols].copy()

            pl1_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']
            pl2_df.columns = ['name', 'flag', 'year_pro', 'weight', 'height', 'hand', 'age']

            players_df = pd.concat([pl1_df, pl2_df], ignore_index=True)

            # Mapear datos existentes
            mapping_cols = ['flag', 'year_pro', 'weight', 'height', 'hand', 'age']
            for col in mapping_cols:
                players_df[col] = players_df['name'].map(player_mappings[col])

            # Predecir valores faltantes
            self._predict_missing_values(players_df, model_weight, 'weight', ['height', 'age'])
            self._predict_missing_values(players_df, model_height, 'height', ['weight', 'age'])
            self._predict_missing_values(players_df, model_age, 'age', ['height', 'weight'])

            current_year = pd.to_datetime('today').year
            players_df['year_pro'] = players_df.apply(
                self._calculate_year_pro,
                axis=1,
                current_year=current_year,
                mean_age_year_pro_diff=mean_age_year_pro_diff
            )

            numeric_cols = ['year_pro', 'weight', 'height', 'age']
            players_df[numeric_cols] = players_df[numeric_cols].fillna(mean_values)

            # Rellenar 'hand' basado en probabilidades
            mask_hand = players_df['hand'].isnull()
            if mask_hand.any():
                hand_options = list(hand_probs.keys())
                hand_probabilities = list(hand_probs.values())
                players_df.loc[mask_hand, 'hand'] = np.random.choice(
                    hand_options,
                    size=mask_hand.sum(),
                    p=hand_probabilities
                )

            # Volver a asignar los datos al DataFrame original
            half = len(players_df) // 2
            pl1_df_filled = players_df.iloc[:half]
            pl2_df_filled = players_df.iloc[half:]

            pl1_df_filled.columns = ['pl1_name', 'pl1_flag', 'pl1_year_pro', 'pl1_weight', 'pl1_height', 'pl1_hand', 'pl1_age']
            pl2_df_filled.columns = ['pl2_name', 'pl2_flag', 'pl2_year_pro', 'pl2_weight', 'pl2_height', 'pl2_hand', 'pl2_age']

            self.df.reset_index(drop=True, inplace=True)
            self.df.update(pl1_df_filled)
            self.df.update(pl2_df_filled)

            self.df.dropna(inplace=True)
