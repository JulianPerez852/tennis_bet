from src.save_results import save_model, save_winner_metrics
from src.parameters import Parameters
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from catboost import CatBoostClassifier

from sklearn.metrics import classification_report

def evaluate_models_classification(X_train, X_test, y_train, y_test):

    modelos = {
        'lrm': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
        'lrb': LogisticRegression(C=0.01, solver='liblinear'),
        'knn': KNeighborsClassifier(n_neighbors = 3),
        'dtc': DecisionTreeClassifier(criterion="entropy", max_depth = 4),
        # 'svm': svm.SVC(kernel='rbf', probability=True),
        'rf': RandomForestClassifier(random_state=42),
        'gnb': GaussianNB(),
        'ctb': CatBoostClassifier(iterations=100, learning_rate=0.01)
        
        }

    dict_models = {}
    for nombre, modelo in modelos.items():
        dict_models[nombre] = modelo.fit(X_train, y_train)

    dict_pred = {}
    resultados_classification_report = {}
    # Iterar sobre los modelos entrenados
    for nombre_modelo, modelo in dict_models.items():
        # Realizar predicciones utilizando el modelo
        y_pred = modelo.predict(X_test)
        # Generar el informe de clasificación
        reporte_clasificacion = classification_report(y_test, y_pred, output_dict=True)
        # Almacenar el informe en el diccionario utilizando el nombre del modelo como clave
        dict_pred[nombre_modelo] = y_pred.tolist()
        resultados_classification_report[nombre_modelo] = reporte_clasificacion
    dict_pred['test'] = y_test.tolist()
    
    # Inicializar listas para almacenar las métricas
    precision_list = []
    recall_list = []
    f1_list = []

    # Iterar sobre los modelos y extraer las métricas de interés
    for modelo in modelos.keys():
        weighted_avg = resultados_classification_report[modelo]['weighted avg']
        precision_list.append(weighted_avg['precision'])
        recall_list.append(weighted_avg['recall'])
        f1_list.append(weighted_avg['f1-score'])

    dict_metrics = pd.DataFrame({'model' : modelos.keys(),
                    'precision' : precision_list,
                    'recall' : recall_list,
                    'f1score' : f1_list})
    
    return dict_models, dict_pred, dict_metrics

def class_competition(dict_models, dict_pred, dict_metrics):
    for index, row in dict_metrics.iterrows():
        contador = 0
        if index == 0:
            best_precision = dict_metrics['precision'][0]
            best_recall = dict_metrics['recall'][0]
            best_f1score = dict_metrics['f1score'][0]
            best_model = dict_metrics['model'][0]
        else:
            precision = dict_metrics['precision'][index]
            recall = dict_metrics['recall'][index]
            f1score = dict_metrics['f1score'][index]
            model = dict_metrics['model'][index]

            if precision > best_precision:
                contador +=1
            if recall > best_recall:
                contador +=1
            if f1score > best_f1score:
                contador +=1

            if contador >= 2:
                best_precision = precision
                best_recall = recall
                best_f1score = f1score
                best_model = model
                    
        winner_metrics = {'model_name':best_model,
                            'precision': [best_precision], 
                            'recall' : [best_recall], 
                            'f1score':  [best_f1score], 
                            'pred': [dict_pred[best_model]], 
                            'test': [dict_pred['test']],
                        }
        
        save_winner_metrics(winner_metrics, Parameters.results_path, 'classification')

        model = dict_models[best_model]

        save_model(model, Parameters.models_path, best_model)
        
        return winner_metrics, model