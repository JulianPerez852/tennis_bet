import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RiskAnalysisMontecarlo():
    def data_ev_sr(self, df, risk_free_rate):
        # # Crear el DataFrame con los datos
        # data = {
        #     'pl1': ['Giron M.', 'Mcdonald M.', 'Kecmanovic M.', 'Nishioka Y.', 'Popyrin A.', 'Halys Q.', 'Safiullin R.', 'Draper J.', 'Kokkinakis T.'],
        #     'pl2': ['Gasquet R.', 'Galan D.E.', 'O Connell C.', 'Rune H.', 'Auger-Aliassime F.', 'Thompson J.', 'Ymer M.', 'Kwon S.W.', 'Cressy M.'],
        #     'Prob_win_pl1': [0.85, 0.60, 0.59, 0.27, 0.63, 0.75, 0.75, 0.10, 0.74],
        #     'Prob_win_pl2': [0.15, 0.40, 0.41, 0.73, 0.37, 0.25, 0.25, 0.90, 0.26],
        #     'Bet_pl1': [1.91, 1.36, 1.57, 3.75, 6.50, 2.10, 2.20, 1.67, 1.91],
        #     'Bet_pl2': [1.91, 3.20, 2.38, 1.29, 1.11, 1.73, 1.67, 2.20, 1.91]
        # }

        # df = pd.DataFrame(data)

        # Calcular el Valor Esperado (EV) para cada jugador
        df['EV_pl1'] = df['Prob_win_pl1'] * df['Bet_pl1'] - (1 - df['Prob_win_pl1'])
        df['EV_pl2'] = df['Prob_win_pl2'] * df['Bet_pl2'] - (1 - df['Prob_win_pl2'])

        # Calcular la desviación estándar de los EVs
        df['Std_Dev_pl1'] = df['Bet_pl1'] * np.sqrt(df['Prob_win_pl1'] * (1 - df['Prob_win_pl1']))
        df['Std_Dev_pl2'] = df['Bet_pl2'] * np.sqrt(df['Prob_win_pl2'] * (1 - df['Prob_win_pl2']))

        # Calcular el Ratio de Sharpe para cada jugador
        df['Sharpe_Ratio_pl1'] = (df['EV_pl1'] - risk_free_rate) / df['Std_Dev_pl1']
        df['Sharpe_Ratio_pl2'] = (df['EV_pl2'] - risk_free_rate) / df['Std_Dev_pl2']
            
        # Calcular la suma total de los mejores EV para la asignación de dinero
        df['Best_Bet_EV'] = df[['EV_pl1', 'EV_pl2']].max(axis=1)
        total_ev_sum = df['Best_Bet_EV'].sum()
        
        return df, total_ev_sum

    # Función para determinar la apuesta en función del EV, Sharpe Ratio y el nivel de riesgo
    def determine_bet(self, row, risk_tolerance, total_money, total_ev_sum):
        if row['EV_pl1'] > row['EV_pl2']:
            best_bet = 'pl1'
            best_ev = row['EV_pl1']
            best_sharpe = row['Sharpe_Ratio_pl1']
            prob_win = row['Prob_win_pl1']
            payout = row['Bet_pl1']
        else:
            best_bet = 'pl2'
            best_ev = row['EV_pl2']
            best_sharpe = row['Sharpe_Ratio_pl2']
            prob_win = row['Prob_win_pl2']
            payout = row['Bet_pl2']
        
        # Ajustar la cantidad a apostar en función del Sharpe Ratio y la tolerancia al riesgo
        if best_sharpe > (1 - risk_tolerance):  # Aumenta la sensibilidad a medida que disminuye el nivel de riesgo
            money_to_bet = (best_ev / total_ev_sum) * total_money
        else:
            money_to_bet = 1  # Apostar solo $1 si el Sharpe Ratio no cumple con el criterio de riesgo
        
        return pd.Series([best_bet, best_ev, best_sharpe, money_to_bet, prob_win, payout])

    # Función de Simulación de Monte Carlo
    def monte_carlo_simulation(self, df, num_simulations=1000, num_bets=10):
        results = []
        for sim in range(num_simulations):
            total_profit = 0
            for i in range(num_bets):
                bet = df.sample(1).iloc[0]
                outcome = np.random.rand() <= bet['Prob_Win']
                if outcome:  # Si gana la apuesta
                    total_profit += bet['Money_to_Bet'] * bet['Payout']
                else:  # Si pierde la apuesta
                    total_profit -= bet['Money_to_Bet']
            results.append(total_profit)
        return results

    def plot_risk_simulation(self, plots_path, 
                            results_path, 
                            monte_carlo_results, 
                            df, 
                            name_monte_carlo_dist,
                            name_ev_comparation,
                            name_sharpe_ratio_comparison, 
                            file_betting_analysis):

        # Generar gráfico de resultados
        plt.figure(figsize=(10, 6))
        plt.hist(monte_carlo_results, bins=50, color='blue', edgecolor='black')
        plt.title('Distribución de Ganancias y Pérdidas (Simulación de Monte Carlo)')
        plt.xlabel('Ganancias/Pérdidas ($)')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(f'{plots_path}\\{name_monte_carlo_dist}')
        plt.close()

        # Crear resumen de la simulación
        monte_carlo_summary = {
            'Promedio de Ganancia/Pérdida': np.mean(monte_carlo_results),
            'Mediana de Ganancia/Pérdida': np.median(monte_carlo_results),
            'Ganancia/Pérdida Máxima': np.max(monte_carlo_results),
            'Ganancia/Pérdida Mínima': np.min(monte_carlo_results),
            'Probabilidad de Ganancia (>0)': np.mean([x > 0 for x in monte_carlo_results])
        }

        # Crear el archivo Excel con el resumen, tabla de resultados y gráficos
        with pd.ExcelWriter(f'{results_path}/{file_betting_analysis}', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Betting Decisions', index=False)
            
            # Agregar resultados de la simulación de Monte Carlo en una hoja separada
            pd.DataFrame(monte_carlo_results, columns=['Ganancia/Pérdida']).to_excel(
                                writer, sheet_name='Monte Carlo Results', index=False)
            
            # Agregar resumen de la simulación en una hoja separada
            pd.DataFrame([monte_carlo_summary]).to_excel(writer, sheet_name='Monte Carlo Summary', index=False)
            
            # Crear un gráfico del EV de cada jugador
            plt.figure(figsize=(10, 6))
            df.plot(kind='bar', x='pl1', y=['EV_pl1', 'EV_pl2'], color=['blue', 'orange'], ax=plt.gca())
            plt.title('Valor Esperado (EV) por Jugador')
            plt.ylabel('EV')
            plt.xlabel('Partidos')
            plt.tight_layout()
            plt.savefig(f'{plots_path}\\{name_ev_comparation}')
            plt.close()
            
            # Crear un gráfico del Sharpe Ratio de cada jugador
            plt.figure(figsize=(10, 6))
            df.plot(kind='bar', 
                    x='pl1', 
                    y=['Sharpe_Ratio_pl1', 'Sharpe_Ratio_pl2'], 
                    color=['green', 'red'], 
                    ax=plt.gca())
            plt.title('Sharpe Ratio por Jugador')
            plt.ylabel('Sharpe Ratio')
            plt.xlabel('Partidos')
            plt.tight_layout()
            plt.savefig(f'{plots_path}\\{name_sharpe_ratio_comparison}')
            plt.close()

            # Insertar gráficos en el archivo Excel
            workbook = writer.book
            worksheet = writer.sheets['Betting Decisions']
            
            # Insertar gráficos usando openpyxl para cargar las imágenes
            from openpyxl.drawing.image import Image
            img1 = Image(f'{plots_path}\\{name_ev_comparation}')
            img2 = Image(f'{plots_path}\\{name_sharpe_ratio_comparison}')
            worksheet.add_image(img1, 'J2')
            worksheet.add_image(img2, 'J20')
            
            # Insertar gráfico de Monte Carlo
            worksheet_mc = writer.sheets['Monte Carlo Summary']
            img_mc = Image(f'{plots_path}\\{name_monte_carlo_dist}')
            worksheet_mc.add_image(img_mc, 'B2')

        # Output de información
        print("Análisis completado. Revisa el archivo 'betting_analysis.xlsx' para ver los resultados.")
        
        return monte_carlo_summary
      
def risk_analysis_montecarlo(df_gold,
                            risk_free_rate, 
                            risk_tolerance, 
                            total_money, 
                            num_simulations, 
                            num_bets,
                            plots_path,
                            results_path,
                            name_monte_carlo_dist,
                            name_ev_comparation,
                            name_sharpe_ratio_comparison,
                            file_betting_analysis):
    
    risk_analysis_montecarlo = RiskAnalysisMontecarlo()
    df_risk, total_ev_sum = risk_analysis_montecarlo.data_ev_sr(df_gold, risk_free_rate)

    # Aplicar la función al DataFrame
    df_risk[['Best_Bet', 
             'Best_Bet_EV', 
             'Best_Bet_Sharpe', 
             'Money_to_Bet', 
             'Prob_Win', 
             'Payout']] = df_risk.apply(risk_analysis_montecarlo.determine_bet, 
                                        axis=1, 
                                        risk_tolerance=risk_tolerance, 
                                        total_money=total_money, 
                                        total_ev_sum=total_ev_sum)

    # Realiza la simulación de Monte Carlo
    monte_carlo_results = risk_analysis_montecarlo.monte_carlo_simulation(df_risk, num_simulations, num_bets)
    monte_carlo_summary = risk_analysis_montecarlo.plot_risk_simulation(plots_path, 
                                                                    results_path, 
                                                                    monte_carlo_results, 
                                                                    df_risk, 
                                                                    name_monte_carlo_dist,
                                                                    name_ev_comparation,
                                                                    name_sharpe_ratio_comparison, 
                                                                    file_betting_analysis)
    return df_risk, monte_carlo_results, monte_carlo_summary