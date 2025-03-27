import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import os
import io

# Пути к файлам
MODEL_PATH = 'artem/catboostmodel_artem.cbm'
DATA_PATH = 'artem/merged_df.csv'

# Загрузка модели
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Модель не найдена по пути: {MODEL_PATH}")
        return None
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

# Загрузка исторических данных
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Данные не найдены по пути: {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    df['dt'] = pd.to_datetime(df['dt'])
    return df

# Обновленная функция создания признаков
def create_features(df):
    df = df.sort_values('dt')
    # Лаги
    for lag in [1, 2, 4, 12, 52]:
        df[f'lag_{lag}'] = df['Цена на арматуру'].shift(lag)
    # Скользящие статистики
    df['rolling_mean_12'] = df['Цена на арматуру'].shift(1).rolling(12).mean()
    df['rolling_std_12'] = df['Цена на арматуру'].shift(1).rolling(12).std()
    # Временные признаки
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['week'] = df['dt'].dt.isocalendar().week
    # Циклические признаки
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Дифференцирование
    df['Цена_diff'] = df['Цена на арматуру'].diff()
    return df

# Генерация прогноза на 6 месяцев (26 недель)
def generate_six_month_forecast(historical_df, model, weeks_ahead=26):
    last_date = historical_df['dt'].max()
    current_date = last_date + timedelta(weeks=1)
    predictions = []
    
    # Получаем список признаков из модели
    feature_names = model.feature_names_
    
    # Создаем копию исторических данных и сортируем
    extended_df = historical_df.copy().sort_values('dt')
    extended_df = create_features(extended_df)
    extended_df = extended_df.set_index('dt')
    
    # Прогнозируем на 26 недель вперед
    for week in range(weeks_ahead):
        forecast_date = current_date + timedelta(weeks=week)
        
        # Создаем признаки для прогноза
        features = {
            'year': forecast_date.year,
            'month': forecast_date.month,
            'week': forecast_date.isocalendar().week,
            'month_sin': np.sin(2 * np.pi * forecast_date.month / 12),
            'month_cos': np.cos(2 * np.pi * forecast_date.month / 12),
        }
        
        # Рассчитываем лаги
        for lag in [1, 2, 4, 12, 52]:
            lag_date = forecast_date - timedelta(weeks=lag)
            available_dates = extended_df.index[extended_df.index <= lag_date]
            if len(available_dates) == 0:
                st.error(f"Нет данных для расчета лага {lag} на дату {forecast_date.strftime('%Y-%m-%d')}")
                return None
            closest_date = available_dates.max()
            features[f'lag_{lag}'] = extended_df.loc[closest_date, 'Цена на арматуру']
        
        # Скользящие статистики
        rolling_dates = extended_df.index[extended_df.index < forecast_date]
        if len(rolling_dates) >= 12:
            features['rolling_mean_12'] = extended_df.loc[rolling_dates[-12:], 'Цена на арматуру'].mean()
            features['rolling_std_12'] = extended_df.loc[rolling_dates[-12:], 'Цена на арматуру'].std()
        else:
            features['rolling_mean_12'] = extended_df['Цена на арматуру'].mean()
            features['rolling_std_12'] = extended_df['Цена на арматуру'].std()
        
        if len(rolling_dates) > 0 and week > 0:
            last_price = extended_df.loc[rolling_dates[-1], 'Цена на арматуру']
            features['Цена_diff'] = prediction - last_price
        else:
            features['Цена_diff'] = 0  # или другое значение по умолчанию
        
        # Прогнозируем цену
        feature_df = pd.DataFrame([features])[feature_names]  # Упорядочиваем по feature_names
        prediction = model.predict(feature_df)[0]
        
        # Добавляем прогноз в датафрейм
        extended_df.loc[forecast_date] = {
            'Цена на арматуру': prediction,
            **features
        }
        
        predictions.append((forecast_date, prediction))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])
# Упрощённая функция рекомендации
def recommend_weeks(predicted_prices, current_price):
    if not predicted_prices:  # Если нет прогнозируемых цен
        return 1  # Минимальная закупка на 1 неделю
    tender = 1
    for p in predicted_prices[1:]:
        if p > current_price:
            tender += 1
        else:
            break
    return tender

# Основное приложение
def main():
    st.title('Прогнозирование цен на арматуру')
    
    # Загрузка модели и данных
    model = load_model()
    historical_df = load_data()
    
    if model and historical_df is not None:
        # Генерируем прогноз на 6 месяцев при запуске
        six_month_forecast = generate_six_month_forecast(historical_df, model, weeks_ahead=26)
        
        if six_month_forecast is not None:
            # Определяем минимальную и максимальную даты
            first_historical_date = historical_df['dt'].min().date()
            last_historical_date = historical_df['dt'].max().date()
            max_date = last_historical_date + timedelta(weeks=26)  # Ограничение на 6 месяцев вперед
            
            # Ввод даты пользователем
            input_date = st.date_input(
                'Выберите начальную дату прогноза (понедельник):',
                min_value=first_historical_date,
                max_value=max_date,
                value=last_historical_date
            )
            
            if input_date.weekday() != 0:
                st.error("Ошибка выбранной даты. Прогноз только по понедельникам")
                return
            
            if st.button('Сформировать прогноз'):
                start_date = pd.to_datetime(input_date)
                
                # Объединяем исторические данные и прогноз на 6 месяцев
                historical_prices = historical_df[['dt', 'Цена на арматуру']].rename(columns={'dt': 'Date', 'Цена на арматуру': 'Price'})
                historical_prices['Type'] = 'Фактическая'
                forecast_prices = six_month_forecast.rename(columns={'Predicted Price': 'Price'})
                forecast_prices['Type'] = 'Прогнозная'
                combined_df = pd.concat([historical_prices, forecast_prices], ignore_index=True)
                combined_df = combined_df.set_index('Date')
                
                # Фильтруем данные на 6 недель, начиная с выбранной даты
                end_date = start_date + timedelta(weeks=5)
                forecast_df = combined_df[(combined_df.index >= start_date) & (combined_df.index <= end_date)]
                
                # Получаем текущую цену
                if start_date in historical_df['dt'].values:
                    current_price = historical_df[historical_df['dt'] == start_date]['Цена на арматуру'].values[0]
                else:
                    current_price = forecast_df['Price'].iloc[0]
                
                # Определяем рекомендацию (только на основе прогнозных цен)
                if not forecast_df[forecast_df['Type'] == 'Прогнозная'].empty:
                    predicted_prices = forecast_df[forecast_df['Type'] == 'Прогнозная']['Price'].tolist()
                    tender_n = recommend_weeks(predicted_prices, current_price)
                else:
                    prices = forecast_df[forecast_df['Type'] == 'Фактическая']['Price'].tolist()
                    tender_n = recommend_weeks(prices, current_price)
                
                # Рассчитываем объем закупки (N) для каждой строки
                n_values = []
                for i in range(len(forecast_df)):
                    row_price = forecast_df['Price'].iloc[i]  # Текущая цена для этой строки
                    remaining_prices = forecast_df['Price'].iloc[i:].tolist()
                    n = recommend_weeks(remaining_prices, row_price)
                    n_values.append(n)

                # Добавляем столбец с объемом закупки (N)
                forecast_df['Объем закупки (N)'] = n_values
                
                # Визуализация
                fig, ax = plt.subplots(figsize=(10, 5))
                actual_data = forecast_df[forecast_df['Type'] == 'Фактическая']
                predicted_data = forecast_df[forecast_df['Type'] == 'Прогнозная']
                if not actual_data.empty:
                    ax.plot(actual_data.index, actual_data['Price'], marker='o', 
                            linestyle='-', linewidth=2, markersize=8, color='green', label='Фактическая цена')
                if not predicted_data.empty:
                    ax.plot(predicted_data.index, predicted_data['Price'], marker='o', 
                            linestyle='--', linewidth=2, markersize=8, color='blue', label='Прогнозная цена')
                ax.axhline(y=current_price, color='red', linestyle='--', 
                           label=f'Текущая цена: {current_price:.2f} руб/т')
                ax.set_xlabel('Дата', fontsize=12)
                ax.set_ylabel('Цена (руб/т)', fontsize=12)
                ax.set_title('Прогноз цен на арматуру (6 недель)', fontsize=14, pad=20)
                ax.legend(prop={'size': 10})
                plt.xticks(forecast_df.index, [d.strftime('%Y-%m-%d') for d in forecast_df.index], rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Вывод рекомендации
                with st.container():
                    st.success(f"Рекомендуемый тендер: {tender_n} недел{'я' if tender_n == 1 else 'и' if 2 <= tender_n <= 4 else 'ь'}")

                # Вывод детализированных данных
                st.subheader("Детали прогноза:")
                with st.container():
                    st.write("**Даты и цены:**")
                    display_df = forecast_df.reset_index()[['Date', 'Price', 'Type', 'Объем закупки (N)']]
                    display_df['Price'] = display_df['Price'].round(2)
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_df.rename(columns={'Price': 'Цена (руб)', 'Type': 'Тип', 'Объем закупки (N)': 'Объем закупки (недели)'}))
                
                # Кнопка для скачивания
                st.subheader("Скачать прогноз:")
                output = io.BytesIO()
                export_df = forecast_df.reset_index()[['Date', 'Price', 'Type', 'Объем закупки (N)']]
                export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
                export_df['Price'] = export_df['Price'].round(2)
                export_df = export_df.rename(columns={'Price': 'Цена (руб)', 'Type': 'Тип', 'Объем закупки (N)': 'Объем закупки (недели)'})
                export_df.to_excel(output, index=False)
                output.seek(0)
                st.download_button(
                    label="Скачать таблицу с прогнозом",
                    data=output,
                    file_name=f"forecast_{start_date.strftime('%Y-%m-%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()