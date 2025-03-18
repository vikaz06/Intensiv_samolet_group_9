import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import re

import pandas as pd
import numpy as np
import re
from datetime import datetime

def auto_data_processing(file_path, output_file='processed_data.xlsx', exclude_columns=[]):
    # Загрузка данных
    df = pd.read_excel(file_path)
    
    # Обрабатываем колонку Date отдельно перед общей обработкой
    date_columns = [col for col in df.columns if col.lower() in ['date', 'дата'] and col not in exclude_columns]
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='raise')
            exclude_columns.append(col)  # Добавляем в исключения после успешного преобразования
        except:
            pass

    # Функция для преобразования специальных числовых форматов
    def convert_special_numbers(val):
        if isinstance(val, str):
            val = val.replace(' ', '').replace('\u202f', '').replace('\xa0', '')
            
            # Обработка процентов
            if '%' in val:
                num = val.replace('%', '')
                try:
                    return float(num) / 100
                except:
                    return val
                
            # Обработка чисел с разделителями
            if re.match(r'^[+-]?\d{1,3}(,\d{3})*(\.\d+)?$', val):
                return float(val.replace(',', ''))
            
            if re.match(r'^[+-]?\d{1,3}(\.\d{3})*(,\d+)?$', val):
                return float(val.replace('.', '').replace(',', '.'))
            
        return val

    # Функция для автоматического определения типов (исключая заданные колонки)
    def detect_and_convert_dtypes(series):
        col_name = series.name
        
        # Пропускаем исключенные колонки
        if col_name in exclude_columns:
            return series
        
        # Предварительная обработка
        series = series.map(convert_special_numbers)
        # Попытка преобразования в число
        try:
            return pd.to_numeric(series, errors='raise')
        except:
            pass
        
        # Проверим на булевый тип
        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({'Yes', 'No', 'True', 'False', 'Y', 'N', 0, 1}):
            return series.replace({'Yes': True, 'No': False, 'Y': True, 'N': False, 'True': True, 'False': False})
        
        # Для категориальных данных с малым числом уникальных значений
        if len(unique_vals) / len(series) < 0.1:
            return series.astype('category')
            
        return series

    # Применяем преобразование типов ко всем колонкам
    df = df.apply(lambda x: detect_and_convert_dtypes(x) if x.name not in exclude_columns else x)
    
    # Функция для заполнения пропущенных значений
    def fill_missing_values(series):
        if pd.api.types.is_numeric_dtype(series):
            # Для числовых данных используем медиану
            return series.fillna(series.median())
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Для дат используем наиболее частую дату
            return series.fillna(series.mode()[0])
        else:
            # Для остальных типов - самое частое значение
            return series.fillna(series.mode()[0])
    
    # Заполняем пропуски
    df = df.apply(fill_missing_values)
    
    # Функция для обработки аномалий
    def remove_anomalies(series):
        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 + 1.5*iqr
            return series.clip(lower=lower_bound, upper=upper_bound)
        return series
    
    # Обрабатываем аномалии для числовых колонок
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(remove_anomalies)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Генерируем отчет о данных
    report = {
        'columns': [],
        'missing_values_before': [],
        'missing_values_after': [],
        'dtypes': [],
        'anomalies_fixed': []
    }
    
    for col in df.columns:
        report['columns'].append(col)
        report['dtypes'].append(str(df[col].dtype))
        report['missing_values_before'].append(df[col].isna().sum())
        report['missing_values_after'].append(0)  # После обработки пропусков
        if col in numeric_cols:
            report['anomalies_fixed'].append((df[col] != remove_anomalies(df[col])).sum())
        else:
            report['anomalies_fixed'].append(0)
    
    # Сохраняем обработанные данные
    df.to_excel(output_file, index=False)
    
    # Выводим отчет
    print("Обработка данных завершена!")
    print(f"Сохранено в файл: {output_file}")
    print("\nОтчет о данных:")
    print(pd.DataFrame(report))
    
    return df

# Использование
file_path = 'CHMF Акции.xlsx'  # Укажите путь к вашему файлу
processed_data = auto_data_processing(file_path, output_file="CHMF Акции_mod.xlsx", exclude_columns=['Date'])

