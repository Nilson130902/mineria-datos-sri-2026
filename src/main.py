"""
===============================================================================
UNIVERSIDAD ESTATAL AMAZÓNICA (UEA)
Asignatura: Minería de Datos (UEA-L-UFPTI-009)
Práctica N° 1: Minería de datos aplicada sobre el Catastro Vehicular SRI 2026
Estudiante: Nilson Alexis López Miranda
===============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS Y EXPLORACIÓN INICIAL (EDA)
# -----------------------------------------------------------------------------
def load_data():
    csv_path = os.path.join('data', 'SRI_Vehiculos_Nuevos_2026.csv')
    if not os.path.exists(csv_path):
        csv_path = 'SRI_Vehiculos_Nuevos_2026.csv'
        
    print("[+] Cargando dataset de matriculaciones de vehículos...")
    df = pd.read_csv(csv_path, sep=';', encoding='latin1')
    print(f"    - Registros totales: {df.shape[0]:,}")
    print(f"    - Variables totales: {df.shape[1]}")
    return df

def exploratory_data_analysis(df):
    print("\n" + "="*50)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*50)
    
    # 1. Nulos por variable
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    eda_df = pd.DataFrame({'Valores_Nulos': null_counts, 'Porcentaje_%': null_pct.round(4)})
    print("\n--- Reporte de Valores Nulos ---")
    print(eda_df[eda_df['Valores_Nulos'] > 0])
    
    # 2. Registros duplicados
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100
    print(f"\n--- Registros Duplicados ---")
    print(f"    - Cantidad: {dup_count:,} ({dup_pct:.2f}%)")

# -----------------------------------------------------------------------------
# 2. PREPROCESAMIENTO Y FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def preprocess_data(df):
    print("\n" + "="*50)
    print("PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS")
    print("="*50)
    
    df_clean = df.copy()
    
    # Conversión de AVALUO a flotante
    df_clean['AVALUO_NUM'] = df_clean['AVALUO'].astype(str).str.replace(',', '.').astype(float)
    
    # Depuración de registros con nulos en variables clave
    clave_cols = ['MARCA', 'CLASE', 'TIPO COMBUSTIBLE', 'CILINDRAJE', 'AVALUO_NUM', 'Tipo Identificac Potenc Prop']
    df_clean = df_clean.dropna(subset=clave_cols)
    
    # Filtrado de outliers en AVALUO (conservar valores entre $500 y $200,000 USD)
    df_clean = df_clean[(df_clean['AVALUO_NUM'] >= 500) & (df_clean['AVALUO_NUM'] <= 200000)]
    
    # Variable Derivada 1: RANGO_CILINDRAJE
    def categorizar_cilindraje(c):
        if c == 0:
            return 'Electrico/Sin Cilindraje'
        elif c <= 1200:
            return 'Bajo (<=1200cc)'
        elif c <= 2000:
            return 'Medio (1201-2000cc)'
        else:
            return 'Alto (>2000cc)'
            
    df_clean['RANGO_CILINDRAJE'] = df_clean['CILINDRAJE'].apply(categorizar_cilindraje)
    
    # Variable Derivada 2: RATIO_AVALUO_CIL
    df_clean['RATIO_AVALUO_CIL'] = np.where(
        df_clean['CILINDRAJE'] > 0, 
        df_clean['AVALUO_NUM'] / df_clean['CILINDRAJE'], 
        0
    )
    
    print("[+] Limpieza y Feature Engineering completados con éxito.")
    print(f"    - Registros filtrados para modelado: {len(df_clean):,}")
    return df_clean

# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# -----------------------------------------------------------------------------
def train_and_evaluate_models(df):
    print("\n" + "="*50)
    print("MODELADO Y EVALUACIÓN DE MINERÍA DE DATOS")
    print("="*50)
    
    # Muestreo representativo de 20,000 registros para velocidad y estabilidad de evaluación
    df_sample = df.sample(n=min(20000, len(df)), random_state=42)
    
    # Definición de X e y (Target: 1 = Persona Jurídica / Empresa, 0 = Persona Natural)
    X = df_sample[['MARCA', 'CLASE', 'TIPO COMBUSTIBLE', 'CILINDRAJE', 'AVALUO_NUM', 'RATIO_AVALUO_CIL']]
    y = (df_sample['Tipo Identificac Potenc Prop'] == 'JURIDICA').astype(int)
    
    # Proporción: 80% Entrenamiento, 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Column Transformers para variables categóricas y numéricas
    categorical_cols = ['MARCA', 'CLASE', 'TIPO COMBUSTIBLE']
    numeric_cols = ['CILINDRAJE', 'AVALUO_NUM', 'RATIO_AVALUO_CIL']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Definición de los 3 Modelos de Minería de Datos
    models = {
        'Árbol de Decisión': DecisionTreeClassifier(max_depth=10, criterion='gini', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        'Regresión Logística': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n[+] Entrenando modelo: {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Entrenamiento
        pipeline.fit(X_train, y_train)
        
        # Predicción
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Validación Cruzada (5-Fold Cross Validation)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        
        # Cálculo de Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Modelo': name,
            'Accuracy': f"{acc*100:.2f}%",
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc, 4),
            'CV F1 Mean': round(cv_scores.mean(), 4)
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("TABLA COMPARATIVA DE RESULTADOS DE LOS MODELOS")
    print("="*70)
    print(res_df.to_string(index=False))
    return res_df

# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    df_raw = load_data()
    exploratory_data_analysis(df_raw)
    df_processed = preprocess_data(df_raw)
    results_table = train_and_evaluate_models(df_processed)