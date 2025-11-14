import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib


def limpieza_basica(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    num = df.select_dtypes(include=["number"]).columns
    cat = df.select_dtypes(include=["object"]).columns

    for c in num:
        df[c].fillna(df[c].median(), inplace=True)
    for c in cat:
        df[c].fillna("Desconocido", inplace=True)

    if "Sales" in df.columns and "Profit" in df.columns:
        df["profit_margin"] = df["Profit"] / df["Sales"].replace({0: np.nan})
        df["profit_margin"].fillna(0, inplace=True)

    return df


def feature_engineering(df):
    df = df.copy()
    fechas = [c for c in df.columns if "date" in c.lower()]

    if fechas:
        c = fechas[0]
        df[c] = pd.to_datetime(df[c], errors="coerce")
        df["order_year"] = df[c].dt.year
        df["order_month"] = df[c].dt.month
        df["order_dow"] = df[c].dt.dayofweek

    if "Discount" in df.columns and "Sales" in df.columns:
        df["sales_adj"] = df["Sales"] * (1 - df["Discount"])

    return df


def seleccionar_features(df, target="Sales"):
    y = df[target]
    X = df.drop(columns=[target])

    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    cat = [c for c in cat if X[c].nunique() <= 20]

    return X[num + cat], y, num, cat


def construir_modelo(num, cat, tipo="rf"):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat)
    ])

    if tipo == "linear":
        model = LinearRegression()
    elif tipo == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def entrenar_y_evaluar(X, y, tipo="rf"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()

    pipeline = construir_modelo(num, cat, tipo)
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred))
    }

    return pipeline, metrics


st.set_page_config(page_title="Ventas App", layout="wide")

st.title("📊 Análisis y Predicción de Ventas")

archivo = st.sidebar.file_uploader("Sube tu dataset", type=["csv"])
modelo_subido = st.sidebar.file_uploader("Cargar modelo .joblib", type=["joblib"])
entrenar = st.sidebar.checkbox("Entrenar modelo aquí")


if archivo:
    df = pd.read_csv(archivo)
    df = limpieza_basica(df)
    df = feature_engineering(df)

    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.subheader("Estadísticas")
    st.write(df.describe())

    if "Region" in df.columns and "Sales" in df.columns:
        st.subheader("Ventas por región")
        st.bar_chart(df.groupby("Region")["Sales"].sum())

    st.subheader("Correlación")
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.header("🔮 Modelo Predictivo")

    if modelo_subido:
        model = joblib.load(modelo_subido)
        st.success("Modelo cargado")

    elif entrenar:
        target = st.selectbox("Target", df.columns)
        tipo = st.selectbox("Modelo", ["linear", "ridge", "rf"])

        if st.button("Entrenar modelo"):
            X, y, num, cat = seleccionar_features(df, target)
            model, metrics = entrenar_y_evaluar(X, y, tipo)
            joblib.dump(model, "model_trained.joblib")
            st.success("Modelo entrenado")
            st.json(metrics)

else:
    st.info("Sube un CSV para empezar.")
