import streamlit as st
import pandas as pd

st.title("📊 Analiza Performanței și Extinderii Organizației")
st.write("Acesta este un dashboard interactiv creat pentru analiza vânzărilor Superstore.")

@st.cache_data 
def incarca_date():
    df = pd.read_csv("date.csv", encoding="windows-1252")
    return df

df = incarca_date()

st.subheader("1. Vizualizarea Datelor Inițiale")
st.dataframe(df.head())

st.subheader("2. Curățarea Datelor (Tratarea valorilor lipsă)")

valori_lipsa_inainte = df.isnull().sum().sum()
st.write(f"Valori lipsă găsite în setul de date inițial: **{valori_lipsa_inainte}**")

df_curat = df.dropna()

valori_lipsa_dupa = df_curat.isnull().sum().sum()
st.write(f"Valori lipsă după procesul de curățare: **{valori_lipsa_dupa}**")


st.subheader("3. Analiza Vânzărilor și a Profitului")

df_grupate = df_curat.groupby('Category')[['Sales', 'Profit']].sum().reset_index()

st.write("Acesta este un tabel agregat cu totalul vânzărilor și profitului pe fiecare categorie în parte:")
st.dataframe(df_grupate)

st.write("Reprezentare grafică a vânzărilor și profitului:")
st.bar_chart(df_grupate, x="Category", y=["Sales", "Profit"])

st.subheader("4. Codificarea Datelor (Pregătire pentru Machine Learning)")
st.write("Algoritmii matematici nu înțeleg cuvinte (cum ar fi 'Furniture' sau 'Technology'). De aceea, trebuie să le transformăm în etichete numerice (0, 1, 2...).")

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_pregatit = df_curat.copy()

df_pregatit['Category_Codificat'] = encoder.fit_transform(df_pregatit['Category'])

st.dataframe(df_pregatit[['Category', 'Category_Codificat']].head(10))


st.subheader("5. Scalarea Datelor pentru Regiuni")
st.write("Pentru a găsi tipare corecte, aducem valorile la aceeași scară matematică (StandardScaler).")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df_state = df_curat.groupby('State')[['Sales', 'Profit']].sum().reset_index()

scaler = StandardScaler()

df_state[['Sales_Scaled', 'Profit_Scaled']] = scaler.fit_transform(df_state[['Sales', 'Profit']])

st.write("Tabelul la nivel de stat, cu valorile originale și cele scalate:")
st.dataframe(df_state.head())

st.subheader("6. Împărțirea regiunilor în Clustere (K-Means)")
st.write("Am folosit algoritmul K-Means pentru a împărți statele în 3 grupuri de performanță (Clustere).")

kmeans = KMeans(n_clusters=3, random_state=42)

df_state['Cluster'] = kmeans.fit_predict(df_state[['Sales_Scaled', 'Profit_Scaled']])

df_state['Cluster'] = df_state['Cluster'].astype(str)

st.write("Graficul de mai jos arată cum au fost grupate statele în funcție de Vânzări și Profit:")

st.scatter_chart(df_state, x='Sales', y='Profit', color='Cluster')



st.subheader("7. Ce influențează Profitul? (Regresie Multiplă)")
st.write("Folosim `statsmodels` pentru a vedea cum influențează Vânzările și Reducerile (Discount) nivelul Profitului.")

import statsmodels.api as sm

X = df_curat[['Sales', 'Discount']]
Y = df_curat['Profit']

X_sm = sm.add_constant(X)

model = sm.OLS(Y, X_sm).fit()

st.write(f"**R-pătrat (R-squared):** {model.rsquared:.2f} (arată cât la sută din variația profitului este explicată de acest model)")

st.text(model.summary())

st.info("💡 **Interpretare economică:** Dacă te uiți în tabel la coeficientul (coef) pentru 'Discount', vei vedea un număr negativ mare. Asta înseamnă că oferirea de reduceri prea mari 'mănâncă' rapid din profitabilitatea organizației.")

st.subheader("8. Harta Profitabilității pe State (Geopandas)")
st.write("Afișăm o hartă a SUA pentru a vizualiza zonele profitabile și cele cu pierderi.")

import geopandas as gpd
import matplotlib.pyplot as plt

url_harta = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

try:
    with st.spinner('Se construiește harta geografică (poate dura câteva secunde)...'):
        harta_sua = gpd.read_file(url_harta)
        
        harta_profit = harta_sua.merge(df_state, left_on="name", right_on="State", how="inner")
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        harta_profit.plot(column='Profit', ax=ax, legend=True, 
                          legend_kwds={'label': "Profit Total pe Stat"},
                          cmap='RdYlGn', 
                          missing_kwds={"color": "lightgrey"})
        
        ax.set_axis_off()
        plt.title("Performanța Financiară (Profit) pe Statele din SUA", fontsize=18)
        
        st.pyplot(fig)
        
except Exception as e:
    st.error(f"A apărut o problemă la încărcarea hărții: {e}")