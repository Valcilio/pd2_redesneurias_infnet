import requests
import json

import streamlit as st
import pandas as pd

# Título do app
st.title("Julgador de Ações")

# Inputs de float
close_price = st.number_input("Close Price", format="%.2f", value=5.0)
dividend_yield = st.number_input("Dividend Yield", format="%.2f", value=4.5)
p_l = st.number_input("P/L", format="%.2f", value=10.0)
p_vp = st.number_input("P/VP", format="%.2f", value=1.5)
margem_liquida = st.number_input("Margem Liquida", format="%.2f", value=15.0)
divida_liq_ebit = st.number_input("Divida Liquida/Ebit", format="%.2f", value=2.0)
liquidez_corrente = st.number_input("Liquidez Corrente", format="%.2f", value=1.5)
cagr_lucros_5_anos = st.number_input("CAGR Lucros 5 Anos", format="%.2f", value=8.0)
liquidez_diaria = st.number_input("Liquidez Diaria", format="%.2f", value=1.2)
vpa = st.number_input("VPA", format="%.2f", value=3.0)
lpa = st.number_input("LPA", format="%.2f", value=0.5)
annual_dividend = st.number_input("Annual Dividend", format="%.2f", value=0.2)

# Inputs de string
setor = st.text_input("Setor", value='Financeiro')
subsetor = st.text_input("Subsetor", value='Bancos')
segmento = st.text_input("Segmento", value='Banco Múltiplo')

df = pd.DataFrame({
    'close_price': [close_price],
    'dividend_yield': [dividend_yield],
    'p/l': [p_l],
    'p/vp': [p_vp],
    'margem_liquida': [margem_liquida],
    'divida_liq/ebit': [divida_liq_ebit],
    'liquidez_corrente': [liquidez_corrente],
    'cagr_lucros_5_anos': [cagr_lucros_5_anos],
    'liquidez_diaria': [liquidez_diaria],
    'vpa': [vpa],
    'lpa': [lpa],
    'annual_dividend': [annual_dividend],
    'setor_atua': [setor],
    'subsetor_atua': [subsetor],
    'segmento_atua': [segmento]
})


if st.button("Gerar Score"):
    json_df = df.to_json()
    header = {'Content-type': 'application/json'}
    framework_data = {'data': json_df}
    framework_json = json.dumps(framework_data)
    response = requests.post('http://127.0.0.1:5000/predict', data=framework_json, headers=header)
    st.write("O Score da Ação é:", json.loads(response.content)['prediction'])