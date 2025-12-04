import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import requests
from io import BytesIO

MODEL_PATH = 'https://raw.githubusercontent.com/smorchkova001-git/ML_models/refs/heads/main/models/model.pkl'
FEATURE_NAMES_PATH = 'https://raw.githubusercontent.com/smorchkova001-git/ML_models/refs/heads/main/models/feature_names.pkl'

@st.cache_resource
def load_model():
    model_response = requests.get(MODEL_PATH)
    model = pickle.load(BytesIO(model_response.content))
    
    feature_names_response = requests.get(FEATURE_NAMES_PATH)
    feature_names = pickle.load(BytesIO(feature_names_response.content))
    
    return model, feature_names

try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

st.title("Коэффициенты модели")

#=====================ГИСТОГРАММА КОЭФФИЦИЕНТОВ=====================
fig = px.histogram(
    x=MODEL.best_estimator_.named_steps['ridge'].coef_, 
    nbins=20,
    color_discrete_sequence=['#ffad8f'])

fig.update_layout(
    xaxis_title='Коэффициенты',
    yaxis_title='Частота'
)

st.plotly_chart(fig)

#===================== ТАБЛИЦА КОЭФФИЦИЕНТОВ=====================
coefficients = MODEL.best_estimator_.named_steps['ridge'].coef_
coef_df = pd.DataFrame({'Признак': FEATURE_NAMES, 'Коэффициент': coefficients.round(0)})

coef_df = coef_df.sort_values('Коэффициент', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)
st.dataframe(coef_df[['Признак', 'Коэффициент']])

# Ссылка на GitHub
with st.sidebar:
    st.markdown("---")
    st.markdown("**👩‍💻 Автор:** Сморчкова Юлиана")
    st.markdown("**🔗 Подробнее на** [GitHub](https://github.com/smorchkova001-git/ML_models/tree/main)")
    st.markdown("---")