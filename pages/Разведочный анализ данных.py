import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud


st.title('Разведочный анализ данных')

# Загружаем данные
df_train = pd.read_csv('https://raw.githubusercontent.com/smorchkova001-git/ML_models/refs/heads/main/df_train_clean.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/smorchkova001-git/ML_models/refs/heads/main/df_test_clean.csv')

#====================ВЫБИРАЕМ ДАТАСЕТ=====================
dataset = st.selectbox('Выберите датасет', options=['df_train', 'df_test'])
if dataset == 'df_train':
    df = df_train
else:
    df = df_test

#=====================ДАТАСЕТ=====================
st.subheader('Шаг 1: Датасет')
cols_selected = st.multiselect('Выберите колонки для отображения', options=df.columns)
if cols_selected:
    st.dataframe(df[cols_selected])
else:
    st.write(df)

#=====================ОСНОВНЫЕ СТАТИСТИКИ=====================
st.subheader('Шаг 2: Основные статистики')
st.write(df.describe())


#=====================ГИСТОГРАММА=====================
st.subheader('Шаг 3: Гистограмма/столбчатая диаграмма')
column = st.selectbox('Выберите колонку', df.columns)
bins = st.slider('Количество интервалов (bins)', 5, 100, 20)

fig, ax = plt.subplots(figsize=(12, 7))

if df[column].dtype == 'object':
    order = df[column].value_counts().index
    sns.countplot(data=df, y=column, ax=ax, color='#4aab6b', order=order)
    ax.set_xlabel('Количество')
    ax.set_ylabel(column)
else:
    sns.histplot(data=df, x=column, bins=bins, ax=ax, color='#ffad8f')

plt.title(f'Распределение: {column}')
plt.tight_layout()
st.pyplot(fig)


#=====================КОРРЕЛЯЦИОННЫЕ МАТРИЦЫ=====================

st.subheader('Шаг 4: Корреляционные матрицы')
corr_type = st.selectbox('Выберите тип корреляции', options=['Пирсон', 'Спирмен'])
METHOD = ['pearson' if corr_type == 'Пирсон' else 'spearman']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(method=METHOD[0], numeric_only=True), cmap="Blues", annot=True, ax=ax)
st.pyplot(fig)


#=====================SCATTER PLOT=====================
st.subheader('Шаг 5: Scatter plot')
x_col = st.selectbox('Выберите колонку для оси X', df.columns)
y_col = st.selectbox('Выберите колонку для оси Y', df.columns, index=list(df.columns).index('selling_price'))
fig = px.scatter(df, x=x_col, y=y_col)
st.plotly_chart(fig)

#=====================ОБЛАКО СЛОВ ДЛЯ МАРОК АВТОМОБИЛЕЙ=====================
st.subheader('Шаг 6: Облако слов для марок автомобилей')

text = ' '.join(df['name'].dropna().astype(str))
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Set3'
    ).generate(text)
    
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

#=====================BOX PLOT ДЛЯ ЦЕЛЕВОГО ПРИЗНАКА=====================
st.subheader('Шаг 7: Box plot для целевого признака')
fig, ax = plt.subplots(figsize=(15, 7))
sns.boxplot(data=df, x='selling_price', ax=ax, color='lightblue')
ax.set_xlabel('selling_price')
plt.tight_layout()
st.pyplot(fig)


# Ссылка на GitHub
with st.sidebar:
    st.markdown("---")
    st.markdown("**👩‍💻 Автор:** Сморчкова Юлиана")
    st.markdown("**🔗 Подробнее на** [GitHub](https://github.com/smorchkova001-git/ML_models/tree/main)")
    st.markdown("---")