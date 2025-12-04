import streamlit as st
import pandas as pd
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

st.title('Предсказание стоимости автомобиля')
st.write('Для оценки стоимости автомобиля введите его характеристики и нажмите кнопку "Оценить стоимость"')
st.markdown('*Если вашей категории нет в списке, выберите "Другое"*')

col1, col2 = st.columns(2)

with col1:
    name_types = ['Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia',
       'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota',
       'Volkswagen', 'Volvo', 'Другое']
    name = st.selectbox('Марка автомобиля (name):', name_types, placeholder='Выберите марку автомобиля')
    
    year = st.number_input('Год выпуска (year):', min_value=1900, value=None, placeholder='Введите значение') 
    km_driven = st.number_input('Пробег (km_driven), км:', min_value=0, value=None, placeholder='Введите значение')
    
    fuel_types = ['CNG', 'Diesel', 'LPG', 'Petrol', 'Другое']
    fuel = st.selectbox('Тип топлива (fuel):', fuel_types, placeholder='Введите значение') 
    
    seller_types = ['Dealer', 'Individual', 'Trustmark Dealer', 'Другое']
    seller_type = st.selectbox('Тип продавца (seller_type):', seller_types, placeholder='Введите значение')
    
    transmission_types = ['Manual', 'Automatic', 'Другое']
    transmission = st.selectbox('Тип коробки передач (transmission):', transmission_types, placeholder='Введите значение') 

with col2:
    owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car', 'Другое']
    owner = st.selectbox('Владелец (owner):', owner_types, placeholder='Введите значение') 

    mileage = st.number_input('Расход топлива (mileage), kmpl:', min_value=0.0, value=None, placeholder='Введите значение')
    engine = st.number_input('Объем двигателя (engine), CC:', min_value=0, value=None, placeholder='Введите значение')
    max_power = st.number_input('Максимальная мощность (max_power), bhp :', min_value=0.0, value=None, placeholder='Введите значение')
    
    seats_types = [2, 4, 5, 6, 7, 8, 9, 10, 14, 'Другое']
    seats = st.selectbox('Количество мест (seats):', seats_types, placeholder='Введите значение')

if st.button("Оценить стоимость", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame([{
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }])
    
    input_data = input_data.replace('Другое', 'NA')

    try:
        prediction = MODEL.predict(input_data)[0]
        st.success(f'### Предсказанная стоимость: **{prediction:.0f}**')

    except Exception as e:
        st.error(f'Ошибка при предсказании: {e}')
        st.write('Проверьте, что все поля заполнены корректно.')


# Ссылка на GitHub
with st.sidebar:
    st.markdown("---")
    st.markdown("**👩‍💻 Автор:** Сморчкова Юлиана")
    st.markdown("**🔗 Подробнее на** [GitHub](https://github.com/smorchkova001-git/ML_models/tree/main)")
    st.markdown("---")