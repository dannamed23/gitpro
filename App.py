import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load the df_sample.csv file
df_sample = pd.read_csv('PISA_df_sample.csv')

# Display the head of the dataframe
print(df_sample.head())

# Load the dataset
@st.cache_resource
def load_data():
    return pd.read_csv('PISA_df_sample.csv')

df = load_data()

user_ids = df['id_cliente'].astype('category').cat.codes.values
item_ids = df['articulo'].astype('category').cat.codes.values
df['user_id'] = user_ids
df['item_id'] = item_ids

interaction_matrix = df.pivot_table(index=['id_cliente', 'trimestre'], columns='articulo', values='unidades_venta', fill_value=0).reset_index()

# Neural Network Model
def build_nn_model(num_users, num_items):
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, 50)(user_input)
    item_embedding = Embedding(num_items, 50)(item_input)
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    concat = Concatenate()([user_vec, item_vec])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1)(dense)
    model = Model([user_input, item_input], output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Train the Neural Network Model
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
X = df[['user_id', 'item_id']].values
y = df['unidades_venta'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nn_model = build_nn_model(num_users, num_items)
nn_model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=64, validation_data=([X_test[:, 0], X_test[:, 1]], y_test))

# KNN Model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(interaction_matrix.iloc[:, 2:].values)

# Streamlit UI
st.title('Pisa product recommendator')

user_id = st.selectbox('Select User ID', df['id_cliente'].unique())
trimester = st.selectbox('Select Trimester', df['trimestre'].unique())
recommendation_type = st.selectbox('Select Recommendation Type', ['Neural Network', 'KNN'])

if st.button('Get Recommendations'):
    if recommendation_type == 'Neural Network':
        # Neural Network Recommendations
        user_idx = df[df['id_cliente'] == user_id]['user_id'].values[0]
        trimester_data = interaction_matrix[interaction_matrix['trimestre'] == trimester].drop(columns=['id_cliente', 'trimestre'])
        item_ids = trimester_data.columns
        user_ids = np.full(len(item_ids), user_idx)
        predictions = nn_model.predict([user_ids, np.arange(len(item_ids))])
        top_items = item_ids[np.argsort(predictions.flatten())[::-1][:5]]
        st.write('Neural Network Recommendations for user:', user_id, 'in trimester:', trimester)
        st.write(top_items)
    else:
        # KNN Recommendations
        user_idx = interaction_matrix[(interaction_matrix['id_cliente'] == user_id) & (interaction_matrix['trimestre'] == trimester)].index[0]
        distances, indices = knn.kneighbors(interaction_matrix.iloc[user_idx, 2:].values.reshape(1, -1), n_neighbors=6)
        similar_users = interaction_matrix.iloc[indices.flatten()[1:], :2]
        recommended_items = interaction_matrix.iloc[similar_users.index, 2:].sum().sort_values(ascending=False).index[:5]
        st.write('KNN Recommendations for user:', user_id, 'in trimester:', trimester)
        st.write(recommended_items)
