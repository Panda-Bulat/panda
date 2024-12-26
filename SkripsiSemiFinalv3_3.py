#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle

page_bg_img = """
<style>
/* Latar belakang utama */
[data-testid="stAppViewContainer"]{
    background-image: url(https://images.unsplash.com/photo-1485963631004-f2f00b1d6606?q=80&w=1975&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D);
    background-size: cover;
    # color: #5A3E36; /* Warna cokelat gelap */
    color : white;
}

[data-testid="stAppViewContainer"]::before {
    content: ""; /* Membuat pseudo-element */
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.75); /* Layer hitam transparan */
    z-index: 0; /* Pastikan layer ini muncul di atas gambar */
    pointer-events: none; /* Supaya layer tidak mengganggu interaksirea pengguna */
}

/* Header transparan */
[data-testid="stHeader"]{
    background-color: rgba(0, 0, 0, 0);
}

/* Kotak melengkung di tengah */
# .central-box {
#         # background-color: rgba(255, 255, 255, 0.9); /* Warna putih transparan */
#         background-color: rgba(0, 0, 0, 0.8);
#         padding: 40px; /* Ruang dalam */
#         border-radius: 20px; /* Sudut melengkung */
#         box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3); /* Efek bayangan */
#         width: 120%; /* Lebar kotak */
#         height: 200vw;
#         margin: 0 auto; /* Pusatkan secara horizontal */
#         margin-top: 45%; /* Jarak dari atas */
#         text-align: center; /* Rata tengah */
#         position: absolute; /* Posisi absolut untuk diatur di belakang */
#         top: 50%; /* Atur posisi vertikal */
#         left: 50%; /* Atur posisi horizontal */
#         transform: translate(-50%, -50%); /* Pusatkan secara sempurna */
#         z-index: 0; /* Letakkan di belakang elemen lain */
# }

/* Teks tetap bersih */
h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    # color: #5A3E36 !important;
    color: white !important;
}

/* Bayangan untuk input box dan tombol */
.stTextInput>div, .stButton>button, .stTextArea>div, select {
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3); /* Tambahkan bayangan */
    border-radius: 8px; /* Membulatkan sudut */
    border: 1px solid #DEB887; /* Warna cokelat muda untuk border */
}

/* Tampilan tombol */
.stButton>button {
    background-color: #F5DEB3; /* Warna tombol */
    # color: #5A3E36 !important;
    color: white !important;
    padding: 10px 15px;
    border-radius: 8px;
}

.stButton>button:hover {
    background-color: #DEB887; /* Warna hover tombol */
    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4); /* Bayangan lebih tebal saat hover */
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# # Tambahkan div wrapper untuk membungkus semua elemen utama
# st.markdown(
#     """
#     <div class="central-box">
#     </div>
#     """,
#     unsafe_allow_html=True
# )
    
nltk.download("punkt", quiet=True)

# Initialize and preprocess data and model
@st.cache_resource
def initialize_data_and_model():
    try:
        # Load dataset
        merged_data = pd.read_csv("merged_dataset.csv", encoding="utf-8")
        merged_data['clean_tokens'] = merged_data['clean_tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Load FastText model menggunakan pickle
        with open("fasttext_model.pkl", "rb") as file:
            fasttext_model = pickle.load(file)

        return merged_data, fasttext_model
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return pd.DataFrame(), None

# Filter by category function
def filter_by_category(data, selected_categories):
    return data[data['kategori'].isin(selected_categories)].reset_index(drop=True)

# Filter by allergens function
def filter_by_allergens(data, user_allergens):
    filtered_data = data.copy()
    for allergen in user_allergens:
        filtered_data = filtered_data[~filtered_data['ingredients'].str.contains(allergen, case=False, na=False)]
    return filtered_data.reset_index(drop=True)

# FastText Recommendation Function
def recommend_recipes_fasttext(data, user_input, fasttext_model, top_n=5):
    user_input_tokens = word_tokenize(user_input)
    user_input_vector = np.zeros(fasttext_model.vector_size)
    num_tokens = 0
    for token in user_input_tokens:
        if token in fasttext_model.wv:
            user_input_vector += fasttext_model.wv[token]
            num_tokens += 1
    if num_tokens > 0:
        user_input_vector /= num_tokens

    recipe_vectors = []
    for recipe_tokens in data['clean_tokens']:
        recipe_vector = np.zeros(fasttext_model.vector_size)
        num_tokens = 0
        for token in recipe_tokens:
            if token in fasttext_model.wv:
                recipe_vector += fasttext_model.wv[token]
                num_tokens += 1
        if num_tokens > 0:
            recipe_vector /= num_tokens
        recipe_vectors.append(recipe_vector)

    recipe_vectors = np.array(recipe_vectors)
    similarities = cosine_similarity([user_input_vector], recipe_vectors)[0]

    max_loves = data['Loves'].max()
    data['normalized_loves'] = data['Loves'] / max_loves
    combined_scores = 0.8 * similarities + 0.2 * data['normalized_loves']

    data['score'] = combined_scores
    recommendations = data.nlargest(top_n, 'score')[['Title', 'Loves', 'score', 'ingredients', 'Steps']]
    return recommendations

# Main application
def main():

    st.markdown(
        """
        <div class="bg-opacity"></div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Recipe Recommendation System")

    if "merged_data" not in st.session_state or "fasttext_model" not in st.session_state:
        # Tampilkan warning hanya saat data belum diinisialisasi
        with st.spinner("Initializing data and model, please wait..."):
            st.session_state.merged_data, st.session_state.fasttext_model = initialize_data_and_model()

    merged_data = st.session_state.merged_data
    fasttext_model = st.session_state.fasttext_model

    # Predefined categories
    available_categories = ["ayam", "ikan", "kambing", "sapi", "tahu", "telur", "tempe", "udang"]

    # Step 1: User selects categories
    st.subheader("Step 1: Pilih Kategori")
    selected_categories = st.multiselect("Pilih kategori", options=available_categories)

    # Step 2: User enters allergens
    st.subheader("Step 2: Pilih bahan alergen")
    st.write("*Jika terdapat typo pada input alergen, hasil rekomendasi bisa saja tidak akurat")
    user_allergens = st.text_input("Masukkan bahan alergen (dipisah tanda koma (,) misal: telur, udang)")

    # Step 3: User enters ingredients
    st.subheader("Step 3: Pilih bahan makanan yang ingin dimasukkan")
    st.write("*Jika terdapat typo pada input bahan makanan, hasil rekomendasi bisa saja berbeda")
    user_ingredients = st.text_input("Masukkan bahan makanan (dipisah tanda koma (,) misal: telur, udang)")

    # Submit button
    if st.button("Submit"):
        if selected_categories and user_allergens and user_ingredients:
            # Step 1: Filter by category
            filtered_data = filter_by_category(merged_data, selected_categories)

            # Step 2: Filter by allergens
            user_allergens = [allergen.strip() for allergen in user_allergens.split(',')]
            filtered_data = filter_by_allergens(filtered_data, user_allergens)

            # Step 3: Use user ingredients to generate recommendations
            user_ingredients = [ingredient.strip() for ingredient in user_ingredients.split(',')]
            user_ingredients_str = ' '.join(user_ingredients)

            # Step 4: Get recommendations
            recommendations = recommend_recipes_fasttext(filtered_data, user_ingredients_str, fasttext_model)

            # Step 5: Display recommendations
            st.subheader("Top Recommended Recipes")
            if not recommendations.empty:
                for idx, row in recommendations.iterrows():
                    with st.expander(f"{idx + 1}. {row['Title']} (Score: {row['score']:.2f})"):
                        # Display Ingredients
                        st.write("**Ingredients:**")
                        ingredients_list = row['ingredients'].split(";")
                        for ing in ingredients_list:
                            if ing.strip():  # Only display non-empty ingredients
                                st.markdown(f"- {ing.strip()}")

                        # Display Steps
                        st.write("**Steps:**")
                        steps_list = row['Steps'].split("--")
                        for step in steps_list:
                            if step.strip():  # Only display non-empty steps
                                st.markdown(f"- {step.strip()}")
            else:
                st.write("No recommendations found. Please adjust your inputs.")
        else:
            st.warning("Please fill out all inputs before submitting!")

if __name__ == "__main__":
    main()
    
