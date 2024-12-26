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
import gdown

page_bg_img = """
<style>
/* Latar belakang utama */
[data-testid="stAppViewContainer"] {
    background-image: url(https://images.unsplash.com/photo-1485963631004-f2f00b1d6606?q=80&w=1975&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D);
    background-size: cover;
    color: white;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    z-index: 0;
    pointer-events: none;
}

/* Header transparan */
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* Gaya untuk semua teks */
h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: "Roboto", sans-serif;
    color: #F5F5F5 !important;
}

/* Gaya input dan tombol */
.stTextInput>div, .stButton>button, .stTextArea>div, select {
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    border: 1px solid #DEB887;
}

/* Tombol dengan warna yang menarik */
.stButton>button {
    background-color: #FFD700; /* Warna emas */
    color: black !important;
    padding: 10px 15px;
    border-radius: 8px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.stButton>button:hover {
    background-color: #FFA500; /* Warna oranye */
    transform: scale(1.05);
}

/* Tampilan subjudul */
.stSubtitle {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 10px;
    color: #FFD700;
}

/* Kotak hasil rekomendasi */
.expander-header {
    background-color: rgba(255, 215, 0, 0.1); /* Warna emas semi transparan */
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #FFD700;
    margin-bottom: 10px;
    box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
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
def load_model_from_drive():
    file_id = "1jJ3lUkbp2dK-TuWsB1c5-18pEen07T_E"  
    url = f"https://drive.google.com/uc?id={file_id}"  # Format URL untuk gdown
    output = "fasttext_model.pkl"  # Nama file output lokal

    gdown.download(url, output, quiet=False)

    try:
        with open(output, "rb") as file:
            fasttext_model = pickle.load(file)
    except Exception as e:
        raise ValueError(f"Failed to load the FastText model. Error: {e}")

    return fasttext_model

@st.cache_resource
def initialize_data_and_model():
    try:
        # Load dataset
        merged_data = pd.read_csv("merged_dataset.csv", encoding="utf-8")
        merged_data['clean_tokens'] = merged_data['clean_tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Load FastText model dari Google Drive
        fasttext_model = load_model_from_drive()

        return merged_data, fasttext_model
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return pd.DataFrame(), None

# Filter by category function
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
        # Tampilkan pesan loading yang lebih rapi
        with st.spinner("Initializing data and model, please wait..."):
            st.session_state.merged_data, st.session_state.fasttext_model = initialize_data_and_model()
    
        # Menampilkan teks kecil menggunakan markdown setelah inisialisasi selesai
        st.markdown('<small>Data and model initialized successfully.</small>', unsafe_allow_html=True)

    merged_data = st.session_state.merged_data
    fasttext_model = st.session_state.fasttext_model

    # Predefined categories
    available_categories = ["ayam", "ikan", "kambing", "sapi", "tahu", "telur", "tempe", "udang"]

    # Step 1: User selects categories
    st.subheader("Step 1: Pilih Kategori")
    selected_categories = st.multiselect("Pilih kategori", options=available_categories)

    # Step 2: User enters allergens
    st.subheader("Step 2: Pilih bahan alergen")
    user_allergens = st.text_input("Masukkan bahan alergen (dipisah tanda koma (,) misal: telur, udang)")
    st.markdown("""
        <p style="font-size: 10px; margin-bottom: 4px;">*Jika terdapat typo pada input alergen, hasil rekomendasi bisa saja tidak akurat.</p>
        <p style="font-size: 10px; margin-bottom: 4px;">*Jika tidak ada, silakan masukkan tanda minus (-).</p>""", unsafe_allow_html=True)

    # Step 3: User enters ingredients
    st.subheader("Step 3: Pilih bahan makanan yang ingin dimasukkan")
    user_ingredients = st.text_input("Masukkan bahan makanan (dipisah tanda koma (,) misal: telur, udang)")
    st.markdown("""
        <p style="font-size: 10px; margin-bottom: 5px;">*Jika terdapat typo pada input bahan makanan, hasil rekomendasi bisa saja berbeda.</p>""", unsafe_allow_html=True)

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
                    with st.expander(f"{idx + 1}. {row['Title']}"):
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
