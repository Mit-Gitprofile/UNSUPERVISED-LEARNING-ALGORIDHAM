import streamlit as st
import pandas as pd
import pickle
import os
import shutil

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="📱 Mobile Recommendation System",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Phone Recommendation System (Unsupervised Learning)")

# ------------------------------------------------
# Load Model and Dataset
# ------------------------------------------------
try:
    with open("model.pkl", "rb") as file:
        model, scaler, numeric_features = pickle.load(file)

    dataset_path = os.path.join(os.path.dirname(__file__), "mobile_recommendation_system_dataset copy.csv")

    if not os.path.exists(dataset_path):
        st.error("❌ Dataset not found in the folder.")
        st.stop()

    # Temporary copy to avoid Permission Denied error
    temp_copy = "temp_dataset_copy.csv"
    shutil.copy(dataset_path, temp_copy)
    df = pd.read_csv(temp_copy)

    st.success("✅ Model and dataset loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Error loading model or dataset: {e}")
    st.stop()

# ------------------------------------------------
# Auto-detect important columns
# ------------------------------------------------
def detect_column(df, keywords):
    """Find the column in df that best matches any of the keywords."""
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

name_col = detect_column(df, ["name", "model", "product"])
rating_col = detect_column(df, ["rating", "review"])
price_col = detect_column(df, ["price", "cost", "amount"])

if not name_col:
    st.error("❌ Could not find 'Name' column. Please check your CSV headers.")
    st.stop()

if not rating_col:
    st.warning("⚠️ 'Rating' column not found. Using default value 0.")
    df["Rating"] = 0
    rating_col = "Rating"

if not price_col:
    st.warning("⚠️ 'Price' column not found. Using default value 0.")
    df["Price"] = 0
    price_col = "Price"

# ------------------------------------------------
# 🧹 Clean and Convert Price Column
# ------------------------------------------------
if price_col in df.columns:
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)  # remove ₹, commas, etc.
        .replace('', '0')
        .astype(float)
    )

# ------------------------------------------------
# Ensure Cluster Column Exists
# ------------------------------------------------
if 'Cluster' not in df.columns and all(col in df.columns for col in numeric_features):
    X_scaled = scaler.transform(df[numeric_features])
    df['Cluster'] = model.predict(X_scaled)

# ------------------------------------------------
# Detect Image URL Column Automatically
# ------------------------------------------------
image_column = None
for col in df.columns:
    if any(x in col.lower() for x in ['image', 'img', 'url', 'link']):
        image_column = col
        break

# ------------------------------------------------
# Extract company names
# ------------------------------------------------
def extract_company(name):
    if isinstance(name, str):
        return name.split()[0].capitalize()
    return "Unknown"

df['Company'] = df[name_col].apply(extract_company)

# ------------------------------------------------
# Display Top 50 Mobiles
# ------------------------------------------------
st.subheader("🏆 Top 50 Mobile Phones")

top_phones = df.head(50).reset_index(drop=True)
rows, cols = 10, 5
img_width = 150

for row_idx in range(rows):
    row_data = top_phones.iloc[row_idx * cols : (row_idx + 1) * cols]
    col_list = st.columns(cols)
    for i, (_, data) in enumerate(row_data.iterrows()):
        with col_list[i]:
            if image_column and not pd.isna(data[image_column]):
                st.image(data[image_column], width=img_width)
            else:
                st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)

            st.markdown(f"**{data[name_col]}**")
            st.markdown(f"⭐ **Rating:** {data[rating_col]}**")
            st.markdown(f"💰 **Price:** ₹{int(data[price_col])}**")

# ------------------------------------------------
# 🔍 Advanced Search Bar (Enhanced)
# ------------------------------------------------
st.markdown("---")
st.subheader("🔎 Advanced Search")

search_option = st.selectbox(
    "Search by:",
    ["Select Option", "Name", "Rating", "Price"],
    index=0
)

# ------------------------------------------------
# 🎯 Smart Recommendation (Improved Version)
# ------------------------------------------------
st.markdown("---")
st.subheader("🎯 Smart Recommendation")

# ----- 1️⃣ Search by Brand (Name) -----
if search_option == "Name":
    st.info("Recommendations will be based on the selected mobile brand.")

    # Extract only unique brand names (first word of each model name)
    brand_names = sorted(df['Company'].unique())

    selected_brand = st.selectbox(
        "Select a Brand:",
        options=[""] + brand_names
    )

    if st.button("🔍 Recommend by Brand"):
        if not selected_brand:
            st.warning("Please select a brand first!")
        else:
            rec_df = df[df['Company'].str.lower() == selected_brand.lower()]
            if rec_df.empty:
                st.error("No mobiles found for this brand.")
            else:
                st.markdown(f"### 🔍 Mobiles from **{selected_brand}**")
                rec_cols = st.columns(5)
                for i, (_, rec) in enumerate(rec_df.head(10).iterrows()):
                    with rec_cols[i % 5]:
                        if image_column and not pd.isna(rec[image_column]):
                            st.image(rec[image_column], width=img_width)
                        else:
                            st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)
                        st.markdown(f"**{rec[name_col]}**")
                        st.markdown(f"⭐ **Rating:** {rec[rating_col]}**")
                        st.markdown(f"💰 **Price:** ₹{int(rec[price_col])}**")

# ----- 2️⃣ Search by Rating -----
elif search_option == "Rating":
    st.info("Recommendations will be based on mobile ratings (1–5).")

    rating_options = [1, 2, 3, 4, 5]
    selected_rating = st.selectbox(
        "Select Rating:",
        options=rating_options
    )

    if st.button("🔁 Recommend by Rating"):
        rec_df = df[df[rating_col].round() == selected_rating]
        if rec_df.empty:
            st.error("No mobiles found with this rating.")
        else:
            st.markdown(f"### 🔍 Mobiles with Rating **{selected_rating}**")
            rec_cols = st.columns(5)
            for i, (_, rec) in enumerate(rec_df.head(10).iterrows()):
                with rec_cols[i % 5]:
                    if image_column and not pd.isna(rec[image_column]):
                        st.image(rec[image_column], width=img_width)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)
                    st.markdown(f"**{rec[name_col]}**")
                    st.markdown(f"⭐ **Rating:** {rec[rating_col]}**")
                    st.markdown(f"💰 **Price:** ₹{int(rec[price_col])}**")

# ----- 3️⃣ Search by Price -----
elif search_option == "Price":
    st.info("Recommendations will be based on price ranges.")

    # Define simple price range options
    price_options = [5000, 10000, 15000, 20000, 25000, 30000,
                     40000, 50000, 60000, 80000, 100000]
    selected_price = st.selectbox(
        "Select Maximum Price (≤):",
        options=price_options
    )

    if st.button("🔁 Recommend by Price"):
        rec_df = df[df[price_col] <= selected_price]
        if rec_df.empty:
            st.error("No mobiles found under this price range.")
        else:
            st.markdown(f"### 💰 Mobiles priced ≤ **₹{selected_price}**")
            rec_cols = st.columns(5)
            for i, (_, rec) in enumerate(rec_df.head(10).iterrows()):
                with rec_cols[i % 5]:
                    if image_column and not pd.isna(rec[image_column]):
                        st.image(rec[image_column], width=img_width)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)
                    st.markdown(f"**{rec[name_col]}**")
                    st.markdown(f"⭐ **Rating:** {rec[rating_col]}**")
                    st.markdown(f"💰 **Price:** ₹{int(rec[price_col])}**")
