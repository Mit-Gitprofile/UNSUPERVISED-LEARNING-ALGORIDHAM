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
            st.markdown(f"⭐ **Rating:** {data[rating_col]}")
            st.markdown(f"💰 **Price:** {data[price_col]}")

# ------------------------------------------------
# Extract company names
# ------------------------------------------------
def extract_company(name):
    if isinstance(name, str):
        return name.split()[0].capitalize()
    return "Unknown"

df['Company'] = df[name_col].apply(extract_company)
companies = sorted(df['Company'].unique())

# ------------------------------------------------
# 🔍 Advanced Search Bar
# ------------------------------------------------
st.markdown("---")
st.subheader("🔎 Advanced Search")

search_option = st.selectbox(
    "Search by:",
    ["Select Option", "Name", "Rating", "Price"],
    index=0
)


# ------------------------------------------------
# 🎯 Dynamic Recommendation Section
# ------------------------------------------------
st.markdown("---")
st.subheader("🎯 Smart Recommendation")

if search_option == "Name":
    st.info("Recommendations will be based on similar names.")
    selected_name = st.selectbox(
        "Select a Mobile Name:",
        options=[""] + sorted(df[name_col].dropna().unique().tolist())
    )

    if st.button("🔍 Recommend Similar Names"):
        if not selected_name:
            st.warning("Please select a mobile name first!")
        else:
            keyword = selected_name.split()[0].lower()
            rec_df = df[df[name_col].str.lower().str.contains(keyword)]
            if rec_df.empty:
                st.error("No similar mobiles found.")
            else:
                st.markdown(f"### 🔍 Mobiles similar to **{selected_name}**")
                rec_cols = st.columns(5)
                for i, (_, rec) in enumerate(rec_df.head(10).iterrows()):
                    with rec_cols[i % 5]:
                        if image_column and not pd.isna(rec[image_column]):
                            st.image(rec[image_column], width=img_width)
                        else:
                            st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)
                        st.markdown(f"**{rec[name_col]}**")
                        st.markdown(f"⭐ **Rating:** {rec[rating_col]}")
                        st.markdown(f"💰 **Price:** {rec[price_col]}")

elif search_option == "Rating":
    st.info("Recommendations will be based on similar ratings.")
    selected_rating = st.selectbox(
        "Select a Rating:",
        options=sorted(df[rating_col].dropna().unique().tolist())
    )

    if st.button("🔁 Recommend by Rating"):
        rec_df = df[df[rating_col] == selected_rating]
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
                    st.markdown(f"⭐ **Rating:** {rec[rating_col]}")
                    st.markdown(f"💰 **Price:** {rec[price_col]}")

elif search_option == "Price":
    st.info("Recommendations will be based on similar prices.")
    selected_price = st.selectbox(
        "Select a Price Range (less than or equal to):",
        options=sorted(df[price_col].dropna().unique().tolist())
    )

    if st.button("🔁 Recommend by Price"):
        rec_df = df[df[price_col] <= selected_price]
        if rec_df.empty:
            st.error("No mobiles found under this price.")
        else:
            st.markdown(f"###  Mobiles with Price ≤ **{selected_price}**")
            rec_cols = st.columns(5)
            for i, (_, rec) in enumerate(rec_df.head(10).iterrows()):
                with rec_cols[i % 5]:
                    if image_column and not pd.isna(rec[image_column]):
                        st.image(rec[image_column], width=img_width)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", width=img_width)
                    st.markdown(f"**{rec[name_col]}**")
                    st.markdown(f"⭐ **Rating:** {rec[rating_col]}")
                    st.markdown(f"💰 **Price:** {rec[price_col]}")
