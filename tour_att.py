import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("Final_tourism.csv")
    df = df[['UserId', 'AttractionId', 'Rating', 'AttractionTypeId', 'CityId', 'CountryId']].dropna()
    df['UserId'] = df['UserId'].astype(str)
    df['AttractionId'] = df['AttractionId'].astype(str)
    return df

df = load_data()

# === Collaborative Filtering ===
def collaborative_filtering(df, user_id, top_n=5):
    user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["AttractionId", "Predicted Rating"])

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    top_users = similar_users.head(10).index

    weighted_ratings = user_item_matrix.loc[top_users].T.dot(similar_users[top_users])
    sim_sum = similar_users[top_users].sum()
    recommendation_scores = weighted_ratings / (sim_sum + 1e-9)

    seen = user_item_matrix.loc[user_id]
    unseen = recommendation_scores[seen[seen == 0].index]
    return unseen.sort_values(ascending=False).head(top_n).reset_index().rename(columns={0: 'Predicted Rating'})

# === Content-Based Filtering ===
def content_based_filtering(df, user_id, top_n=5):
    attraction_features = df.drop_duplicates('AttractionId')[
        ['AttractionId', 'AttractionTypeId', 'CityId', 'CountryId']
    ].set_index('AttractionId')

    scaler = StandardScaler()
    attraction_scaled = scaler.fit_transform(attraction_features)
    attraction_similarity = pd.DataFrame(
        cosine_similarity(attraction_scaled),
        index=attraction_features.index,
        columns=attraction_features.index
    )

    user_data = df[df['UserId'] == user_id]
    if user_data.empty:
        return pd.DataFrame(columns=["AttractionId", "Similarity Score"])

    top_rated = user_data.sort_values(by='Rating', ascending=False).iloc[0]['AttractionId']
    sim_scores = attraction_similarity[top_rated].sort_values(ascending=False)
    sim_scores = sim_scores.drop(labels=user_data['AttractionId'].values, errors='ignore')

    return sim_scores.head(top_n).reset_index().rename(columns={top_rated: 'Similarity Score'})

# === Streamlit Interface ===
st.title("🏝️ Tourism Attraction Recommender System")

st.markdown("Get attraction suggestions based on other users with similar tastes.")

user_ids = sorted(df['UserId'].unique())
selected_user = st.selectbox("Choose a User ID", user_ids)
num_recs = st.slider("Number of Recommendations", 1, 10, 5)

st.markdown("### Choose Recommendation Method:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Collaborative Filtering"):
        cf_results = collaborative_filtering(df, selected_user, num_recs)
        if cf_results.empty:
            st.warning("No collaborative recommendations available.")
        else:
            st.subheader("🔷 Collaborative Recommendations")
            st.dataframe(cf_results.rename(columns={"index": "AttractionId"}))

with col2:
    if st.button("Content-Based Filtering"):
        cb_results = content_based_filtering(df, selected_user, num_recs)
        if cb_results.empty:
            st.warning("No content-based recommendations available.")
        else:
            st.subheader("🔶 Content-Based Recommendations")
            st.dataframe(cb_results)
