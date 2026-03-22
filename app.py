import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time

st.title("推荐系统演示")
# st.markdown("The result is shown below: ")

st.sidebar.header("设置")
show_raw = st.sidebar.checkbox("原始数据", value=False)


# read data and find medalists
path = Path(__file__).parent / 'movies.csv'
movies = pd.read_csv(path)

path = Path(__file__).parent / 'ratings.csv'
ratings = pd.read_csv(path)
ratings_table = ratings.pivot(index='userId', columns='movieId', values='rating')

if show_raw:
    if show_raw:
        tab1, tab2 = st.tabs(["主界面", "原始数据"])
        with tab2:
            st.dataframe(movies)
            st.dataframe(ratings_table)


# Step 1: Define your items
items = movies['chinese_title']

# Step 2: Let user select up to 5 items
selected_items = st.multiselect(
    "选择一些电影进行评分（最多5部）:",
    options=items,
    max_selections=5
)

# Step 3: Create sliders for selected items
ratings_dict = {}

if selected_items:
    st.subheader("请对这些电影进行评分（0-5分）")

    for item in selected_items:
        ratings_dict[item] = st.slider(
            f"Rate {item}",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.5
        )

if ratings_dict:
    btn = st.button("Start Calculation")

    if btn:
        with st.spinner("⏳ 开始分析... 请稍候..."):
            time.sleep(1)  # simulate long computation

        with st.spinner("⏳正在匹配100,575名用户数据..."):
            time.sleep(3)

        rated_idx = []
        rated_movie_title = []
        my_ratings = []

        for item, rating in ratings_dict.items():
            idx = movies[movies['chinese_title'] == item]['movieId'].values[0]
            # print(idx)
            rated_idx.append(idx)
            rated_movie_title.append(item)
            my_ratings.append(rating)
            # st.write(f"{item}: {rating}")

        ratings_on_slected_movies = ratings_table.iloc[:, rated_idx]
        fully_rated = ratings_on_slected_movies.dropna()
        fully_rated.columns = rated_movie_title

        st.subheader(f"找到以下{fully_rated.shape[0]}名用户对相同电影进行评分：")
        st.dataframe(fully_rated)

        with st.spinner("⏳正在匹配最接近的用户评分..."):
            time.sleep(3)

        dists = np.sqrt(np.sum((fully_rated - my_ratings) ** 2, axis=1))
        row = np.argmin(dists)
        user_id = fully_rated.index[row]
        # st.write(f"Our ratings: {my_ratings}\n{user_id}'s ratings: {fully_rated.loc[user_id].values}")
        rated_movie_names = movies[movies['movieId'].isin(rated_idx)]['chinese_title']
        compare_table = pd.DataFrame([rated_movie_names.values, my_ratings, fully_rated.loc[user_id].values]).T
        compare_table.columns = ['电影', '我的评分', '最接近的用户评分']
        st.subheader(f"匹配到用户(ID: {user_id})评分和您最一致，对比如下：")
        st.dataframe(compare_table)

        with st.spinner("⏳生成推荐报告..."):
            time.sleep(3)

        st.subheader(f"以下是用户(ID: {user_id})观看过的电影评分：")
        closest_rater = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)
        final_result = closest_rater.merge(movies, on='movieId')[['userId', 'chinese_title', 'rating']]
        st.dataframe(final_result)


        st.success("✅ Done!")