import streamlit as st
from streamlit_javascript import st_javascript
import json
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
# import helper
# from ollama import chat
from openai import OpenAI

def cosine_similarity_weighted(u, v, alpha=10):
    mask = (u != 0) & (v != 0)
    overlap = np.sum(mask)

    if overlap == 0:
        return 0

    u_common = u[mask]
    v_common = v[mask]

    sim = np.dot(u_common, v_common) / (
            np.linalg.norm(u_common) * np.linalg.norm(v_common)
    )

    # significance weighting
    weight = overlap / (overlap + alpha)

    return sim * weight

def cosine_with_all(target, matrix):
    sims = []
    for user in matrix:
        sims.append(cosine_similarity_weighted(target, user))
    return np.array(sims)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # api_key="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"
    api_key="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"
)

model = "qwen/qwen3.6-plus:free"

st.set_page_config(page_title="AI in Action (Recommendation Systems)", layout="wide")

st.sidebar.header("控制台")

if "tab" not in st.session_state:
    st.session_state.tab = "用户界面"

# 1. Initialize Session State
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_bot_mode" not in st.session_state:
    st.session_state.last_bot_mode = None
def close_chat():
    st.session_state.chat_visible = False

# 2. Sidebar Toggle
with st.sidebar:
    st.checkbox(
        "AI助手",
        key="chat_visible"
    )

# --- navigation ---
tab = st.sidebar.radio("前后台切换", ["用户界面", "后台分析"],
               index=["用户界面", "后台分析"].index(st.session_state.tab))

# read data and find medalists
path = Path(__file__).parent / 'movies.csv'
movies = pd.read_csv(path)

path = Path(__file__).parent / 'ratings.csv'
ratings = pd.read_csv(path)
ratings_table = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_table_filled = ratings_table.fillna(0)

# 1. Initialize session state to store the data once it arrives
if "watch_data" not in st.session_state:
    st.session_state.watch_data = None

# 2. Track a "version" number to force the JS to re-run on demand
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "current_section" not in st.session_state:
    st.session_state.current_section = None
if "last_section" not in st.session_state:
    st.session_state.last_section = None

if "instruction_mode" not in st.session_state:
    st.session_state.instruction_mode = True

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

# tab1, tab2 = st.tabs(["用户界面", "分析台"])

# with tab1:
if tab == '用户界面':
    st.title("用户界面")
    # st.markdown("The result is shown below: ")

    videos = list(zip(movies['link'], movies['chinese_title']))

    html = f"""
    <style>
    .grid {{height:600px; overflow-y:scroll; }}
    .row {{ display:flex; margin-bottom:12px; }}
    .item {{ flex:0 0 25%; padding:8px; cursor:pointer; }}
    .thumb {{ position:relative; padding-bottom:56.25%; }}
    .thumb img {{ position:absolute; width:100%; height:100%; object-fit:cover; }}
    .title {{ text-align:center; margin-top:6px; }}
    .overlay {{
        display:none; position:fixed; top:0; left:0;
        width:100%; height:100%; background:rgba(0,0,0,0.85);
        justify-content:center; align-items:center;
    }}
    .player-wrapper {{ width:70%; max-width:900px; }}
    .timer {{ color:white; text-align:center; margin-bottom:10px; }}
    .player {{ width:100%; aspect-ratio:16/9; }}
    </style>
    
    <div class="grid">
    """

    # Build grid
    for i in range(0, len(videos), 3):
        html += '<div class="row">'
        for j in range(3):
            if i + j < len(videos):
                url, name = videos[i + j]
                vid = url.split("/")[-1]

                html += f"""
                <div class="item" onclick="openPlayer('{url}', '{name}')">
                    <div class="thumb">
                        <img src="https://img.youtube.com/vi/{vid}/0.jpg">
                    </div>
                    <div class="title">{name}</div>
                </div>
                """
        html += "</div>"

    html += "</div>"

    # Overlay + JS
    html += """
    <div class="overlay" id="overlay" onclick="closePlayer(event)">
        <div class="player-wrapper">
            <div id="timer" class="timer">⏱️ Watching: 0s</div>
            <div class="player" id="playerContainer"></div>
        </div>
    </div>
    
    <script>
    let startTime = null;
    let timerInterval = null;
    let currentMovie = null;
    
    function openPlayer(url, name) {
        const overlay = document.getElementById("overlay");
        const container = document.getElementById("playerContainer");
    
        currentMovie = name;
    
        container.innerHTML = `
            <iframe src="${url}?autoplay=1"
            width="100%" height="100%"
            frameborder="0" allow="autoplay"></iframe>
        `;
    
        startTime = Date.now();
    
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime)/1000);
            document.getElementById("timer").innerText =
                "⏱️ Watching: " + elapsed + "s";
        }, 1000);
    
        overlay.style.display = "flex";
    }
    
    function closePlayer(event) {
        if (event.target.id === "overlay") {
            stopPlayer();
        }
    }
    
    function stopPlayer() {
        const elapsed = Math.floor((Date.now() - startTime)/1000);
    
        // --- Get existing data ---
        let data = localStorage.getItem("watch_data");
        data = data ? JSON.parse(data) : {};
    
        // --- Accumulate watch time ---
        if (data[currentMovie]) {
            data[currentMovie] += elapsed;
        } else {
            data[currentMovie] = elapsed;
        }
         
        // --- Save back to localStorage ---
        localStorage.setItem("watch_data", JSON.stringify(data));
        // Debug (optional)
        console.log("Saved:", data);
        
        
    
        // --- Cleanup player ---
        document.getElementById("playerContainer").innerHTML = "";
        clearInterval(timerInterval);
        document.getElementById("timer").innerText = "⏱️ Watching: 0s";
        document.getElementById("overlay").style.display = "none";
    }
    </script>
    """

    components.html(html, height=650)

    # if st.button("分析"):
    #     st.session_state.query_count += 1
    #     # We clear the old data so the UI feels responsive
    #     st.session_state.watch_data = None

    if st.button("清空数据"):
        st_javascript("localStorage.removeItem('watch_data')")
        st.success("Data cleared!")


# with tab2:
elif tab == '后台分析':
    # st.markdown("The result is shown below: ")

    sections = ['后台数据', '匹配兴趣', '推荐报告']
    selected_section = st.sidebar.radio("目录", sections)

    data = st_javascript(
        f"window.localStorage.getItem('watch_data')",
        # key=f"js_query_{st.session_state.query_count}"
    )

    if data is not None and data != 0:
        st.session_state.watch_data = data

    if st.session_state.watch_data:
        data_obj = json.loads(st.session_state.watch_data)
    else:
        st.info("没有记录！")

    if selected_section == sections[0]:
        st.title("后台数据")
        st.session_state.current_section = sections[0]
        try:
            watch_data = json.loads(data)
            st.success("✅ 观看数据已加载!")

            # --- Show raw data ---
            # st.subheader("原始数据")
            # st.write(watch_data)

            # --- Convert to DataFrame ---
            df = pd.DataFrame(
                list(watch_data.items()),
                columns=["Movie", "Watch Time (s)"]
            ).sort_values(by="Watch Time (s)", ascending=False)

            # --- Metrics ---
            total_time = df["Watch Time (s)"].sum()
            avg_time = df["Watch Time (s)"].mean()
            top_movie = df.iloc[0]["Movie"]

            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ 总观看时长", f"{total_time}s")
            col2.metric("📈 平均观看时长", f"{avg_time:.1f}s")
            col3.metric("🏆 最长观看时长视频", top_movie)

            st.divider()

            # --- Charts ---
            st.subheader("📊 视频观看时长分布")
            fig = px.bar(
                df,
                x="Movie",
                y="Watch Time (s)"
            )

            fig.update_layout(
                xaxis_title="视频",
                yaxis_title="观看时长（秒）",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 原始数据")
            st.dataframe(df)

        except:
            st.info("请先浏览用户界面的视频信息！")

    if selected_section == sections[1]:
        st.title("匹配兴趣")
        st.session_state.current_section = sections[1]
        # try:
        watch_data = json.loads(data)

        # with st.spinner("⏳ 开始分析... 请稍候..."):
        #     time.sleep(1)  # simulate long computation
        #
        # with st.spinner("⏳正在匹配100,575名用户数据..."):
        #     time.sleep(1)

        watched_movies = []
        my_ratings = []
        to_order = []
        for k, v in watch_data.items():
            idx = movies[movies['chinese_title'] == k]['movieId'].values[0]
            to_order.append((idx, k, v))
        to_order.sort()
        
        for idx, k, v in to_order:
            watched_movies.append(k)
            my_ratings.append(v)

        rated_idx = movies[movies['chinese_title'].isin(watched_movies)]['movieId'].tolist()
        target = np.zeros(ratings_table.shape[1])
        target[rated_idx] = my_ratings

        sim_scores = cosine_with_all(target, ratings_table_filled.values)
        user_id = sim_scores.argmax()
        top_k = 3
        user_ids = np.argpartition(sim_scores, -top_k)[-top_k:]

        # st.write(user_id, user_ids, rated_idx)
        # st.write(sim_scores)
        my_ratings = np.array(my_ratings)
        compare_table = pd.DataFrame([watched_movies,
                                      (my_ratings - my_ratings.min()) / (my_ratings.max() - my_ratings.min()) * 5,
                                      *[ratings_table.loc[user_id].values[rated_idx] for user_id in user_ids]]).T
        compare_table.columns = ['电影', '我的兴趣'] + [f'相似用户{i+1}' for i in range(top_k)]
        
        st.dataframe(compare_table)
        st.subheader("我的兴趣和最接近的用户的匹配度")

        for i in range(top_k):
            fig = px.scatter(
                compare_table,
                x="我的兴趣", y=f"相似用户{i+1}",
                trendline="ols",
                hover_data='电影',
                range_x=[-1, 6],
                range_y=[0, 6]
            )
            
            fig.update_layout(
                xaxis_title="我的兴趣",
                yaxis_title=f"相似用户{i+1}(ID{user_ids[i]}))",
            )

            st.plotly_chart(fig, use_container_width=True)
            

        # with st.spinner("⏳生成电影匹配度列表..."):
        #     time.sleep(3)

        st.subheader(f"以下是相似用户观看过的视频评分：")

        final_result = movies[['movieId', 'chinese_title']]
        for i, user_id in enumerate(user_ids):
            closest_rater = ratings[ratings['userId'] == user_id]
            final_result = final_result.merge(closest_rater, on='movieId', how='outer')
        final_result = final_result[['movieId', 'chinese_title', 'rating_x', 'rating_y', 'rating']]
        final_result.columns = ['电影ID', '视频名']+[f"用户{user_ids[i]}" for i in range(top_k)]

        st.dataframe(final_result)

        st.session_state.analysis_results.extend([compare_table, final_result])
        st.success("✅ 分析完成!")
        # except:
        #     st.info("请先浏览用户界面的视频信息！")
    if selected_section == sections[2]:
        st.title("推荐报告")
        st.session_state.current_section = sections[2]

        if st.session_state.analysis_results:
            # for df in st.session_state.analysis_results:
            #     st.dataframe(df)
            compare_table, final_result = st.session_state.analysis_results[-2:]

            final_result_dict = final_result.to_dict(orient='list')
            del final_result_dict['电影ID']

            prompt = f'你现在有一个python字典，内容是：{final_result_dict}。字典里有4个元素，key为“视频名”的元素对应一个视频列表，剩下三个元素的key对应兴趣爱好相似的三个人，这三个key对应的value是三个人对这个视频列表中视频分别的评分（评分从低到高为1-5，可能有空值NaN）。总结一下这个字典中数据的具体情况，尤其是每个人对每部电影的评分，从而从中筛选出一些可以推荐的电影，形成一个推荐列表，里面有每一部考虑了三人评分后推荐的电影，并附上推荐理由。列出详细的分析和思考过程，同时带上数据佐证。'

            with st.spinner("⏳AI助手正在生成回复..."):
                # response = chat(
                #     model='deepseek-r1:1.5b',
                #     messages=[{'role': 'user', 'content': prompt}],
                # )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

            response = response.choices[0].message.content
            st.write(response)

            st.session_state.analysis_results = []
        else:
            st.markdown('还未计算分析结果，可以去“匹配兴趣”选项中进行分析！')



if st.session_state.last_bot_mode != tab:
    if tab == '用户界面':
        greeting = "当前页面模拟短视频平台的用户界面，当用户点击视频进行观看时，平台会记录用户行为数据（视频播放次数，时间，是否点赞，关注，评论等），方便后台进行分析和推荐新内容。"
    elif tab == '后台分析':
        greeting = "当前页面模拟后台分析过程，即短视频平台的推荐算法会对用户数据进行分析，目的是更准确地推送用户喜好的内容，让用户尽可能久地在线使用平台。"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    # Update the tracker so it doesn't loop infinitely
    st.session_state.last_bot_mode = tab
else:
    if st.session_state.last_section != st.session_state.current_section:
        greeting = f'你当前在“{st.session_state.current_section}”页面！'
        if st.session_state.current_section == sections[0]:
            greeting += '\n\n这里你可以看到后台数据，平台会记录用户与平台的各种互动数据，为简化分析，这里仅展示不同视频片段的观看时长。'
        elif st.session_state.current_section == sections[1]:
            greeting += '\n\n这里展示的是推荐算法的核心，协同过滤算法。'
        elif st.session_state.current_section == sections[2]:
            greeting += '\n\n根据你的观看偏好和数据库中类似用户的比对，系统会生成最终结果。'
        st.session_state.last_section = st.session_state.current_section
        st.session_state.messages.append({"role": "assistant", "content": greeting})

# 4. The Floating Chat Logic
if st.session_state.chat_visible:
    # CSS to position the window and style the 'X' button
    st.markdown("""
            <style>
            [data-testid="stVerticalBlock"] > div:has(div.floating-window) {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 300px;
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                z-index: 1000;
                padding: 15px;
            }
            .stButton>button {
                border: none;
                background: transparent;
                float: right;
                font-size: 20px;
            }
            /* 3. Change Font Size of the Input Box */
            .stChatInput textarea {
                font-size: 12px !important;
            }
        
            /* 4. Change Font Size of the Header/Title */
            .floating-chat-box h3 {
                font-size: 12px !important;
            }
            [data-testid="stChatMessage"] p {
                font-size: 12px !important; /* Adjust this number as needed */
                line-height: 1.5;
            }
            </style>
        """, unsafe_allow_html=True)

    # We use a wrapper div with a custom class to target it with CSS
    with st.container():
        st.markdown('<div class="floating-window">', unsafe_allow_html=True)

        # Header Row: Title and Close Button
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.subheader("AI助手")
        with col2:
            st.button("✖️", key="close_chat_btn", on_click=close_chat)

        # KEY FIX: Using a fixed-height container keeps the chat INSIDE the window
        chat_container = st.container(height=300)

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])


        # Chat input is placed at the bottom of the floating container
        if prompt := st.chat_input("请输入..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container.chat_message("user"):
                st.write(prompt)

            # Simulated AI logic
            with st.spinner("⏳AI助手正在思考回复..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

            response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})
            with chat_container.chat_message("assistant"):
                st.write(response)

            # Force refresh to show new message immediately
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
