import streamlit as st
import pandas as pd
import re
from collections import Counter
from groq import Groq

st.set_page_config(page_title="AI CS Analyzer Pro", layout="wide")
st.title("🎓 AI CS Analyzer Pro")

# ===============================
# API KEY
# ===============================
groq_key = st.sidebar.text_input("Groq API Key", type="password")

# ===============================
# Upload Student Data
# ===============================
st.header("📊 Student Performance")

file = st.file_uploader("Upload Student CSV", type="csv")

if file:
    df = pd.read_csv(file)
    st.write(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    courses = st.multiselect("Select Courses", numeric_cols, default=numeric_cols[:-1])
    final_col = st.selectbox("Final Grade Column", numeric_cols)

    weak_t = st.slider("Weak <", 40, 70, 60)
    avg_t = st.slider("Average <", 60, 90, 75)

    if st.button("Analyze Students"):

        df["Level"] = df[final_col].apply(
            lambda x: "Weak" if x < weak_t else "Average" if x < avg_t else "Excellent"
        )

        averages = df[courses].mean()

        weak_courses = averages[averages < weak_t]
        avg_courses  = averages[(averages >= weak_t) & (averages < avg_t)]
        good_courses = averages[averages >= avg_t]

        st.subheader("📊 Classification")

        st.write("🔴 Weak:", list(weak_courses.index))
        st.write("🟡 Average:", list(avg_courses.index))
        st.write("🟢 Good:", list(good_courses.index))

        # ===============================
        # 🎯 AI Recommendations
        # ===============================
        if groq_key:
            client = Groq(api_key=groq_key)

            prompt = f"""
You are a senior CS professor.

We analyzed student grades.

Weak courses: {list(weak_courses.index)}
Average courses: {list(avg_courses.index)}

For EACH weak course:
1. Why students struggle
2. Teaching improvements
3. YouTube course suggestions
4. Practice platforms (LeetCode, Codeforces, etc.)
5. Specific exercises ideas

Be VERY specific per course.
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("🤖 AI Recommendations")
            st.write(response.choices[0].message.content)

# ===============================
# 💼 Job Market NLP Analysis
# ===============================
st.header("💼 Job Market Analysis (NLP)")

job_file = st.file_uploader("Upload Jobs CSV", type="csv", key="jobs")

# 🔥 Skills Dictionary (Advanced)
SKILLS = [
    "python","java","c++","javascript","react","node","django","flask",
    "machine learning","deep learning","tensorflow","pytorch",
    "sql","mongodb","docker","kubernetes","aws","azure",
    "data science","nlp","computer vision","linux","git"
]

def extract_skills(text):
    found = []
    for skill in SKILLS:
        if re.search(rf"\b{skill}\b", text):
            found.append(skill)
    return found

if job_file:
    jobs = pd.read_csv(job_file)

    text_col = st.selectbox("Select Description Column", jobs.columns)

    corpus = jobs[text_col].dropna().str.lower()

    all_skills = []

    for text in corpus:
        all_skills.extend(extract_skills(text))

    counter = Counter(all_skills)

    top_skills = counter.most_common(15)

    st.subheader("🔥 Top Skills in Market")
    st.write(top_skills)

    # ===============================
    # 🎯 Curriculum Comparison
    # ===============================
    CURRENT_PLAN = [
        "Programming","OOP","Data Structures","Algorithms",
        "Databases","Operating Systems","Networks",
        "Artificial Intelligence"
    ]

    if groq_key:
        client = Groq(api_key=groq_key)

        prompt = f"""
You are a CS curriculum expert.

Top market skills:
{top_skills}

Current curriculum:
{CURRENT_PLAN}

Provide:

1. Missing skills (IMPORTANT)
2. Courses to improve
3. New courses to add
4. Career paths based on skills

Be practical and realistic.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("🎓 Curriculum Recommendations")
        st.write(response.choices[0].message.content)
