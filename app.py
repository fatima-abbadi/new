import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# تحميل NLP
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI CS Analyzer Pro", layout="wide")
st.title("🎓 AI CS Analyzer Pro")

# =============================
# 🧠 NLP: تنظيف النص
# =============================
def preprocess(text):
    doc = nlp(str(text).lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])

# =============================
# 🎯 تجاهل الأعمدة الغلط
# =============================
IGNORE_COLUMNS = ["rank", "id", "student", "name"]

def filter_courses(columns):
    valid = []
    for col in columns:
        if any(x in col.lower() for x in IGNORE_COLUMNS):
            continue
        valid.append(col)
    return valid

# =============================
# 📚 Resources (ذكية)
# =============================
RESOURCES = {
    "programming": {
        "yt": ["Python Full Course - freeCodeCamp", "CS50 Harvard"],
        "practice": ["https://leetcode.com", "https://hackerrank.com"],
        "tips": ["Practice daily", "Build small projects"]
    },
    "algorithms": {
        "yt": ["Abdul Bari Algorithms", "MIT Algorithms"],
        "practice": ["https://codeforces.com"],
        "tips": ["Focus on DP & greedy", "Solve contests"]
    },
    "network": {
        "yt": ["Networking - Cisco", "Gate Smashers"],
        "practice": ["https://packettracerlabs.com"],
        "tips": ["Use simulators", "Practice protocols"]
    },
    "database": {
        "yt": ["SQL Full Course", "Database Systems"],
        "practice": ["https://sqlzoo.net"],
        "tips": ["Practice queries", "Design ER diagrams"]
    }
}

def detect_field(course):
    c = course.lower()
    if "program" in c:
        return "programming"
    if "algorithm" in c:
        return "algorithms"
    if "network" in c:
        return "network"
    if "data" in c:
        return "database"
    return "programming"

# =============================
# 📊 رفع البيانات
# =============================
file = st.file_uploader("Upload Student Dataset", type="csv")

if file:
    df = pd.read_csv(file)

    st.success(f"Loaded {len(df)} rows")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = filter_courses(numeric_cols)

    courses = st.multiselect("Select Courses", numeric_cols, default=numeric_cols)

    final_col = st.selectbox("Final Grade Column", numeric_cols)

    weak = st.slider("Weak <", 40, 70, 60)
    avg  = st.slider("Average <", 60, 90, 75)

    if st.button("Analyze"):

        avg_scores = df[courses].mean().sort_values()

        weak_c   = avg_scores[avg_scores < weak]
        med_c    = avg_scores[(avg_scores >= weak) & (avg_scores < avg)]
        good_c   = avg_scores[avg_scores >= avg]

        st.subheader("📊 Classification")

        st.write("🔴 Weak:", list(weak_c.index))
        st.write("🟡 Average:", list(med_c.index))
        st.write("🟢 Good:", list(good_c.index))

        # =============================
        # 📈 Plotly
        # =============================
        fig = px.bar(
            x=avg_scores.values,
            y=avg_scores.index,
            orientation='h',
            title="Course Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # 🎯 Recommendations
        # =============================
        st.subheader("🎯 Smart Recommendations")

        for course, val in avg_scores.items():

            if val < weak:
                level = "🔴"
                label = "ضعيف"
            elif val < avg:
                level = "🟡"
                label = "متوسط"
            else:
                level = "🟢"
                label = "جيد"

            st.markdown(f"### {level} {course} ({label})")

            field = detect_field(course)
            res = RESOURCES[field]

            st.write("🎥 YouTube:")
            for y in res["yt"]:
                st.write("-", y)

            st.write("💻 Practice:")
            for p in res["practice"]:
                st.write("-", p)

            st.write("📌 تحسين:")
            for t in res["tips"]:
                st.write("-", t)

        # =============================
        # 💼 Job Roles Prediction
        # =============================
        st.subheader("💼 Predicted Job Roles")

        roles = []

        for course in courses:
            field = detect_field(course)
            if field == "programming":
                roles.append("Software Engineer")
            elif field == "algorithms":
                roles.append("Competitive Programmer")
            elif field == "network":
                roles.append("Network Engineer")
            elif field == "database":
                roles.append("Data Analyst")

        roles = list(set(roles))
        st.write("🎯 Suitable Jobs:", roles)

        # =============================
        # 📊 TF-IDF (NLP)
        # =============================
        st.subheader("🧠 NLP Skill Extraction")

        text_data = " ".join(courses)
        processed = preprocess(text_data)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([processed])

        features = vectorizer.get_feature_names_out()

        st.write("Top Keywords:", list(features[:10]))
