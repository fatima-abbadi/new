import streamlit as st
import pandas as pd
import numpy as np
import spacy

# تحميل NLP
nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
st.title("🎓 AI CS Analyzer Pro")

# ========= Helpers =========

def classify(avg, weak_t, avg_t):
    if avg < weak_t:
        return "Weak"
    elif avg < avg_t:
        return "Average"
    return "Good"

def difficulty_index(series, threshold):
    return (series < threshold).sum() / len(series)

# NLP استخراج كلمات
def extract_keywords(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

# توصيات جاهزة (مهم 🔥)
COURSE_RESOURCES = {
    "operating": {
        "youtube": [
            "https://www.youtube.com/watch?v=26QPDBe-NB8",
            "https://www.youtube.com/watch?v=6i2C0Q3Y4Xw"
        ],
        "practice": [
            "https://leetcode.com/problemset/",
            "https://hackerrank.com/domains/os"
        ],
        "tips": [
            "Focus on process scheduling",
            "Use OS simulators",
            "Mini OS projects"
        ]
    },
    "database": {
        "youtube": [
            "https://www.youtube.com/watch?v=HXV3zeQKqGY"
        ],
        "practice": [
            "https://leetcode.com/problemset/database/"
        ],
        "tips": [
            "Practice SQL queries",
            "Work on real DB projects"
        ]
    },
    "network": {
        "youtube": [
            "https://www.youtube.com/watch?v=qiQR5rTSshw"
        ],
        "practice": [
            "https://hackerrank.com/domains/networking"
        ],
        "tips": [
            "Use packet tracer",
            "Simulate networks"
        ]
    }
}

def get_recommendations(course):
    keywords = extract_keywords(course)

    for key in COURSE_RESOURCES:
        if key in keywords:
            return COURSE_RESOURCES[key]

    # fallback عام
    return {
        "youtube": ["https://www.youtube.com/results?search_query=" + course],
        "practice": [
            "https://leetcode.com",
            "https://hackerrank.com"
        ],
        "tips": [
            "Increase practical exercises",
            "Use real-world projects"
        ]
    }

# ========= UI =========

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected = st.multiselect("Select Courses", numeric_cols, default=numeric_cols)

    weak_t = st.slider("Weak <", 40, 70, 60)
    avg_t  = st.slider("Average <", 60, 90, 75)

    if st.button("Analyze"):

        results = []

        for col in selected:
            avg = df[col].mean()
            std = df[col].std()
            diff = difficulty_index(df[col], weak_t)

            level = classify(avg, weak_t, avg_t)

            results.append({
                "course": col,
                "avg": avg,
                "std": std,
                "difficulty": diff,
                "level": level
            })

        res_df = pd.DataFrame(results).sort_values("avg")

        st.dataframe(res_df)

        st.divider()

        # ========= عرض احترافي =========
        for _, row in res_df.iterrows():

            if row["level"] == "Weak":
                st.error(f"🔴 {row['course']} (ضعيف)")
            elif row["level"] == "Average":
                st.warning(f"🟡 {row['course']} (متوسط)")
            else:
                st.success(f"🟢 {row['course']} (جيد)")

            rec = get_recommendations(row["course"])

            st.markdown("🎥 YouTube:")
            for y in rec["youtube"]:
                st.write(f"- {y}")

            st.markdown("💻 Practice:")
            for p in rec["practice"]:
                st.write(f"- {p}")

            st.markdown("📌 تحسين التدريس:")
            for t in rec["tips"]:
                st.write(f"- {t}")

            st.divider()
