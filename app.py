import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="AI CS Analyzer Pro", layout="wide")
st.title("🎓 AI CS Analyzer Pro")

# =========================
# 🎯 Smart Category Detection
# =========================
def detect_category(course):
    course = course.lower()

    if any(x in course for x in ["os", "operating"]):
        return "systems"
    elif any(x in course for x in ["network"]):
        return "networks"
    elif any(x in course for x in ["data", "database"]):
        return "data"
    elif any(x in course for x in ["algorithm", "programming"]):
        return "problem_solving"
    elif any(x in course for x in ["ai", "machine"]):
        return "ai"
    else:
        return "general"

# =========================
# 🎯 Resources by Category
# =========================
RESOURCES = {
    "systems": {
        "youtube": [
            "https://www.youtube.com/watch?v=26QPDBe-NB8"
        ],
        "practice": [
            "https://leetcode.com"
        ],
        "tips": [
            "Focus on processes and memory",
            "Use simulators",
            "Do small OS projects"
        ]
    },
    "networks": {
        "youtube": [
            "https://www.youtube.com/watch?v=qiQR5rTSshw"
        ],
        "practice": [
            "https://hackerrank.com"
        ],
        "tips": [
            "Use packet tracer",
            "Visualize protocols",
            "Do labs"
        ]
    },
    "problem_solving": {
        "youtube": [
            "https://www.youtube.com/watch?v=8hly31xKli0"
        ],
        "practice": [
            "https://codeforces.com",
            "https://leetcode.com"
        ],
        "tips": [
            "Solve problems daily",
            "Focus on patterns",
            "Practice contests"
        ]
    },
    "data": {
        "youtube": [
            "https://www.youtube.com/watch?v=HXV3zeQKqGY"
        ],
        "practice": [
            "https://kaggle.com"
        ],
        "tips": [
            "Work on datasets",
            "Practice SQL",
            "Build mini projects"
        ]
    },
    "ai": {
        "youtube": [
            "https://www.youtube.com/watch?v=aircAruvnKk"
        ],
        "practice": [
            "https://kaggle.com"
        ],
        "tips": [
            "Understand math first",
            "Practice models",
            "Build AI projects"
        ]
    },
    "general": {
        "youtube": [
            "https://www.youtube.com"
        ],
        "practice": [
            "https://leetcode.com"
        ],
        "tips": [
            "Revise basics",
            "Practice regularly",
            "Ask questions"
        ]
    }
}

# =========================
# 📊 Upload CSV
# =========================
file = st.file_uploader("Upload Student CSV", type="csv")

if file:
    df = pd.read_csv(file)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    courses = st.multiselect("Select Courses", numeric_cols, default=numeric_cols[:-1])
    final_col = st.selectbox("Final Grade", numeric_cols)

    weak_t = st.slider("Weak <", 40, 70, 60)
    avg_t = st.slider("Average <", 60, 90, 75)

    if st.button("Analyze"):

        averages = df[courses].mean()
        stds = df[courses].std()

        weak = averages[averages < weak_t]
        avg  = averages[(averages >= weak_t) & (averages < avg_t)]
        good = averages[averages >= avg_t]

        st.subheader("📊 Classification")

        st.write("🔴 Weak:", list(weak.index))
        st.write("🟡 Average:", list(avg.index))
        st.write("🟢 Good:", list(good.index))

        # =========================
        # 🎯 Dynamic Recommendations
        # =========================
        st.subheader("🎯 Smart Recommendations")

        for course in weak.index:
            category = detect_category(course)
            res = RESOURCES[category]

            st.markdown(f"### 🔴 {course} (ضعيف)")
            st.write(f"📂 Category: {category}")

            st.markdown("🎥 YouTube:")
            for y in res["youtube"]:
                st.write(y)

            st.markdown("💻 Practice:")
            for p in res["practice"]:
                st.write(p)

            st.markdown("📌 Teaching Improvements:")
            for t in res["tips"]:
                st.write(f"- {t}")
