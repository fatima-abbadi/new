import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

import re

st.set_page_config(page_title="AI CS Analyzer PRO", layout="wide")
st.title("🎓 AI CS Department Analyzer — PRO Version")

# ═══════════════════════════════
# Sidebar
# ═══════════════════════════════
groq_key = st.sidebar.text_input("Groq API Key", type="password")

tab1, tab2 = st.tabs(["📊 Students (ML)", "💼 Job Market (NLP)"])

# ═══════════════════════════════
# TAB 1 — ML STUDENT ANALYSIS
# ═══════════════════════════════
with tab1:
    st.header("📊 Student Performance (Machine Learning)")

    file = st.file_uploader("Upload Student CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        courses = st.multiselect("Select course features", numeric_cols[:-1], default=numeric_cols[:-1])
        target  = st.selectbox("Final Grade", numeric_cols)

        weak_t = st.slider("Weak <", 40, 70, 60)
        avg_t  = st.slider("Average <", 60, 90, 75)

        if st.button("Train ML Model"):

            def classify(x):
                if x < weak_t: return "Weak"
                elif x < avg_t: return "Average"
                else: return "Excellent"

            df['Label'] = df[target].apply(classify)

            X = df[courses]
            y = df['Label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            st.success(f"Model Accuracy: {acc*100:.2f}%")

            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            # Feature importance
            importances = pd.Series(model.feature_importances_, index=courses).sort_values()

            fig, ax = plt.subplots()
            importances.plot(kind='barh', ax=ax)
            st.pyplot(fig)

# ═══════════════════════════════
# TAB 2 — NLP JOB ANALYSIS
# ═══════════════════════════════
with tab2:
    st.header("💼 Job Market Analysis (NLP TF-IDF)")

    jobs_file = st.file_uploader("Upload Jobs CSV", type="csv")

    if jobs_file:
        jobs = pd.read_csv(jobs_file)
        st.dataframe(jobs.head())

        text_cols = jobs.select_dtypes(include='object').columns.tolist()
        desc_col = st.selectbox("Description Column", text_cols)

        if st.button("Analyze Market (NLP)"):

            text_data = jobs[desc_col].dropna().astype(str)

            # تنظيف النص
            text_data = text_data.str.lower().apply(lambda x: re.sub(r'[^a-z0-9 ]',' ',x))

            # TF-IDF + ngrams
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1,2),
                max_features=200
            )

            X = vectorizer.fit_transform(text_data)

            terms = vectorizer.get_feature_names_out()
            scores = np.asarray(X.mean(axis=0)).ravel()

            skill_df = pd.DataFrame({
                "Skill": terms,
                "Score": scores
            }).sort_values(by="Score", ascending=False).head(20)

            st.subheader("Top Skills (TF-IDF)")

            fig2, ax2 = plt.subplots()
            ax2.barh(skill_df['Skill'], skill_df['Score'])
            st.pyplot(fig2)

            st.dataframe(skill_df)

            # AI recommendations
            if groq_key:
                with st.spinner("Generating AI insights..."):
                    client = Groq(api_key=groq_key)

                    skills_text = "\n".join([
                        f"{row.Skill}: {row.Score:.3f}"
                        for _, row in skill_df.iterrows()
                    ])

                    prompt = f"""
Top job market skills:
{skills_text}

Generate:
1. Curriculum gaps
2. New CS courses
3. Priority skills
"""

                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content":prompt}]
                    )

                    st.markdown(resp.choices[0].message.content)
