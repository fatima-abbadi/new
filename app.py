"""
CS Department Intelligence System
- Tab 1: Student performance analysis + Random Forest ML
- Tab 2: Job market analysis using TF-IDF NLP
- Tab 3: Dynamic AI institutional report per course
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from groq import Groq
import re
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="CS Department Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
    padding:2rem 2.5rem; border-radius:12px;
    margin-bottom:2rem; border-left:4px solid #3b82f6;
}
.main-header h1 { color:#f1f5f9; font-size:1.8rem; font-weight:600; margin:0; font-family:'IBM Plex Mono',monospace; }
.main-header p  { color:#94a3b8; margin:0.5rem 0 0; font-size:0.9rem; }
.metric-card { background:#0f172a; border:1px solid #1e3a5f; border-radius:10px; padding:1.2rem; text-align:center; margin-bottom:4px; }
.metric-card .val { font-size:1.7rem; font-weight:600; font-family:'IBM Plex Mono',monospace; color:#3b82f6; }
.metric-card .lbl { font-size:0.75rem; color:#64748b; margin-top:0.3rem; }
.sec { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#3b82f6;
       letter-spacing:.08em; text-transform:uppercase;
       border-bottom:1px solid #1e3a5f; padding-bottom:.5rem; margin-bottom:1.2rem; }
.box-red   { background:#1a0a0a; border:1px solid #dc2626; border-radius:8px; padding:.8rem 1rem; color:#fca5a5; font-size:.85rem; margin-bottom:.5rem; }
.box-amber { background:#1a1200; border:1px solid #d97706; border-radius:8px; padding:.8rem 1rem; color:#fcd34d; font-size:.85rem; margin-bottom:.5rem; }
.box-green { background:#001a0a; border:1px solid #16a34a; border-radius:8px; padding:.8rem 1rem; color:#86efac; font-size:.85rem; margin-bottom:.5rem; }
.box-blue  { background:#0a0f1a; border:1px solid #3b82f6; border-radius:8px; padding:.8rem 1rem; color:#93c5fd; font-size:.85rem; margin-bottom:.5rem; }
.stButton>button { background:#1e3a5f; color:#f1f5f9; border:1px solid #3b82f6;
    border-radius:8px; font-family:'IBM Plex Mono',monospace; font-size:.85rem; padding:.6rem 1.5rem; width:100%; }
.stButton>button:hover { background:#3b82f6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🎓 CS Department Intelligence System</h1>
  <p>TF-IDF NLP · Random Forest ML · Dynamic Per-Course AI Recommendations · Institutional Decision Support</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Free at console.groq.com")
    st.markdown("---")
    st.markdown("**Pipeline:**")
    st.markdown("1. Upload any grades CSV")
    st.markdown("2. RF trains 80%, tests 20%")
    st.markdown("3. Upload any jobs CSV")
    st.markdown("4. TF-IDF extracts skills")
    st.markdown("5. AI report per course")

# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Student Performance",
    "💼 Job Market NLP",
    "📋 Institutional Report"
])

# ╔══════════════════════════════════════╗
# ║  TAB 1 — STUDENT PERFORMANCE        ║
# ╚══════════════════════════════════════╝
with tab1:
    st.markdown('<p class="sec">01 · Student Performance — Random Forest Analysis</p>', unsafe_allow_html=True)
    st.markdown('<div class="box-blue">Works with <b>any</b> CSV. Columns auto-detected. Model trains on 80% → tests on 20%. No leakage: classification applied after split.</div>', unsafe_allow_html=True)

    up_stu = st.file_uploader("📂 Upload Student Grades CSV", type="csv", key="stu")

    if up_stu:
        df_raw = pd.read_csv(up_stu)
        st.success(f"✅ {len(df_raw)} students · {len(df_raw.columns)} columns")

        with st.expander("👁 Preview"):
            st.dataframe(df_raw.head())

        st.markdown("---")
        numeric_cols = df_raw.select_dtypes(include='number').columns.tolist()
        # auto-exclude non-course columns
        skip_kw = ['id','rank','student','semester','final','total','gpa','grade','index']
        auto_courses = [c for c in numeric_cols
                        if not any(k in c.lower() for k in skip_kw)]

        c1, c2 = st.columns(2)
        with c1:
            course_cols = st.multiselect("Course columns (features):", numeric_cols, default=auto_courses)
        with c2:
            final_col = st.selectbox("Final grade column (target):", numeric_cols,
                index=numeric_cols.index('Final_Grade') if 'Final_Grade' in numeric_cols else len(numeric_cols)-1)

        t1, t2 = st.columns(2)
        with t1: weak_t = st.slider("Weak below:", 30, 70, 60)
        with t2: avg_t  = st.slider("Average below:", 65, 90, 75)

        if st.button("🔍 Run ML Analysis", key="b1"):
            if len(course_cols) < 2: st.error("Need ≥2 course columns."); st.stop()

            df = df_raw.copy()

            # ── Descriptive (all data) ──
            avgs      = df[course_cols].mean().sort_values()
            stds      = df[course_cols].std()
            fail_rate = {c: (df[c] < weak_t).mean() * 100 for c in course_cols}
            weak_c    = avgs[avgs < weak_t]
            mid_c     = avgs[(avgs >= weak_t) & (avgs < avg_t)]
            good_c    = avgs[avgs >= avg_t]

            def classify(g):
                if g < weak_t:  return 'Weak'
                elif g < avg_t: return 'Average'
                return 'Excellent'

            df['Level'] = df[final_col].apply(classify)
            counts = df['Level'].value_counts()

            # ── ML: split THEN label (no leakage) ──
            X = df[course_cols];  y_raw = df[final_col]
            X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_raw, test_size=0.2, random_state=42)
            y_tr = yr_tr.apply(classify)
            y_te = yr_te.apply(classify)

            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_tr, y_tr)
            y_pr = rf.predict(X_te)

            acc  = accuracy_score(y_te, y_pr)
            prec = precision_score(y_te, y_pr, average='weighted', zero_division=0)
            rec  = recall_score(y_te, y_pr,    average='weighted', zero_division=0)
            f1   = f1_score(y_te, y_pr,        average='weighted', zero_division=0)
            cv   = cross_val_score(rf, X_tr, y_tr, cv=5, scoring='accuracy').mean()
            imps = pd.Series(rf.feature_importances_, index=course_cols).sort_values(ascending=False)

            # ── Display results ──
            st.markdown("---")
            m1,m2,m3,m4 = st.columns(4)
            for col,(l,v) in zip([m1,m2,m3,m4],[
                ("Students", len(df)),
                ("🔴 Weak",    f"{counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.0f}%)"),
                ("🟡 Average", f"{counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.0f}%)"),
                ("🟢 Excellent",f"{counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.0f}%)")
            ]):
                col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Course Average Scores**")
                bc=['#dc2626' if v<weak_t else '#d97706' if v<avg_t else '#16a34a' for v in avgs.values]
                fig, ax = plt.subplots(figsize=(8, max(5, len(course_cols)*.45)))
                fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#0f172a')
                bars = ax.barh(avgs.index, avgs.values, color=bc, height=.6)
                for b,v in zip(bars,avgs.values):
                    ax.text(v+.5, b.get_y()+b.get_height()/2, f'{v:.1f}', va='center', fontsize=8, color='#94a3b8')
                ax.axvline(avgs.mean(), color='#3b82f6', linestyle='--', lw=1.2)
                ax.set_xlim(0,110); ax.tick_params(colors='#94a3b8',labelsize=8); ax.spines[:].set_color('#1e3a5f')
                r=mpatches.Patch(color='#dc2626',label=f'Weak (<{weak_t})')
                a=mpatches.Patch(color='#d97706',label=f'Mid ({weak_t}–{avg_t})')
                g=mpatches.Patch(color='#16a34a',label=f'Good (>{avg_t})')
                ax.legend(handles=[r,a,g], fontsize=7, facecolor='#0f172a', labelcolor='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with ch2:
                st.markdown("**Level Distribution**")
                fig2, ax2 = plt.subplots(figsize=(5,5))
                fig2.patch.set_facecolor('#0f172a'); ax2.set_facecolor('#0f172a')
                lc={'Weak':'#dc2626','Average':'#d97706','Excellent':'#16a34a'}
                wedges,texts,auto = ax2.pie(counts.values, labels=counts.index,
                    colors=[lc.get(l,'#3b82f6') for l in counts.index],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor':'#0f172a','linewidth':2})
                for t in texts: t.set_color('#94a3b8')
                for a in auto:  a.set_color('white'); a.set_fontsize(9)
                plt.tight_layout(); st.pyplot(fig2); plt.close()

            # Course status
            st.markdown("---")
            cw,cm,cg = st.columns(3)
            with cw:
                st.markdown("**🔴 Weak**")
                if len(weak_c):
                    for c,v in weak_c.items():
                        st.markdown(f'<div class="box-red"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
                else: st.success("None")
            with cm:
                st.markdown("**🟡 Medium**")
                if len(mid_c):
                    for c,v in mid_c.items():
                        st.markdown(f'<div class="box-amber"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
                else: st.success("None")
            with cg:
                st.markdown("**🟢 Good**")
                for c,v in good_c.items():
                    st.markdown(f'<div class="box-green"><b>{c}</b> · Avg:{v:.1f}</div>', unsafe_allow_html=True)

            # ML metrics
            st.markdown("---")
            st.markdown("#### 🤖 Random Forest Metrics")
            st.caption(f"Train: {len(X_tr)} students (80%) · Test: {len(X_te)} students (20%) · 5-fold CV on train set")

            r1,r2,r3,r4,r5 = st.columns(5)
            for col,(l,v) in zip([r1,r2,r3,r4,r5],[
                ("Accuracy",f"{acc*100:.1f}%"),("Precision",f"{prec*100:.1f}%"),
                ("Recall",f"{rec*100:.1f}%"),("F1-Score",f"{f1*100:.1f}%"),("CV Acc.",f"{cv*100:.1f}%")
            ]):
                col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Confusion matrix
            st.markdown("**Confusion Matrix (test set)**")
            lbls = sorted(y_te.unique())
            cm_m = confusion_matrix(y_te, y_pr, labels=lbls)
            fig3, ax3 = plt.subplots(figsize=(5,4))
            fig3.patch.set_facecolor('#0f172a'); ax3.set_facecolor('#0f172a')
            ax3.imshow(cm_m, cmap='Blues')
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    clr = 'white' if cm_m[i,j] > cm_m.max()/2 else '#94a3b8'
                    ax3.text(j,i,str(cm_m[i,j]),ha='center',va='center',fontsize=13,fontweight='bold',color=clr)
            ax3.set_xticks(range(len(lbls))); ax3.set_xticklabels(lbls,color='#94a3b8',fontsize=9)
            ax3.set_yticks(range(len(lbls))); ax3.set_yticklabels(lbls,color='#94a3b8',fontsize=9)
            ax3.set_xlabel('Predicted',color='#94a3b8'); ax3.set_ylabel('Actual',color='#94a3b8')
            ax3.spines[:].set_color('#1e3a5f')
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            # Feature importance
            st.markdown("**Feature Importance — which courses influence final grade most**")
            top_imp = imps.sort_values(ascending=True)
            fig4, ax4 = plt.subplots(figsize=(8, max(4, len(course_cols)*.4)))
            fig4.patch.set_facecolor('#0f172a'); ax4.set_facecolor('#0f172a')
            ic = ['#3b82f6' if v > imps.mean() else '#334155' for v in top_imp.values]
            ax4.barh(top_imp.index, top_imp.values, color=ic, height=.6)
            ax4.axvline(imps.mean(), color='#d97706', linestyle='--', lw=1)
            ax4.tick_params(colors='#94a3b8',labelsize=8); ax4.spines[:].set_color('#1e3a5f')
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            # Save
            st.session_state.update({
                'avgs':avgs,'weak_c':weak_c,'mid_c':mid_c,'good_c':good_c,
                'counts':counts,'n_stu':len(df),'n_tr':len(X_tr),'n_te':len(X_te),
                'met':{'acc':acc,'prec':prec,'rec':rec,'f1':f1,'cv':cv},
                'imps':imps,'fr':fail_rate,'wt':weak_t,'at':avg_t,'cc':course_cols
            })
            st.success("✅ Analysis done → Tab 2")
    else:
        st.info("📂 Upload any student grades CSV.")

# ╔══════════════════════════════════════╗
# ║  TAB 2 — JOB MARKET NLP             ║
# ╚══════════════════════════════════════╝
with tab2:
    st.markdown('<p class="sec">02 · Job Market — TF-IDF NLP Skill Extraction</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box-blue">
    <b>NLP pipeline:</b><br>
    1. Tokenise & clean job descriptions (lowercase, remove punctuation)<br>
    2. Build TF-IDF matrix over all job postings<br>
    3. Sum TF-IDF scores per skill term → weighted demand score<br>
    4. Count raw frequency → mention count<br>
    5. Combine both signals to rank skills
    </div>
    """, unsafe_allow_html=True)

    up_job = st.file_uploader("📂 Upload Job Listings CSV", type="csv", key="job")

    if up_job:
        with st.spinner("Loading..."):
            jobs_df = pd.read_csv(up_job, on_bad_lines='skip')
        st.success(f"✅ {len(jobs_df)} job listings")

        text_cols = jobs_df.select_dtypes(include='object').columns.tolist()
        desc_col = st.selectbox("Job description column:", text_cols,
            index=text_cols.index('jobdescription') if 'jobdescription' in text_cols else 0)

        extra = st.text_input("Extra skills to track (comma-separated):",
                              placeholder="rust, llm, chatgpt, cybersecurity")

        if st.button("🔍 Run NLP Analysis", key="b2"):

            with st.spinner("Running TF-IDF NLP..."):

                # ── Skill taxonomy (cs-focused) ──
                TAXONOMY = {
                    "Programming Languages": [
                        "python","java","javascript","c++","c#","kotlin","swift",
                        "go","typescript","php","ruby","scala","rust","r programming"
                    ],
                    "Web & Mobile": [
                        "react","angular","vue","node.js","django","flask","spring",
                        "html5","css3","android","ios","flutter","next.js","express"
                    ],
                    "Data Science & AI": [
                        "machine learning","deep learning","tensorflow","pytorch",
                        "pandas","numpy","sql","nosql","mongodb","apache spark",
                        "hadoop","data science","nlp","tableau","power bi",
                        "data analysis","scikit-learn","keras"
                    ],
                    "Cloud & DevOps": [
                        "aws","azure","google cloud","docker","kubernetes",
                        "jenkins","ci/cd","terraform","linux","git","devops",
                        "ansible","microservices","serverless"
                    ],
                    "Cybersecurity": [
                        "cybersecurity","penetration testing","network security",
                        "encryption","identity access management","firewall",
                        "siem","ethical hacking","vulnerability assessment",
                        "security audit","zero trust"
                    ],
                    "Databases": [
                        "mysql","postgresql","oracle","sql server","redis",
                        "elasticsearch","cassandra","database design","data modeling",
                        "data warehouse"
                    ],
                    "Software Engineering": [
                        "agile","scrum","rest api","microservices","design patterns",
                        "software architecture","unit testing","test driven development",
                        "version control","code review","continuous integration"
                    ],
                    "Networking": [
                        "networking","tcp/ip","routing","switching","vpn",
                        "network administration","cisco","wireshark","network protocols",
                        "load balancing"
                    ],
                    "Emerging Tech": [
                        "blockchain","iot","embedded systems","computer vision",
                        "robotics","large language model","generative ai","ar/vr",
                        "edge computing","5g"
                    ]
                }

                if extra.strip():
                    TAXONOMY["Custom"] = [s.strip().lower() for s in extra.split(',') if s.strip()]

                # ── Clean text ──
                raw_texts = jobs_df[desc_col].dropna().tolist()
                def clean(t):
                    t = str(t).lower()
                    t = re.sub(r'[^a-z0-9\s/+#.]', ' ', t)
                    return re.sub(r'\s+', ' ', t).strip()
                clean_texts = [clean(t) for t in raw_texts]

                # ── TF-IDF on full corpus ──
                # each job = one document
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 3),   # unigrams, bigrams, trigrams
                    min_df=2,             # skill must appear in ≥2 jobs
                    max_df=0.95,          # ignore if in >95% of jobs
                    sublinear_tf=True     # log scaling
                )
                tfidf_matrix = vectorizer.fit_transform(clean_texts)
                feature_names = vectorizer.get_feature_names_out()

                # ── Score each skill ──
                # TF-IDF score = sum of TF-IDF weights across all docs
                tfidf_sums = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                tfidf_dict = dict(zip(feature_names, tfidf_sums))

                all_text = ' '.join(clean_texts)
                skill_results = {}
                for cat, skills in TAXONOMY.items():
                    for skill in skills:
                        skill_clean = clean(skill)
                        # raw frequency count
                        freq = all_text.count(skill_clean)
                        # TF-IDF weighted score
                        tfidf_score = tfidf_dict.get(skill_clean, 0.0)
                        # job coverage: % of jobs mentioning this skill
                        jobs_with = sum(1 for t in clean_texts if skill_clean in t)
                        coverage  = jobs_with / len(clean_texts) * 100
                        if freq > 0:
                            skill_results[skill] = {
                                'category':    cat,
                                'freq':        freq,
                                'tfidf_score': round(float(tfidf_score), 2),
                                'job_coverage': round(coverage, 1),
                                'jobs_with':   jobs_with
                            }

                # Sort by TF-IDF score (more meaningful than raw count)
                sorted_by_tfidf = sorted(
                    skill_results.items(),
                    key=lambda x: x[1]['tfidf_score'],
                    reverse=True
                )
                top30 = sorted_by_tfidf[:30]

            # ── Charts ──
            CAT_COLORS = {
                'Programming Languages':'#3b82f6','Web & Mobile':'#16a34a',
                'Data Science & AI':'#7c3aed','Cloud & DevOps':'#d97706',
                'Cybersecurity':'#dc2626','Databases':'#0891b2',
                'Software Engineering':'#059669','Networking':'#9333ea',
                'Emerging Tech':'#f59e0b','Custom':'#ec4899'
            }

            st.markdown("#### Top 20 Skills — ranked by TF-IDF score")
            names = [s[0] for s in top30[:20]]
            scores= [s[1]['tfidf_score'] for s in top30[:20]]
            covs  = [s[1]['job_coverage'] for s in top30[:20]]
            cats  = [s[1]['category'] for s in top30[:20]]
            bclrs = [CAT_COLORS.get(c,'#64748b') for c in cats]

            fig5, ax5 = plt.subplots(figsize=(10,8))
            fig5.patch.set_facecolor('#0f172a'); ax5.set_facecolor('#0f172a')
            bars5 = ax5.barh(names, scores, color=bclrs, height=.6)
            for b,v,cov in zip(bars5,scores,covs):
                ax5.text(v+.1, b.get_y()+b.get_height()/2,
                         f'{v:.1f} | {cov:.0f}% jobs', va='center', fontsize=7, color='#94a3b8')
            ax5.tick_params(colors='#94a3b8',labelsize=8); ax5.spines[:].set_color('#1e3a5f')
            ax5.set_xlabel('TF-IDF Score (higher = more important)',color='#94a3b8')
            handles=[mpatches.Patch(color=v,label=k) for k,v in CAT_COLORS.items() if k in cats]
            ax5.legend(handles=handles,fontsize=7,facecolor='#0f172a',labelcolor='#94a3b8',loc='lower right')
            plt.tight_layout(); st.pyplot(fig5); plt.close()

            # Stats
            s1,s2,s3,s4 = st.columns(4)
            for col,(l,v) in zip([s1,s2,s3,s4],[
                ("Total Jobs",len(jobs_df)),("Skills Found",len(skill_results)),
                ("Top Skill",top30[0][0] if top30 else '—'),
                ("Top TF-IDF",f"{top30[0][1]['tfidf_score']:.1f}" if top30 else 0)
            ]):
                col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.2rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Full skills table
            st.markdown("---")
            st.markdown("#### All Extracted Skills")
            skill_df = pd.DataFrame([
                {
                    'Skill': s,
                    'Category': d['category'],
                    'TF-IDF Score': d['tfidf_score'],
                    'Job Coverage %': d['job_coverage'],
                    'Jobs Mentioning': d['jobs_with'],
                    'Raw Freq.': d['freq']
                }
                for s, d in sorted_by_tfidf
            ])
            st.dataframe(skill_df, use_container_width=True)

            # Category breakdown
            st.markdown("---")
            st.markdown("#### Demand by Category (TF-IDF weighted)")
            cat_totals = {}
            for _,d in skill_results.items():
                cat_totals[d['category']] = cat_totals.get(d['category'],0) + d['tfidf_score']
            total_all = sum(cat_totals.values()) or 1
            for cat,total in sorted(cat_totals.items(),key=lambda x:x[1],reverse=True):
                pct = int(total/total_all*100)
                st.progress(pct, text=f"**{cat}** — TF-IDF total {total:.1f} ({pct}%)")

            # Save
            st.session_state.update({
                'top30':top30,'skill_results':skill_results,
                'n_jobs':len(jobs_df),'sorted_skills':sorted_by_tfidf
            })
            st.success("✅ NLP analysis done → Tab 3")
    else:
        st.info("📂 Upload a job listings CSV.")

# ╔══════════════════════════════════════╗
# ║  TAB 3 — INSTITUTIONAL REPORT       ║
# ╚══════════════════════════════════════╝
with tab3:
    st.markdown('<p class="sec">03 · Institutional Report — Dynamic AI Per-Course Recommendations</p>', unsafe_allow_html=True)

    has_s = 'weak_c'  in st.session_state
    has_j = 'top30'   in st.session_state

    if not has_s: st.warning("⚠️ Complete Tab 1 first.")
    if not has_j: st.warning("⚠️ Complete Tab 2 first.")

    if has_s and has_j:
        # unpack
        wc   = st.session_state['weak_c']
        mc   = st.session_state['mid_c']
        gc   = st.session_state['good_c']
        top30= st.session_state['top30']
        sr   = st.session_state['skill_results']
        cnt  = st.session_state['counts']
        ns   = st.session_state['n_stu']
        ntr  = st.session_state['n_tr']
        nte  = st.session_state['n_te']
        met  = st.session_state['met']
        imps = st.session_state['imps']
        fr   = st.session_state['fr']
        nj   = st.session_state['n_jobs']
        wt   = st.session_state['wt']
        at   = st.session_state['at']
        cc   = st.session_state['cc']

        # summary cards
        g1,g2,g3,g4,g5 = st.columns(5)
        for col,(l,v) in zip([g1,g2,g3,g4,g5],[
            ("Students",ns),("Train/Test",f"{ntr}/{nte}"),
            ("Weak Courses",len(wc)),("Jobs Analyzed",nj),("F1",f"{met['f1']*100:.1f}%")
        ]):
            col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.2rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # metrics explanation
        with st.expander("📐 How metrics were calculated"):
            st.markdown(f"""
**Train/Test Split:**
- 80% ({ntr} students) → train Random Forest
- 20% ({nte} students) → evaluate (never seen during training)
- Classification into Weak/Average/Excellent applied **after** split → no data leakage

**Why Random Forest?**
Appeared in 6/8 related studies; handles small datasets; gives Feature Importance

| Metric | Value | Formula |
|--------|-------|---------|
| Accuracy  | {met['acc']*100:.1f}% | (TP+TN) / all predictions |
| Precision | {met['prec']*100:.1f}% | TP / (TP+FP) — of predicted positives, how many are correct |
| Recall    | {met['rec']*100:.1f}% | TP / (TP+FN) — of actual positives, how many did we find |
| F1-Score  | {met['f1']*100:.1f}% | 2·P·R/(P+R) — harmonic mean |
| CV Accuracy | {met['cv']*100:.1f}% | 5-fold mean on training set — stability check |

**NLP (TF-IDF):**
- TF-IDF = Term Frequency × Inverse Document Frequency
- High score = skill appears often but is specific to certain job types (more signal)
- Job coverage % = percentage of all job postings mentioning this skill
            """)

        # weak courses
        st.markdown("#### 🔴 Courses Requiring Immediate Action")
        if len(wc):
            for c,v in wc.items():
                imp_v = imps.get(c, 0)
                st.markdown(
                    f'<div class="box-red"><b>{c}</b> — '
                    f'Avg: {v:.1f} · Fail rate: {fr.get(c,0):.1f}% · '
                    f'RF Importance: {imp_v:.3f}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("No weak courses.")

        # top skills
        st.markdown("---")
        st.markdown("#### 💼 Top 15 Market Skills (by TF-IDF)")
        t_df = pd.DataFrame([
            (s, d['category'], d['tfidf_score'], d['job_coverage'], d['jobs_with'])
            for s,d in top30[:15]
        ], columns=['Skill','Category','TF-IDF Score','Job Coverage %','Jobs Mentioning'])
        st.dataframe(t_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🤖 AI Institutional Report — Per-Course, Per-Dataset")
        st.markdown('<div class="box-blue">The AI reads your actual course names, fail rates, RF importance scores, and TF-IDF skill demand scores to generate fully dynamic recommendations. Different data → different report.</div>', unsafe_allow_html=True)

        if not groq_key:
            st.error("⚠️ Add Groq API Key in the sidebar.")
        else:
            if st.button("🚀 Generate Full Institutional Report", key="b3"):
                with st.spinner("Generating dynamic per-course report..."):
                    try:
                        client = Groq(api_key=groq_key)

                        weak_detail = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%, rf_importance={imps.get(c,0):.3f}"
                            for c,v in wc.items()
                        ]) or "None"

                        mid_detail = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%"
                            for c,v in mc.items()
                        ]) or "None"

                        top5_imp = "\n".join([f"- {c}: {v:.3f}" for c,v in imps.head(5).items()])

                        top20_skills = "\n".join([
                            f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, coverage={d['job_coverage']:.1f}% of jobs"
                            for s,d in top30[:20]
                        ])

                        existing = ", ".join(cc)

                        prompt = f"""You are a senior academic consultant writing a formal institutional report for the Head of a Computer Science Department.

=== STUDENT DATA ({ns} students) ===
Random Forest: Train={ntr} | Test={nte}
Accuracy={met['acc']*100:.1f}% | Precision={met['prec']*100:.1f}% | Recall={met['rec']*100:.1f}% | F1={met['f1']*100:.1f}% | CV={met['cv']*100:.1f}%
Thresholds: Weak<{wt} | Average {wt}-{at} | Excellent>{at}
Distribution: Weak={cnt.get('Weak',0)} | Average={cnt.get('Average',0)} | Excellent={cnt.get('Excellent',0)}

WEAK COURSES:
{weak_detail}

MEDIUM COURSES:
{mid_detail}

TOP 5 MOST INFLUENTIAL (RF Feature Importance):
{top5_imp}

CURRENT COURSES IN DATASET:
{existing}

=== JOB MARKET ({nj} listings, TF-IDF NLP analysis) ===
TOP 20 SKILLS BY TF-IDF SCORE:
{top20_skills}

=== REPORT STRUCTURE ===
Write a formal institutional report with EXACTLY these sections:

## 1. Executive Summary
Key numbers and 2–3 sentence overview.

## 2. Individual Course Recommendations
For EACH weak course, a dedicated subsection:

### [Exact Course Name]
- **Why students struggle:** specific analysis based on fail rate and RF importance
- **3 Teaching improvements:** concrete methods for THIS specific course topic
- **Learning Resources (specific, not generic):**
  * YouTube channels: name 2–3 real channels focused on THIS topic
    Example: for Operating Systems → "Brian Will OS lectures", "Neso Academy OS playlist", "MIT 6.004 Lectures"
  * Practice platforms: specific platform + specific resource
    Example: for Algorithms → "LeetCode medium problems tagged 'graphs'", "HackerRank algorithm certification"
  * Textbook: 1–2 well-known books for THIS exact topic
- **Expected impact:** how many students could move from Weak to Average if implemented
- **Priority:** High / Medium

## 3. Curriculum–Market Gap Analysis
Based on TF-IDF data:
- Skills with high TF-IDF score NOT in existing courses (list with scores)
- Estimated current curriculum coverage %
- Most critical gaps ranked by TF-IDF score

## 4. Recommended New Courses (based on gap analysis)
4–5 new courses, each:
- **Course name**
- **Skills covered** (from TF-IDF gap analysis, with actual scores)
- **Market evidence:** TF-IDF score and job coverage %
- **Suggested year/semester**
- **Priority:** High / Medium / Low

## 5. Priority Action Plan
Numbered list from most urgent to least. Timeline for each (Next semester / 1 year / 2–3 years).

Be specific. Use actual course names and numbers from the data. No generic advice."""

                        resp = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=3500
                        )
                        recs = resp.choices[0].message.content
                        st.markdown(recs)

                        # Download
                        st.markdown("---")
                        report = (
                            f"CS DEPARTMENT INSTITUTIONAL REPORT\n{'='*60}\n\n"
                            f"Students={ns} | Train={ntr} | Test={nte} | Jobs={nj}\n"
                            f"Accuracy={met['acc']*100:.1f}% | F1={met['f1']*100:.1f}%\n\n"
                            f"WEAK COURSES:\n{weak_detail}\n\n"
                            f"TOP MARKET SKILLS (TF-IDF):\n{top20_skills}\n\n"
                            f"{'='*60}\nAI RECOMMENDATIONS\n{'='*60}\n\n{recs}"
                        )
                        st.download_button(
                            "📥 Download Full Report",
                            data=report,
                            file_name="cs_dept_report.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌ {str(e)}")
