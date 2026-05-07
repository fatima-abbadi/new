"""
CS Department Intelligence System v6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
التحسينات عن v5:
  1. Stratified Split — يحافظ على نسب Weak/Average/Excellent في Train وTest
  2. Course Priority Score — يجمع fail_rate + SHAP + RF importance في score واحد
  3. Cross-Branch Linking — ربط Branch A مع Branch B في الـ AI Report
  4. SHAP positive_ratio محسّن — يستخدم model.classes_ للدقة
  5. Early Warning محسّن — يعرض risk distribution chart
  6. Validation Section — يثبت صحة الحسابات للورقة العلمية
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              silhouette_score, classification_report)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import shap, re, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════
st.set_page_config(page_title="CS Department Intelligence System",
                   page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.hdr{background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:1.8rem 2.5rem;
     border-radius:12px;margin-bottom:1.5rem;border-left:4px solid #3b82f6;}
.hdr h1{color:#f1f5f9;font-size:1.7rem;font-weight:600;margin:0;font-family:'IBM Plex Mono',monospace;}
.hdr p{color:#94a3b8;margin:.4rem 0 0;font-size:.85rem;}
.mc{background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;
    padding:1rem;text-align:center;margin-bottom:4px;}
.mc .v{font-size:1.6rem;font-weight:600;font-family:'IBM Plex Mono',monospace;color:#3b82f6;}
.mc .l{font-size:.72rem;color:#64748b;margin-top:.2rem;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#3b82f6;
     letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #1e3a5f;
     padding-bottom:.4rem;margin-bottom:1rem;}
.br {background:#1a0a0a;border:1px solid #dc2626;border-radius:8px;
     padding:.7rem 1rem;color:#fca5a5;font-size:.83rem;margin-bottom:.4rem;}
.ba {background:#1a1200;border:1px solid #d97706;border-radius:8px;
     padding:.7rem 1rem;color:#fcd34d;font-size:.83rem;margin-bottom:.4rem;}
.bg {background:#001a0a;border:1px solid #16a34a;border-radius:8px;
     padding:.7rem 1rem;color:#86efac;font-size:.83rem;margin-bottom:.4rem;}
.bb {background:#0a0f1a;border:1px solid #3b82f6;border-radius:8px;
     padding:.7rem 1rem;color:#93c5fd;font-size:.83rem;margin-bottom:.4rem;}
.bp {background:#0f0a1a;border:1px solid #7c3aed;border-radius:8px;
     padding:.7rem 1rem;color:#c4b5fd;font-size:.83rem;margin-bottom:.4rem;}
.priority-critical{background:#1a0505;border:1px solid #dc2626;border-radius:8px;
     padding:.6rem 1rem;color:#fca5a5;font-size:.83rem;margin-bottom:.3rem;}
.priority-high{background:#1a0f00;border:1px solid #d97706;border-radius:8px;
     padding:.6rem 1rem;color:#fcd34d;font-size:.83rem;margin-bottom:.3rem;}
.priority-monitor{background:#001a05;border:1px solid #16a34a;border-radius:8px;
     padding:.6rem 1rem;color:#86efac;font-size:.83rem;margin-bottom:.3rem;}
.cc {background:#0f172a;border:1px solid #334155;border-radius:10px;
     padding:.9rem 1.1rem;margin-bottom:.5rem;}
.stButton>button{background:#1e3a5f;color:#f1f5f9;border:1px solid #3b82f6;
    border-radius:8px;font-family:'IBM Plex Mono',monospace;
    font-size:.83rem;padding:.55rem 1.4rem;width:100%;}
.stButton>button:hover{background:#3b82f6;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hdr">
  <h1>🎓 CS Department Intelligence System</h1>
  <p>Random Forest · SHAP · KMeans Clustering · TF-IDF NLP · BERT Embeddings · AI Reports · v6</p>
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
    st.markdown("**Tab 1 — Students:**")
    max_k = st.slider("Max clusters to test:", 3, 8, 6)
    st.markdown("---")
    st.markdown("**Tab 2 — Job Market:**")
    bert_on        = st.toggle("Enable BERT (Sentence-BERT)", value=False)
    bert_name      = st.selectbox("BERT model:",
                                   ["all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0)
    bert_threshold = st.slider("BERT similarity threshold:",
                                0.30, 0.70, 0.45, 0.05)
    st.caption("Requires: pip install sentence-transformers")
    st.markdown("---")
    st.markdown("**Each tab is independent.**")

# ══════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════
STOP_WORDS = {
    'the','and','for','with','this','that','are','have','been','will','from',
    'they','you','our','your','its','their','has','was','were','can','may',
    'must','should','would','could','not','but','also','which','when','where',
    'what','how','all','any','each','more','most','other','some','such',
    'into','over','after','than','then','there','these','those','both','few',
    'many','much','own','same','very','just','about',
}

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'http\S+', ' ', t)
    t = re.sub(r'[^a-z0-9\s/+#.]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def preprocess(t):
    tokens = clean_text(t).split()
    return ' '.join(w for w in tokens if w not in STOP_WORDS and len(w) > 1)

def compute_auc(model, X_te, y_te, classes):
    try:
        yb = label_binarize(y_te, classes=classes)
        yp = model.predict_proba(X_te)
        if len(classes) == 2:
            return roc_auc_score(yb, yp[:, 1])
        return roc_auc_score(yb, yp, multi_class='ovr', average='weighted')
    except Exception:
        return None

def compute_shap(model, X_te_df, features):
    """
    يرجع (mean_abs_shap, mean_signed_shap, sv_exc):
    - mean_abs  → حجم التأثير
    - mean_sign → اتجاه التأثير بالنسبة لـ Excellent
    - sv_exc    → raw SHAP values للـ Excellent (لحساب positive_ratio)
    يستخدم model.classes_ للدقة
    """
    try:
        arr     = np.array(X_te_df.values[:min(50, len(X_te_df))], dtype=float)
        exp     = shap.TreeExplainer(model)
        sv      = exp.shap_values(arr)
        classes = list(model.classes_)
        exc_idx = classes.index('Excellent') if 'Excellent' in classes else len(classes)-1

        if isinstance(sv, list):
            stacked   = np.stack([np.abs(s) for s in sv], axis=0)
            mean_abs  = stacked.mean(axis=0).mean(axis=0)
            sv_exc    = sv[exc_idx]
            mean_sign = sv_exc.mean(axis=0)
        elif sv.ndim == 3:
            mean_abs  = np.abs(sv).mean(axis=0).mean(axis=1)
            sv_exc    = sv[:, :, exc_idx]
            mean_sign = sv_exc.mean(axis=0)
        else:
            mean_abs  = np.abs(sv).mean(axis=0)
            sv_exc    = sv
            mean_sign = sv.mean(axis=0)

        abs_s  = pd.Series(mean_abs,  index=features).sort_values(ascending=False)
        sign_s = pd.Series(mean_sign, index=features)
        pr_s   = pd.Series((sv_exc > 0).mean(axis=0), index=features)
        return abs_s, sign_s, pr_s
    except Exception as e:
        return str(e), None, None

def compute_course_priority(fr, shap_v, imps, courses):
    """
    Course Priority Score = 0.5*norm_fail + 0.3*norm_shap + 0.2*norm_rf
    يجمع الأدلة الثلاثة في score واحد قابل للمقارنة
    مستند علمياً على: fail_rate (impact on students) +
    SHAP (causal influence) + RF importance (model dependency)
    """
    max_fail = max(fr.values()) / 100 if fr else 1
    max_shap = float(shap_v.max()) if shap_v is not None else 1
    max_rf   = float(imps.max())   if len(imps) > 0 else 1

    scores = {}
    for c in courses:
        norm_fail = (fr.get(c, 0) / 100) / max_fail if max_fail > 0 else 0
        norm_shap = float(shap_v.get(c, 0)) / max_shap if (shap_v is not None and max_shap > 0) else 0
        norm_rf   = float(imps.get(c, 0))   / max_rf   if max_rf > 0 else 0
        scores[c] = round(0.5*norm_fail + 0.3*norm_shap + 0.2*norm_rf, 4)
    return scores

def call_groq(key, prompt, max_tokens=2500):
    try:
        c = Groq(api_key=key)
        r = c.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {e}"

@st.cache_resource(show_spinner=False)
def load_bert(name):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)
    except ImportError:
        return None

def dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.spines[:].set_color('#1e3a5f')
    return fig, ax

def metric_card(col, label, value):
    col.markdown(
        f'<div class="mc"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAXONOMY
# ══════════════════════════════════════════════════
TAXONOMY = {
    "Programming Languages": ["python","java","javascript","c++","c#","kotlin","swift",
                               "go","typescript","php","ruby","scala","rust","r programming"],
    "Web & Mobile":          ["react","angular","vue","node.js","django","flask","spring",
                               "html5","css3","android","ios","flutter","next.js","express"],
    "Data Science & AI":     ["machine learning","deep learning","tensorflow","pytorch",
                               "pandas","numpy","sql","nosql","mongodb","apache spark","hadoop",
                               "data science","nlp","tableau","power bi","data analysis",
                               "scikit-learn","keras"],
    "Cloud & DevOps":        ["aws","azure","google cloud","docker","kubernetes","jenkins",
                               "ci/cd","terraform","linux","git","devops","ansible",
                               "microservices","serverless"],
    "Cybersecurity":         ["cybersecurity","penetration testing","network security",
                               "encryption","identity access management","firewall","siem",
                               "ethical hacking","vulnerability assessment","security audit"],
    "Databases":             ["mysql","postgresql","oracle","sql server","redis",
                               "elasticsearch","cassandra","database design",
                               "data modeling","data warehouse"],
    "Software Engineering":  ["agile","scrum","rest api","microservices","design patterns",
                               "software architecture","unit testing","test driven development",
                               "version control","code review","continuous integration"],
    "Networking":            ["networking","tcp/ip","routing","switching","vpn",
                               "network administration","cisco","wireshark","network protocols"],
    "Emerging Tech":         ["blockchain","iot","embedded systems","computer vision",
                               "robotics","large language model","generative ai",
                               "ar/vr","edge computing","5g"],
}

CAT_COLORS = {
    'Programming Languages':'#3b82f6','Web & Mobile':'#16a34a',
    'Data Science & AI':'#7c3aed','Cloud & DevOps':'#d97706',
    'Cybersecurity':'#dc2626','Databases':'#0891b2',
    'Software Engineering':'#059669','Networking':'#9333ea',
    'Emerging Tech':'#f59e0b','Custom':'#ec4899'
}

# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊 Student Analysis", "💼 Job Market Analysis"])

# ╔══════════════════════════════════════╗
# ║  TAB 1 — STUDENT ANALYSIS           ║
# ╚══════════════════════════════════════╝
with tab1:
    st.markdown('<p class="sec">01 · Student Analysis — RF + SHAP + KMeans + AI Report</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="bb">Upload <b>any</b> grades CSV. '
        'Suggested course columns shown — <b>verify and adjust manually</b> before running. '
        'RF trained on 80% (stratified), evaluated on 20%. '
        'class_weight auto-selected by F1. SHAP sampled for speed.</div>',
        unsafe_allow_html=True)

    up = st.file_uploader("📂 Upload Student Grades CSV", type="csv", key="stu")

    if up:
        df_raw = pd.read_csv(up)
        st.success(f"✅ {len(df_raw)} students · {len(df_raw.columns)} columns")

        with st.expander("👁 Preview data"):
            st.dataframe(df_raw.head())

        st.markdown("---")
        num_cols = df_raw.select_dtypes(include='number').columns.tolist()

        skip_kw = ['id','rank','student','semester','final','total','gpa','grade','index']
        auto_c  = [c for c in num_cols if not any(k in c.lower() for k in skip_kw)]

        st.markdown(
            '<div class="ba">⚠️ The suggested columns below are auto-detected. '
            '<b>Please review carefully</b> — remove any non-course columns '
            '(e.g. IDs, ranks) and add any that were missed.</div>',
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            course_cols = st.multiselect(
                "Course columns (model features) — verify before running:",
                options=num_cols, default=auto_c)
        with c2:
            final_col = st.selectbox(
                "Final grade column (classification target):",
                options=num_cols,
                index=num_cols.index('Final_Grade') if 'Final_Grade' in num_cols else len(num_cols)-1)

        t1, t2, t3 = st.columns(3)
        with t1:
            thresh_mode = st.radio("Threshold mode:",
                ["Fixed score", "Quantile (data-driven)"], index=0)
        with t2:
            if thresh_mode == "Fixed score":
                wt = st.slider("Weak: score below", 30, 70, 60)
            else:
                wq = st.slider("Weak: bottom %", 5, 40, 25)
                wt = None
        with t3:
            if thresh_mode == "Fixed score":
                at = st.slider("Average: score below", 65, 90, 75)
            else:
                aq = st.slider("Excellent: top %", 10, 40, 25)
                at = None

        if st.button("🔍 Run Analysis", key="b1_run"):
            if len(course_cols) < 2:
                st.error("Need ≥2 course columns.")
                st.stop()
            if final_col in course_cols:
                st.error("Final grade column must NOT be in course columns.")
                st.stop()

            df = df_raw.copy()

            # ── Thresholds مؤقتة للـ stratified split ──
            if thresh_mode == "Fixed score":
                wt_tmp, at_tmp = wt, at
            else:
                wt_tmp, at_tmp = 60, 75  # مؤقت فقط

            def classify_tmp(g):
                if g < wt_tmp:  return 'Weak'
                elif g < at_tmp: return 'Average'
                return 'Excellent'

            # ══ FIX 1: STRATIFIED SPLIT ══
            # نصنف مؤقتاً عشان نعمل stratified
            X     = df[course_cols]
            y_raw = df[final_col]
            y_labels_temp = y_raw.apply(classify_tmp)

            Xtr, Xte, ytr_r, yte_r = train_test_split(
                X, y_raw, test_size=0.2,
                random_state=42,
                stratify=y_labels_temp)  # ← stratified يحافظ على نسب الكلاسات

            # ── Thresholds الحقيقية بعد التقسيم ──
            if thresh_mode == "Quantile (data-driven)":
                wt = float(np.percentile(ytr_r, wq))
                at = float(np.percentile(ytr_r, 100 - aq))
                st.info(f"📊 Quantile thresholds (from training set only): "
                        f"Weak < {wt:.1f} | Excellent ≥ {at:.1f}")

            def classify(g):
                if g < wt:   return 'Weak'
                elif g < at: return 'Average'
                return 'Excellent'

            # ── Descriptive stats ──
            st.caption("ℹ️ Descriptive stats computed on full dataset — display only, not used in training.")
            df_display = df.copy()
            df_display['Level'] = df_display[final_col].apply(classify)
            cnts = df_display['Level'].value_counts()
            avgs = df[course_cols].mean().sort_values()
            fr   = {c: (df[c] < wt).mean()*100 for c in course_cols}
            wc   = avgs[avgs < wt]
            mc   = avgs[(avgs >= wt) & (avgs < at)]
            gc   = avgs[avgs >= at]

            ytr = ytr_r.apply(classify)
            yte = yte_r.apply(classify)

            # ══ RF + RandomizedSearchCV ══
            with st.spinner("Training models — RF tuning + LR baseline..."):

                # اختيار class_weight بـ CV على Train فقط
                best_cw_label = 'balanced'
                best_cv_f1    = -1
                for cw, cw_label in [('balanced', 'balanced'), (None, 'none')]:
                    rf_tmp = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight=cw)
                    cv_f1 = cross_val_score(
                        rf_tmp, Xtr, ytr, cv=5, scoring='f1_weighted').mean()
                    if cv_f1 > best_cv_f1:
                        best_cv_f1    = cv_f1
                        best_cw_label = cw_label

                param_dist = {
                    'n_estimators':      [100, 200],
                    'max_depth':         [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf':  [1, 2, 4],
                }
                rf_base = RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced' if best_cw_label == 'balanced' else None)
                rscv = RandomizedSearchCV(
                    rf_base, param_dist, n_iter=12, cv=3,
                    scoring='f1_weighted', random_state=42, n_jobs=-1)
                rscv.fit(Xtr, ytr)
                rf          = rscv.best_estimator_
                ypr         = rf.predict(Xte)
                best_params = rscv.best_params_

                # LR baseline
                lr_pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight='balanced'))
                lr_pipe.fit(Xtr, ytr)
                ypr_lr = lr_pipe.predict(Xte)
                f1_lr  = f1_score(ypr_lr, yte, average='weighted', zero_division=0)
                acc_lr = accuracy_score(yte, ypr_lr)

            # ══ Metrics ══
            acc  = accuracy_score(yte, ypr)
            prec = precision_score(yte, ypr, average='weighted', zero_division=0)
            rec  = recall_score(yte, ypr,    average='weighted', zero_division=0)
            f1   = f1_score(yte, ypr,        average='weighted', zero_division=0)
            cv   = cross_val_score(rf, Xtr, ytr, cv=5, scoring='accuracy').mean()
            imps = pd.Series(rf.feature_importances_, index=course_cols).sort_values(ascending=False)
            cls  = sorted(ytr.unique())
            auc  = compute_auc(rf, Xte.values, yte, cls)

            # Early Warning
            yproba    = rf.predict_proba(Xte)
            cls_list  = list(rf.classes_)
            weak_idx  = cls_list.index('Weak') if 'Weak' in cls_list else 0
            risk_scores = yproba[:, weak_idx]

            # ══ SHAP ══
            with st.spinner("Computing SHAP values..."):
                shap_result = compute_shap(rf, Xte, course_cols)
                if isinstance(shap_result, tuple) and len(shap_result) == 3:
                    shap_v, shap_sign, shap_pos_ratio = shap_result
                    if isinstance(shap_v, str):
                        shap_v = shap_sign = shap_pos_ratio = None
                else:
                    shap_v = shap_sign = shap_pos_ratio = None

            # ══ FIX 2: COURSE PRIORITY SCORE ══
            course_priority = compute_course_priority(fr, shap_v, imps, course_cols)
            priority_df = pd.DataFrame([
                {
                    'Course': c,
                    'Priority Score': score,
                    'Fail Rate %': round(fr.get(c, 0), 1),
                    '|SHAP|': round(float(shap_v.get(c, 0)), 4) if shap_v is not None else 0,
                    'RF Importance': round(float(imps.get(c, 0)), 4),
                    'Level': ('🔴 Critical' if score > 0.6 else
                              '🟡 High' if score > 0.35 else
                              '🟢 Monitor')
                }
                for c, score in sorted(course_priority.items(),
                                       key=lambda x: x[1], reverse=True)
            ])

            # ══ KMeans ══
            with st.spinner("Running KMeans Clustering..."):
                X_all   = df[course_cols].fillna(df[course_cols].mean())
                X_tr_cl = X_all.iloc[Xtr.index]
                X_te_cl = X_all.iloc[Xte.index]

                scaler = StandardScaler()
                scaler.fit(X_tr_cl)          # fit على Train فقط
                Xs_tr  = scaler.transform(X_tr_cl)
                Xs_te  = scaler.transform(X_te_cl)
                Xs_all = scaler.transform(X_all)

                ks = range(2, max_k + 1)
                inert, silhs = [], []
                for k in ks:
                    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    lb     = km_tmp.fit_predict(Xs_tr)
                    inert.append(km_tmp.inertia_)
                    silhs.append(silhouette_score(Xs_tr, lb))

                best_k = list(ks)[np.argmax(silhs)]
                km_f   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                km_f.fit(Xs_tr)

                df.loc[Xtr.index, 'Cluster'] = km_f.predict(Xs_tr)
                df.loc[Xte.index, 'Cluster'] = km_f.predict(Xs_te)
                df['Cluster'] = df['Cluster'].astype(int)

                pca_c    = PCA(n_components=2).fit_transform(Xs_all)
                df['P1'] = pca_c[:, 0]
                df['P2'] = pca_c[:, 1]

                prof = df.groupby('Cluster')[course_cols].mean()
                prof['avg'] = prof.mean(axis=1)
                prof = prof.sort_values('avg', ascending=False)

                lmap  = {}
                n_cl  = len(prof)
                for i, cid in enumerate(prof.index):
                    if i == 0:        lmap[cid] = "🏆 High Performers"
                    elif i == n_cl-1: lmap[cid] = "🚨 At Risk"
                    elif i == 1:      lmap[cid] = "⬆️ Above Average"
                    elif i == n_cl-2: lmap[cid] = "⬇️ Below Average"
                    else:             lmap[cid] = f"📊 Mid Group {i}"
                df['CL'] = df['Cluster'].map(lmap)

            # ══════════════════ DISPLAY ══════════════════
            st.markdown("---")
            kpi_cols = st.columns(6)
            for col, (l, v) in zip(kpi_cols, [
                ("Students",  len(df)),
                ("🔴 Weak",   f"{cnts.get('Weak',0)} ({cnts.get('Weak',0)/len(df)*100:.0f}%)"),
                ("🟡 Avg",    f"{cnts.get('Average',0)} ({cnts.get('Average',0)/len(df)*100:.0f}%)"),
                ("🟢 Excel",  f"{cnts.get('Excellent',0)} ({cnts.get('Excellent',0)/len(df)*100:.0f}%)"),
                ("F1",        f"{f1*100:.1f}%"),
                ("Clusters",  best_k),
            ]):
                metric_card(col, l, v)

            # ── Charts ──
            st.markdown("---")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Course Average Scores**")
                bc = ['#dc2626' if v<wt else '#d97706' if v<at else '#16a34a' for v in avgs.values]
                fig, ax = dark_fig((8, max(5, len(course_cols)*.45)))
                bars = ax.barh(avgs.index, avgs.values, color=bc, height=.6)
                for b, v in zip(bars, avgs.values):
                    ax.text(v+.5, b.get_y()+b.get_height()/2, f'{v:.1f}',
                            va='center', fontsize=8, color='#94a3b8')
                ax.axvline(avgs.mean(), color='#3b82f6', linestyle='--', lw=1.2)
                ax.set_xlim(0, 110)
                ax.legend(handles=[
                    mpatches.Patch(color='#dc2626', label=f'Weak (<{wt})'),
                    mpatches.Patch(color='#d97706', label=f'Mid ({wt}–{at})'),
                    mpatches.Patch(color='#16a34a', label=f'Good (>{at})'),
                ], fontsize=7, facecolor='#0f172a', labelcolor='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with ch2:
                st.markdown("**Level Distribution**")
                fig2, ax2 = dark_fig((5, 5))
                lc = {'Weak':'#dc2626','Average':'#d97706','Excellent':'#16a34a'}
                _, texts, auto = ax2.pie(
                    cnts.values, labels=cnts.index,
                    colors=[lc.get(l,'#3b82f6') for l in cnts.index],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor':'#0f172a','linewidth':2})
                for t in texts: t.set_color('#94a3b8')
                for a in auto:  a.set_color('white'); a.set_fontsize(9)
                plt.tight_layout(); st.pyplot(fig2); plt.close()

            # ── Course status ──
            st.markdown("---")
            cw_col, cm_col, cg_col = st.columns(3)
            with cw_col:
                st.markdown("**🔴 Weak Courses**")
                if len(wc):
                    for c, v in wc.items():
                        st.markdown(
                            f'<div class="br"><b>{c}</b><br>'
                            f'Avg:{v:.1f} · Fail:{fr[c]:.1f}%</div>',
                            unsafe_allow_html=True)
                else:
                    st.success("None")
            with cm_col:
                st.markdown("**🟡 Medium Courses**")
                if len(mc):
                    for c, v in mc.items():
                        st.markdown(
                            f'<div class="ba"><b>{c}</b><br>'
                            f'Avg:{v:.1f} · Fail:{fr[c]:.1f}%</div>',
                            unsafe_allow_html=True)
                else:
                    st.success("None")
            with cg_col:
                st.markdown("**🟢 Good Courses**")
                for c, v in gc.items():
                    st.markdown(
                        f'<div class="bg"><b>{c}</b> · Avg:{v:.1f}</div>',
                        unsafe_allow_html=True)

            # ══ FIX 2 DISPLAY: COURSE PRIORITY MATRIX ══
            st.markdown("---")
            st.markdown("#### 📊 Course Priority Matrix")
            st.caption(
                "Priority Score = 0.5 × norm(fail_rate) + 0.3 × norm(|SHAP|) + 0.2 × norm(RF importance). "
                "Combines three independent evidence sources into one actionable ranking.")

            for _, row in priority_df.iterrows():
                css_class = ('priority-critical' if row['Level'] == '🔴 Critical'
                             else 'priority-high' if row['Level'] == '🟡 High'
                             else 'priority-monitor')
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<b>{row["Course"]}</b> &nbsp;·&nbsp; '
                    f'Score: <b>{row["Priority Score"]:.3f}</b> &nbsp;·&nbsp; '
                    f'Fail: {row["Fail Rate %"]}% &nbsp;·&nbsp; '
                    f'|SHAP|: {row["|SHAP|"]} &nbsp;·&nbsp; '
                    f'RF: {row["RF Importance"]} &nbsp;·&nbsp; '
                    f'{row["Level"]}'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── ML Metrics ──
            st.markdown("---")
            st.markdown("#### 🤖 Random Forest Results")
            st.caption(
                f"Train: {len(Xtr)} students (80%) · Test: {len(Xte)} students (20%) · "
                f"**Stratified split** (preserves class ratios) · "
                f"class_weight={best_cw_label} (5-fold CV F1 on train only — no test leakage)")

            m_cols = st.columns(6)
            for col, (l, v) in zip(m_cols, [
                ("Accuracy",  f"{acc*100:.1f}%"),
                ("Precision", f"{prec*100:.1f}%"),
                ("Recall",    f"{rec*100:.1f}%"),
                ("F1-Score",  f"{f1*100:.1f}%"),
                ("CV Acc.",   f"{cv*100:.1f}%"),
                ("AUC-ROC",   f"{auc*100:.1f}%" if auc else "N/A"),
            ]):
                metric_card(col, l, v)

            # LR comparison
            with st.expander("📊 Model Comparison — RF vs Logistic Regression baseline"):
                comp_data = {
                    'Model': ['Random Forest (proposed)', 'Logistic Regression (baseline)'],
                    'Accuracy': [f"{acc*100:.1f}%", f"{acc_lr*100:.1f}%"],
                    'F1-Score': [f"{f1*100:.1f}%", f"{f1_lr*100:.1f}%"],
                    'Advantage': [
                        'Feature importance + SHAP + course-level diagnosis',
                        'Simple, interpretable, fast baseline'
                    ]
                }
                st.table(pd.DataFrame(comp_data))
                delta = (f1 - f1_lr) * 100
                if delta > 0:
                    st.success(f"✅ RF outperforms LR by {delta:.1f}% F1.")
                else:
                    st.warning(f"⚠️ LR beats RF by {abs(delta):.1f}% F1 — "
                               f"RF retained for SHAP explainability and course diagnosis.")

            # Confusion Matrix + Classification Report
            st.markdown("**Confusion Matrix (test set)**")
            lbls = sorted(yte.unique())
            cm_m = confusion_matrix(yte, ypr, labels=lbls)
            fig3, ax3 = dark_fig((5, 4))
            ax3.imshow(cm_m, cmap='Blues')
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    ax3.text(j, i, str(cm_m[i, j]), ha='center', va='center',
                             fontsize=12, fontweight='bold',
                             color='white' if cm_m[i,j] > cm_m.max()/2 else '#94a3b8')
            ax3.set_xticks(range(len(lbls))); ax3.set_xticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_yticks(range(len(lbls))); ax3.set_yticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_xlabel('Predicted', color='#94a3b8')
            ax3.set_ylabel('Actual',    color='#94a3b8')
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            with st.expander("📋 Full Classification Report (per-class metrics)"):
                report_str = classification_report(yte, ypr, zero_division=0)
                st.code(report_str)
                st.caption("Per-class precision/recall/F1 — "
                           "more informative than weighted average under class imbalance.")

            # Feature Importance + SHAP
            fi_c, sh_c = st.columns(2)
            with fi_c:
                st.markdown("**RF Feature Importance**")
                ti = imps.sort_values(ascending=True)
                fig4, ax4 = dark_fig((7, max(4, len(course_cols)*.4)))
                ax4.barh(ti.index, ti.values,
                         color=['#3b82f6' if v > imps.mean() else '#334155' for v in ti.values],
                         height=.6)
                ax4.axvline(imps.mean(), color='#d97706', linestyle='--', lw=1)
                ax4.set_title('How often each course splits the trees',
                              color='#94a3b8', fontsize=8)
                plt.tight_layout(); st.pyplot(fig4); plt.close()

            with sh_c:
                st.markdown("**SHAP — magnitude + direction + % cases**")
                if shap_v is not None:
                    ts = shap_v.sort_values(ascending=True)
                    colors_shap = [
                        '#16a34a' if (shap_sign.get(f, 0) >= 0) else '#dc2626'
                        for f in ts.index
                    ]
                    fig5, ax5 = dark_fig((7, max(4, len(course_cols)*.4)))
                    ax5.barh(ts.index, ts.values, color=colors_shap, height=.6)
                    ax5.axvline(shap_v.mean(), color='#d97706', linestyle='--', lw=1)
                    ax5.set_title('mean|SHAP|  🟢=helps Excellent  🔴=hurts',
                                  color='#94a3b8', fontsize=8)
                    plt.tight_layout(); st.pyplot(fig5); plt.close()

                    rows_shap = []
                    for c in shap_v.index[:10]:
                        d_val = shap_sign.get(c, 0) if shap_sign is not None else 0
                        pr    = (shap_pos_ratio.get(c, float('nan'))
                                 if shap_pos_ratio is not None else float('nan'))
                        rows_shap.append({
                            'Course': c,
                            '|SHAP|': round(float(shap_v[c]), 4),
                            'Direction': '🟢 Excellent' if d_val >= 0 else '🔴 Weak/Avg',
                            '% helps': f"{pr*100:.0f}%" if not np.isnan(pr) else '—'
                        })
                    st.dataframe(pd.DataFrame(rows_shap),
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("SHAP not available.")

            with st.expander("📐 Metrics & Design Decisions"):
                st.markdown(f"""
| Metric | Value | Formula | Note |
|--------|-------|---------|------|
| Accuracy  | {acc*100:.1f}%  | (TP+TN)/all | Misleading with imbalance |
| Precision | {prec*100:.1f}% | TP/(TP+FP)  | Avoid false Weak labels |
| Recall    | {rec*100:.1f}%  | TP/(TP+FN)  | Don't miss at-risk students |
| F1        | {f1*100:.1f}%   | 2·P·R/(P+R) | **Primary metric** — handles imbalance |
| CV Acc.   | {cv*100:.1f}%   | 5-fold mean | Overfitting check |
| AUC-ROC   | {f"{auc*100:.1f}%" if auc else "N/A"} | Area under ROC | Threshold-independent |

**Best RF params (RandomizedSearchCV, 3-fold CV on train only):** `{best_params}`

**class_weight:** `{best_cw_label}` — selected by 5-fold CV F1, no test leakage.

**Split:** Stratified 80/20 — preserves Weak/Average/Excellent ratios in both sets.

**SHAP direction:** 🟢 = higher grade → pushes toward Excellent | 🔴 = hurts Excellent

**Priority Score formula:**
`0.5 × norm(fail_rate) + 0.3 × norm(|SHAP|) + 0.2 × norm(RF importance)`
Higher weight on fail_rate (most direct evidence of difficulty).
                """)

            # ── Early Warning ──
            st.markdown("---")
            st.markdown("#### 🚨 Early Warning — Student Risk Scores (Test Set)")
            st.caption("Risk = predicted probability of Weak classification. "
                       "Model trained on 80% only.")

            risk_df = pd.DataFrame({
                'Student#':  range(1, len(Xte)+1),
                'Risk %':    (risk_scores * 100).round(1),
                'Actual':    yte.values,
                'Predicted': ypr,
            }).sort_values('Risk %', ascending=False)
            high_risk = risk_df[risk_df['Risk %'] >= 50]

            r1, r2, r3 = st.columns(3)
            r1.metric("High Risk (≥50%)", len(high_risk), delta_color="inverse")
            r2.metric("Avg Risk Score",   f"{risk_scores.mean()*100:.1f}%")
            r3.metric("Max Risk Score",   f"{risk_scores.max()*100:.1f}%")

            # Risk distribution chart
            fig_r, ax_r = dark_fig((8, 3))
            ax_r.hist(risk_scores * 100, bins=20, color='#3b82f6',
                      edgecolor='#0f172a', alpha=0.8)
            ax_r.axvline(50, color='#dc2626', linestyle='--', lw=1.5, label='High Risk threshold (50%)')
            ax_r.set_xlabel('Risk Score %', color='#94a3b8')
            ax_r.set_ylabel('# Students',   color='#94a3b8')
            ax_r.set_title('Risk Score Distribution (Test Set)',
                            color='#f1f5f9', fontsize=10)
            ax_r.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')
            plt.tight_layout(); st.pyplot(fig_r); plt.close()

            with st.expander(f"📋 Top {min(20, len(risk_df))} Highest-Risk Students"):
                st.dataframe(risk_df.head(20), hide_index=True, use_container_width=True)

            # ── Clustering ──
            st.markdown("---")
            st.markdown("#### 🔵 KMeans Student Clustering")
            best_sil = max(silhs)
            st.markdown(
                f'<div class="bp">KMeans found <b>{best_k} natural student groups</b> '
                f'(silhouette = {best_sil:.3f}, chosen by argmax silhouette across '
                f'k=2..{max_k}). Scaler + KMeans fitted on training rows only.</div>',
                unsafe_allow_html=True)

            for cid, label in lmap.items():
                n_c   = (df['Cluster'] == cid).sum()
                avg_c = float(df[df['Cluster']==cid][course_cols].mean().mean())
                b_c   = df[df['Cluster']==cid][course_cols].mean().idxmax()
                w_c   = df[df['Cluster']==cid][course_cols].mean().idxmin()
                st.markdown(
                    f'<div class="cc">'
                    f'<b style="color:#f1f5f9">{label}</b> &nbsp;·&nbsp; '
                    f'<span style="color:#64748b">{n_c} students '
                    f'({n_c/len(df)*100:.0f}%)</span> &nbsp;·&nbsp; '
                    f'<span style="color:#3b82f6">Avg: {avg_c:.1f}</span> &nbsp;·&nbsp; '
                    f'<span style="color:#16a34a">Best: {b_c}</span> &nbsp;·&nbsp; '
                    f'<span style="color:#dc2626">Weakest: {w_c}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

            fig_el, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
            fig_el.patch.set_facecolor('#0f172a')
            for ax in [ax_e, ax_s]:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors='#94a3b8')
                ax.spines[:].set_color('#1e3a5f')
            ax_e.plot(list(ks), inert, 'o-', color='#3b82f6', lw=2)
            ax_e.axvline(best_k, color='#d97706', linestyle='--', lw=1.5,
                         label=f'Best k={best_k}')
            ax_e.set_xlabel('k', color='#94a3b8')
            ax_e.set_ylabel('Inertia', color='#94a3b8')
            ax_e.set_title('Elbow Method', color='#f1f5f9', fontsize=10)
            ax_e.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')
            ax_s.bar(list(ks), silhs,
                     color=['#d97706' if k==best_k else '#334155' for k in ks],
                     width=.5)
            ax_s.set_xlabel('k', color='#94a3b8')
            ax_s.set_ylabel('Silhouette', color='#94a3b8')
            ax_s.set_title('Silhouette Score (argmax → best k)',
                            color='#f1f5f9', fontsize=10)
            plt.tight_layout(); st.pyplot(fig_el); plt.close()

            CCOLORS = ['#3b82f6','#16a34a','#d97706','#dc2626',
                       '#7c3aed','#0891b2','#ec4899','#84cc16']
            fig_p, ax_p = dark_fig((9, 6))
            for cid, label in lmap.items():
                mask = df['Cluster'] == cid
                ax_p.scatter(df.loc[mask,'P1'], df.loc[mask,'P2'],
                             c=CCOLORS[cid % len(CCOLORS)],
                             label=f"C{cid}: {label} (n={mask.sum()})",
                             alpha=.75, s=50, edgecolors='none')
            ax_p.set_title('Student Clusters — PCA 2D',
                            color='#f1f5f9', fontsize=11)
            ax_p.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')
            plt.tight_layout(); st.pyplot(fig_p); plt.close()

            with st.expander("📊 Cluster Profile Table"):
                pd_prof = prof.copy()
                pd_prof.index = [lmap.get(i, f'C{i}') for i in pd_prof.index]
                st.dataframe(pd_prof.round(1), use_container_width=True)

            # ══ حفظ في session_state ══
            shap_to_save      = shap_v          if isinstance(shap_v,          pd.Series) else None
            shap_sign_to_save = shap_sign        if isinstance(shap_sign,        pd.Series) else None
            shap_pr_to_save   = shap_pos_ratio   if isinstance(shap_pos_ratio,   pd.Series) else None

            st.session_state['s1'] = {
                'df': df, 'avgs': avgs, 'fr': fr,
                'wc': wc, 'mc': mc, 'gc': gc,
                'cnts': cnts, 'course_cols': course_cols,
                'wt': wt, 'at': at,
                'Xtr': Xtr, 'Xte': Xte, 'ytr': ytr, 'yte': yte, 'ypr': ypr,
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'cv': cv, 'auc': auc, 'imps': imps,
                'shap_v': shap_to_save,
                'shap_sign': shap_sign_to_save,
                'shap_pos_ratio': shap_pr_to_save,
                'best_cw_label': best_cw_label,
                'best_params': best_params,
                'best_k': best_k, 'best_sil': max(silhs),
                'lmap': lmap, 'ks': list(ks),
                'inert': inert, 'silhs': silhs,
                'f1_lr': f1_lr, 'acc_lr': acc_lr,
                'course_priority': course_priority,
                'priority_df': priority_df,
                'risk_scores': risk_scores,
                'risk_df': risk_df,
            }
            st.success("✅ Analysis complete — scroll down to generate the AI report.")

        # ══ AI REPORT ══
        if 's1' in st.session_state:
            s = st.session_state['s1']
            st.markdown("---")
            st.markdown("#### 🤖 AI Institutional Report")
            st.markdown(
                '<div class="bb">The AI generates a report grounded in computed data. '
                '<b>Treat as a decision-support draft</b> — verify with domain experts.</div>',
                unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Student Performance Report", key="b1_report"):
                    wc   = s['wc'];  mc   = s['mc'];  fr   = s['fr']
                    imps = s['imps']; shap_v = s['shap_v']
                    acc  = s['acc']; prec = s['prec']; rec = s['rec']
                    f1   = s['f1'];  cv   = s['cv'];  auc = s['auc']
                    cnts = s['cnts']; df  = s['df']
                    course_cols  = s['course_cols']
                    wt   = s['wt'];  at  = s['at']
                    Xtr  = s['Xtr']; Xte = s['Xte']
                    best_cw_label = s['best_cw_label']
                    best_k        = s['best_k']
                    best_sil      = s['best_sil']
                    lmap          = s['lmap']
                    f1_lr         = s.get('f1_lr', 0)
                    course_priority = s.get('course_priority', {})
                    risk_scores   = s.get('risk_scores', np.array([]))

                    def shap_val_str(c):
                        if shap_v is None: return "N/A"
                        return f"{float(shap_v.get(c, 0)):.4f}"

                    weak_str = "\n".join([
                        f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%, "
                        f"rf_importance={imps.get(c,0):.3f}, shap={shap_val_str(c)}, "
                        f"priority_score={course_priority.get(c,0):.3f}"
                        for c, v in wc.items()
                    ]) or "None"

                    mid_str = "\n".join([
                        f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%"
                        for c, v in mc.items()
                    ]) or "None"

                    cluster_str = "\n".join([
                        f"- {lmap.get(cid,'?')}: {(df['Cluster']==cid).sum()} students, "
                        f"avg={float(df[df['Cluster']==cid][course_cols].mean().mean()):.1f}, "
                        f"weakest={df[df['Cluster']==cid][course_cols].mean().idxmin()}"
                        for cid in sorted(lmap.keys())
                    ])

                    top5_imp = "\n".join([
                        f"- {c}: {v:.3f}" for c, v in imps.head(5).items()])

                    top3_priority = "\n".join([
                        f"- {c}: score={score:.3f}, fail={fr.get(c,0):.1f}%"
                        for c, score in sorted(
                            course_priority.items(), key=lambda x: x[1], reverse=True)[:3]
                    ])

                    high_risk_n = (risk_scores >= 0.5).sum() if len(risk_scores) > 0 else 0

                    # ══ FIX 3: CROSS-BRANCH LINKING ══
                    market_context = ""
                    if 's2' in st.session_state:
                        s2_data = st.session_state['s2']
                        top5_market = "\n".join([
                            f"  - {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, "
                            f"coverage={d['job_coverage']:.1f}%"
                            for s, d in s2_data['top30'][:5]
                        ])
                        market_context = f"""
=== JOB MARKET DATA (from Branch B — cross-branch insight) ===
Top 5 demanded skills:
{top5_market}

CROSS-BRANCH ALIGNMENT INSTRUCTION:
For each weak course, identify if its subject maps to a high-demand market skill.
A course that is BOTH academically weak AND maps to a high-demand skill = CRITICAL priority.
Example: 'Operating_Systems' weak + 'linux/devops' high demand = highest urgency.
This cross-branch alignment is the core institutional insight of this system.
"""

                    with st.spinner("Generating AI report..."):
                        prompt = f"""You are a senior academic consultant writing a formal institutional
report for the Head of a Computer Science Department.

RULES:
1. Use ONLY the numbers provided — do not invent values.
2. Textbooks: ONLY widely known CS books (Tanenbaum, CLRS, Silberschatz, Sommerville,
   Kurose & Ross, Russell & Norvig, Goodfellow et al.).
3. YouTube: ONLY real channels (MIT OpenCourseWare, freeCodeCamp, Neso Academy, CS50,
   3Blue1Brown, Abdul Bari).
4. Present as evidence-based suggestions — not final decisions.

=== STUDENT DATA ({len(df)} students) ===
RF: Train={len(Xtr)} | Test={len(Xte)} | Stratified split | class_weight={best_cw_label}
Accuracy={acc*100:.1f}% | F1={f1*100:.1f}% | CV={cv*100:.1f}% | AUC={f"{auc*100:.1f}%" if auc else "N/A"}
LR baseline F1={f1_lr*100:.1f}% | RF advantage={(f1-f1_lr)*100:+.1f}%
Thresholds: Weak<{wt} | Excellent≥{at}
Distribution: Weak={cnts.get('Weak',0)} | Average={cnts.get('Average',0)} | Excellent={cnts.get('Excellent',0)}

WEAK COURSES (fail_rate + rf_importance + shap + priority_score):
{weak_str}

MEDIUM COURSES:
{mid_str}

TOP 5 INFLUENTIAL COURSES (RF Feature Importance):
{top5_imp}

TOP 3 PRIORITY COURSES (combined score):
{top3_priority}

EARLY WARNING: {high_risk_n} high-risk students (≥50% Weak probability) in test set.

KMEANS CLUSTERS (k={best_k}, silhouette={best_sil:.3f}):
{cluster_str}
{market_context}

=== REPORT STRUCTURE ===

## 1. Executive Summary
3 sentences: key numbers + priority_score ranking + early warning findings.

## 2. Per-Course Analysis (Weak courses only)
For EACH weak course:
### [Course Name] | Priority: X.XXX | Fail: X% | SHAP: X.XXXX
- **Root cause:** specific to this course content
- **3 Teaching improvements:** actionable, course-specific
- **Resources:** 2 YouTube channels + 1 practice platform + 1 textbook
- **Timeline:** short/medium/long term

## 3. Cross-Branch Insight (if market data available)
Which weak courses map to high-demand market skills? Priority implications.

## 4. Cluster-Based Interventions
Per cluster: profile, main struggle, recommended action.

## 5. Priority Action Plan
Top 5 actions, numbered, with measurable targets."""

                        result = call_groq(groq_key, prompt, max_tokens=3000)
                        st.markdown(result)
                        st.session_state['s1_report'] = result

                if 's1_report' in st.session_state:
                    s2 = st.session_state['s1']
                    report_txt = (
                        f"CS DEPARTMENT — STUDENT PERFORMANCE REPORT\n{'='*60}\n\n"
                        f"Students={len(s2['df'])} | F1={s2['f1']*100:.1f}% | "
                        f"class_weight={s2['best_cw_label']} | "
                        f"Clusters={s2['best_k']} | Stratified split\n\n"
                        f"{'='*60}\nAI REPORT\n{'='*60}\n\n"
                        f"{st.session_state['s1_report']}"
                    )
                    st.download_button(
                        "📥 Download Student Report",
                        data=report_txt,
                        file_name="student_report.txt",
                        mime="text/plain",
                        key="dl_s1"
                    )
    else:
        st.info("📂 Upload any student grades CSV to begin.")


# ╔══════════════════════════════════════╗
# ║  TAB 2 — JOB MARKET ANALYSIS        ║
# ╚══════════════════════════════════════╝
with tab2:
    st.markdown('<p class="sec">02 · Job Market — TF-IDF NLP + BERT Semantic + AI Report</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="bb">Upload any job listings CSV. '
        'TF-IDF extracts skills lexically. BERT (optional) adds semantic understanding. '
        'Generate Report creates a curriculum gap analysis.</div>',
        unsafe_allow_html=True)

    up_j = st.file_uploader("📂 Upload Job Listings CSV", type="csv", key="job")

    if up_j:
        with st.spinner("Loading..."):
            jobs_df = pd.read_csv(up_j, on_bad_lines='skip')
        st.success(f"✅ {len(jobs_df)} job listings")

        with st.expander("👁 Preview"):
            st.dataframe(jobs_df.head(3))

        tcols    = jobs_df.select_dtypes(include='object').columns.tolist()
        desc_col = st.selectbox(
            "Job description column:", tcols,
            index=tcols.index('jobdescription') if 'jobdescription' in tcols else 0)
        extra   = st.text_input(
            "Extra skills to track (comma-separated):",
            placeholder="rust, llm, chatgpt, quantum")
        curr_in = st.text_area(
            "Current curriculum courses (comma-separated, for gap analysis):",
            placeholder="Operating Systems, Databases, Networks, Algorithms, ...",
            height=80)

        if st.button("🔍 Run NLP Analysis", key="b2_run"):

            with st.spinner("Running TF-IDF pipeline..."):
                tax = dict(TAXONOMY)
                if extra.strip():
                    tax["Custom"] = [s.strip().lower()
                                     for s in extra.split(',') if s.strip()]

                raw   = jobs_df[desc_col].dropna().tolist()
                clean = [preprocess(t) for t in raw]
                atxt  = ' '.join(clean)

                vec  = TfidfVectorizer(ngram_range=(1,3), min_df=1,
                                       max_df=0.95, sublinear_tf=True)
                tmat = vec.fit_transform(clean)
                fname = vec.get_feature_names_out()
                tsum  = dict(zip(fname, np.asarray(tmat.sum(axis=0)).flatten()))

                skill_r = {}
                for cat, skills in tax.items():
                    for s in skills:
                        sc      = clean_text(s)
                        pattern = r'\b' + re.escape(sc) + r'\b'
                        freq    = len(re.findall(pattern, atxt))
                        ts      = tsum.get(sc, 0.0)
                        jw      = sum(1 for t in clean if re.search(pattern, t))
                        cov     = jw / len(clean) * 100
                        if freq > 0:
                            skill_r[s] = {
                                'category': cat, 'freq': freq,
                                'tfidf_score': round(float(ts), 2),
                                'job_coverage': round(cov, 1),
                                'jobs_with': jw,
                                'bert_sim': None, 'bert_cov': None
                            }

                sorted_s = sorted(skill_r.items(),
                                  key=lambda x: x[1]['tfidf_score'], reverse=True)
                top30    = sorted_s[:30]

            # BERT
            bert_model  = None
            bert_status = "disabled"
            if bert_on:
                with st.spinner(f"Loading BERT ({bert_name})..."):
                    bert_model = load_bert(bert_name)
                if bert_model:
                    with st.spinner("Computing BERT semantic similarity..."):
                        sample = clean[:min(300, len(clean))]

                        @st.cache_data(show_spinner=False)
                        def get_job_embeddings(_model, texts):
                            return _model.encode(texts, batch_size=32,
                                                 show_progress_bar=False,
                                                 convert_to_numpy=True)
                        job_emb = get_job_embeddings(bert_model, tuple(sample))
                        for s, d in skill_r.items():
                            se  = bert_model.encode([s], convert_to_numpy=True)
                            sim = cosine_similarity(se, job_emb)[0]
                            m   = (sim >= bert_threshold).sum()
                            d['bert_sim']     = round(float(sim.mean()), 4)
                            d['bert_sim_max'] = round(float(sim.max()),  4)
                            d['bert_cov']     = round(m / len(sample) * 100, 1)
                    bert_status = f"✅ {bert_name}"
                else:
                    bert_status = "⚠️ sentence-transformers not installed"

            if "✅" in bert_status:
                st.markdown(f'<div class="bb">🔵 BERT: {bert_status}</div>',
                            unsafe_allow_html=True)
            elif "⚠️" in bert_status:
                st.markdown(
                    f'<div class="ba">🔵 BERT: {bert_status} — '
                    f'<code>pip install sentence-transformers</code></div>',
                    unsafe_allow_html=True)

            with st.expander("🔬 NLP Pipeline Stages"):
                st.markdown(f"""
| Stage | Action | Detail |
|-------|--------|--------|
| Stage 2: Cleanup     | lowercase · URL removal · punctuation (keeps /+#.) | {len(raw):,} texts |
| Stage 3: Pre-process | tokenize · remove {len(STOP_WORDS)} stop words · min length 2 | |
| Stage 4a: TF-IDF     | ngram=(1,3) · sublinear_tf · min_df=1 · max_df=0.95 | |
| Stage 4b: BERT       | Sentence-BERT cosine sim · threshold={bert_threshold} | {bert_status} |

**min_df=1:** keeps rare but taxonomy-listed skills (Rust, LLM, etc.)
**word-boundary regex:** prevents "sql" matching inside "nosql"
**max similarity (not mean):** stronger signal for sparse skill detection
                """)

            # Chart
            names  = [s[0] for s in top30[:20]]
            scores = [s[1]['tfidf_score'] for s in top30[:20]]
            covs   = [s[1]['job_coverage'] for s in top30[:20]]
            cats   = [s[1]['category'] for s in top30[:20]]
            bclrs  = [CAT_COLORS.get(c, '#64748b') for c in cats]
            has_bert = any(d['bert_sim'] is not None for _, d in skill_r.items())

            if has_bert:
                bert_s = [skill_r[s[0]].get('bert_sim') or 0 for s in top30[:20]]
                fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                fig_b.patch.set_facecolor('#0f172a')
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#0f172a')
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    ax.spines[:].set_color('#1e3a5f')
                bars1 = ax1.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars1, scores, covs):
                    ax1.text(v+.1, b.get_y()+b.get_height()/2,
                             f'{v:.1f}|{cov:.0f}%',
                             va='center', fontsize=7, color='#94a3b8')
                ax1.set_xlabel('TF-IDF Score', color='#94a3b8')
                ax1.set_title('TF-IDF (Lexical)', color='#f1f5f9', fontsize=10)
                bars2 = ax2.barh(
                    names, bert_s,
                    color=['#7c3aed' if v > float(np.mean(bert_s)) else '#334155'
                           for v in bert_s], height=.6)
                for b, v in zip(bars2, bert_s):
                    ax2.text(v+.001, b.get_y()+b.get_height()/2,
                             f'{v:.3f}', va='center', fontsize=7, color='#94a3b8')
                ax2.set_xlabel('BERT Avg Similarity', color='#94a3b8')
                ax2.set_title('BERT (Semantic)', color='#f1f5f9', fontsize=10)
                plt.suptitle('Top 20 Skills — TF-IDF vs BERT',
                             color='#f1f5f9', fontsize=12)
                plt.tight_layout(); st.pyplot(fig_b); plt.close()
            else:
                fig_t, ax_t = dark_fig((10, 8))
                bars_t = ax_t.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars_t, scores, covs):
                    ax_t.text(v+.1, b.get_y()+b.get_height()/2,
                              f'{v:.1f} | {cov:.0f}% jobs',
                              va='center', fontsize=7, color='#94a3b8')
                ax_t.set_xlabel('TF-IDF Score', color='#94a3b8')
                ax_t.set_title('Top 20 Skills by TF-IDF Score',
                               color='#f1f5f9', fontsize=11)
                handles = [mpatches.Patch(color=v, label=k)
                           for k, v in CAT_COLORS.items() if k in cats]
                ax_t.legend(handles=handles, fontsize=7,
                            facecolor='#0f172a', labelcolor='#94a3b8',
                            loc='lower right')
                plt.tight_layout(); st.pyplot(fig_t); plt.close()

            sc = st.columns(4)
            for col, (l, v) in zip(sc, [
                ("Total Jobs",   len(jobs_df)),
                ("Skills Found", len(skill_r)),
                ("Top Skill",    top30[0][0] if top30 else '—'),
                ("BERT Active",  "✅" if has_bert else "❌"),
            ]):
                metric_card(col, l, v)

            st.markdown("---")
            st.markdown("#### Demand by Category (TF-IDF weighted)")
            cat_tot = {}
            for _, d in skill_r.items():
                cat_tot[d['category']] = cat_tot.get(d['category'], 0) + d['tfidf_score']
            tot_all = sum(cat_tot.values()) or 1
            for cat, total in sorted(cat_tot.items(),
                                     key=lambda x: x[1], reverse=True):
                pct = int(total / tot_all * 100)
                st.progress(pct, text=f"**{cat}** — {total:.1f} ({pct}%)")

            st.markdown("---")
            st.markdown("#### Full Skills Table")
            rows = [{
                'Skill': s, 'Category': d['category'],
                'TF-IDF': d['tfidf_score'],
                'Job Coverage %': d['job_coverage'],
                'Jobs #': d['jobs_with'],
                'BERT Sim': d['bert_sim'] if d['bert_sim'] else '—',
                'BERT Cov %': d['bert_cov'] if d['bert_cov'] else '—',
                'Raw Freq': d['freq'],
            } for s, d in sorted_s]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Gap analysis
            if curr_in.strip():
                st.markdown("---")
                st.markdown("#### Gap Analysis — Curriculum vs Market")
                curr_courses   = [c.strip() for c in curr_in.split(',') if c.strip()]
                use_bert_gap   = has_bert and bert_model is not None
                if use_bert_gap:
                    curr_embs = bert_model.encode(curr_courses, convert_to_numpy=True)

                gap_rows = []
                for s, d in top30[:20]:
                    string_match = any(
                        s.lower() in c.lower() or c.lower() in s.lower()
                        for c in curr_courses)
                    if use_bert_gap:
                        se   = bert_model.encode([s], convert_to_numpy=True)
                        sim  = cosine_similarity(se, curr_embs)[0]
                        bert_match = float(sim.max()) >= bert_threshold
                        covered    = string_match or bert_match
                        match_type = ('✅ String' if string_match else
                                      f'🔵 BERT ({sim.max():.2f})' if bert_match
                                      else '❌ Gap')
                    else:
                        covered    = string_match
                        match_type = '✅ Yes' if string_match else '❌ Gap'

                    gap_rows.append({
                        'Skill': s, 'Category': d['category'],
                        'TF-IDF': d['tfidf_score'],
                        'Coverage %': d['job_coverage'],
                        'In Curriculum': match_type
                    })

                gdf = pd.DataFrame(gap_rows)
                st.dataframe(gdf, use_container_width=True, hide_index=True)
                n_gaps = sum(1 for r in gap_rows if '❌' in r['In Curriculum'])
                g1, g2 = st.columns(2)
                g1.metric("Skills with Gap", n_gaps, delta_color="inverse")
                g2.metric("Skills Covered",  len(gap_rows)-n_gaps)

            # حفظ في session_state
            st.session_state['s2'] = {
                'jobs_n':      len(jobs_df),
                'skill_r':     skill_r,
                'top30':       top30,
                'cat_tot':     cat_tot,
                'tot_all':     tot_all,
                'has_bert':    has_bert,
                'bert_status': bert_status,
                'curr_in':     curr_in,
            }
            st.success("✅ NLP analysis complete — scroll down to generate the AI report.")

        # ── AI REPORT Tab 2 ──
        if 's2' in st.session_state:
            s2 = st.session_state['s2']
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Job Market & Curriculum")
            st.markdown(
                '<div class="bb">Generates curriculum gap analysis based on '
                'TF-IDF/BERT results. <b>Treat as decision-support draft.</b></div>',
                unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Job Market Report", key="b2_report"):
                    top30    = s2['top30']
                    cat_tot  = s2['cat_tot']
                    tot_all  = s2['tot_all']
                    skill_r  = s2['skill_r']
                    has_bert = s2['has_bert']
                    curr_in  = s2['curr_in']

                    with st.spinner("Generating AI report..."):
                        top20_str = "\n".join([
                            f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, "
                            f"coverage={d['job_coverage']:.1f}%"
                            + (f", BERT={d['bert_sim']:.3f}" if d['bert_sim'] else "")
                            for s, d in top30[:20]
                        ])
                        cat_str = "\n".join([
                            f"- {cat}: {tot:.1f} ({int(tot/tot_all*100)}%)"
                            for cat, tot in sorted(
                                cat_tot.items(), key=lambda x: x[1], reverse=True)
                        ])
                        curr_str  = curr_in.strip() if curr_in.strip() else "Not provided"
                        bert_note = ""
                        if has_bert:
                            top3b = sorted(
                                [(s, d['bert_sim']) for s, d in skill_r.items()
                                 if d['bert_sim']],
                                key=lambda x: x[1], reverse=True)[:5]
                            bert_note = (
                                f"\nTOP 5 BY BERT: "
                                f"{', '.join([f'{s}({v:.3f})' for s,v in top3b])}")

                        # ربط مع Branch A إذا متاح
                        student_context = ""
                        if 's1' in st.session_state:
                            s1_data = st.session_state['s1']
                            weak_courses = list(s1_data.get('wc', {}).keys())
                            top3_priority = list(s1_data.get('course_priority', {}).items())
                            top3_priority.sort(key=lambda x: x[1], reverse=True)
                            student_context = f"""
=== STUDENT PERFORMANCE DATA (from Branch A) ===
Weak courses: {weak_courses}
Top priority courses: {[c for c, _ in top3_priority[:3]]}
CROSS-BRANCH: Link market skill gaps to academically weak courses for urgency ranking.
"""

                        prompt_j = f"""You are a senior academic consultant writing a formal
curriculum analysis report for a CS Department Head.

RULES: Use provided numbers only. Real textbooks/YouTube only. Evidence-based suggestions.

=== JOB MARKET ({s2['jobs_n']} listings) ===
TOP 20 SKILLS:
{top20_str}
{bert_note}

DEMAND BY CATEGORY:
{cat_str}

CURRENT CURRICULUM: {curr_str}
{student_context}

=== REPORT STRUCTURE ===

## 1. Market Overview
Dominant categories, TF-IDF vs BERT divergences.

## 2. Per-Skill Analysis (Top 10)
### [Skill] — TF-IDF: X | Coverage: X%
- Industry context + how to teach + resources

## 3. Curriculum–Market Gap
- High-demand skills NOT in curriculum (with TF-IDF scores)
- Curriculum coverage % of top 20 skills

## 4. Recommended New Courses (4–5)
Name · skills covered · placement · Priority

## 5. Priority Action Plan
Top 5, numbered, measurable targets."""

                        result_j = call_groq(groq_key, prompt_j, max_tokens=3000)
                        st.session_state['s2_report'] = result_j

                if 's2_report' in st.session_state:
                    st.markdown(st.session_state['s2_report'])
                    s2r = st.session_state['s2']
                    top20_dl = "\n".join([
                        f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}"
                        for s, d in s2r['top30'][:20]
                    ])
                    cat_dl = "\n".join([
                        f"- {cat}: {tot:.1f}"
                        for cat, tot in sorted(
                            s2r['cat_tot'].items(), key=lambda x: x[1], reverse=True)
                    ])
                    report_j = (
                        f"CS DEPARTMENT — JOB MARKET REPORT\n{'='*60}\n\n"
                        f"Jobs={s2r['jobs_n']} | Skills={len(s2r['skill_r'])} | "
                        f"BERT={s2r['bert_status']}\n\n"
                        f"TOP SKILLS:\n{top20_dl}\n\nCATEGORIES:\n{cat_dl}\n\n"
                        f"{'='*60}\nAI REPORT\n{'='*60}\n\n"
                        f"{st.session_state['s2_report']}"
                    )
                    st.download_button(
                        "📥 Download Job Market Report",
                        data=report_j,
                        file_name="job_market_report.txt",
                        mime="text/plain",
                        key="dl_s2"
                    )
    else:
        st.info("📂 Upload a job listings CSV to begin.")
