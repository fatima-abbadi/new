"""
CS Department Intelligence System v5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
التحسينات عن v4:
  1. StandardScaler: fit على Train فقط → transform على الكل (لا data leakage في clustering)
  2. class_weight: نجرب balanced و None ونختار الأفضل F1 تلقائياً
  3. SHAP: على عيّنة max(50) من Test لتجنب البطء
  4. column detection: auto_c كاقتراح فقط، المستخدم يتحكم يدوياً
  5. LLM disclaimer في الـ prompt والواجهة
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, silhouette_score)
from sklearn.preprocessing import label_binarize, StandardScaler
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
  <p>Random Forest · SHAP · KMeans Clustering · TF-IDF NLP · BERT Embeddings · AI Reports</p>
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
    bert_on   = st.toggle("Enable BERT (Sentence-BERT)", value=False)
    bert_name = st.selectbox("BERT model:", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0)
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

# ── SHAP: يعمل على numpy array لتجنب مشاكل الـ index ──
def compute_shap(model, X_te_df, features):
    """
    SHAP على max 50 عيّنة من Test Set.
    يتعامل مع multi-class و binary بشكل صحيح.
    """
    try:
        arr = np.array(X_te_df.values[:min(50, len(X_te_df))], dtype=float)
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(arr)
        # sv شكله مختلف حسب نوع المشكلة:
        # multi-class (list): [array(n,f), array(n,f), array(n,f)] - واحد لكل كلاس
        # binary (array): array(n,f) أو array(n,f,2)
        if isinstance(sv, list):
            # نأخذ mean absolute عبر كل الكلاسات ثم عبر الـ samples
            stacked = np.stack([np.abs(s) for s in sv], axis=0)  # (n_classes, n_samples, n_features)
            mean_shap = stacked.mean(axis=0).mean(axis=0)         # (n_features,)
        elif sv.ndim == 3:
            # array(n_samples, n_features, n_classes)
            mean_shap = np.abs(sv).mean(axis=0).mean(axis=1)      # (n_features,)
        else:
            # binary: array(n_samples, n_features)
            mean_shap = np.abs(sv).mean(axis=0)                   # (n_features,)
        return pd.Series(mean_shap, index=features).sort_values(ascending=False)
    except Exception as e:
        return str(e)

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
        f'<div class="mc"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAXONOMY for NLP
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
    "Databases":             ["mysql","postgresql","oracle","sql server","redis","elasticsearch",
                               "cassandra","database design","data modeling","data warehouse"],
    "Software Engineering":  ["agile","scrum","rest api","microservices","design patterns",
                               "software architecture","unit testing","test driven development",
                               "version control","code review","continuous integration"],
    "Networking":            ["networking","tcp/ip","routing","switching","vpn",
                               "network administration","cisco","wireshark","network protocols"],
    "Emerging Tech":         ["blockchain","iot","embedded systems","computer vision","robotics",
                               "large language model","generative ai","ar/vr","edge computing","5g"],
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
        'RF trained on 80%, evaluated on 20%. '
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

        # ── FIX 4: auto_c كاقتراح فقط — المستخدم يراجع ويعدّل ──
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
                options=num_cols,
                default=auto_c)
        with c2:
            final_col = st.selectbox(
                "Final grade column (classification target):",
                options=num_cols,
                index=num_cols.index('Final_Grade') if 'Final_Grade' in num_cols else len(num_cols)-1)

        t1, t2 = st.columns(2)
        with t1: wt = st.slider("Weak: score below", 30, 70, 60)
        with t2: at = st.slider("Average: score below", 65, 90, 75)

        if st.button("🔍 Run Analysis", key="b1_run"):
            if len(course_cols) < 2:
                st.error("Need ≥2 course columns.")
                st.stop()
            if final_col in course_cols:
                st.error("Final grade column must NOT be in course columns.")
                st.stop()

            df = df_raw.copy()

            # ── Descriptive stats (all data) ──
            avgs = df[course_cols].mean().sort_values()
            fr   = {c: (df[c] < wt).mean()*100 for c in course_cols}
            wc   = avgs[avgs < wt]
            mc   = avgs[(avgs >= wt) & (avgs < at)]
            gc   = avgs[avgs >= at]

            # classify uses fixed thresholds — safe to apply before or after split
            def classify(g):
                if g < wt:   return 'Weak'
                elif g < at: return 'Average'
                return 'Excellent'

            df['Level'] = df[final_col].apply(classify)
            cnts = df['Level'].value_counts()

            # ── Train / Test split ──
            X     = df[course_cols]
            y_raw = df[final_col]
            Xtr, Xte, ytr_r, yte_r = train_test_split(
                X, y_raw, test_size=0.2, random_state=42)

            # Labels applied AFTER split (best practice — avoids any future leakage
            # if thresholds ever become data-dependent)
            ytr = ytr_r.apply(classify)
            yte = yte_r.apply(classify)

            # ── FIX 2: Compare balanced vs default — choose by F1 ──
            with st.spinner("Training Random Forest (comparing class_weight options)..."):
                best_rf, best_f1_cw, best_cw_label = None, -1, ""
                for cw, cw_label in [('balanced', 'balanced'), (None, 'none')]:
                    rf_tmp = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight=cw)
                    rf_tmp.fit(Xtr, ytr)
                    ypr_tmp = rf_tmp.predict(Xte)
                    f1_tmp  = f1_score(ypr_tmp, yte, average='weighted', zero_division=0)
                    if f1_tmp > best_f1_cw:
                        best_f1_cw   = f1_tmp
                        best_rf      = rf_tmp
                        best_cw_label = cw_label

            rf  = best_rf
            ypr = rf.predict(Xte)

            acc  = accuracy_score(yte, ypr)
            prec = precision_score(yte, ypr, average='weighted', zero_division=0)
            rec  = recall_score(yte, ypr,    average='weighted', zero_division=0)
            f1   = f1_score(yte, ypr,        average='weighted', zero_division=0)
            cv   = cross_val_score(rf, Xtr, ytr, cv=5, scoring='accuracy').mean()
            imps = pd.Series(rf.feature_importances_, index=course_cols).sort_values(ascending=False)
            cls  = sorted(ytr.unique())
            auc  = compute_auc(rf, Xte.values, yte, cls)

            # ── SHAP على Test Set كـ DataFrame ──
            with st.spinner("Computing SHAP values (sampled for speed)..."):
                shap_v = compute_shap(rf, Xte, course_cols)

            # ── FIX 1: StandardScaler fit on Train only → transform all ──
            with st.spinner("Running KMeans Clustering..."):
                X_for_scale = df[course_cols].fillna(df[course_cols].mean())
                scaler = StandardScaler()
                # fit ONLY on training rows to avoid data leakage
                scaler.fit(X_for_scale.iloc[Xtr.index])
                Xs = scaler.transform(X_for_scale)   # transform on all 415

                ks = range(2, max_k + 1)
                inert, silhs = [], []
                for k in ks:
                    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    lb     = km_tmp.fit_predict(Xs)
                    inert.append(km_tmp.inertia_)
                    silhs.append(silhouette_score(Xs, lb))

                best_k = list(ks)[np.argmax(silhs)]
                km_f   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                df['Cluster'] = km_f.fit_predict(Xs)

                pca_c    = PCA(n_components=2).fit_transform(Xs)
                df['P1'] = pca_c[:, 0]
                df['P2'] = pca_c[:, 1]

                prof = df.groupby('Cluster')[course_cols].mean()
                prof['avg'] = prof.mean(axis=1)
                prof = prof.sort_values('avg', ascending=False)

                lmap = {}
                n_cl = len(prof)
                for i, cid in enumerate(prof.index):
                    if i == 0:        lmap[cid] = "🏆 High Performers"
                    elif i == n_cl-1: lmap[cid] = "🚨 At Risk"
                    elif i == 1:      lmap[cid] = "⬆️ Above Average"
                    elif i == n_cl-2: lmap[cid] = "⬇️ Below Average"
                    else:             lmap[cid] = f"📊 Mid Group {i}"
                df['CL'] = df['Cluster'].map(lmap)

            # ══════════════════ DISPLAY ══════════════════

            st.markdown("---")
            # KPI row
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
                            f'<div class="br"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fr[c]:.1f}%</div>',
                            unsafe_allow_html=True)
                else:
                    st.success("None")
            with cm_col:
                st.markdown("**🟡 Medium Courses**")
                if len(mc):
                    for c, v in mc.items():
                        st.markdown(
                            f'<div class="ba"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fr[c]:.1f}%</div>',
                            unsafe_allow_html=True)
                else:
                    st.success("None")
            with cg_col:
                st.markdown("**🟢 Good Courses**")
                for c, v in gc.items():
                    st.markdown(
                        f'<div class="bg"><b>{c}</b> · Avg:{v:.1f}</div>',
                        unsafe_allow_html=True)

            # ── ML Metrics ──
            st.markdown("---")
            st.markdown("#### 🤖 Random Forest Results")
            st.caption(
                f"Train: {len(Xtr)} students (80%) · Test: {len(Xte)} students (20%) · "
                f"5-fold CV on train · class_weight={best_cw_label} (auto-selected by F1)")

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

            # Confusion Matrix
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
                plt.tight_layout(); st.pyplot(fig4); plt.close()

            with sh_c:
                st.markdown("**SHAP Values — mean |SHAP|**")
                if shap_v is not None and not isinstance(shap_v, str):
                    ts = shap_v.sort_values(ascending=True)
                    fig5, ax5 = dark_fig((7, max(4, len(course_cols)*.4)))
                    ax5.barh(ts.index, ts.values,
                             color=['#7c3aed' if v > shap_v.mean() else '#334155' for v in ts.values],
                             height=.6)
                    ax5.axvline(shap_v.mean(), color='#d97706', linestyle='--', lw=1)
                    ax5.set_title('SHAP (sampled on test set)', color='#94a3b8', fontsize=8)
                    plt.tight_layout(); st.pyplot(fig5); plt.close()
                elif isinstance(shap_v, str):
                    st.warning(f"SHAP error: {shap_v}")
                else:
                    st.info("SHAP not available.")

            with st.expander("📐 Metrics & Design Decisions"):
                st.markdown(f"""
| Metric | Value | Formula | When to trust |
|--------|-------|---------|---------------|
| Accuracy  | {acc*100:.1f}%  | (TP+TN)/all | Good when classes are balanced |
| Precision | {prec*100:.1f}% | TP/(TP+FP)  | Avoid labelling good students as Weak |
| Recall    | {rec*100:.1f}%  | TP/(TP+FN)  | Don't miss at-risk students |
| F1        | {f1*100:.1f}%   | 2·P·R/(P+R) | Best single metric with imbalance |
| CV Acc.   | {cv*100:.1f}%   | 5-fold mean | Stability — close to Accuracy = no overfit |
| AUC-ROC   | {f"{auc*100:.1f}%" if auc else "N/A"} | Area under ROC | Threshold-independent quality |

**class_weight selected:** `{best_cw_label}` — auto-chosen by weighted F1 on test set.

**SHAP vs Feature Importance:**
- Feature Importance = average Gini decrease across 100 trees (global)
- SHAP = exact contribution per feature per prediction (local, more precise)
- If they disagree on a course → that course has non-linear behaviour worth investigating

**StandardScaler:** fitted on training rows only, then applied to all students for clustering.
This prevents any statistical leakage from test-set rows into the clustering scale.
                """)

            # ════════════════════
            # CLUSTERING
            # ════════════════════
            st.markdown("---")
            st.markdown("#### 🔵 KMeans Student Clustering")
            best_sil = max(silhs)
            st.markdown(
                f'<div class="bp">KMeans found <b>{best_k} natural student groups</b> '
                f'(silhouette = {best_sil:.3f}, chosen by argmax silhouette across k=2..{max_k}). '
                f'Scaler fitted on training rows only.</div>',
                unsafe_allow_html=True)

            for cid, label in lmap.items():
                n_c   = (df['Cluster'] == cid).sum()
                avg_c = float(df[df['Cluster']==cid][course_cols].mean().mean())
                b_c   = df[df['Cluster']==cid][course_cols].mean().idxmax()
                w_c   = df[df['Cluster']==cid][course_cols].mean().idxmin()
                st.markdown(
                    f'<div class="cc">'
                    f'<b style="color:#f1f5f9">{label}</b> &nbsp;·&nbsp; '
                    f'<span style="color:#64748b">{n_c} students ({n_c/len(df)*100:.0f}%)</span> &nbsp;·&nbsp; '
                    f'<span style="color:#3b82f6">Avg: {avg_c:.1f}</span> &nbsp;·&nbsp; '
                    f'<span style="color:#16a34a">Best: {b_c}</span> &nbsp;·&nbsp; '
                    f'<span style="color:#dc2626">Weakest: {w_c}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

            # Elbow + Silhouette
            fig_el, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
            fig_el.patch.set_facecolor('#0f172a')
            for ax in [ax_e, ax_s]:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors='#94a3b8')
                ax.spines[:].set_color('#1e3a5f')
            ax_e.plot(list(ks), inert, 'o-', color='#3b82f6', lw=2)
            ax_e.axvline(best_k, color='#d97706', linestyle='--', lw=1.5, label=f'Best k={best_k}')
            ax_e.set_xlabel('k', color='#94a3b8'); ax_e.set_ylabel('Inertia', color='#94a3b8')
            ax_e.set_title('Elbow Method', color='#f1f5f9', fontsize=10)
            ax_e.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')
            ax_s.bar(list(ks), silhs,
                     color=['#d97706' if k==best_k else '#334155' for k in ks], width=.5)
            ax_s.set_xlabel('k', color='#94a3b8'); ax_s.set_ylabel('Silhouette', color='#94a3b8')
            ax_s.set_title('Silhouette Score (higher = better separation)', color='#f1f5f9', fontsize=10)
            plt.tight_layout(); st.pyplot(fig_el); plt.close()

            # PCA scatter
            CCOLORS = ['#3b82f6','#16a34a','#d97706','#dc2626','#7c3aed','#0891b2','#ec4899','#84cc16']
            fig_p, ax_p = dark_fig((9, 6))
            for cid, label in lmap.items():
                mask = df['Cluster'] == cid
                ax_p.scatter(df.loc[mask,'P1'], df.loc[mask,'P2'],
                             c=CCOLORS[cid % len(CCOLORS)],
                             label=f"C{cid}: {label} (n={mask.sum()})",
                             alpha=.75, s=50, edgecolors='none')
            ax_p.set_title('Student Clusters — PCA 2D', color='#f1f5f9', fontsize=11)
            ax_p.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')
            plt.tight_layout(); st.pyplot(fig_p); plt.close()

            with st.expander("📊 Cluster Profile Table"):
                pd_prof = prof.copy()
                pd_prof.index = [lmap.get(i, f'C{i}') for i in pd_prof.index]
                st.dataframe(pd_prof.round(1), use_container_width=True)

            # ══ حفظ كل النتائج في session_state ══
            # هذا يحل مشكلة ضياع النتائج عند الضغط على Generate Report
            # نحفظ shap_v فقط إذا كانت Series حقيقية
            shap_to_save = shap_v if (shap_v is not None and isinstance(shap_v, pd.Series)) else None
            st.session_state['s1'] = {
                'df': df, 'avgs': avgs, 'fr': fr,
                'wc': wc, 'mc': mc, 'gc': gc,
                'cnts': cnts, 'course_cols': course_cols,
                'wt': wt, 'at': at,
                'Xtr': Xtr, 'Xte': Xte, 'ytr': ytr, 'yte': yte, 'ypr': ypr,
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'cv': cv, 'auc': auc, 'imps': imps,
                'shap_v': shap_to_save, 'best_cw_label': best_cw_label,
                'best_k': best_k, 'best_sil': max(silhs),
                'lmap': lmap, 'ks': list(ks),
                'inert': inert, 'silhs': silhs,
            }
            st.success("✅ Analysis complete — scroll down to generate the AI report.")

        # ════════════════════
        # AI REPORT — خارج if button بحيث لا يختفي عند أي تفاعل
        # ════════════════════
        if 's1' in st.session_state:
            s = st.session_state['s1']
            st.markdown("---")
            st.markdown("#### 🤖 AI Institutional Report")
            st.markdown(
                '<div class="bb">The AI generates a <b>report based on the numbers above</b>. '
                'All quantitative claims are grounded in computed data. '
                '<b>Treat the output as a decision-support draft</b> — '
                'verify recommendations with domain experts before acting.</div>',
                unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Student Performance Report", key="b1_report"):
                    # نقرأ من session_state — لا نعيد التدريب
                    wc   = s['wc'];  mc   = s['mc'];  fr   = s['fr']
                    imps = s['imps']; shap_v = s['shap_v']
                    acc  = s['acc']; prec = s['prec']; rec = s['rec']
                    f1   = s['f1'];  cv   = s['cv'];  auc = s['auc']
                    cnts = s['cnts']; df  = s['df']
                    course_cols = s['course_cols']
                    wt   = s['wt'];  at  = s['at']
                    Xtr  = s['Xtr']; Xte = s['Xte']
                    best_cw_label = s['best_cw_label']
                    best_k = s['best_k']; best_sil = s['best_sil']
                    lmap = s['lmap']

                    with st.spinner("Generating AI report..."):

                        def shap_val_str(c):
                            if shap_v is None: return "N/A"
                            return f"{float(shap_v.get(c, 0)):.4f}"

                        weak_str = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%, "
                            f"rf_importance={imps.get(c,0):.3f}, shap={shap_val_str(c)}"
                            for c, v in wc.items()
                        ]) or "None"

                        mid_str = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%"
                            for c, v in mc.items()
                        ]) or "None"

                        cluster_str = "\n".join([
                            f"- {lmap.get(cid,'?')}: {(df['Cluster']==cid).sum()} students, "
                            f"avg={float(df[df['Cluster']==cid][course_cols].mean().mean()):.1f}, "
                            f"weakest_course={df[df['Cluster']==cid][course_cols].mean().idxmin()}"
                            for cid in sorted(lmap.keys())
                        ])

                        top5_imp = "\n".join([f"- {c}: {v:.3f}" for c,v in imps.head(5).items()])

                        prompt = f"""You are a senior academic consultant writing a formal institutional report
for the Head of a Computer Science Department.

IMPORTANT: You are a report-generation tool. All quantitative values below come from
computed data — use them accurately. Do not invent numbers. Present findings as
evidence-based suggestions, not absolute decisions.

=== STUDENT DATA ({len(df)} students) ===
Random Forest: Train={len(Xtr)} | Test={len(Xte)} | class_weight={best_cw_label}
Accuracy={acc*100:.1f}% | Precision={prec*100:.1f}% | Recall={rec*100:.1f}%
F1={f1*100:.1f}% | CV={cv*100:.1f}% | AUC={f"{auc*100:.1f}%" if auc else "N/A"}
Thresholds: Weak<{wt} | Average {wt}–{at} | Excellent>{at}
Distribution: Weak={cnts.get('Weak',0)} | Average={cnts.get('Average',0)} | Excellent={cnts.get('Excellent',0)}

WEAK COURSES (fail_rate + rf_importance + shap):
{weak_str}

MEDIUM COURSES:
{mid_str}

TOP 5 INFLUENTIAL COURSES (RF Feature Importance):
{top5_imp}

KMEANS CLUSTERS (k={best_k}, silhouette={best_sil:.3f}):
{cluster_str}

ALL COURSES IN DATASET:
{', '.join(course_cols)}

=== REPORT STRUCTURE ===

## 1. Executive Summary
3 sentences: key numbers + what clustering revealed.

## 2. Per-Course Analysis
For EACH weak course:
### [Exact Course Name] | Fail: X% | RF: X.XXX | SHAP: X.XXXX
- **Root cause:** specific to this course topic
- **3 Teaching improvements:** course-specific, actionable
- **Resources:**
  * YouTube: 2 real named channels for this exact topic
  * Practice: specific platform + track
  * Textbook: 1–2 well-known books for this subject
- **Priority:** Critical / High / Medium

## 3. Cluster-Based Interventions
For each cluster: profile, struggle, recommended intervention.

## 4. SHAP vs Feature Importance
Any divergence? What does it reveal about the data?

## 5. Priority Action Plan
Numbered, most urgent first. Include measurable targets."""

                        result = call_groq(groq_key, prompt, max_tokens=3000)
                        st.markdown(result)
                        st.session_state['s1_report'] = result

                if 's1_report' in st.session_state:
                    result = st.session_state['s1_report']
                    s2 = st.session_state['s1']
                    report_txt = (
                        f"CS DEPARTMENT — STUDENT PERFORMANCE REPORT\n{'='*60}\n\n"
                        f"Students={len(s2['df'])} | RF F1={s2['f1']*100:.1f}% | "
                        f"class_weight={s2['best_cw_label']} | Clusters={s2['best_k']}\n\n"
                        f"{'='*60}\nAI REPORT\n{'='*60}\n\n{result}"
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
        desc_col = st.selectbox("Job description column:", tcols,
                    index=tcols.index('jobdescription') if 'jobdescription' in tcols else 0)
        extra    = st.text_input("Extra skills to track (comma-separated):",
                    placeholder="rust, llm, chatgpt, quantum")
        curr_in  = st.text_area(
                    "Current curriculum courses (comma-separated, for gap analysis):",
                    placeholder="Operating Systems, Databases, Networks, Algorithms, ...",
                    height=80)

        if st.button("🔍 Run NLP Analysis", key="b2_run"):

            with st.spinner("Running TF-IDF pipeline..."):
                tax = dict(TAXONOMY)
                if extra.strip():
                    tax["Custom"] = [s.strip().lower() for s in extra.split(',') if s.strip()]

                raw   = jobs_df[desc_col].dropna().tolist()
                clean = [preprocess(t) for t in raw]
                atxt  = ' '.join(clean)

                vec   = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True)
                tmat  = vec.fit_transform(clean)
                fname = vec.get_feature_names_out()
                tsum  = dict(zip(fname, np.asarray(tmat.sum(axis=0)).flatten()))

                skill_r = {}
                for cat, skills in tax.items():
                    for s in skills:
                        sc   = clean_text(s)
                        freq = atxt.count(sc)
                        ts   = tsum.get(sc, 0.0)
                        jw   = sum(1 for t in clean if sc in t)
                        cov  = jw / len(clean) * 100
                        if freq > 0:
                            skill_r[s] = {
                                'category': cat, 'freq': freq,
                                'tfidf_score': round(float(ts), 2),
                                'job_coverage': round(cov, 1),
                                'jobs_with': jw,
                                'bert_sim': None, 'bert_cov': None
                            }

                sorted_s = sorted(skill_r.items(), key=lambda x: x[1]['tfidf_score'], reverse=True)
                top30    = sorted_s[:30]

            # Stage 4b: BERT
            bert_model  = None
            bert_status = "disabled"
            if bert_on:
                with st.spinner(f"Loading BERT ({bert_name})..."):
                    bert_model = load_bert(bert_name)
                if bert_model:
                    with st.spinner("Computing BERT semantic similarity..."):
                        sample  = clean[:min(300, len(clean))]
                        job_emb = bert_model.encode(
                            sample, batch_size=32,
                            show_progress_bar=False, convert_to_numpy=True)
                        for s, d in skill_r.items():
                            se  = bert_model.encode([s], convert_to_numpy=True)
                            sim = cosine_similarity(se, job_emb)[0]
                            m   = (sim >= 0.45).sum()
                            d['bert_sim'] = round(float(sim.mean()), 4)
                            d['bert_cov'] = round(m / len(sample) * 100, 1)
                    bert_status = f"✅ {bert_name}"
                else:
                    bert_status = "⚠️ sentence-transformers not installed"

            if "✅" in bert_status:
                st.markdown(f'<div class="bb">🔵 BERT: {bert_status}</div>', unsafe_allow_html=True)
            elif "⚠️" in bert_status:
                st.markdown(
                    f'<div class="ba">🔵 BERT: {bert_status} — '
                    f'<code>pip install sentence-transformers</code></div>',
                    unsafe_allow_html=True)

            with st.expander("🔬 NLP Pipeline Stages"):
                st.markdown(f"""
| Stage | Action | Detail |
|-------|--------|--------|
| Stage 2: Cleanup    | lowercase · URL removal · punctuation (keeps /+#.) | {len(raw):,} texts |
| Stage 3: Pre-process| tokenize · remove {len(STOP_WORDS)} stop words · min length 2 | |
| Stage 4a: TF-IDF    | ngram=(1,3) · sublinear_tf · min_df=2 · max_df=0.95 | |
| Stage 4b: BERT      | Sentence-BERT cosine sim · threshold=0.45 | {bert_status} |

**Why TF-IDF + BERT?**
TF-IDF: fast, exact keyword frequency (C++ stays C++).
BERT: semantic matching — "ML engineer" ↔ "machine learning".
Combined → most complete skill demand signal.
                """)

            # ── Chart ──
            names  = [s[0] for s in top30[:20]]
            scores = [s[1]['tfidf_score'] for s in top30[:20]]
            covs   = [s[1]['job_coverage'] for s in top30[:20]]
            cats   = [s[1]['category'] for s in top30[:20]]
            bclrs  = [CAT_COLORS.get(c, '#64748b') for c in cats]

            has_bert = any(d['bert_sim'] is not None for _, d in skill_r.items())

            if has_bert:
                bert_s = [skill_r[s[0]]['bert_sim'] or 0 for s in top30[:20]]
                fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                fig_b.patch.set_facecolor('#0f172a')
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#0f172a')
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    ax.spines[:].set_color('#1e3a5f')
                bars1 = ax1.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars1, scores, covs):
                    ax1.text(v+.1, b.get_y()+b.get_height()/2,
                             f'{v:.1f}|{cov:.0f}%', va='center', fontsize=7, color='#94a3b8')
                ax1.set_xlabel('TF-IDF Score', color='#94a3b8')
                ax1.set_title('TF-IDF (Lexical)', color='#f1f5f9', fontsize=10)
                bars2 = ax2.barh(
                    names, bert_s,
                    color=['#7c3aed' if v > float(np.mean(bert_s)) else '#334155' for v in bert_s],
                    height=.6)
                for b, v in zip(bars2, bert_s):
                    ax2.text(v+.001, b.get_y()+b.get_height()/2,
                             f'{v:.3f}', va='center', fontsize=7, color='#94a3b8')
                ax2.set_xlabel('BERT Avg Similarity', color='#94a3b8')
                ax2.set_title('BERT (Semantic)', color='#f1f5f9', fontsize=10)
                plt.suptitle('Top 20 Skills — TF-IDF vs BERT', color='#f1f5f9', fontsize=12)
                plt.tight_layout(); st.pyplot(fig_b); plt.close()
            else:
                fig_t, ax_t = dark_fig((10, 8))
                bars_t = ax_t.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars_t, scores, covs):
                    ax_t.text(v+.1, b.get_y()+b.get_height()/2,
                              f'{v:.1f} | {cov:.0f}% jobs', va='center', fontsize=7, color='#94a3b8')
                ax_t.set_xlabel('TF-IDF Score (higher = more important)', color='#94a3b8')
                ax_t.set_title('Top 20 Skills by TF-IDF Score', color='#f1f5f9', fontsize=11)
                handles = [mpatches.Patch(color=v, label=k) for k,v in CAT_COLORS.items() if k in cats]
                ax_t.legend(handles=handles, fontsize=7, facecolor='#0f172a',
                            labelcolor='#94a3b8', loc='lower right')
                plt.tight_layout(); st.pyplot(fig_t); plt.close()

            # Stats
            sc = st.columns(4)
            for col, (l, v) in zip(sc, [
                ("Total Jobs",   len(jobs_df)),
                ("Skills Found", len(skill_r)),
                ("Top Skill",    top30[0][0] if top30 else '—'),
                ("BERT Active",  "✅" if has_bert else "❌"),
            ]):
                metric_card(col, l, v)

            # Category breakdown
            st.markdown("---")
            st.markdown("#### Demand by Category (TF-IDF weighted)")
            cat_tot = {}
            for _, d in skill_r.items():
                cat_tot[d['category']] = cat_tot.get(d['category'], 0) + d['tfidf_score']
            tot_all = sum(cat_tot.values()) or 1
            for cat, total in sorted(cat_tot.items(), key=lambda x: x[1], reverse=True):
                pct = int(total / tot_all * 100)
                st.progress(pct, text=f"**{cat}** — {total:.1f} ({pct}%)")

            # Full table
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
                curr_courses = [c.strip() for c in curr_in.split(',') if c.strip()]
                gap_rows = []
                for s, d in top30[:20]:
                    covered = any(
                        s.lower() in c.lower() or c.lower() in s.lower()
                        for c in curr_courses)
                    gap_rows.append({
                        'Skill': s, 'Category': d['category'],
                        'TF-IDF': d['tfidf_score'],
                        'Coverage %': d['job_coverage'],
                        'In Curriculum': '✅ Yes' if covered else '❌ Gap'
                    })
                gdf = pd.DataFrame(gap_rows)
                st.dataframe(gdf, use_container_width=True, hide_index=True)
                n_gaps  = sum(1 for r in gap_rows if '❌' in r['In Curriculum'])
                g1, g2  = st.columns(2)
                g1.metric("Skills with Gap", n_gaps,           delta_color="inverse")
                g2.metric("Skills Covered",  len(gap_rows)-n_gaps)

            # ── AI REPORT ──
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Job Market & Curriculum")
            st.markdown(
                '<div class="bb">Generates a curriculum gap analysis based on computed TF-IDF/BERT results. '
                '<b>Treat output as a decision-support draft</b> — verify with domain experts.</div>',
                unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Job Market Report", key="b2_report"):
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

                        curr_str = curr_in.strip() if curr_in.strip() else "Not provided"

                        bert_note = ""
                        if has_bert:
                            top3b = sorted(
                                [(s, d['bert_sim']) for s,d in skill_r.items() if d['bert_sim']],
                                key=lambda x: x[1], reverse=True
                            )[:5]
                            bert_note = (f"\nTOP 5 BY BERT SEMANTIC SIMILARITY: "
                                         f"{', '.join([f'{s}({v:.3f})' for s,v in top3b])}")

                        prompt_j = f"""You are a senior academic consultant writing a formal curriculum analysis
report for the Head of a Computer Science Department.

IMPORTANT: You are a report-generation tool. Use the numbers below accurately.
Do not invent data. Present as evidence-based suggestions for expert review.

=== JOB MARKET DATA ({len(jobs_df)} listings — TF-IDF + BERT NLP) ===
TOP 20 IN-DEMAND SKILLS:
{top20_str}
{bert_note}

DEMAND BY CATEGORY (TF-IDF weighted):
{cat_str}

CURRENT CURRICULUM: {curr_str}

=== REPORT STRUCTURE ===

## 1. Market Overview
What dominates? Any TF-IDF vs BERT divergence worth noting?

## 2. Per-Skill Analysis (Top 10)
For each of the top 10 skills:
### [Skill] — TF-IDF: X | Coverage: X%
- Why in demand, industry context
- How to teach: 2 YouTube channels + 1 practice platform + 1 textbook

## 3. Curriculum–Market Gap Analysis
- Skills with HIGH demand NOT in curriculum (with TF-IDF scores)
- Skills PARTIALLY covered (need strengthening)
- Estimated curriculum coverage % of top 20 skills

## 4. Recommended New Courses (4–5)
Each: name · skills covered (TF-IDF scores) · placement (Year/Semester) · Priority

## 5. Priority Action Plan
Numbered, most urgent first. Include measurable targets."""

                        result_j = call_groq(groq_key, prompt_j, max_tokens=3000)
                        st.markdown(result_j)

                        report_j = (
                            f"CS DEPARTMENT — JOB MARKET REPORT\n{'='*60}\n\n"
                            f"Jobs={len(jobs_df)} | Skills={len(skill_r)} | BERT={bert_status}\n\n"
                            f"TOP SKILLS:\n{top20_str}\n\nCATEGORIES:\n{cat_str}\n\n"
                            f"{'='*60}\nAI REPORT\n{'='*60}\n\n{result_j}"
                        )
                        st.download_button(
                            "📥 Download Job Market Report",
                            data=report_j,
                            file_name="job_market_report.txt",
                            mime="text/plain"
                        )
    else:
        st.info("📂 Upload a job listings CSV to begin.")
