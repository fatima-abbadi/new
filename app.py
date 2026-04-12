"""
CS Department Intelligence System v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tab 1 — Student Performance:
    Random Forest + KMeans Clustering + SHAP + AUC
    → Independent AI Report (5 sub-tabs)

Tab 2 — Job Market NLP:
    TF-IDF + BERT Semantic Similarity
    → Independent AI Report (4 sub-tabs)

Each tab is fully independent — no cross-dependency.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, silhouette_score
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import shap
import re
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS Department Intelligence System v4",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1a1040 100%);
    padding:2rem 2.5rem; border-radius:12px;
    margin-bottom:2rem; border-left:4px solid #3b82f6;
}
.main-header h1 {
    color:#f1f5f9; font-size:1.8rem; font-weight:600;
    margin:0; font-family:'IBM Plex Mono',monospace;
}
.main-header p { color:#94a3b8; margin:.5rem 0 0; font-size:.9rem; }

.badge {
    display:inline-block; border-radius:6px;
    padding:2px 10px; font-size:.75rem;
    font-family:'IBM Plex Mono',monospace; margin:2px;
}
.badge-blue   { background:#0a1628; border:1px solid #3b82f6; color:#93c5fd; }
.badge-purple { background:#1a0a2e; border:1px solid #7c3aed; color:#c4b5fd; }
.badge-green  { background:#052e16; border:1px solid #16a34a; color:#86efac; }
.badge-amber  { background:#1a1200; border:1px solid #d97706; color:#fcd34d; }

.metric-card {
    background:#0f172a; border:1px solid #1e3a5f;
    border-radius:10px; padding:1.2rem;
    text-align:center; margin-bottom:4px;
}
.metric-card .val {
    font-size:1.6rem; font-weight:600;
    font-family:'IBM Plex Mono',monospace; color:#3b82f6;
}
.metric-card .lbl { font-size:.75rem; color:#64748b; margin-top:.3rem; }

.sec {
    font-family:'IBM Plex Mono',monospace; font-size:.85rem; color:#3b82f6;
    letter-spacing:.08em; text-transform:uppercase;
    border-bottom:1px solid #1e3a5f; padding-bottom:.5rem; margin-bottom:1.2rem;
}

.box-red    { background:#1a0a0a; border:1px solid #dc2626; border-radius:8px; padding:.8rem 1rem; color:#fca5a5; font-size:.85rem; margin-bottom:.5rem; }
.box-amber  { background:#1a1200; border:1px solid #d97706; border-radius:8px; padding:.8rem 1rem; color:#fcd34d; font-size:.85rem; margin-bottom:.5rem; }
.box-green  { background:#001a0a; border:1px solid #16a34a; border-radius:8px; padding:.8rem 1rem; color:#86efac; font-size:.85rem; margin-bottom:.5rem; }
.box-blue   { background:#0a0f1a; border:1px solid #3b82f6; border-radius:8px; padding:.8rem 1rem; color:#93c5fd; font-size:.85rem; margin-bottom:.5rem; }
.box-purple { background:#0f0a1a; border:1px solid #7c3aed; border-radius:8px; padding:.8rem 1rem; color:#c4b5fd; font-size:.85rem; margin-bottom:.5rem; }
.box-teal   { background:#001a1a; border:1px solid #0d9488; border-radius:8px; padding:.8rem 1rem; color:#5eead4; font-size:.85rem; margin-bottom:.5rem; }

.cluster-card {
    background:#0f172a; border:1px solid #334155;
    border-radius:10px; padding:1rem 1.2rem; margin-bottom:.6rem;
}

.stButton>button {
    background:#1e3a5f; color:#f1f5f9; border:1px solid #3b82f6;
    border-radius:8px; font-family:'IBM Plex Mono',monospace;
    font-size:.85rem; padding:.6rem 1.5rem; width:100%;
}
.stButton>button:hover { background:#3b82f6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🎓 CS Department Intelligence System <span style="color:#7c3aed">v4</span></h1>
  <p>Two independent modules · Each generates its own full AI report</p>
</div>
<span class="badge badge-blue">TF-IDF NLP</span>
<span class="badge badge-purple">BERT Embeddings</span>
<span class="badge badge-green">Random Forest + SHAP</span>
<span class="badge badge-purple">KMeans Clustering</span>
<span class="badge badge-amber">Career Recommendations</span>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Free at console.groq.com")

    st.markdown("---")
    st.markdown("### 🔵 BERT Settings")
    bert_enabled = st.toggle("Enable BERT", value=True)
    bert_model_name = st.selectbox(
        "Model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
    )
    st.caption("MiniLM = fastest · mpnet = best quality")

    st.markdown("---")
    st.markdown("### 🟣 Clustering")
    max_k = st.slider("Max clusters to test:", 3, 8, 6)

    st.markdown("---")
    st.markdown("**Two independent tabs:**")
    st.markdown("- Tab 1 → Student analysis + report")
    st.markdown("- Tab 2 → Job market analysis + report")


# ══════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════

def call_groq(api_key: str, prompt: str, max_tokens: int = 2500) -> str:
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Stage 2: Clean ──
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s/+#.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ── Stage 3: Pre-process ──
STOP_WORDS = {
    "the","and","for","with","this","that","are","have","been","will","from",
    "they","you","our","your","its","their","has","was","were","can","may",
    "must","should","would","could","not","but","also","which","when","where",
    "what","how","all","any","each","more","most","other","some","such",
    "into","over","after","than","then","there","these","those","both","few",
    "many","much","own","same","very","just","about",
}

def preprocess_text(t: str) -> str:
    tokens = clean_text(t).split()
    return " ".join(w for w in tokens if w not in STOP_WORDS and len(w) > 1)


# ── Stage 6: AUC ──
def compute_auc(model, X_te, y_te, classes):
    try:
        y_bin   = label_binarize(y_te, classes=classes)
        y_proba = model.predict_proba(X_te)
        if len(classes) == 2:
            return roc_auc_score(y_bin, y_proba[:, 1])
        return roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        return None


# ── Stage 6: SHAP ──
def compute_shap(model, X_tr, X_te, feature_names):
    try:
        explainer   = shap.TreeExplainer(model)
        sv          = explainer.shap_values(X_te)
        mean_abs    = np.mean([np.abs(s) for s in sv], axis=0) if isinstance(sv, list) else np.abs(sv)
        return pd.Series(mean_abs.mean(axis=0), index=feature_names).sort_values(ascending=False)
    except Exception:
        return None


# ── BERT loader ──
@st.cache_resource(show_spinner=False)
def load_bert(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        return None


def bert_scores(model, texts: list, skills: list) -> dict:
    if model is None:
        return {}
    job_embs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    out = {}
    for skill in skills:
        s_emb   = model.encode([skill], convert_to_numpy=True)
        sims    = cosine_similarity(s_emb, job_embs)[0]
        matched = (sims >= 0.45).sum()
        out[skill] = {
            "bert_avg":  float(sims.mean()),
            "bert_cov":  float(matched / len(texts) * 100),
        }
    return out


# ── KMeans Clustering ──
def run_clustering(df: pd.DataFrame, course_cols: list, max_k: int = 6):
    X = df[course_cols].fillna(df[course_cols].mean())
    scaler   = StandardScaler()
    Xs       = scaler.fit_transform(X)
    ks       = list(range(2, max_k + 1))
    inertias = []
    sils     = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, lb))

    best_k  = ks[int(np.argmax(sils))]
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df      = df.copy()
    df["Cluster"] = km_best.fit_predict(Xs)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]

    profile = df.groupby("Cluster")[course_cols].mean()
    profile["Overall"] = profile.mean(axis=1)
    profile = profile.sort_values("Overall", ascending=False)

    n = len(profile)
    labels = {}
    for rank, cid in enumerate(profile.index):
        if rank == 0:           labels[cid] = "🏆 High Performers"
        elif rank == n - 1:     labels[cid] = "🚨 At Risk"
        elif rank == 1:         labels[cid] = "⬆️ Above Average"
        elif rank == n - 2:     labels[cid] = "⬇️ Below Average"
        else:                   labels[cid] = f"📊 Middle Group {rank}"

    df["Cluster_Label"] = df["Cluster"].map(labels)
    return {
        "df": df, "best_k": best_k, "pca": pca, "profile": profile,
        "labels": labels, "ks": ks, "inertias": inertias, "sils": sils, "Xs": Xs,
    }


def plot_clusters(cd: dict, course_cols: list):
    COLORS = ["#3b82f6","#16a34a","#d97706","#dc2626","#7c3aed","#0891b2","#ec4899","#84cc16"]
    df = cd["df"]

    # ── Elbow + Silhouette ──
    fig, (ae, ас) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    for ax in [ae, ас]:
        ax.set_facecolor("#0f172a"); ax.spines[:].set_color("#1e3a5f")
        ax.tick_params(colors="#94a3b8")

    ae.plot(cd["ks"], cd["inertias"], "o-", color="#3b82f6", lw=2)
    ae.axvline(cd["best_k"], color="#d97706", ls="--", lw=1.5, label=f"Best k={cd['best_k']}")
    ae.set_title("Elbow Method", color="#f1f5f9", fontsize=10)
    ae.set_xlabel("k", color="#94a3b8"); ae.set_ylabel("Inertia", color="#94a3b8")
    ae.legend(fontsize=8, facecolor="#0f172a", labelcolor="#94a3b8")

    bc = ["#d97706" if k == cd["best_k"] else "#334155" for k in cd["ks"]]
    ас.bar(cd["ks"], cd["sils"], color=bc, width=0.5)
    ас.set_title("Silhouette Score", color="#f1f5f9", fontsize=10)
    ас.set_xlabel("k", color="#94a3b8"); ас.set_ylabel("Silhouette", color="#94a3b8")

    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── PCA Scatter ──
    fig2, ax = plt.subplots(figsize=(9, 6))
    fig2.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    for cid, label in cd["labels"].items():
        mask = df["Cluster"] == cid
        ax.scatter(df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],
                   c=COLORS[cid % len(COLORS)], label=f"C{cid}: {label} (n={mask.sum()})",
                   alpha=0.75, s=55, edgecolors="none")
    ax.set_title("Student Clusters (PCA 2D)", color="#f1f5f9", fontsize=11)
    ax.set_xlabel(f"PC1 ({cd['pca'].explained_variance_ratio_[0]*100:.1f}%)", color="#94a3b8")
    ax.set_ylabel(f"PC2 ({cd['pca'].explained_variance_ratio_[1]*100:.1f}%)", color="#94a3b8")
    ax.tick_params(colors="#64748b"); ax.spines[:].set_color("#1e3a5f")
    ax.legend(fontsize=8, facecolor="#0f172a", labelcolor="#94a3b8")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Heatmap ──
    prof = cd["profile"][course_cols] if all(c in cd["profile"].columns for c in course_cols) else cd["profile"].drop(columns=["Overall"], errors="ignore")
    fig3, ax3 = plt.subplots(figsize=(max(10, len(course_cols) * 0.7), max(4, cd["best_k"] * 0.9)))
    fig3.patch.set_facecolor("#0f172a"); ax3.set_facecolor("#0f172a")
    arr = prof.values
    im  = ax3.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ylabels = [f"C{cid}: {cd['labels'].get(cid,'?')}" for cid in prof.index]
    ax3.set_yticks(range(len(prof))); ax3.set_yticklabels(ylabels, color="#f1f5f9", fontsize=8)
    ax3.set_xticks(range(len(prof.columns)))
    ax3.set_xticklabels(prof.columns, rotation=45, ha="right", color="#94a3b8", fontsize=7)
    ax3.spines[:].set_color("#1e3a5f")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax3.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center", fontsize=7,
                     color="white" if arr[i,j] < 50 else "#0f172a")
    plt.colorbar(im, ax=ax3, label="Avg Grade")
    ax3.set_title("Cluster Heatmap", color="#f1f5f9", fontsize=10)
    plt.tight_layout(); st.pyplot(fig3); plt.close()


# ── Career Profiles ──
CAREERS = {
    "Data Scientist":        {"kw": ["data","statistic","python","ml","algorithm","math","database"],      "emoji": "📊"},
    "ML/AI Engineer":        {"kw": ["machine","learning","neural","ai","algorithm","python","deep"],       "emoji": "🤖"},
    "Backend Developer":     {"kw": ["java","database","network","os","algorithm","software","api"],        "emoji": "⚙️"},
    "Full Stack Developer":  {"kw": ["web","javascript","database","frontend","backend","html","css"],      "emoji": "🌐"},
    "DevOps / Cloud":        {"kw": ["network","os","linux","cloud","system","security","infra"],           "emoji": "☁️"},
    "Cybersecurity Analyst": {"kw": ["security","network","crypto","os","algorithm","forensic","ethical"],  "emoji": "🔐"},
}

def career_match(grades: pd.Series, course_cols: list) -> list:
    scores = {}
    for career, info in CAREERS.items():
        score = 0.0
        for c in course_cols:
            cn = c.lower()
            for kw in info["kw"]:
                if kw in cn:
                    score += float(grades[c]) / 100.0
                    break
        scores[career] = score / max(len(info["kw"]), 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]


# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📊 Student Performance",
    "💼 Job Market NLP",
])


# ╔══════════════════════════════════════════════════╗
# ║  TAB 1 — STUDENT PERFORMANCE                    ║
# ║  Sub-tabs: Analysis · Clustering · Career Rec · ║
# ║            Metrics · AI Report                  ║
# ╚══════════════════════════════════════════════════╝
with tab1:
    st.markdown('<p class="sec">01 · Student Performance — RF + Clustering + Career Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-green">✦ INDEPENDENT — No Tab 2 required</span>', unsafe_allow_html=True)

    up_stu = st.file_uploader("📂 Upload Student Grades CSV", type="csv", key="stu")

    if not up_stu:
        st.info("📂 Upload any student grades CSV to begin.")
        st.stop()

    df_raw = pd.read_csv(up_stu)
    st.success(f"✅ {len(df_raw)} students · {len(df_raw.columns)} columns")

    with st.expander("👁 Preview"):
        st.dataframe(df_raw.head())

    st.markdown("---")
    num_cols  = df_raw.select_dtypes(include="number").columns.tolist()
    skip_kw   = ["id","rank","student","semester","final","total","gpa","grade","index"]
    auto_cols = [c for c in num_cols if not any(k in c.lower() for k in skip_kw)]

    c1, c2 = st.columns(2)
    with c1:
        course_cols = st.multiselect("Course columns:", num_cols, default=auto_cols)
    with c2:
        final_col = st.selectbox(
            "Final grade column:",
            num_cols,
            index=num_cols.index("Final_Grade") if "Final_Grade" in num_cols else len(num_cols) - 1,
        )

    t1, t2 = st.columns(2)
    with t1: weak_t = st.slider("Weak below:", 30, 70, 60)
    with t2: avg_t  = st.slider("Average below:", 65, 90, 75)

    if not st.button("🔍 Run Full Analysis", key="b1", type="primary"):
        st.stop()

    if len(course_cols) < 2:
        st.error("Need ≥2 course columns.")
        st.stop()

    df = df_raw.copy()

    # ── Descriptive ──
    avgs      = df[course_cols].mean().sort_values()
    fail_rate = {c: (df[c] < weak_t).mean() * 100 for c in course_cols}
    weak_c    = avgs[avgs < weak_t]
    mid_c     = avgs[(avgs >= weak_t) & (avgs < avg_t)]
    good_c    = avgs[avgs >= avg_t]

    def classify(g):
        return "Weak" if g < weak_t else "Average" if g < avg_t else "Excellent"

    df["Level"] = df[final_col].apply(classify)
    counts = df["Level"].value_counts()

    # ── RF ──
    X = df[course_cols]; y_raw = df[final_col]
    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    y_tr = yr_tr.apply(classify); y_te = yr_te.apply(classify)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    y_pr = rf.predict(X_te)

    acc   = accuracy_score(y_te, y_pr)
    prec  = precision_score(y_te, y_pr, average="weighted", zero_division=0)
    rec   = recall_score(y_te, y_pr,    average="weighted", zero_division=0)
    f1    = f1_score(y_te, y_pr,        average="weighted", zero_division=0)
    cv    = cross_val_score(rf, X_tr, y_tr, cv=5, scoring="accuracy").mean()
    imps  = pd.Series(rf.feature_importances_, index=course_cols).sort_values(ascending=False)
    auc   = compute_auc(rf, X_te.values, y_te, sorted(y_tr.unique()))

    # ── SHAP ──
    with st.spinner("Computing SHAP..."):
        shap_vals = compute_shap(rf, X_tr.values, X_te.values, course_cols)

    # ── Clustering ──
    with st.spinner("Running KMeans Clustering..."):
        cd = run_clustering(df, course_cols, max_k)
    df = cd["df"]

    auc_str = f"{auc*100:.1f}%" if auc else "N/A"

    # ════════════════════════════════════════
    # SUB-TABS
    # ════════════════════════════════════════
    s1, s2, s3, s4, s5 = st.tabs([
        "📊 Performance Analysis",
        "🔵 Student Clustering",
        "🎯 Career Recommendations",
        "📐 Metrics & SHAP",
        "🤖 AI Report",
    ])

    # ── S1: Performance Analysis ──
    with s1:
        st.markdown('<p class="sec">Performance Analysis — Course averages & level distribution</p>', unsafe_allow_html=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        for col,(l,v) in zip([m1,m2,m3,m4,m5],[
            ("Students",   len(df)),
            ("🔴 Weak",     f"{counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.0f}%)"),
            ("🟡 Average",  f"{counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.0f}%)"),
            ("🟢 Excellent",f"{counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.0f}%)"),
            ("Clusters",   cd["best_k"]),
        ]):
            col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Course Averages**")
            bc  = ["#dc2626" if v < weak_t else "#d97706" if v < avg_t else "#16a34a" for v in avgs.values]
            fig, ax = plt.subplots(figsize=(8, max(5, len(course_cols) * .45)))
            fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
            bars = ax.barh(avgs.index, avgs.values, color=bc, height=.6)
            for b, v in zip(bars, avgs.values):
                ax.text(v + .5, b.get_y() + b.get_height()/2, f"{v:.1f}",
                        va="center", fontsize=8, color="#94a3b8")
            ax.axvline(avgs.mean(), color="#3b82f6", ls="--", lw=1.2)
            ax.set_xlim(0, 110); ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.spines[:].set_color("#1e3a5f")
            ax.legend(handles=[
                mpatches.Patch(color="#dc2626", label=f"Weak (<{weak_t})"),
                mpatches.Patch(color="#d97706", label=f"Mid ({weak_t}–{avg_t})"),
                mpatches.Patch(color="#16a34a", label=f"Good (>{avg_t})"),
            ], fontsize=7, facecolor="#0f172a", labelcolor="#94a3b8")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with ch2:
            st.markdown("**Level Distribution**")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            fig2.patch.set_facecolor("#0f172a"); ax2.set_facecolor("#0f172a")
            lc = {"Weak":"#dc2626","Average":"#d97706","Excellent":"#16a34a"}
            wedges, texts, autos = ax2.pie(
                counts.values, labels=counts.index,
                colors=[lc.get(l, "#3b82f6") for l in counts.index],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor":"#0f172a","linewidth":2},
            )
            for t in texts:  t.set_color("#94a3b8")
            for a in autos:  a.set_color("white"); a.set_fontsize(9)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        cw, cm, cg = st.columns(3)
        with cw:
            st.markdown("**🔴 Weak Courses**")
            for c, v in weak_c.items():
                st.markdown(f'<div class="box-red"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
            if not len(weak_c): st.success("None")
        with cm:
            st.markdown("**🟡 Medium Courses**")
            for c, v in mid_c.items():
                st.markdown(f'<div class="box-amber"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
            if not len(mid_c): st.success("None")
        with cg:
            st.markdown("**🟢 Good Courses**")
            for c, v in good_c.items():
                st.markdown(f'<div class="box-green"><b>{c}</b> · Avg:{v:.1f}</div>', unsafe_allow_html=True)

    # ── S2: Clustering ──
    with s2:
        st.markdown('<p class="sec">KMeans Clustering — natural student groups</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="box-purple"><b>{cd["best_k"]} clusters</b> selected by max Silhouette Score ({max(cd["sils"]):.3f}). '
                    f'Unlike manual thresholds, clustering finds hidden patterns across all courses together.</div>', unsafe_allow_html=True)

        for cid, label in cd["labels"].items():
            n_c    = (df["Cluster"] == cid).sum()
            avg_c  = float(df[df["Cluster"] == cid][course_cols].mean().mean())
            best_c = df[df["Cluster"] == cid][course_cols].mean().idxmax()
            wst_c  = df[df["Cluster"] == cid][course_cols].mean().idxmin()
            st.markdown(f"""
<div class="cluster-card">
  <b style="color:#f1f5f9">{label}</b> &nbsp;·&nbsp;
  <span style="color:#64748b">{n_c} students ({n_c/len(df)*100:.0f}%)</span> &nbsp;·&nbsp;
  <span style="color:#3b82f6">Avg: {avg_c:.1f}</span> &nbsp;·&nbsp;
  <span style="color:#16a34a">Best: {best_c}</span> &nbsp;·&nbsp;
  <span style="color:#dc2626">Weakest: {wst_c}</span>
</div>""", unsafe_allow_html=True)

        plot_clusters(cd, course_cols)

        with st.expander("📊 Full Cluster Profile Table"):
            p = cd["profile"].copy()
            p.index = [cd["labels"].get(i, f"Cluster {i}") for i in p.index]
            st.dataframe(p.round(1), use_container_width=True)

    # ── S3: Career Recommendations ──
    with s3:
        st.markdown('<p class="sec">Career Path Recommendations — per student</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-teal">Select a student to see their top 3 career path matches, courses to strengthen, and peer comparison vs their cluster.</div>', unsafe_allow_html=True)

        id_cols    = [c for c in df_raw.columns if any(k in c.lower() for k in ["id","name","student","no"])]
        id_col     = id_cols[0] if id_cols else None
        stu_labels = df_raw[id_col].astype(str).tolist() if id_col else [f"Student {i+1}" for i in range(len(df))]

        sel = st.selectbox("Select student:", stu_labels)
        idx = stu_labels.index(sel)
        row = df.iloc[idx]
        grades = row[course_cols]

        top_careers = career_match(grades, course_cols)
        stu_cluster = int(row["Cluster"])
        stu_label   = cd["labels"].get(stu_cluster, f"Cluster {stu_cluster}")
        stu_avg     = float(grades.mean())
        cluster_avg = float(cd["profile"].loc[stu_cluster, "Overall"]) if "Overall" in cd["profile"].columns else stu_avg

        # Career cards
        c1c, c2c, c3c = st.columns(3)
        for col, (career, score) in zip([c1c, c2c, c3c], top_careers):
            pct = int(score * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            col.markdown(f"""
<div class="cluster-card">
  <div style="font-size:1.5rem">{CAREERS[career]['emoji']}</div>
  <div style="color:#f1f5f9;font-weight:600;font-size:.9rem">{career}</div>
  <div style="color:#3b82f6;font-family:monospace;font-size:.8rem">{bar} {pct}%</div>
</div>""", unsafe_allow_html=True)

        # Course status
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**🔴 Strengthen**")
            wk = [(c, float(grades[c])) for c in course_cols if grades[c] < weak_t]
            for c, g in sorted(wk, key=lambda x: x[1])[:4]:
                st.markdown(f'<div class="box-red"><b>{c}</b> — {g:.0f}/100</div>', unsafe_allow_html=True)
            if not wk: st.success("No weak courses!")
        with cols[1]:
            st.markdown("**🟡 Improve**")
            md = [(c, float(grades[c])) for c in course_cols if weak_t <= grades[c] < avg_t]
            for c, g in sorted(md, key=lambda x: x[1])[:4]:
                st.markdown(f'<div class="box-amber"><b>{c}</b> — {g:.0f}/100</div>', unsafe_allow_html=True)
            if not md: st.success("All above average!")
        with cols[2]:
            st.markdown("**🟢 Strengths**")
            st_c = [(c, float(grades[c])) for c in course_cols if grades[c] >= avg_t]
            for c, g in sorted(st_c, key=lambda x: x[1], reverse=True)[:4]:
                st.markdown(f'<div class="box-green"><b>{c}</b> — {g:.0f}/100</div>', unsafe_allow_html=True)

        # Peer comparison
        diff  = stu_avg - cluster_avg
        sign  = "+" if diff >= 0 else ""
        color = "box-green" if diff >= 0 else "box-red"
        st.markdown(f'<div class="{color}">📊 <b>Your average:</b> {stu_avg:.1f} · <b>Cluster avg:</b> {cluster_avg:.1f} · <b>Diff:</b> {sign}{diff:.1f} · <b>Group:</b> {stu_label}</div>', unsafe_allow_html=True)

        # All students summary
        with st.expander("📋 All Students — Career + Cluster Summary"):
            rows = []
            for i, r in df.iterrows():
                cid   = int(r["Cluster"])
                g     = r[course_cols]
                tc    = career_match(g, course_cols)
                sid   = str(df_raw.iloc[i][id_col]) if id_col else f"S{i+1}"
                rows.append({
                    "Student":         sid,
                    "Cluster":         cd["labels"].get(cid, f"C{cid}"),
                    "Avg Grade":       round(float(g.mean()), 1),
                    "Top Career":      tc[0][0] if tc else "—",
                    "Weak Courses":    sum(1 for c in course_cols if g[c] < weak_t),
                    "RF Level":        r.get("Level", "—"),
                })
            smdf = pd.DataFrame(rows)
            st.dataframe(smdf, use_container_width=True, hide_index=True)
            st.download_button("📥 Download CSV", smdf.to_csv(index=False),
                               "student_summary.csv", "text/csv")

    # ── S4: Metrics & SHAP ──
    with s4:
        st.markdown('<p class="sec">Model Metrics — Random Forest + AUC + SHAP</p>', unsafe_allow_html=True)

        r1c,r2c,r3c,r4c,r5c,r6c = st.columns(6)
        for col,(l,v) in zip([r1c,r2c,r3c,r4c,r5c,r6c],[
            ("Accuracy",  f"{acc*100:.1f}%"),
            ("Precision", f"{prec*100:.1f}%"),
            ("Recall",    f"{rec*100:.1f}%"),
            ("F1-Score",  f"{f1*100:.1f}%"),
            ("CV Acc.",   f"{cv*100:.1f}%"),
            ("AUC-ROC",   auc_str),
        ]):
            col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Metrics Explained")
        st.markdown(f"""
| Metric | Value | Formula | Meaning |
|--------|-------|---------|---------|
| Accuracy   | {acc*100:.1f}%  | (TP+TN)/all       | Overall correct predictions |
| Precision  | {prec*100:.1f}% | TP/(TP+FP)        | Of predicted weak, truly weak % |
| Recall     | {rec*100:.1f}%  | TP/(TP+FN)        | Of all weak, how many caught |
| F1-Score   | {f1*100:.1f}%   | 2·P·R/(P+R)       | Precision-recall balance |
| CV Acc.    | {cv*100:.1f}%   | 5-fold mean       | Stability, not overfitting |
| AUC-ROC    | {auc_str}       | Area under ROC    | Quality vs threshold changes |
| Silhouette | {max(cd['sils']):.3f} | cluster separation | Clustering quality (0–1) |

**Train/Test:** 80% ({len(X_tr)}) train · 20% ({len(X_te)}) test · labels applied **after** split (no leakage).
        """)

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        lbls = sorted(y_te.unique())
        cm_m = confusion_matrix(y_te, y_pr, labels=lbls)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor("#0f172a"); ax3.set_facecolor("#0f172a")
        ax3.imshow(cm_m, cmap="Blues")
        for i in range(len(lbls)):
            for j in range(len(lbls)):
                ax3.text(j, i, str(cm_m[i,j]), ha="center", va="center", fontsize=13,
                         fontweight="bold", color="white" if cm_m[i,j] > cm_m.max()/2 else "#94a3b8")
        ax3.set_xticks(range(len(lbls))); ax3.set_xticklabels(lbls, color="#94a3b8", fontsize=9)
        ax3.set_yticks(range(len(lbls))); ax3.set_yticklabels(lbls, color="#94a3b8", fontsize=9)
        ax3.set_xlabel("Predicted", color="#94a3b8"); ax3.set_ylabel("Actual", color="#94a3b8")
        ax3.spines[:].set_color("#1e3a5f")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        # RF Importance vs SHAP
        fi_c, sh_c = st.columns(2)
        with fi_c:
            st.markdown("**Feature Importance (RF)**")
            top_imp = imps.sort_values(ascending=True)
            fig4, ax4 = plt.subplots(figsize=(7, max(4, len(course_cols) * .4)))
            fig4.patch.set_facecolor("#0f172a"); ax4.set_facecolor("#0f172a")
            ic = ["#3b82f6" if v > imps.mean() else "#334155" for v in top_imp.values]
            ax4.barh(top_imp.index, top_imp.values, color=ic, height=.6)
            ax4.axvline(imps.mean(), color="#d97706", ls="--", lw=1)
            ax4.tick_params(colors="#94a3b8", labelsize=8); ax4.spines[:].set_color("#1e3a5f")
            ax4.set_title("RF Feature Importance", color="#94a3b8", fontsize=9)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        with sh_c:
            st.markdown("**SHAP Values**")
            if shap_vals is not None:
                sv_s = shap_vals.sort_values(ascending=True)
                fig5, ax5 = plt.subplots(figsize=(7, max(4, len(course_cols) * .4)))
                fig5.patch.set_facecolor("#0f172a"); ax5.set_facecolor("#0f172a")
                sc = ["#7c3aed" if v > shap_vals.mean() else "#334155" for v in sv_s.values]
                ax5.barh(sv_s.index, sv_s.values, color=sc, height=.6)
                ax5.axvline(shap_vals.mean(), color="#d97706", ls="--", lw=1)
                ax5.tick_params(colors="#94a3b8", labelsize=8); ax5.spines[:].set_color("#1e3a5f")
                ax5.set_title("Mean |SHAP| per Course", color="#94a3b8", fontsize=9)
                plt.tight_layout(); st.pyplot(fig5); plt.close()
            else:
                st.info("SHAP skipped.")

    # ── S5: AI Report ──
    with s5:
        st.markdown('<p class="sec">AI Report — full institutional analysis from student data</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-purple">The AI reads your actual data: course names, fail rates, RF importance, SHAP values, cluster profiles, and career path alignment. Fully dynamic — no generic advice.</div>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
        else:
            if st.button("🤖 Generate Student Report", key="b1_report"):
                with st.spinner("Generating AI report..."):

                    weak_str = "\n".join([
                        f"- {c}: avg={v:.1f}, fail={fail_rate.get(c,0):.1f}%, rf_imp={imps.get(c,0):.3f}"
                        for c, v in weak_c.items()
                    ]) or "None"

                    cluster_str = "\n".join([
                        f"- {cd['labels'].get(cid,'?')}: {(df['Cluster']==cid).sum()} students, "
                        f"avg={float(df[df['Cluster']==cid][course_cols].mean().mean()):.1f}, "
                        f"weakest={df[df['Cluster']==cid][course_cols].mean().idxmin()}"
                        for cid in sorted(cd["labels"])
                    ])

                    shap_str = "\n".join([f"- {c}: {v:.4f}" for c, v in (shap_vals.head(5).items() if shap_vals is not None else [])])

                    prompt = f"""You are a senior academic consultant writing a formal report for the Head of a Computer Science Department.

=== STUDENT DATA ({len(df)} students) ===
Random Forest: Accuracy={acc*100:.1f}% | F1={f1*100:.1f}% | AUC={auc_str} | CV={cv*100:.1f}%
Thresholds: Weak<{weak_t} | Average {weak_t}–{avg_t} | Excellent>{avg_t}
Distribution: Weak={counts.get('Weak',0)} | Average={counts.get('Average',0)} | Excellent={counts.get('Excellent',0)}

KMEANS CLUSTERS (k={cd['best_k']}, silhouette={max(cd['sils']):.3f}):
{cluster_str}

WEAK COURSES:
{weak_str}

TOP 5 SHAP (most influential courses):
{shap_str}

ALL COURSES: {', '.join(course_cols)}

=== REPORT STRUCTURE ===

## 1. Executive Summary
Key numbers + what clustering revealed. 3-4 sentences.

## 2. Cluster Interventions
For each cluster:
### [Cluster Label] — N students
- Profile: what these students are good/bad at
- Intervention strategy tailored to this group
- Career paths they're positioned for

## 3. Per-Course Recommendations
For EACH weak course:
### [Course Name] | Fail: X% | SHAP: X.XXXX
- Why students struggle
- 3 concrete teaching improvements
- Resources: 2 YouTube channels + 1 platform + 1 textbook
- Priority: Critical / High

## 4. Priority Action Plan
Numbered, most urgent first. Include timeline.

Use exact course names and actual numbers from the data. Be specific."""

                    result = call_groq(groq_key, prompt, max_tokens=2800)
                    st.markdown(result)

                    report_txt = (
                        f"CS DEPT — STUDENT REPORT\n{'='*60}\n\n"
                        f"Students={len(df)} | F1={f1*100:.1f}% | AUC={auc_str} | Clusters={cd['best_k']}\n\n"
                        f"CLUSTERS:\n{cluster_str}\n\nWEAK COURSES:\n{weak_str}\n\n"
                        f"{'='*60}\nAI REPORT\n{'='*60}\n\n{result}"
                    )
                    st.download_button("📥 Download Student Report",
                                       report_txt, "student_report.txt", "text/plain")

    # ── Save to session ──
    st.session_state.update({
        "s1_done": True, "weak_c": weak_c, "mid_c": mid_c,
        "counts": counts, "n_stu": len(df), "n_tr": len(X_tr), "n_te": len(X_te),
        "met": {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cv": cv, "auc": auc},
        "imps": imps, "fr": fail_rate, "wt": weak_t, "at": avg_t, "cc": course_cols,
        "cluster_data": cd,
    })


# ╔══════════════════════════════════════════════════╗
# ║  TAB 2 — JOB MARKET NLP                        ║
# ║  Sub-tabs: Skills · BERT · Gap Table · Report  ║
# ╚══════════════════════════════════════════════════╝
with tab2:
    st.markdown('<p class="sec">02 · Job Market — TF-IDF + BERT Semantic Similarity</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-green">✦ INDEPENDENT — No Tab 1 required</span>', unsafe_allow_html=True)

    TAXONOMY = {
        "Programming Languages": ["python","java","javascript","c++","c#","kotlin","swift","go","typescript","php","ruby","scala","rust"],
        "Web & Mobile":          ["react","angular","vue","node.js","django","flask","spring","html5","css3","android","ios","flutter","next.js"],
        "Data Science & AI":     ["machine learning","deep learning","tensorflow","pytorch","pandas","numpy","sql","nosql","mongodb","data science","nlp","tableau","power bi","scikit-learn"],
        "Cloud & DevOps":        ["aws","azure","google cloud","docker","kubernetes","jenkins","ci/cd","terraform","linux","git","devops","microservices"],
        "Cybersecurity":         ["cybersecurity","penetration testing","network security","encryption","firewall","siem","ethical hacking","vulnerability assessment","zero trust"],
        "Databases":             ["mysql","postgresql","oracle","sql server","redis","elasticsearch","cassandra","database design","data warehouse"],
        "Software Engineering":  ["agile","scrum","rest api","design patterns","software architecture","unit testing","version control","continuous integration"],
        "Networking":            ["networking","tcp/ip","routing","switching","vpn","cisco","wireshark","network protocols","load balancing"],
        "Emerging Tech":         ["blockchain","iot","embedded systems","computer vision","large language model","generative ai","edge computing","5g"],
    }

    up_job = st.file_uploader("📂 Upload Job Listings CSV", type="csv", key="job")

    if not up_job:
        st.info("📂 Upload a job listings CSV to begin.")
        st.stop()

    with st.spinner("Loading..."):
        jobs_df = pd.read_csv(up_job, on_bad_lines="skip")
    st.success(f"✅ {len(jobs_df)} job listings")

    text_cols = jobs_df.select_dtypes(include="object").columns.tolist()
    desc_col  = st.selectbox(
        "Job description column:", text_cols,
        index=text_cols.index("jobdescription") if "jobdescription" in text_cols else 0,
    )
    extra = st.text_input("Extra skills to track:", placeholder="rust, llm, cybersecurity")

    if not st.button("🔍 Run NLP Analysis", key="b2", type="primary"):
        st.stop()

    if extra.strip():
        TAXONOMY["Custom"] = [s.strip().lower() for s in extra.split(",") if s.strip()]

    with st.spinner("Running TF-IDF (Stage 4a)..."):
        raw_texts     = jobs_df[desc_col].dropna().tolist()
        clean_texts_p = [preprocess_text(t) for t in raw_texts]

        vec  = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True)
        tmat = vec.fit_transform(clean_texts_p)
        fnames  = vec.get_feature_names_out()
        tsums   = np.asarray(tmat.sum(axis=0)).flatten()
        tdict   = dict(zip(fnames, tsums))
        all_txt = " ".join(clean_texts_p)

        skill_results = {}
        for cat, skills in TAXONOMY.items():
            for skill in skills:
                sc   = clean_text(skill)
                freq = all_txt.count(sc)
                ts   = tdict.get(sc, 0.0)
                jw   = sum(1 for t in clean_texts_p if sc in t)
                cov  = jw / len(clean_texts_p) * 100
                if freq > 0:
                    skill_results[skill] = {
                        "category": cat, "freq": freq,
                        "tfidf": round(float(ts), 2),
                        "cov": round(cov, 1), "jobs": jw,
                        "bert_sim": None, "bert_cov": None,
                    }

        sorted_skills = sorted(skill_results.items(), key=lambda x: x[1]["tfidf"], reverse=True)
        top30         = sorted_skills[:30]

    # ── BERT ──
    bert_model = None
    bert_ok    = False
    if bert_enabled:
        with st.spinner("Loading BERT model..."):
            bert_model = load_bert(bert_model_name)
        if bert_model:
            sample = clean_texts_p[:min(500, len(clean_texts_p))]
            with st.spinner(f"Computing BERT similarity ({len(skill_results)} skills)..."):
                br = bert_scores(bert_model, sample, list(skill_results.keys()))
            for sk, bd in br.items():
                if sk in skill_results:
                    skill_results[sk]["bert_sim"] = round(bd["bert_avg"], 4)
                    skill_results[sk]["bert_cov"] = round(bd["bert_cov"], 1)
            bert_ok = True

    has_bert = bert_ok and any(v["bert_sim"] is not None for v in skill_results.values())
    bert_msg = f"✅ {bert_model_name}" if has_bert else "⚠️ Not available — install sentence-transformers"
    (st.markdown(f'<div class="box-blue">🔵 BERT: {bert_msg}</div>', unsafe_allow_html=True)
     if has_bert else
     st.markdown(f'<div class="box-amber">🔵 BERT: {bert_msg}</div>', unsafe_allow_html=True))

    cat_totals = {}
    for _, d in skill_results.items():
        cat_totals[d["category"]] = cat_totals.get(d["category"], 0) + d["tfidf"]

    # ════════════════════════════════════════
    # SUB-TABS
    # ════════════════════════════════════════
    j1, j2, j3, j4 = st.tabs([
        "📊 Skills Analysis",
        "🔵 BERT Comparison",
        "🔗 Curriculum Gap",
        "🤖 AI Report",
    ])

    CAT_COLORS = {
        "Programming Languages":"#3b82f6","Web & Mobile":"#16a34a",
        "Data Science & AI":"#7c3aed","Cloud & DevOps":"#d97706",
        "Cybersecurity":"#dc2626","Databases":"#0891b2",
        "Software Engineering":"#059669","Networking":"#9333ea",
        "Emerging Tech":"#f59e0b","Custom":"#ec4899",
    }

    # ── J1: Skills Analysis ──
    with j1:
        st.markdown('<p class="sec">TF-IDF Skill Extraction — top 20 skills by demand</p>', unsafe_allow_html=True)

        s1c,s2c,s3c,s4c = st.columns(4)
        for col,(l,v) in zip([s1c,s2c,s3c,s4c],[
            ("Total Jobs",   len(jobs_df)),
            ("Skills Found", len(skill_results)),
            ("Top Skill",    top30[0][0] if top30 else "—"),
            ("BERT Active",  "✅" if has_bert else "❌"),
        ]):
            col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

        names  = [s[0] for s in top30[:20]]
        scores = [s[1]["tfidf"] for s in top30[:20]]
        covs   = [s[1]["cov"] for s in top30[:20]]
        cats   = [s[1]["category"] for s in top30[:20]]
        bclrs  = [CAT_COLORS.get(c,"#64748b") for c in cats]

        fig6, ax6 = plt.subplots(figsize=(10, 8))
        fig6.patch.set_facecolor("#0f172a"); ax6.set_facecolor("#0f172a")
        bars = ax6.barh(names, scores, color=bclrs, height=.6)
        for b, v, cov in zip(bars, scores, covs):
            ax6.text(v+.1, b.get_y()+b.get_height()/2,
                     f"{v:.1f} | {cov:.0f}% jobs", va="center", fontsize=7, color="#94a3b8")
        ax6.tick_params(colors="#94a3b8", labelsize=8); ax6.spines[:].set_color("#1e3a5f")
        ax6.set_xlabel("TF-IDF Score", color="#94a3b8")
        ax6.set_title("Top 20 Skills by TF-IDF Score", color="#f1f5f9")
        ax6.legend(
            handles=[mpatches.Patch(color=v, label=k) for k,v in CAT_COLORS.items() if k in cats],
            fontsize=7, facecolor="#0f172a", labelcolor="#94a3b8", loc="lower right",
        )
        plt.tight_layout(); st.pyplot(fig6); plt.close()

        st.markdown("#### Demand by Category")
        total_all = sum(cat_totals.values()) or 1
        for cat, total in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
            pct = int(total/total_all*100)
            st.progress(pct, text=f"**{cat}** — {total:.1f} ({pct}%)")

        st.markdown("#### All Extracted Skills")
        st.dataframe(pd.DataFrame([{
            "Skill": s, "Category": d["category"],
            "TF-IDF": d["tfidf"], "Coverage %": d["cov"],
            "Jobs": d["jobs"], "BERT Sim": d["bert_sim"] or "—",
        } for s, d in sorted_skills]), use_container_width=True)

        with st.expander("🔬 Pipeline Stages Explained"):
            st.markdown(f"""
| Stage | Description | Detail |
|-------|-------------|--------|
| Stage 2: Clean    | URLs · Punctuation · Whitespace | {len(raw_texts):,} texts |
| Stage 3: Pre-proc | Tokenization · Stop words ({len(STOP_WORDS)}) | Preprocessed |
| Stage 4a: TF-IDF  | ngram=(1,3) · sublinear_tf · min_df=2 | Matrix ready |
| Stage 4b: BERT    | Cosine similarity · threshold=0.45 | {bert_model_name} |

**Why TF-IDF + BERT together?**
TF-IDF is fast and exact (keyword counting). BERT understands context —
it knows "ML engineer" relates to "machine learning". Combined gives the most complete picture.
            """)

    # ── J2: BERT Comparison ──
    with j2:
        st.markdown('<p class="sec">BERT Semantic Similarity — contextual skill scoring</p>', unsafe_allow_html=True)

        if not has_bert:
            st.markdown('<div class="box-amber">⚠️ BERT not active. Install: <code>pip install sentence-transformers</code> and re-run.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="box-blue">🔵 Model: <b>{bert_model_name}</b> · {len([v for v in skill_results.values() if v["bert_sim"]])} skills scored · threshold=0.45</div>', unsafe_allow_html=True)

            bert_sorted = sorted(
                [(s, d) for s, d in skill_results.items() if d["bert_sim"] is not None],
                key=lambda x: x[1]["bert_sim"], reverse=True,
            )[:20]

            b_names  = [s for s,_ in bert_sorted]
            b_scores = [d["bert_sim"] for _,d in bert_sorted]
            b_covs   = [d["bert_cov"] for _,d in bert_sorted]
            b_cats   = [d["category"] for _,d in bert_sorted]
            b_clrs   = [CAT_COLORS.get(c,"#64748b") for c in b_cats]

            fig7, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(16, 8))
            fig7.patch.set_facecolor("#0f172a")

            for ax in [ax_t, ax_b]:
                ax.set_facecolor("#0f172a"); ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.spines[:].set_color("#1e3a5f")

            # TF-IDF (same order as BERT for comparison)
            tfidf_vals = [skill_results[s]["tfidf"] if s in skill_results else 0 for s in b_names]
            t_clrs     = [CAT_COLORS.get(skill_results.get(s,{}).get("category",""),"#334155") for s in b_names]
            ax_t.barh(b_names, tfidf_vals, color=t_clrs, height=.6)
            ax_t.set_xlabel("TF-IDF Score", color="#94a3b8")
            ax_t.set_title("TF-IDF (Lexical)", color="#f1f5f9", fontsize=10)

            # BERT
            brt_clrs = ["#7c3aed" if v > float(np.mean(b_scores)) else "#334155" for v in b_scores]
            bars_b   = ax_b.barh(b_names, b_scores, color=brt_clrs, height=.6)
            for b, v, cov in zip(bars_b, b_scores, b_covs):
                ax_b.text(v+.001, b.get_y()+b.get_height()/2,
                          f"{v:.3f} | {cov:.0f}%", va="center", fontsize=7, color="#94a3b8")
            ax_b.set_xlabel("BERT Avg Similarity", color="#94a3b8")
            ax_b.set_title("BERT Semantic (Contextual)", color="#f1f5f9", fontsize=10)

            plt.suptitle("TF-IDF vs BERT — Top 20 Skills", color="#f1f5f9", fontsize=12)
            plt.tight_layout(); st.pyplot(fig7); plt.close()

            # Ranking comparison table
            st.markdown("#### TF-IDF vs BERT Ranking Agreement")
            tfidf_r = {s: i+1 for i, (s,_) in enumerate(sorted_skills[:20])}
            bert_r  = {s: i+1 for i, (s,_) in enumerate(bert_sorted)}
            cmp_rows = []
            for s, _ in sorted_skills[:20]:
                tr   = tfidf_r.get(s, "—")
                br   = bert_r.get(s, "—")
                diff = (tr - br) if isinstance(tr,int) and isinstance(br,int) else 0
                cmp_rows.append({
                    "Skill": s, "TF-IDF Rank": tr, "BERT Rank": br,
                    "Diff": f"{'↑' if diff>0 else '↓' if diff<0 else '='}{abs(diff)}",
                    "Agreement": "✅" if abs(diff)<=3 else "⚠️ Diverge",
                    "TF-IDF": skill_results.get(s,{}).get("tfidf","—"),
                    "BERT Sim": skill_results.get(s,{}).get("bert_sim","—"),
                })
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # ── J3: Curriculum Gap ──
    with j3:
        st.markdown('<p class="sec">Curriculum–Market Gap — what market wants vs what exists</p>', unsafe_allow_html=True)

        existing = st.session_state.get("cc", [])
        if not existing:
            st.markdown('<div class="box-amber">💡 Run Tab 1 first to automatically compare against your actual course list. Showing generic gap analysis.</div>', unsafe_allow_html=True)

        gap_rows = []
        for s, d in top30[:20]:
            covered = any(
                s.lower() in c.lower() or c.lower() in s.lower()
                for c in existing
            ) if existing else False
            gap_rows.append({
                "Skill":       s,
                "Category":    d["category"],
                "TF-IDF":      d["tfidf"],
                "Coverage %":  d["cov"],
                "BERT Sim":    d.get("bert_sim") or "—",
                "In Curriculum": "✅ Yes" if covered else "❌ Gap",
            })
        gdf = pd.DataFrame(gap_rows)
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        gc1, gc2 = st.columns(2)
        gc1.metric("Skills with Gap",  sum(1 for r in gap_rows if "❌" in r["In Curriculum"]), delta_color="inverse")
        gc2.metric("Skills Covered",   sum(1 for r in gap_rows if "✅" in r["In Curriculum"]))

    # ── J4: AI Report ──
    with j4:
        st.markdown('<p class="sec">AI Report — market analysis + curriculum recommendations</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-purple">The AI uses TF-IDF scores, BERT similarity, job coverage %, and category demand to generate a targeted curriculum improvement plan.</div>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
        else:
            r_sub1, r_sub2, r_sub3 = st.tabs([
                "📊 Skills Deep Dive",
                "📚 New Courses",
                "🚀 Action Plan",
            ])

            top20_str = "\n".join([
                f"- {s} ({d['category']}): TF-IDF={d['tfidf']:.1f}, cov={d['cov']:.1f}%"
                + (f", BERT={d['bert_sim']:.3f}" if d.get("bert_sim") else "")
                for s, d in top30[:20]
            ])
            cat_str = "\n".join([
                f"- {cat}: {total:.1f}" for cat, total
                in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
            ])
            existing = st.session_state.get("cc", [])
            courses_str = ", ".join(existing) if existing else "Not provided (run Tab 1 first)"

            with r_sub1:
                st.markdown("**Top 10 Skills — Deep Analysis**")
                if st.button("🤖 Generate Skills Analysis", key="b2_skills"):
                    with st.spinner("Generating..."):
                        result_skills = call_groq(groq_key, f"""Analyse top 10 skills from job market data for CS department head.

TOP 20 SKILLS (TF-IDF + BERT):
{top20_str}

DEMAND BY CATEGORY:
{cat_str}

For each of the top 10 skills:
### [Skill] — TF-IDF: X | Coverage: X% | BERT: X
- What it's used for in industry
- Why high demand
- Teaching approach (theory vs practical ratio)
- Resources: 2 real YouTube channels + 1 platform + 1 book
- Connection to other skills (skill cluster)""", max_tokens=2500)
                        st.session_state["r_skills"] = result_skills
                if "r_skills" in st.session_state:
                    st.markdown(st.session_state["r_skills"])
                    st.download_button("📥 Download", st.session_state["r_skills"],
                                       "skills_analysis.txt", "text/plain", key="dl_sk")

            with r_sub2:
                st.markdown("**Recommended New Courses**")
                if st.button("🤖 Generate New Courses", key="b2_new"):
                    with st.spinner("Generating..."):
                        result_new = call_groq(groq_key, f"""Recommend 5 new CS courses based on market data.

CURRENT CURRICULUM: {courses_str}
TOP 20 MARKET SKILLS: {top20_str}

For each course:
## Course N: [Name]
**Skills covered:** (from TF-IDF data with scores)
**Market evidence:** TF-IDF total, coverage %, trend
**Why needed:** gap reasoning
**Placement:** Year X, Semester Y (Required/Elective)
**Prerequisites:** from existing courses
**Priority:** High/Medium/Low""", max_tokens=2000)
                        st.session_state["r_new"] = result_new
                if "r_new" in st.session_state:
                    st.markdown(st.session_state["r_new"])
                    st.download_button("📥 Download", st.session_state["r_new"],
                                       "new_courses.txt", "text/plain", key="dl_nw")

            with r_sub3:
                st.markdown("**Priority Action Plan**")
                if st.button("🚀 Generate Action Plan", key="b2_action"):
                    with st.spinner("Generating..."):
                        prev = "\n".join([
                            f"{k}:\n{st.session_state[k][:500]}"
                            for k in ["r_skills","r_new"] if k in st.session_state
                        ])
                        result_action = call_groq(groq_key, f"""Priority action plan for CS dept based on job market analysis.

Jobs analyzed: {len(jobs_df)}
Top skills: {', '.join([s for s,_ in top30[:8]])}
CURRENT CURRICULUM: {courses_str}
{prev}

## Executive Summary (3-4 sentences)

## 🔴 IMMEDIATE (Next Semester)
3-4 actions with target, expected impact, owner.

## 🟡 SHORT-TERM (1 Year)
3-4 actions.

## 🟢 MEDIUM-TERM (1-3 Years)
2-3 actions including new courses.

## Success Metrics
% of top-20 skills covered target, timeline.""", max_tokens=2000)
                        st.session_state["r_action_j"] = result_action

                if "r_action_j" in st.session_state:
                    st.markdown(st.session_state["r_action_j"])

                    full = (
                        f"CS DEPT — JOB MARKET REPORT\n{'='*60}\n\n"
                        f"Jobs={len(jobs_df)}\nTop skills: {', '.join([s for s,_ in top30[:10]])}\n\n"
                        f"{'='*60}\nSKILLS ANALYSIS\n{'='*60}\n{st.session_state.get('r_skills','Not generated')}\n\n"
                        f"{'='*60}\nNEW COURSES\n{'='*60}\n{st.session_state.get('r_new','Not generated')}\n\n"
                        f"{'='*60}\nACTION PLAN\n{'='*60}\n{st.session_state.get('r_action_j','Not generated')}"
                    )
                    st.download_button("📥 Download Full Market Report",
                                       full, "job_market_report.txt", "text/plain", key="dl_full_j")

    # ── Save to session ──
    st.session_state.update({
        "top30": top30, "skill_results": skill_results,
        "n_jobs": len(jobs_df), "cat_totals": cat_totals,
    })
