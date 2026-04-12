"""
CS Department Intelligence System v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New in v3:
  ✦ BERT Embeddings (Sentence-BERT) بدل TF-IDF فقط
  ✦ Student Clustering (KMeans + PCA + Elbow)
  ✦ Recommendation System (Career Path + Course Suggestions)

Tabs:
  Tab 1 — Student Performance: Random Forest + Clustering + Recommendations
  Tab 2 — Job Market NLP:      TF-IDF + BERT Semantic Similarity
  Tab 3 — Combined Report:     Full institutional analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="CS Department Intelligence System v3",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
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
.main-header h1 { color:#f1f5f9; font-size:1.8rem; font-weight:600; margin:0; font-family:'IBM Plex Mono',monospace; }
.main-header p  { color:#94a3b8; margin:0.5rem 0 0; font-size:0.9rem; }
.badge-v3 {
    display:inline-block; background:#1a0a2e; border:1px solid #7c3aed;
    color:#c4b5fd; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-family:'IBM Plex Mono',monospace; margin:2px;
}
.badge-independent {
    display:inline-block; background:#052e16; border:1px solid #16a34a;
    color:#86efac; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-family:'IBM Plex Mono',monospace; margin-bottom:1rem;
}
.metric-card {
    background:#0f172a; border:1px solid #1e3a5f;
    border-radius:10px; padding:1.2rem;
    text-align:center; margin-bottom:4px;
}
.metric-card .val { font-size:1.7rem; font-weight:600; font-family:'IBM Plex Mono',monospace; color:#3b82f6; }
.metric-card .lbl { font-size:0.75rem; color:#64748b; margin-top:0.3rem; }
.sec {
    font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#3b82f6;
    letter-spacing:.08em; text-transform:uppercase;
    border-bottom:1px solid #1e3a5f; padding-bottom:.5rem; margin-bottom:1.2rem;
}
.box-red   { background:#1a0a0a; border:1px solid #dc2626; border-radius:8px; padding:.8rem 1rem; color:#fca5a5; font-size:.85rem; margin-bottom:.5rem; }
.box-amber { background:#1a1200; border:1px solid #d97706; border-radius:8px; padding:.8rem 1rem; color:#fcd34d; font-size:.85rem; margin-bottom:.5rem; }
.box-green { background:#001a0a; border:1px solid #16a34a; border-radius:8px; padding:.8rem 1rem; color:#86efac; font-size:.85rem; margin-bottom:.5rem; }
.box-blue  { background:#0a0f1a; border:1px solid #3b82f6; border-radius:8px; padding:.8rem 1rem; color:#93c5fd; font-size:.85rem; margin-bottom:.5rem; }
.box-purple{ background:#0f0a1a; border:1px solid #7c3aed; border-radius:8px; padding:.8rem 1rem; color:#c4b5fd; font-size:.85rem; margin-bottom:.5rem; }
.box-teal  { background:#001a1a; border:1px solid #0d9488; border-radius:8px; padding:.8rem 1rem; color:#5eead4; font-size:.85rem; margin-bottom:.5rem; }
.cluster-card {
    background:#0f172a; border:1px solid #334155; border-radius:10px;
    padding:1rem 1.2rem; margin-bottom:.6rem;
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
  <h1>🎓 CS Department Intelligence System <span style="color:#7c3aed">v3</span></h1>
  <p>
    <span style="color:#3b82f6">TF-IDF NLP</span> ·
    <span style="color:#7c3aed">BERT Semantic Embeddings</span> ·
    <span style="color:#16a34a">Random Forest + Clustering</span> ·
    <span style="color:#d97706">Recommendation System</span> ·
    Institutional Decision Support
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<span class="badge-v3">✦ NEW: BERT Embeddings</span>
<span class="badge-v3">✦ NEW: Student Clustering</span>
<span class="badge-v3">✦ NEW: Recommendation System</span>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Free at console.groq.com")
    st.markdown("---")
    st.markdown("### 🔵 BERT Settings")
    bert_enabled = st.toggle("Enable BERT (Sentence-BERT)", value=True)
    st.caption("Requires `sentence-transformers` installed.\nIf not installed, falls back to TF-IDF.")
    bert_model_name = st.selectbox(
        "BERT Model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        index=0
    )
    st.caption("all-MiniLM-L6-v2 = fastest (384d)\nall-mpnet-base-v2 = best quality (768d)")
    st.markdown("---")
    st.markdown("### 🟣 Clustering Settings")
    max_k = st.slider("Max clusters to test:", 3, 8, 6)
    st.markdown("---")
    st.markdown("**Pipeline v3:**")
    st.markdown("```\nCSV Input\n  ↓\nStage 2: Clean\n  ↓\nStage 3: Pre-process\n  ↓\nStage 4a: TF-IDF\nStage 4b: BERT Embed\n  ↓\nStage 5: Random Forest\n       + KMeans Cluster\n  ↓\nStage 6: AUC + SHAP\n       + Rec. System\n  ↓\nAI Report\n```")

# ══════════════════════════════════════════════════
# BERT LOADER (cached)
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_bert_model(model_name: str):
    """Load Sentence-BERT model — cached so it loads once."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        return None


def bert_skill_scores(model, clean_texts: list, skills: list) -> dict:
    """
    BERT Semantic Similarity:
    لكل مهارة، نحسب cosine similarity بينها وبين كل وظيفة
    ونرجع: average similarity score + job coverage %
    """
    if model is None:
        return {}

    results = {}
    # Encode all job descriptions once (efficient)
    job_embeddings = model.encode(
        clean_texts, batch_size=32,
        show_progress_bar=False, convert_to_numpy=True
    )

    for skill in skills:
        skill_emb = model.encode([skill], convert_to_numpy=True)
        sims = cosine_similarity(skill_emb, job_embeddings)[0]  # shape: (n_jobs,)
        threshold = 0.45  # threshold للـ semantic match
        matched = (sims >= threshold).sum()
        results[skill] = {
            'bert_avg_sim':  float(sims.mean()),
            'bert_max_sim':  float(sims.max()),
            'bert_coverage': float(matched / len(clean_texts) * 100),
            'bert_jobs':     int(matched),
        }
    return results


# ══════════════════════════════════════════════════
# GROQ HELPER
# ══════════════════════════════════════════════════
def call_groq(api_key: str, prompt: str, max_tokens: int = 2500) -> str:
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ══════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════

# ── Stage 2: Enhanced Text Cleaning ──
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r'http\S+', ' ', t)
    t = re.sub(r'[^a-z0-9\s/+#.]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ── Stage 3: Stop Word Removal + Tokenization ──
STOP_WORDS = {
    'the','and','for','with','this','that','are','have','been',
    'will','from','they','you','our','your','its','their','has',
    'was','were','can','may','must','should','would','could',
    'not','but','also','which','when','where','what','how',
    'all','any','each','more','most','other','some','such',
    'into','over','after','than','then','there','these','those',
    'both','few','many','much','own','same','very','just','about',
}

def preprocess_text(t: str) -> str:
    tokens = clean_text(t).split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [w for w in tokens if len(w) > 1]
    return ' '.join(tokens)


# ── Stage 6: AUC ──
def compute_auc(model, X_te, y_te, classes):
    try:
        y_bin   = label_binarize(y_te, classes=classes)
        y_proba = model.predict_proba(X_te)
        if len(classes) == 2:
            return roc_auc_score(y_bin, y_proba[:, 1])
        return roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        return None


# ── Stage 6: SHAP ──
def compute_shap(model, X_tr, X_te, feature_names):
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values)
        mean_shap = pd.Series(
            mean_abs.mean(axis=0),
            index=feature_names
        ).sort_values(ascending=False)
        return mean_shap
    except Exception:
        return None


# ══════════════════════════════════════════════════
# NEW — STUDENT CLUSTERING (KMeans + PCA + Elbow)
# ══════════════════════════════════════════════════

def run_student_clustering(df: pd.DataFrame, course_cols: list, max_k: int = 6):
    """
    Stage 5b — KMeans Student Clustering
    ─────────────────────────────────────
    1. StandardScaler → normalize grades
    2. Elbow method  → find best k (inertia + silhouette)
    3. KMeans fit    → assign cluster labels
    4. PCA (2D)      → for visualization
    5. Cluster profiling → mean grades per cluster
    Returns: df with 'Cluster' column, scaler, kmeans, pca, elbow data, silhouette scores
    """
    X = df[course_cols].copy().fillna(df[course_cols].mean())

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + Silhouette
    inertias    = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        silhouettes.append(sil)

    # Best k = max silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]

    # Final KMeans
    kmeans  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df      = df.copy()
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA 2D
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df['PCA1'] = coords[:, 0]
    df['PCA2'] = coords[:, 1]

    # Cluster profiles
    profile = df.groupby('Cluster')[course_cols].mean()
    profile['Overall_Avg'] = profile.mean(axis=1)
    profile = profile.sort_values('Overall_Avg', ascending=False)

    # Auto-label clusters based on overall average
    n = len(profile)
    labels_map = {}
    for rank, cid in enumerate(profile.index):
        if rank == 0:
            labels_map[cid] = "🏆 High Performers"
        elif rank == n - 1:
            labels_map[cid] = "🚨 At Risk"
        elif rank == 1:
            labels_map[cid] = "⬆️ Above Average"
        elif rank == n - 2:
            labels_map[cid] = "⬇️ Below Average"
        else:
            labels_map[cid] = f"📊 Cluster {rank+1}"

    df['Cluster_Label'] = df['Cluster'].map(labels_map)

    return {
        'df':           df,
        'best_k':       best_k,
        'kmeans':       kmeans,
        'pca':          pca,
        'scaler':       scaler,
        'profile':      profile,
        'labels_map':   labels_map,
        'inertias':     list(inertias),
        'silhouettes':  list(silhouettes),
        'k_range':      list(k_range),
        'X_scaled':     X_scaled,
    }


def plot_clustering_results(cluster_data: dict, course_cols: list):
    """Plot elbow curve, silhouette scores, and PCA scatter."""
    df          = cluster_data['df']
    best_k      = cluster_data['best_k']
    k_range     = cluster_data['k_range']
    inertias    = cluster_data['inertias']
    silhouettes = cluster_data['silhouettes']
    labels_map  = cluster_data['labels_map']

    CLUSTER_COLORS = [
        '#3b82f6','#16a34a','#d97706','#dc2626',
        '#7c3aed','#0891b2','#ec4899','#84cc16'
    ]

    # ── Row 1: Elbow + Silhouette ──
    fig1, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
    fig1.patch.set_facecolor('#0f172a')
    for ax in [ax_e, ax_s]:
        ax.set_facecolor('#0f172a')

    # Elbow
    ax_e.plot(k_range, inertias, 'o-', color='#3b82f6', lw=2, ms=7)
    ax_e.axvline(best_k, color='#d97706', linestyle='--', lw=1.5, label=f'Best k={best_k}')
    ax_e.set_xlabel('Number of Clusters (k)', color='#94a3b8')
    ax_e.set_ylabel('Inertia (SSE)', color='#94a3b8')
    ax_e.set_title('Elbow Method', color='#f1f5f9', fontsize=10)
    ax_e.tick_params(colors='#94a3b8')
    ax_e.spines[:].set_color('#1e3a5f')
    ax_e.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8')

    # Silhouette
    bar_colors = ['#d97706' if k == best_k else '#334155' for k in k_range]
    ax_s.bar(k_range, silhouettes, color=bar_colors, width=0.5)
    ax_s.set_xlabel('Number of Clusters (k)', color='#94a3b8')
    ax_s.set_ylabel('Silhouette Score', color='#94a3b8')
    ax_s.set_title('Silhouette Score (higher = better)', color='#f1f5f9', fontsize=10)
    ax_s.tick_params(colors='#94a3b8')
    ax_s.spines[:].set_color('#1e3a5f')

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── Row 2: PCA Scatter ──
    fig2, ax_p = plt.subplots(figsize=(9, 6))
    fig2.patch.set_facecolor('#0f172a')
    ax_p.set_facecolor('#0f172a')

    for cid, label in labels_map.items():
        mask = df['Cluster'] == cid
        clr  = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax_p.scatter(
            df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'],
            c=clr, label=f"C{cid}: {label} (n={mask.sum()})",
            alpha=0.75, s=55, edgecolors='none'
        )

    ax_p.set_xlabel(f'PC1 ({cluster_data["pca"].explained_variance_ratio_[0]*100:.1f}% var)', color='#94a3b8')
    ax_p.set_ylabel(f'PC2 ({cluster_data["pca"].explained_variance_ratio_[1]*100:.1f}% var)', color='#94a3b8')
    ax_p.set_title('Student Clusters (PCA 2D)', color='#f1f5f9', fontsize=11)
    ax_p.tick_params(colors='#64748b')
    ax_p.spines[:].set_color('#1e3a5f')
    ax_p.legend(fontsize=8, facecolor='#0f172a', labelcolor='#94a3b8', loc='best')

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Row 3: Cluster Profile Heatmap ──
    profile = cluster_data['profile'].drop(columns=['Overall_Avg'], errors='ignore')
    if len(profile.columns) > 0:
        fig3, ax_h = plt.subplots(figsize=(max(10, len(course_cols)*0.7), max(4, best_k*0.9)))
        fig3.patch.set_facecolor('#0f172a')
        ax_h.set_facecolor('#0f172a')

        data_arr = profile[course_cols].values if all(c in profile.columns for c in course_cols) else profile.values
        im = ax_h.imshow(data_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        cluster_labels = [f"C{cid}: {labels_map.get(cid,'?')}" for cid in profile.index]
        ax_h.set_yticks(range(len(profile)))
        ax_h.set_yticklabels(cluster_labels, color='#f1f5f9', fontsize=8)
        ax_h.set_xticks(range(len(profile.columns)))
        ax_h.set_xticklabels(profile.columns, rotation=45, ha='right', color='#94a3b8', fontsize=7)
        ax_h.spines[:].set_color('#1e3a5f')

        for i in range(data_arr.shape[0]):
            for j in range(data_arr.shape[1]):
                ax_h.text(j, i, f'{data_arr[i,j]:.0f}',
                          ha='center', va='center', fontsize=7,
                          color='white' if data_arr[i,j] < 50 else '#0f172a')

        plt.colorbar(im, ax=ax_h, label='Average Grade')
        ax_h.set_title('Cluster Profile Heatmap (avg grade per course)', color='#f1f5f9', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ══════════════════════════════════════════════════
# NEW — RECOMMENDATION SYSTEM
# ══════════════════════════════════════════════════

# Career profiles — based on skill demand from typical job market
CAREER_PROFILES = {
    "Data Scientist": {
        "skills":      ["python", "machine learning", "sql", "pandas", "tensorflow", "data science", "numpy", "scikit-learn", "tableau"],
        "description": "Builds ML models, analyzes data, creates insights",
        "emoji":       "📊",
        "courses_kw":  ["data", "statistic", "python", "ml", "algorithm", "math", "database"],
    },
    "ML/AI Engineer": {
        "skills":      ["python", "tensorflow", "pytorch", "deep learning", "machine learning", "nlp", "computer vision", "keras", "aws"],
        "description": "Deploys and optimizes ML models in production",
        "emoji":       "🤖",
        "courses_kw":  ["machine", "learning", "neural", "ai", "algorithm", "python", "deep"],
    },
    "Backend Developer": {
        "skills":      ["java", "python", "sql", "rest api", "docker", "microservices", "spring", "postgresql", "redis"],
        "description": "Builds server-side APIs and scalable backend systems",
        "emoji":       "⚙️",
        "courses_kw":  ["java", "database", "network", "os", "algorithm", "software", "api"],
    },
    "Full Stack Developer": {
        "skills":      ["javascript", "react", "node.js", "sql", "html5", "css3", "rest api", "typescript", "mongodb"],
        "description": "Develops both frontend and backend web applications",
        "emoji":       "🌐",
        "courses_kw":  ["web", "javascript", "database", "frontend", "backend", "html", "css"],
    },
    "DevOps / Cloud Engineer": {
        "skills":      ["aws", "docker", "kubernetes", "linux", "ci/cd", "terraform", "azure", "git", "ansible"],
        "description": "Manages infrastructure, deployment pipelines, and cloud platforms",
        "emoji":       "☁️",
        "courses_kw":  ["network", "os", "linux", "cloud", "system", "security", "infra"],
    },
    "Cybersecurity Analyst": {
        "skills":      ["cybersecurity", "network security", "penetration testing", "linux", "encryption", "firewall", "ethical hacking", "siem"],
        "description": "Protects systems from threats and vulnerabilities",
        "emoji":       "🔐",
        "courses_kw":  ["security", "network", "crypto", "os", "algorithm", "forensic", "ethical"],
    },
    "Mobile Developer": {
        "skills":      ["kotlin", "swift", "flutter", "android", "ios", "react native", "firebase", "rest api"],
        "description": "Builds mobile applications for Android and iOS platforms",
        "emoji":       "📱",
        "courses_kw":  ["mobile", "android", "ios", "java", "kotlin", "swift", "ui"],
    },
}


def recommend_for_student(
    student_row: pd.Series,
    course_cols: list,
    top30_skills: list,
    cluster_label: str,
    cluster_profile: pd.DataFrame,
    weak_t: float = 60,
    avg_t: float = 75,
) -> dict:
    """
    Content-Based Recommendation System
    ─────────────────────────────────────
    1. Map course names → skills using keyword matching
    2. Score each career profile against student grades
    3. Recommend top 3 career paths
    4. Recommend courses to strengthen
    5. Identify market skills the student is positioned for
    """

    grades = student_row[course_cols]
    norm   = grades / 100.0  # normalize 0-1

    # ── Map courses → inferred skills (simple keyword match) ──
    def course_to_skills(course_name: str) -> list:
        cn = course_name.lower()
        skill_keywords = {
            "python":      ["python"],
            "java":        ["java"],
            "javascript":  ["javascript", "js"],
            "sql":         ["sql", "database", "db"],
            "machine learning": ["machine learning", "ml", "ai"],
            "deep learning":    ["deep learning", "neural"],
            "networking":       ["network", "tcp", "routing"],
            "security":         ["security", "cyber", "cryptograph"],
            "web":              ["web", "html", "css", "react"],
            "mobile":           ["mobile", "android", "ios"],
            "data":             ["data", "statistic", "analytics"],
            "cloud":            ["cloud", "aws", "azure", "devops"],
            "os":               ["operating", "linux", "system"],
            "algorithm":        ["algorithm", "data structure"],
            "software":         ["software engineering", "agile", "scrum"],
        }
        matched = []
        for skill, keywords in skill_keywords.items():
            if any(kw in cn for kw in keywords):
                matched.append(skill)
        return matched if matched else [cn.split()[0]]

    student_skills = {}
    for c in course_cols:
        s = course_to_skills(c)
        g = float(grades[c])
        for sk in s:
            student_skills[sk] = max(student_skills.get(sk, 0), g)

    # ── Score each career path ──
    career_scores = {}
    for career, profile in CAREER_PROFILES.items():
        score    = 0.0
        matched  = 0
        total    = len(profile['courses_kw'])
        for kw in profile['courses_kw']:
            # direct match in student skills
            for sk, grade in student_skills.items():
                if kw in sk or sk in kw:
                    score   += grade / 100.0
                    matched += 1
                    break
        career_scores[career] = (score / total) if total > 0 else 0.0

    top_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # ── Courses to strengthen ──
    weak_courses   = [(c, float(grades[c])) for c in course_cols if grades[c] < weak_t]
    medium_courses = [(c, float(grades[c])) for c in course_cols if weak_t <= grades[c] < avg_t]
    strong_courses = [(c, float(grades[c])) for c in course_cols if grades[c] >= avg_t]

    weak_courses.sort(key=lambda x: x[1])
    medium_courses.sort(key=lambda x: x[1])
    strong_courses.sort(key=lambda x: x[1], reverse=True)

    # ── Market skills alignment ──
    top_market_skills = [s for s, _ in top30_skills[:15]]
    aligned_skills   = []
    gap_skills       = []

    for msk in top_market_skills:
        msk_l = msk.lower()
        covered = any(
            msk_l in c.lower() or c.lower() in msk_l or
            any(kw in c.lower() for kw in msk_l.split()[:2])
            for c in strong_courses
        )
        if covered:
            aligned_skills.append(msk)
        else:
            gap_skills.append(msk)

    # ── Peer comparison (vs cluster average) ──
    student_avg = float(grades.mean())
    try:
        cluster_avg = float(cluster_profile.loc[
            cluster_profile.index[0], 'Overall_Avg'
        ] if 'Overall_Avg' in cluster_profile.columns else
        cluster_profile.mean().mean())
    except Exception:
        cluster_avg = student_avg

    return {
        'top_careers':     top_careers,
        'career_scores':   career_scores,
        'weak_courses':    weak_courses[:3],
        'medium_courses':  medium_courses[:3],
        'strong_courses':  strong_courses[:3],
        'aligned_skills':  aligned_skills[:5],
        'gap_skills':      gap_skills[:5],
        'student_avg':     student_avg,
        'cluster_label':   cluster_label,
        'cluster_avg':     cluster_avg,
    }


def render_student_recommendation(rec: dict, student_id: str):
    """Render recommendation card for a single student."""

    st.markdown(f"#### 🎯 Recommendations for Student: `{student_id}`")

    # Career paths
    c1, c2, c3 = st.columns(3)
    for col, (career, score) in zip([c1, c2, c3], rec['top_careers']):
        profile = CAREER_PROFILES[career]
        pct     = int(score * 100)
        bar     = "█" * (pct // 10) + "░" * (10 - pct // 10)
        col.markdown(f"""
<div class="cluster-card">
<div style="font-size:1.5rem">{profile['emoji']}</div>
<div style="color:#f1f5f9; font-weight:600; font-size:.9rem">{career}</div>
<div style="color:#3b82f6; font-family:monospace; font-size:.8rem">{bar} {pct}%</div>
<div style="color:#64748b; font-size:.75rem; margin-top:.3rem">{profile['description']}</div>
</div>
""", unsafe_allow_html=True)

    # Courses
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**🔴 Courses to Strengthen**")
        if rec['weak_courses']:
            for c, g in rec['weak_courses']:
                st.markdown(f'<div class="box-red"><b>{c}</b> — {g:.0f}/100</div>', unsafe_allow_html=True)
        else:
            st.success("No weak courses!")

    with cols[1]:
        st.markdown("**🟡 Courses to Improve**")
        if rec['medium_courses']:
            for c, g in rec['medium_courses']:
                st.markdown(f'<div class="box-amber"><b>{c}</b> — {g:.0f}/100</div>', unsafe_allow_html=True)
        else:
            st.success("All above average!")

    with cols[2]:
        st.markdown("**🟢 Your Strengths**")
        if rec['strong_courses']:
            for c, g in rec['strong_courses']:
                st.markdown(f'<div class="box-green"><b>{c}</b> — {g:.0f}/100 ✓</div>', unsafe_allow_html=True)

    # Market alignment
    col_a, col_g = st.columns(2)
    with col_a:
        st.markdown("**💼 Market Skills You're Ready For**")
        if rec['aligned_skills']:
            for s in rec['aligned_skills']:
                st.markdown(f'<div class="box-green">✅ {s}</div>', unsafe_allow_html=True)
        else:
            st.info("Focus on strengthening weak courses first.")

    with col_g:
        st.markdown("**📚 Market Skills to Develop**")
        if rec['gap_skills']:
            for s in rec['gap_skills']:
                st.markdown(f'<div class="box-amber">📌 {s}</div>', unsafe_allow_html=True)

    # Peer comparison
    diff = rec['student_avg'] - rec['cluster_avg']
    sign = "+" if diff >= 0 else ""
    color = "box-green" if diff >= 0 else "box-red"
    st.markdown(f"""
<div class="{color}">
📊 <b>Peer Comparison:</b> Your average = <b>{rec['student_avg']:.1f}</b> |
Cluster average = <b>{rec['cluster_avg']:.1f}</b> |
Difference = <b>{sign}{diff:.1f}</b> |
Group: <b>{rec['cluster_label']}</b>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Student Performance",
    "💼 Job Market NLP",
    "📋 Combined Report",
])


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — STUDENT PERFORMANCE + CLUSTERING + RECS   ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<p class="sec">01 · Student Performance — Random Forest + KMeans Clustering + Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge-independent">✦ INDEPENDENT — No other tab required</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box-blue">
    <b>v3 additions:</b> KMeans Clustering identifies natural student groups (no manual thresholds) ·
    PCA visualization · Recommendation System suggests career paths + courses to strengthen per student.
    </div>
    """, unsafe_allow_html=True)

    up_stu = st.file_uploader("📂 Upload Student Grades CSV", type="csv", key="stu")

    if up_stu:
        df_raw = pd.read_csv(up_stu)
        st.success(f"✅ {len(df_raw)} students · {len(df_raw.columns)} columns")

        with st.expander("👁 Preview"):
            st.dataframe(df_raw.head())

        st.markdown("---")
        numeric_cols = df_raw.select_dtypes(include='number').columns.tolist()
        skip_kw = ['id','rank','student','semester','final','total','gpa','grade','index']
        auto_courses = [c for c in numeric_cols if not any(k in c.lower() for k in skip_kw)]

        c1, c2 = st.columns(2)
        with c1:
            course_cols = st.multiselect("Course columns (features):", numeric_cols, default=auto_courses)
        with c2:
            final_col = st.selectbox(
                "Final grade column (target):", numeric_cols,
                index=numeric_cols.index('Final_Grade') if 'Final_Grade' in numeric_cols else len(numeric_cols)-1
            )

        t1, t2 = st.columns(2)
        with t1: weak_t = st.slider("Weak below:", 30, 70, 60)
        with t2: avg_t  = st.slider("Average below:", 65, 90, 75)

        if st.button("🔍 Run Full ML Analysis (RF + Clustering)", key="b1"):
            if len(course_cols) < 2:
                st.error("Need ≥2 course columns.")
                st.stop()

            df = df_raw.copy()

            # ── Descriptive stats ──
            avgs      = df[course_cols].mean().sort_values()
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

            # ── Stage 5a: Random Forest ──
            X = df[course_cols]; y_raw = df[final_col]
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
            classes  = sorted(y_tr.unique())
            auc_val  = compute_auc(rf, X_te.values, y_te, classes)

            # ── Stage 6: SHAP ──
            with st.spinner("Computing SHAP values..."):
                shap_vals = compute_shap(rf, X_tr.values, X_te.values, course_cols)

            # ── Stage 5b: NEW — KMeans Clustering ──
            with st.spinner("Running KMeans Clustering..."):
                cluster_data = run_student_clustering(df, course_cols, max_k=max_k)

            df = cluster_data['df']

            # ── KPI cards ──
            st.markdown("---")
            m1,m2,m3,m4,m5 = st.columns(5)
            for col,(l,v) in zip([m1,m2,m3,m4,m5],[
                ("Students",     len(df)),
                ("🔴 Weak",       f"{counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.0f}%)"),
                ("🟡 Average",    f"{counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.0f}%)"),
                ("🟢 Excellent",  f"{counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.0f}%)"),
                ("Clusters",     cluster_data['best_k']),
            ]):
                col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # ── Charts ──
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Course Average Scores**")
                bc = ['#dc2626' if v < weak_t else '#d97706' if v < avg_t else '#16a34a' for v in avgs.values]
                fig, ax = plt.subplots(figsize=(8, max(5, len(course_cols) * .45)))
                fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#0f172a')
                bars = ax.barh(avgs.index, avgs.values, color=bc, height=.6)
                for b, v in zip(bars, avgs.values):
                    ax.text(v+.5, b.get_y()+b.get_height()/2, f'{v:.1f}',
                            va='center', fontsize=8, color='#94a3b8')
                ax.axvline(avgs.mean(), color='#3b82f6', linestyle='--', lw=1.2)
                ax.set_xlim(0, 110)
                ax.tick_params(colors='#94a3b8', labelsize=8)
                ax.spines[:].set_color('#1e3a5f')
                r_p = mpatches.Patch(color='#dc2626', label=f'Weak (<{weak_t})')
                a_p = mpatches.Patch(color='#d97706', label=f'Mid ({weak_t}–{avg_t})')
                g_p = mpatches.Patch(color='#16a34a', label=f'Good (>{avg_t})')
                ax.legend(handles=[r_p,a_p,g_p], fontsize=7, facecolor='#0f172a', labelcolor='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with ch2:
                st.markdown("**Level Distribution**")
                fig2, ax2 = plt.subplots(figsize=(5,5))
                fig2.patch.set_facecolor('#0f172a'); ax2.set_facecolor('#0f172a')
                lc = {'Weak':'#dc2626','Average':'#d97706','Excellent':'#16a34a'}
                wedges, texts, auto = ax2.pie(
                    counts.values, labels=counts.index,
                    colors=[lc.get(l,'#3b82f6') for l in counts.index],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor':'#0f172a','linewidth':2}
                )
                for t in texts: t.set_color('#94a3b8')
                for a in auto:  a.set_color('white'); a.set_fontsize(9)
                plt.tight_layout(); st.pyplot(fig2); plt.close()

            # ── Course status boxes ──
            st.markdown("---")
            cw, cm, cg = st.columns(3)
            with cw:
                st.markdown("**🔴 Weak**")
                if len(weak_c):
                    for c, v in weak_c.items():
                        st.markdown(f'<div class="box-red"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.success("None")
            with cm:
                st.markdown("**🟡 Medium**")
                if len(mid_c):
                    for c, v in mid_c.items():
                        st.markdown(f'<div class="box-amber"><b>{c}</b><br>Avg:{v:.1f} · Fail:{fail_rate[c]:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.success("None")
            with cg:
                st.markdown("**🟢 Good**")
                for c, v in good_c.items():
                    st.markdown(f'<div class="box-green"><b>{c}</b> · Avg:{v:.1f}</div>', unsafe_allow_html=True)

            # ── ML Metrics ──
            st.markdown("---")
            st.markdown("#### 🤖 Random Forest Metrics — Stage 5a & 6")
            r1,r2,r3,r4,r5,r6 = st.columns(6)
            auc_display = f"{auc_val*100:.1f}%" if auc_val else "N/A"
            for col,(l,v) in zip([r1,r2,r3,r4,r5,r6],[
                ("Accuracy",  f"{acc*100:.1f}%"),
                ("Precision", f"{prec*100:.1f}%"),
                ("Recall",    f"{rec*100:.1f}%"),
                ("F1-Score",  f"{f1*100:.1f}%"),
                ("CV Acc.",   f"{cv*100:.1f}%"),
                ("AUC-ROC",   auc_display),
            ]):
                col.markdown(f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Confusion matrix
            lbls = sorted(y_te.unique())
            cm_m = confusion_matrix(y_te, y_pr, labels=lbls)
            fig3, ax3 = plt.subplots(figsize=(5,4))
            fig3.patch.set_facecolor('#0f172a'); ax3.set_facecolor('#0f172a')
            ax3.imshow(cm_m, cmap='Blues')
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    clr = 'white' if cm_m[i,j] > cm_m.max()/2 else '#94a3b8'
                    ax3.text(j, i, str(cm_m[i,j]), ha='center', va='center',
                             fontsize=13, fontweight='bold', color=clr)
            ax3.set_xticks(range(len(lbls))); ax3.set_xticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_yticks(range(len(lbls))); ax3.set_yticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_xlabel('Predicted', color='#94a3b8')
            ax3.set_ylabel('Actual',    color='#94a3b8')
            ax3.spines[:].set_color('#1e3a5f')
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            # Feature Importance vs SHAP
            fi_col, shap_col = st.columns(2)
            with fi_col:
                st.markdown("**Feature Importance (Stage 5a)**")
                top_imp = imps.sort_values(ascending=True)
                fig4, ax4 = plt.subplots(figsize=(7, max(4, len(course_cols)*.4)))
                fig4.patch.set_facecolor('#0f172a'); ax4.set_facecolor('#0f172a')
                ic = ['#3b82f6' if v > imps.mean() else '#334155' for v in top_imp.values]
                ax4.barh(top_imp.index, top_imp.values, color=ic, height=.6)
                ax4.axvline(imps.mean(), color='#d97706', linestyle='--', lw=1)
                ax4.tick_params(colors='#94a3b8', labelsize=8)
                ax4.spines[:].set_color('#1e3a5f')
                ax4.set_title('RF Feature Importance', color='#94a3b8', fontsize=9)
                plt.tight_layout(); st.pyplot(fig4); plt.close()

            with shap_col:
                st.markdown("**SHAP Values (Stage 6)**")
                if shap_vals is not None:
                    shap_sorted = shap_vals.sort_values(ascending=True)
                    fig_shap, ax_shap = plt.subplots(figsize=(7, max(4, len(course_cols)*.4)))
                    fig_shap.patch.set_facecolor('#0f172a'); ax_shap.set_facecolor('#0f172a')
                    sc = ['#7c3aed' if v > shap_vals.mean() else '#334155' for v in shap_sorted.values]
                    ax_shap.barh(shap_sorted.index, shap_sorted.values, color=sc, height=.6)
                    ax_shap.axvline(shap_vals.mean(), color='#d97706', linestyle='--', lw=1)
                    ax_shap.tick_params(colors='#94a3b8', labelsize=8)
                    ax_shap.spines[:].set_color('#1e3a5f')
                    ax_shap.set_title('Mean |SHAP| per Course', color='#94a3b8', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig_shap); plt.close()
                else:
                    st.info("SHAP skipped.")

            # ═══════════════════════════════════════
            # NEW — CLUSTERING SECTION
            # ═══════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🔵 Stage 5b — KMeans Student Clustering")
            st.markdown(f"""
<div class="box-purple">
<b>KMeans found {cluster_data['best_k']} natural student groups</b> (selected by max Silhouette Score).<br>
Unlike manual thresholds, clustering discovers hidden patterns in how students perform across courses together.
</div>
""", unsafe_allow_html=True)

            # Cluster summary cards
            for cid, label in cluster_data['labels_map'].items():
                n_stu   = (df['Cluster'] == cid).sum()
                avg_all = float(df[df['Cluster'] == cid][course_cols].mean().mean())
                best_c  = df[df['Cluster'] == cid][course_cols].mean().idxmax()
                worst_c = df[df['Cluster'] == cid][course_cols].mean().idxmin()
                st.markdown(f"""
<div class="cluster-card">
<b style="color:#f1f5f9">{label}</b> &nbsp;·&nbsp;
<span style="color:#64748b">{n_stu} students ({n_stu/len(df)*100:.0f}%)</span> &nbsp;·&nbsp;
<span style="color:#3b82f6">Avg: {avg_all:.1f}</span> &nbsp;·&nbsp;
<span style="color:#16a34a">Best: {best_c}</span> &nbsp;·&nbsp;
<span style="color:#dc2626">Weakest: {worst_c}</span>
</div>
""", unsafe_allow_html=True)

            # Clustering plots (elbow, silhouette, PCA, heatmap)
            plot_clustering_results(cluster_data, course_cols)

            # Cluster breakdown expander
            with st.expander("📊 Full Cluster Profile Table"):
                profile_display = cluster_data['profile'].copy()
                profile_display.index = [cluster_data['labels_map'].get(i, f'Cluster {i}') for i in profile_display.index]
                st.dataframe(profile_display.round(1), use_container_width=True)

            # ═══════════════════════════════════════
            # NEW — RECOMMENDATION SYSTEM SECTION
            # ═══════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🟢 Stage 6b — Individual Student Recommendation System")
            st.markdown("""
<div class="box-teal">
Select any student to get a personalised recommendation: career path alignment,
courses to strengthen, market skills they're ready for, and peer comparison.
</div>
""", unsafe_allow_html=True)

            # Student selector
            # Try to find a student ID column
            id_candidates = [c for c in df_raw.columns if any(k in c.lower() for k in ['id','name','student','no','num'])]
            if id_candidates:
                id_col  = id_candidates[0]
                stu_ids = df_raw[id_col].astype(str).tolist()
            else:
                id_col  = None
                stu_ids = [f"Student {i+1}" for i in range(len(df))]

            selected_stu = st.selectbox("🔍 Select a student:", stu_ids, key="stu_select")
            sel_idx = stu_ids.index(selected_stu)
            student_row  = df.iloc[sel_idx]

            # Get cluster info for this student
            stu_cluster  = int(student_row['Cluster'])
            stu_label    = cluster_data['labels_map'].get(stu_cluster, f'Cluster {stu_cluster}')
            stu_profile  = cluster_data['profile']

            # Get top30 from session if available, else empty
            top30_for_rec = st.session_state.get('top30', [])

            rec = recommend_for_student(
                student_row   = student_row,
                course_cols   = course_cols,
                top30_skills  = top30_for_rec,
                cluster_label = stu_label,
                cluster_profile = stu_profile,
                weak_t        = weak_t,
                avg_t         = avg_t,
            )

            render_student_recommendation(rec, selected_stu)

            # Batch recommendations summary
            with st.expander("📋 All Students — Cluster + Career Path Summary"):
                rows = []
                for i, row in df.iterrows():
                    cid   = int(row['Cluster'])
                    label = cluster_data['labels_map'].get(cid, f'C{cid}')
                    stu_id_val = str(df_raw.iloc[i][id_col]) if id_col else f"Student {i+1}"

                    r = recommend_for_student(
                        student_row   = row,
                        course_cols   = course_cols,
                        top30_skills  = top30_for_rec,
                        cluster_label = label,
                        cluster_profile = stu_profile,
                        weak_t        = weak_t,
                        avg_t         = avg_t,
                    )
                    top_career = r['top_careers'][0][0] if r['top_careers'] else "—"
                    rows.append({
                        'Student':         stu_id_val,
                        'Cluster':         label,
                        'Avg Grade':       round(r['student_avg'], 1),
                        'Top Career Path': top_career,
                        'Weak Courses':    len(r['weak_courses']),
                        'Strong Courses':  len(r['strong_courses']),
                        'RF Level':        row.get('Level', '—'),
                    })
                summary_df = pd.DataFrame(rows)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                csv_out = summary_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Student Summary CSV",
                    data=csv_out,
                    file_name="student_cluster_recommendations.csv",
                    mime="text/csv"
                )

            # ── Save to session ──
            st.session_state.update({
                'avgs': avgs, 'weak_c': weak_c, 'mid_c': mid_c, 'good_c': good_c,
                'counts': counts, 'n_stu': len(df), 'n_tr': len(X_tr), 'n_te': len(X_te),
                'met': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cv': cv, 'auc': auc_val},
                'imps': imps, 'shap_vals': shap_vals,
                'fr': fail_rate, 'wt': weak_t, 'at': avg_t, 'cc': course_cols,
                'cluster_data': cluster_data,
            })

            # ── AI Report ──
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Student Performance + Clustering")
            st.markdown('<div class="box-purple">AI report now includes cluster insights + per-cluster recommendations. No job market data required.</div>', unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Student Performance Report", key="b1_report"):
                    with st.spinner("Generating report..."):

                        weak_detail_str = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fail_rate.get(c,0):.1f}%, rf_importance={imps.get(c,0):.3f}"
                            for c, v in weak_c.items()
                        ]) or "None"

                        cluster_summary = "\n".join([
                            f"- {cluster_data['labels_map'].get(cid,'?')}: "
                            f"{(df['Cluster']==cid).sum()} students, "
                            f"avg={float(df[df['Cluster']==cid][course_cols].mean().mean()):.1f}"
                            for cid in sorted(cluster_data['labels_map'].keys())
                        ])

                        prompt_student = f"""You are a senior academic consultant writing a formal report for the Head of a Computer Science Department.

=== STUDENT DATA ({len(df)} students) ===
Random Forest: Accuracy={acc*100:.1f}% | F1={f1*100:.1f}% | AUC={f"{auc_val*100:.1f}%" if auc_val else "N/A"}
Thresholds: Weak<{weak_t} | Average {weak_t}–{avg_t} | Excellent>{avg_t}
Distribution: Weak={counts.get('Weak',0)} | Average={counts.get('Average',0)} | Excellent={counts.get('Excellent',0)}

KMEANS CLUSTERING (best k={cluster_data['best_k']} by silhouette score):
{cluster_summary}

WEAK COURSES:
{weak_detail_str}

TOP 5 MOST INFLUENTIAL (RF):
{chr(10).join([f"- {c}: {v:.3f}" for c, v in imps.head(5).items()])}

ALL COURSES:
{', '.join(course_cols)}

=== REPORT STRUCTURE ===

## 1. Executive Summary
Key numbers: student performance + what clustering revealed about hidden student groups.

## 2. Cluster Analysis
For each cluster: who are these students, what do they struggle with, what's the intervention strategy.

## 3. Per-Course Analysis & Recommendations
For EACH weak and medium course:
### [Course Name]
- Diagnosis, 3 teaching improvements, resources (YouTube + platform + textbook), priority.

## 4. Recommendation System Insights
What career paths are most of the weak students positioned for? What should they focus on?

## 5. Priority Action Plan
Numbered, most urgent first."""

                        result = call_groq(groq_key, prompt_student, max_tokens=2500)
                        st.markdown(result)

                        report_txt = (
                            f"CS DEPARTMENT — STUDENT PERFORMANCE REPORT v3\n{'='*60}\n\n"
                            f"Students={len(df)} | RF F1={f1*100:.1f}% | Clusters={cluster_data['best_k']}\n\n"
                            f"CLUSTER SUMMARY:\n{cluster_summary}\n\n"
                            f"WEAK COURSES:\n{weak_detail_str}\n\n"
                            f"{'='*60}\nAI RECOMMENDATIONS\n{'='*60}\n\n{result}"
                        )
                        st.download_button(
                            "📥 Download Student Report",
                            data=report_txt,
                            file_name="student_performance_report_v3.txt",
                            mime="text/plain"
                        )
    else:
        st.info("📂 Upload any student grades CSV to begin.")


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — JOB MARKET NLP: TF-IDF + BERT             ║
# ╚══════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<p class="sec">02 · Job Market — TF-IDF + BERT Semantic Embeddings + Independent AI Report</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge-independent">✦ INDEPENDENT — No other tab required</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box-blue">
    <b>v3 NLP pipeline:</b> TF-IDF (lexical matching) <b>+</b> BERT Semantic Similarity (contextual understanding).<br>
    BERT understands that "ML engineer" and "machine learning" are the same concept — TF-IDF doesn't.
    </div>
    """, unsafe_allow_html=True)

    up_job = st.file_uploader("📂 Upload Job Listings CSV", type="csv", key="job")

    if up_job:
        with st.spinner("Loading..."):
            jobs_df = pd.read_csv(up_job, on_bad_lines='skip')
        st.success(f"✅ {len(jobs_df)} job listings")

        text_cols = jobs_df.select_dtypes(include='object').columns.tolist()
        desc_col = st.selectbox(
            "Job description column:", text_cols,
            index=text_cols.index('jobdescription') if 'jobdescription' in text_cols else 0
        )
        extra = st.text_input(
            "Extra skills to track (comma-separated):",
            placeholder="rust, llm, chatgpt, cybersecurity"
        )

        if st.button("🔍 Run NLP Analysis (TF-IDF + BERT)", key="b2"):
            with st.spinner("Running TF-IDF NLP pipeline..."):

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
                        "network administration","cisco","wireshark",
                        "network protocols","load balancing"
                    ],
                    "Emerging Tech": [
                        "blockchain","iot","embedded systems","computer vision",
                        "robotics","large language model","generative ai","ar/vr",
                        "edge computing","5g"
                    ]
                }

                if extra.strip():
                    TAXONOMY["Custom"] = [s.strip().lower() for s in extra.split(',') if s.strip()]

                raw_texts     = jobs_df[desc_col].dropna().tolist()
                cleaned_texts = [clean_text(t) for t in raw_texts]
                clean_texts   = [preprocess_text(t) for t in raw_texts]

                # Stage 4a: TF-IDF
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=2, max_df=0.95,
                    sublinear_tf=True,
                )
                tfidf_matrix  = vectorizer.fit_transform(clean_texts)
                feature_names = vectorizer.get_feature_names_out()
                tfidf_sums    = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                tfidf_dict    = dict(zip(feature_names, tfidf_sums))

                all_text = ' '.join(clean_texts)
                skill_results = {}
                all_skills_flat = [s for skills in TAXONOMY.values() for s in skills]

                for cat, skills in TAXONOMY.items():
                    for skill in skills:
                        sc    = clean_text(skill)
                        freq  = all_text.count(sc)
                        tscore= tfidf_dict.get(sc, 0.0)
                        jw    = sum(1 for t in clean_texts if sc in t)
                        cov   = jw / len(clean_texts) * 100
                        if freq > 0:
                            skill_results[skill] = {
                                'category':     cat,
                                'freq':         freq,
                                'tfidf_score':  round(float(tscore), 2),
                                'job_coverage': round(cov, 1),
                                'jobs_with':    jw,
                                'bert_avg_sim':  None,
                                'bert_coverage': None,
                            }

                sorted_by_tfidf = sorted(
                    skill_results.items(),
                    key=lambda x: x[1]['tfidf_score'], reverse=True
                )
                top30 = sorted_by_tfidf[:30]

            # Stage 4b: BERT (if enabled)
            bert_results = {}
            bert_model   = None
            bert_status  = "disabled"

            if bert_enabled:
                with st.spinner("🔵 Loading BERT model (first time may take ~30s)..."):
                    bert_model = load_bert_model(bert_model_name)

                if bert_model is not None:
                    with st.spinner(f"🔵 Computing BERT semantic similarity for {len(skill_results)} skills..."):
                        # Sample for speed (BERT is slow on large datasets)
                        sample_texts = clean_texts[:min(500, len(clean_texts))]
                        bert_results = bert_skill_scores(bert_model, sample_texts, list(skill_results.keys()))

                    # Merge BERT scores into skill_results
                    for skill, bdata in bert_results.items():
                        if skill in skill_results:
                            skill_results[skill]['bert_avg_sim']  = round(bdata['bert_avg_sim'], 4)
                            skill_results[skill]['bert_coverage'] = round(bdata['bert_coverage'], 1)

                    bert_status = f"✅ {bert_model_name} · {len(bert_results)} skills scored"
                else:
                    bert_status = "⚠️ sentence-transformers not installed — TF-IDF only"
            else:
                bert_status = "Disabled in sidebar"

            # ── BERT status ──
            if "✅" in bert_status:
                st.markdown(f'<div class="box-blue">🔵 <b>BERT Status:</b> {bert_status}</div>', unsafe_allow_html=True)
            elif "⚠️" in bert_status:
                st.markdown(f'<div class="box-amber">🔵 <b>BERT Status:</b> {bert_status}<br>Install: <code>pip install sentence-transformers</code></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="box-amber">🔵 <b>BERT:</b> {bert_status}</div>', unsafe_allow_html=True)

            # ── Pipeline explanation ──
            with st.expander("🔬 Pipeline Stages 2 → 4b"):
                st.markdown(f"""
| Stage | What happened | Detail |
|-------|--------------|--------|
| **Stage 2: Cleanup**    | URLs · Punctuation · Whitespace   | `{len(raw_texts):,}` texts |
| **Stage 3: Pre-process**| Tokenization · Stop word removal  | `{len(STOP_WORDS)}` stop words |
| **Stage 4a: TF-IDF**   | ngram=(1,3) · sublinear_tf · min_df=2 | Matrix: `{len(raw_texts):,}` docs |
| **Stage 4b: BERT**     | Sentence-BERT cosine similarity   | `{bert_model_name}` · threshold=0.45 |

**TF-IDF vs BERT:**
| | TF-IDF | BERT |
|--|--------|------|
| Understands context | ❌ word counting | ✅ semantic meaning |
| Speed | ✅ instant | ⚠️ slower |
| "ML" = "machine learning" | ❌ | ✅ |
| Best for | exact keyword frequency | semantic similarity |
| Combined → | **hybrid scoring** gives the most complete picture |
                """)

            # ── Charts ──
            CAT_COLORS = {
                'Programming Languages':'#3b82f6','Web & Mobile':'#16a34a',
                'Data Science & AI':'#7c3aed','Cloud & DevOps':'#d97706',
                'Cybersecurity':'#dc2626','Databases':'#0891b2',
                'Software Engineering':'#059669','Networking':'#9333ea',
                'Emerging Tech':'#f59e0b','Custom':'#ec4899'
            }

            # Combined TF-IDF + BERT chart
            has_bert = any(v['bert_avg_sim'] is not None for v in skill_results.values())

            names  = [s[0] for s in top30[:20]]
            scores = [s[1]['tfidf_score'] for s in top30[:20]]
            covs   = [s[1]['job_coverage'] for s in top30[:20]]
            cats   = [s[1]['category'] for s in top30[:20]]
            bclrs  = [CAT_COLORS.get(c,'#64748b') for c in cats]

            if has_bert:
                bert_sims = [skill_results[s[0]].get('bert_avg_sim') or 0 for s in top30[:20]]
                bert_covs = [skill_results[s[0]].get('bert_coverage') or 0 for s in top30[:20]]

                fig5, (ax5, ax_bert) = plt.subplots(1, 2, figsize=(16, 8))
                fig5.patch.set_facecolor('#0f172a')

                # TF-IDF
                ax5.set_facecolor('#0f172a')
                bars5 = ax5.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars5, scores, covs):
                    ax5.text(v+.1, b.get_y()+b.get_height()/2,
                             f'{v:.1f} | {cov:.0f}%', va='center', fontsize=7, color='#94a3b8')
                ax5.tick_params(colors='#94a3b8', labelsize=8)
                ax5.spines[:].set_color('#1e3a5f')
                ax5.set_xlabel('TF-IDF Score', color='#94a3b8')
                ax5.set_title('TF-IDF Scores (Lexical)', color='#f1f5f9', fontsize=10)

                # BERT
                ax_bert.set_facecolor('#0f172a')
                bert_clrs = ['#7c3aed' if v > np.mean(bert_sims) else '#334155' for v in bert_sims]
                bars_bert = ax_bert.barh(names, bert_sims, color=bert_clrs, height=.6)
                for b, v, cov in zip(bars_bert, bert_sims, bert_covs):
                    ax_bert.text(v+.001, b.get_y()+b.get_height()/2,
                                 f'{v:.3f} | {cov:.0f}%', va='center', fontsize=7, color='#94a3b8')
                ax_bert.tick_params(colors='#94a3b8', labelsize=8)
                ax_bert.spines[:].set_color('#1e3a5f')
                ax_bert.set_xlabel('BERT Avg Similarity', color='#94a3b8')
                ax_bert.set_title('BERT Semantic Similarity', color='#f1f5f9', fontsize=10)

                plt.suptitle('Top 20 Skills — TF-IDF vs BERT', color='#f1f5f9', fontsize=12)
                plt.tight_layout(); st.pyplot(fig5); plt.close()

                # ── BERT vs TF-IDF ranking comparison ──
                st.markdown("#### 🔵 TF-IDF vs BERT — Ranking Comparison (top 15)")
                tfidf_ranks = {s: i+1 for i, (s, _) in enumerate(sorted_by_tfidf[:15])}
                bert_sorted = sorted(
                    [(s, d['bert_avg_sim']) for s, d in skill_results.items() if d['bert_avg_sim'] is not None],
                    key=lambda x: x[1], reverse=True
                )
                bert_ranks = {s: i+1 for i, (s, _) in enumerate(bert_sorted[:15])}

                cmp_rows = []
                for s, _ in sorted_by_tfidf[:15]:
                    tr = tfidf_ranks.get(s, '—')
                    br = bert_ranks.get(s, '—')
                    diff = (tr - br) if isinstance(tr, int) and isinstance(br, int) else 0
                    cmp_rows.append({
                        'Skill':          s,
                        'TF-IDF Rank':    tr,
                        'BERT Rank':      br,
                        'Rank Diff':      f"{'↑' if diff > 0 else '↓' if diff < 0 else '='}{abs(diff)}",
                        'Agreement':      '✅' if abs(diff) <= 3 else '⚠️ Diverge',
                        'TF-IDF Score':   skill_results[s]['tfidf_score'],
                        'BERT Sim':       skill_results[s]['bert_avg_sim'],
                    })
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

            else:
                # TF-IDF only chart
                fig5, ax5 = plt.subplots(figsize=(10, 8))
                fig5.patch.set_facecolor('#0f172a'); ax5.set_facecolor('#0f172a')
                bars5 = ax5.barh(names, scores, color=bclrs, height=.6)
                for b, v, cov in zip(bars5, scores, covs):
                    ax5.text(v+.1, b.get_y()+b.get_height()/2,
                             f'{v:.1f} | {cov:.0f}% jobs', va='center', fontsize=7, color='#94a3b8')
                ax5.tick_params(colors='#94a3b8', labelsize=8)
                ax5.spines[:].set_color('#1e3a5f')
                ax5.set_xlabel('TF-IDF Score', color='#94a3b8')
                ax5.set_title('Top 20 Skills by TF-IDF Score', color='#f1f5f9')
                handles = [mpatches.Patch(color=v, label=k) for k,v in CAT_COLORS.items() if k in cats]
                ax5.legend(handles=handles, fontsize=7, facecolor='#0f172a', labelcolor='#94a3b8', loc='lower right')
                plt.tight_layout(); st.pyplot(fig5); plt.close()

            # Stats
            s1,s2,s3,s4 = st.columns(4)
            for col,(l,v) in zip([s1,s2,s3,s4],[
                ("Total Jobs",   len(jobs_df)),
                ("Skills Found", len(skill_results)),
                ("Top Skill",    top30[0][0] if top30 else '—'),
                ("BERT Active",  "✅ Yes" if has_bert else "❌ No"),
            ]):
                col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Full skills table
            st.markdown("---")
            st.markdown("#### All Extracted Skills")
            skill_rows = []
            for s, d in sorted_by_tfidf:
                skill_rows.append({
                    'Skill':          s,
                    'Category':       d['category'],
                    'TF-IDF Score':   d['tfidf_score'],
                    'Job Coverage %': d['job_coverage'],
                    'Jobs Mentioning':d['jobs_with'],
                    'BERT Sim':       d.get('bert_avg_sim', '—'),
                    'BERT Coverage %':d.get('bert_coverage', '—'),
                    'Raw Freq.':      d['freq'],
                })
            skill_df = pd.DataFrame(skill_rows)
            st.dataframe(skill_df, use_container_width=True)

            # Category breakdown
            st.markdown("---")
            st.markdown("#### Demand by Category")
            cat_totals = {}
            for _, d in skill_results.items():
                cat_totals[d['category']] = cat_totals.get(d['category'], 0) + d['tfidf_score']
            total_all = sum(cat_totals.values()) or 1
            for cat, total in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
                pct = int(total/total_all*100)
                st.progress(pct, text=f"**{cat}** — TF-IDF total {total:.1f} ({pct}%)")

            # ── Save ──
            st.session_state.update({
                'top30': top30, 'skill_results': skill_results,
                'n_jobs': len(jobs_df), 'sorted_skills': sorted_by_tfidf,
                'cat_totals': cat_totals,
            })

            # ── AI Report ──
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Job Market Analysis")
            st.markdown('<div class="box-purple">Analysis based on TF-IDF + BERT data. No student grades needed.</div>', unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("📋 Generate Job Market Report", key="b2_report"):
                    with st.spinner("Generating..."):

                        bert_note = ""
                        if has_bert:
                            top3_bert = sorted(
                                [(s, d['bert_avg_sim']) for s, d in skill_results.items() if d['bert_avg_sim']],
                                key=lambda x: x[1], reverse=True
                            )[:5]
                            bert_note = f"\nTOP 5 BY BERT SEMANTIC SIMILARITY: {', '.join([f'{s}({v:.3f})' for s,v in top3_bert])}"

                        top20_str = "\n".join([
                            f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, coverage={d['job_coverage']:.1f}%"
                            + (f", BERT={d['bert_avg_sim']:.3f}" if d.get('bert_avg_sim') else "")
                            for s, d in top30[:20]
                        ])

                        cat_summary = "\n".join([
                            f"- {cat}: {total:.1f}"
                            for cat, total in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
                        ])

                        prompt_jobs = f"""You are a senior academic consultant writing a formal report for a CS Department Head.

=== JOB MARKET ({len(jobs_df)} listings, TF-IDF + BERT NLP) ===
TOP 20 SKILLS:
{top20_str}
{bert_note}

DEMAND BY CATEGORY:
{cat_summary}

=== REPORT ===

## 1. Market Overview
Key findings. What dominates? TF-IDF vs BERT — any divergence worth noting?

## 2. Top Skills Analysis
For each of the top 10 skills:
### [Skill] — TF-IDF: X | Coverage: X%
- Industry use, why in demand, teaching approach, 2 YouTube + 1 platform + 1 textbook

## 3. Recommended New Courses (4–5)
Each: name, skills covered, market evidence, year/semester, priority.

## 4. Curriculum Gaps
High TF-IDF/BERT skills missing from traditional curricula. Rank by urgency.

## 5. Action Plan"""

                        result_jobs = call_groq(groq_key, prompt_jobs, max_tokens=2500)
                        st.markdown(result_jobs)

                        st.download_button(
                            "📥 Download Job Market Report",
                            data=result_jobs,
                            file_name="job_market_report_v3.txt",
                            mime="text/plain"
                        )
    else:
        st.info("📂 Upload a job listings CSV to begin.")


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — COMBINED REPORT                            ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<p class="sec">03 · Combined Institutional Report — Student + Clustering + Job Market</p>', unsafe_allow_html=True)

    has_s = 'weak_c' in st.session_state
    has_j = 'top30'  in st.session_state

    col_s, col_j = st.columns(2)
    with col_s:
        if has_s:
            st.markdown('<div class="box-green">✅ Student data ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-red">⬜ Student data — complete Tab 1 first</div>', unsafe_allow_html=True)
    with col_j:
        if has_j:
            st.markdown('<div class="box-green">✅ Job market data ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-red">⬜ Job market data — complete Tab 2 first</div>', unsafe_allow_html=True)

    if not (has_s and has_j):
        st.info("Complete both Tab 1 and Tab 2 to unlock the combined report.")
        st.stop()

    # unpack
    wc           = st.session_state['weak_c']
    mc           = st.session_state['mid_c']
    top30        = st.session_state['top30']
    sr           = st.session_state['skill_results']
    cnt          = st.session_state['counts']
    ns           = st.session_state['n_stu']
    ntr          = st.session_state['n_tr']
    nte          = st.session_state['n_te']
    met          = st.session_state['met']
    imps         = st.session_state['imps']
    fr           = st.session_state['fr']
    nj           = st.session_state['n_jobs']
    wt           = st.session_state['wt']
    at           = st.session_state['at']
    cc           = st.session_state['cc']
    cluster_data = st.session_state.get('cluster_data', None)
    cat_totals   = st.session_state.get('cat_totals', {})

    weak_detail = "\n".join([
        f"- {c}: avg={v:.1f}, fail_rate={fr.get(c,0):.1f}%, rf_importance={imps.get(c,0):.3f}"
        for c,v in wc.items()
    ]) or "None"

    top20_skills = "\n".join([
        f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, coverage={d['job_coverage']:.1f}%"
        for s,d in top30[:20]
    ])

    auc_display = f"{met['auc']*100:.1f}%" if met.get('auc') else "N/A"

    # KPI bar
    g1,g2,g3,g4,g5,g6 = st.columns(6)
    for col,(l,v) in zip([g1,g2,g3,g4,g5,g6],[
        ("Students",    ns),
        ("Train/Test",  f"{ntr}/{nte}"),
        ("Weak Courses",len(wc)),
        ("Jobs",        nj),
        ("F1 / AUC",    f"{met['f1']*100:.1f}% / {auc_display}"),
        ("Clusters",    cluster_data['best_k'] if cluster_data else "—"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    r1, r2, r3, r4, r5 = st.tabs([
        "📊 Overview & Metrics",
        "🔴 Course Recommendations",
        "💼 Market Gap Analysis",
        "📚 New Courses to Add",
        "🚀 Action Plan",
    ])

    # ── SUB-TAB 1: Overview ──
    with r1:
        st.markdown('<p class="sec">Overview — Student + Clustering + Market Summary</p>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🎓 Student Distribution")
            for level, color in [('Weak','box-red'),('Average','box-amber'),('Excellent','box-green')]:
                n = cnt.get(level, 0)
                pct = n/ns*100 if ns else 0
                st.markdown(f'<div class="{color}"><b>{level}</b> — {n} students ({pct:.0f}%)</div>', unsafe_allow_html=True)

            if cluster_data:
                st.markdown("#### 🔵 Cluster Summary")
                for cid, label in cluster_data['labels_map'].items():
                    n_c = (cluster_data['df']['Cluster'] == cid).sum()
                    st.markdown(f'<div class="box-purple"><b>{label}</b> — {n_c} students</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 💼 Top 10 Market Skills")
            t_df = pd.DataFrame([
                (s, d['category'], d['tfidf_score'],
                 f"{d['job_coverage']:.1f}%",
                 d.get('bert_avg_sim', '—'))
                for s,d in top30[:10]
            ], columns=['Skill','Category','TF-IDF','Coverage','BERT Sim'])
            st.dataframe(t_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📐 Metrics Reference")
        st.markdown(f"""
| Metric | Value | Meaning |
|--------|-------|---------|
| RF Accuracy   | {met['acc']*100:.1f}%  | Correct predictions overall |
| RF F1-Score   | {met['f1']*100:.1f}%   | Precision-recall balance |
| CV Accuracy   | {met['cv']*100:.1f}%   | 5-fold stability |
| AUC-ROC       | {auc_display}          | Quality regardless of threshold |
| Best k        | {cluster_data['best_k'] if cluster_data else '—'}  | Silhouette-optimal cluster count |
| Silhouette    | {max(cluster_data['silhouettes']):.3f if cluster_data else '—'}  | Cluster separation (0–1, higher=better) |
        """)

    # ── SUB-TAB 2: Course Recommendations ──
    with r2:
        st.markdown('<p class="sec">Per-Course Recommendations + Cluster Context</p>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
        else:
            if st.button("🔍 Generate Per-Course Recommendations", key="b3_courses"):
                with st.spinner("Generating..."):

                    cluster_context = ""
                    if cluster_data:
                        lines = []
                        for cid, label in cluster_data['labels_map'].items():
                            n_c    = (cluster_data['df']['Cluster'] == cid).sum()
                            avg_c  = float(cluster_data['df'][cluster_data['df']['Cluster']==cid][cc].mean().mean())
                            worst  = cluster_data['profile'].loc[cid, cc].idxmin() if cid in cluster_data['profile'].index else "—"
                            lines.append(f"  - {label}: {n_c} students, avg={avg_c:.1f}, worst_course={worst}")
                        cluster_context = "STUDENT CLUSTERS:\n" + "\n".join(lines)

                    prompt_courses = f"""You are a senior academic consultant for a CS department.

=== DATA ===
Students: {ns} | Thresholds: Weak<{wt} | Average {wt}–{at}
{cluster_context}

WEAK COURSES:
{weak_detail}

TOP 5 RF IMPORTANT:
{chr(10).join([f"- {c}: {v:.3f}" for c, v in imps.head(5).items()])}

TOP 20 MARKET SKILLS:
{top20_skills}

=== INSTRUCTIONS ===
For EACH weak course:

### [Course Name] | Fail: X% | RF: X.XXX

**Cluster Impact:** Which student groups are most affected?
**Why students struggle:** 2-3 specific reasons.
**3 Teaching Improvements:** course-specific.
**Resources:** 2 YouTube channels + 1 platform + 1 textbook.
**Market Relevance:** related TF-IDF skills with scores.
**Priority:** Critical / High / Medium"""

                    result_courses = call_groq(groq_key, prompt_courses, max_tokens=2500)
                    st.session_state['report_courses'] = result_courses

            if 'report_courses' in st.session_state:
                st.markdown(st.session_state['report_courses'])
                st.download_button(
                    "📥 Download", data=st.session_state['report_courses'],
                    file_name="course_recommendations_v3.txt", mime="text/plain", key="dl_c"
                )

    # ── SUB-TAB 3: Market Gap ──
    with r3:
        st.markdown('<p class="sec">Curriculum–Market Gap Analysis</p>', unsafe_allow_html=True)

        gap_rows = []
        for s, d in top30[:20]:
            covered = any(s.lower() in c.lower() or c.lower() in s.lower() for c in cc)
            bert_v  = d.get('bert_avg_sim')
            gap_rows.append({
                'Skill':          s,
                'Category':       d['category'],
                'TF-IDF Score':   d['tfidf_score'],
                'Job Coverage %': d['job_coverage'],
                'BERT Sim':       f"{bert_v:.3f}" if bert_v else "—",
                'In Curriculum':  '✅ Yes' if covered else '❌ Gap',
            })
        gap_df = pd.DataFrame(gap_rows)
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

        gaps   = [r for r in gap_rows if '❌' in r['In Curriculum']]
        covers = [r for r in gap_rows if '✅' in r['In Curriculum']]
        st.columns(2)[0].metric("Skills with Gap",  len(gaps),   delta_color="inverse")
        st.columns(2)[1].metric("Skills Covered",   len(covers))

        st.markdown("---")
        if groq_key:
            if st.button("🔍 Generate Gap Analysis", key="b3_gap"):
                with st.spinner("..."):
                    prompt_gap = f"""Gap analysis for a CS department.

CURRENT COURSES: {', '.join(cc)}

TOP 20 MARKET SKILLS (TF-IDF + BERT):
{top20_skills}

Write:
## Gap Analysis

### Skills NOT Covered (High Priority Gaps)
Each: TF-IDF score, coverage %, why matters, absorb into existing course OR needs new course.

### Skills Partially Covered (Need Strengthening)

### Skills Well Covered ✅

### Summary: % of top-20 skills covered vs not."""

                    result_gap = call_groq(groq_key, prompt_gap, max_tokens=2000)
                    st.session_state['report_gap'] = result_gap

            if 'report_gap' in st.session_state:
                st.markdown(st.session_state['report_gap'])
                st.download_button("📥 Download", data=st.session_state['report_gap'],
                    file_name="gap_analysis_v3.txt", mime="text/plain", key="dl_g")

    # ── SUB-TAB 4: New Courses ──
    with r4:
        st.markdown('<p class="sec">Recommended New Courses</p>', unsafe_allow_html=True)

        if groq_key:
            if st.button("🔍 Generate New Courses", key="b3_newcourses"):
                with st.spinner("..."):
                    prompt_new = f"""Recommend 5 new courses for a CS department based on market data.

CURRENT: {', '.join(cc)}
TOP 20 MARKET: {top20_skills}
WEAK COURSES: {weak_detail}

For each course:
## Course N: [Name]
**Skills:** (list from TF-IDF with scores)
**Market Evidence:** TF-IDF total, job coverage %, trend
**Why Needed:** gap between curriculum and market
**Placement:** Year X, Semester Y (Required/Elective)
**Prerequisites:** from current curriculum
**Priority:** High/Medium/Low"""

                    result_new = call_groq(groq_key, prompt_new, max_tokens=2000)
                    st.session_state['report_new'] = result_new

            if 'report_new' in st.session_state:
                st.markdown(st.session_state['report_new'])
                st.download_button("📥 Download", data=st.session_state['report_new'],
                    file_name="new_courses_v3.txt", mime="text/plain", key="dl_n")

    # ── SUB-TAB 5: Action Plan ──
    with r5:
        st.markdown('<p class="sec">Priority Action Plan</p>', unsafe_allow_html=True)

        if groq_key:
            if st.button("🚀 Generate Full Action Plan", key="b3_action"):
                with st.spinner("Synthesising all findings..."):

                    cluster_context = ""
                    if cluster_data:
                        cluster_context = f"Clusters: {cluster_data['best_k']} groups found. " \
                            f"At-Risk group: {(cluster_data['df']['Cluster_Label']=='🚨 At Risk').sum() if 'Cluster_Label' in cluster_data['df'] else '?'} students."

                    prev = ""
                    for key, title in [('report_courses','COURSES'),('report_gap','GAP'),('report_new','NEW COURSES')]:
                        if key in st.session_state:
                            prev += f"\n{title}:\n{st.session_state[key][:600]}\n"

                    prompt_action = f"""Priority action plan for CS department head.

DATA SUMMARY:
Students={ns} | F1={met['f1']*100:.1f}% | AUC={auc_display} | Jobs={nj}
{cluster_context}
Weak courses: {', '.join(list(wc.keys()) if hasattr(wc,'keys') else wc.index.tolist())}
Top market skills: {', '.join([s for s,_ in top30[:8]])}
{prev}

## Executive Summary (3-4 sentences)

## Priority Action Plan

### 🔴 IMMEDIATE (Next Semester)
3-4 actions: what, target course, expected impact, owner.

### 🟡 SHORT-TERM (1 Year)
3-4 actions. Same format.

### 🟢 MEDIUM-TERM (1-3 Years)
2-3 actions. Include cluster-based interventions.

## Success Metrics
Measurable targets: fail rate reduction %, curriculum coverage %, cluster At-Risk reduction %."""

                    result_action = call_groq(groq_key, prompt_action, max_tokens=2000)
                    st.session_state['report_action'] = result_action

            if 'report_action' in st.session_state:
                st.markdown(st.session_state['report_action'])

                full_report = (
                    f"CS DEPARTMENT — FULL INSTITUTIONAL REPORT v3\n{'='*60}\n\n"
                    f"Students={ns} | Jobs={nj} | F1={met['f1']*100:.1f}% | Clusters={cluster_data['best_k'] if cluster_data else '—'}\n\n"
                    f"{'='*60}\n1. COURSE RECOMMENDATIONS\n{'='*60}\n{st.session_state.get('report_courses','Not generated')}\n\n"
                    f"{'='*60}\n2. GAP ANALYSIS\n{'='*60}\n{st.session_state.get('report_gap','Not generated')}\n\n"
                    f"{'='*60}\n3. NEW COURSES\n{'='*60}\n{st.session_state.get('report_new','Not generated')}\n\n"
                    f"{'='*60}\n4. ACTION PLAN\n{'='*60}\n{st.session_state.get('report_action','Not generated')}"
                )
                st.download_button(
                    "📥 Download Full Report (All Sections)",
                    data=full_report,
                    file_name="full_institutional_report_v3.txt",
                    mime="text/plain",
                    key="dl_full"
                )
        else:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
