"""
CS Department Intelligence System v2
- Tab 1: Student performance analysis + Random Forest → Independent Report
- Tab 2: Job market analysis using TF-IDF NLP → Independent Report
- Tab 3: Combined report (optional — only if both tabs done)

Each tab generates its own full report independently.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
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
.stButton>button {
    background:#1e3a5f; color:#f1f5f9; border:1px solid #3b82f6;
    border-radius:8px; font-family:'IBM Plex Mono',monospace;
    font-size:.85rem; padding:.6rem 1.5rem; width:100%;
}
.stButton>button:hover { background:#3b82f6; }
.badge-independent {
    display:inline-block; background:#052e16; border:1px solid #16a34a;
    color:#86efac; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-family:'IBM Plex Mono',monospace;
    margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🎓 CS Department Intelligence System</h1>
  <p>TF-IDF NLP · Random Forest ML · Independent Per-Tab AI Reports · Institutional Decision Support</p>
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
    st.markdown("**Each tab works independently:**")
    st.markdown("- Tab 1 → Student report alone")
    st.markdown("- Tab 2 → Job market report alone")
    st.markdown("- Tab 3 → Combined (needs both)")
    st.markdown("---")
    st.markdown("**Full Pipeline:**")
    st.markdown("```\nCSV Input\n  ↓\nStage 2: Clean\n  ↓\nStage 3: Pre-process\n  ↓\nStage 4: TF-IDF\n  ↓\nStage 5: Random Forest\n  ↓\nStage 6: AUC + SHAP\n  ↓\nAI Report\n```")

# ══════════════════════════════════════════════════
# GROQ HELPER
# ══════════════════════════════════════════════════
def call_groq(api_key: str, prompt: str, max_tokens: int = 2500) -> str:
    """استدعاء Groq API مع معالجة الأخطاء."""
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
    """
    Stage 2 — Enhanced Cleanup (بناءً على المحاضرة):
    - إزالة URLs
    - إزالة punctuation
    - تنظيف المسافات الزائدة
    """
    t = str(t).lower()
    t = re.sub(r'http\S+', ' ', t)           # إزالة URLs
    t = re.sub(r'[^a-z0-9\s/+#.]', ' ', t)  # إزالة punctuation
    t = re.sub(r'\s+', ' ', t).strip()       # تنظيف المسافات
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
    """
    Stage 3 — Pre-processing (بناءً على المحاضرة):
    - Tokenization: تقسيم النص لكلمات
    - Stop Word Removal: حذف الكلمات الشائعة التي لا تعطي معلومة
    - إعادة التجميع للـ TF-IDF
    """
    tokens = clean_text(t).split()                      # Tokenization
    tokens = [w for w in tokens if w not in STOP_WORDS] # Stop Word Removal
    tokens = [w for w in tokens if len(w) > 1]          # حذف حروف منفردة
    return ' '.join(tokens)


# ── Stage 6: AUC Calculation ──
def compute_auc(model, X_te, y_te, classes):
    """
    Stage 6 — AUC (بناءً على المحاضرة):
    AUC = مساحة تحت منحنى ROC
    يقيس جودة النموذج بغض النظر عن عتبة القرار.
    نستخدم OvR (One vs Rest) للـ multi-class.
    """
    try:
        y_bin   = label_binarize(y_te, classes=classes)
        y_proba = model.predict_proba(X_te)
        if len(classes) == 2:
            return roc_auc_score(y_bin, y_proba[:, 1])
        return roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        return None


# ── Stage 6: SHAP Values ──
def compute_shap(model, X_tr, X_te, feature_names):
    """
    Stage 6 — SHAP (بناءً على المحاضرة):
    SHAP = SHapley Additive exPlanations
    يشرح لماذا النموذج اتخذ قراراً معيناً لكل طالب.
    أقوى من Feature Importance لأنه يعطي اتجاه التأثير.
    TreeExplainer مخصص للـ Random Forest — سريع وموثوق.
    """
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        # لو multi-class: خذ المتوسط المطلق عبر كل الفئات
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
# TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Student Performance",
    "💼 Job Market NLP",
    "📋 Combined Report"
])


# ╔══════════════════════════════════════════════╗
# ║  TAB 1 — STUDENT PERFORMANCE + OWN REPORT  ║
# ╚══════════════════════════════════════════════╝
with tab1:
    st.markdown('<p class="sec">01 · Student Performance — Random Forest + Independent AI Report</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge-independent">✦ INDEPENDENT — No other tab required</span>', unsafe_allow_html=True)
    st.markdown('<div class="box-blue">Upload any student grades CSV. The AI report is generated from student data alone — no job market data needed.</div>', unsafe_allow_html=True)

    up_stu = st.file_uploader("📂 Upload Student Grades CSV", type="csv", key="stu")

    if up_stu:
        df_raw = pd.read_csv(up_stu)
        st.success(f"✅ {len(df_raw)} students · {len(df_raw.columns)} columns")

        with st.expander("👁 Preview"):
            st.dataframe(df_raw.head())

        st.markdown("---")
        numeric_cols = df_raw.select_dtypes(include='number').columns.tolist()
        skip_kw = ['id','rank','student','semester','final','total','gpa','grade','index']
        auto_courses = [c for c in numeric_cols
                        if not any(k in c.lower() for k in skip_kw)]

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

        if st.button("🔍 Run ML Analysis", key="b1"):
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

            # ── ML ──
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

            # ── Stage 6: AUC ──
            classes  = sorted(y_tr.unique())
            auc_val  = compute_auc(rf, X_te.values, y_te, classes)

            # ── Stage 6: SHAP ──
            with st.spinner("Computing SHAP values..."):
                shap_vals = compute_shap(rf, X_tr.values, X_te.values, course_cols)

            # ── KPI cards ──
            st.markdown("---")
            m1,m2,m3,m4 = st.columns(4)
            for col,(l,v) in zip([m1,m2,m3,m4],[
                ("Students", len(df)),
                ("🔴 Weak",    f"{counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.0f}%)"),
                ("🟡 Average", f"{counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.0f}%)"),
                ("🟢 Excellent",f"{counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.0f}%)")
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
                r = mpatches.Patch(color='#dc2626', label=f'Weak (<{weak_t})')
                a = mpatches.Patch(color='#d97706', label=f'Mid ({weak_t}–{avg_t})')
                g = mpatches.Patch(color='#16a34a', label=f'Good (>{avg_t})')
                ax.legend(handles=[r,a,g], fontsize=7, facecolor='#0f172a', labelcolor='#94a3b8')
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
            st.markdown("#### 🤖 Random Forest Metrics — Stage 5 & 6")
            st.caption(f"Train: {len(X_tr)} (80%) · Test: {len(X_te)} (20%) · 5-fold CV")

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

            # ── Stage 6: Metrics Explanation ──
            with st.expander("📐 How each metric was calculated (Stage 6)"):
                auc_formula = f"{auc_val*100:.1f}%" if auc_val else "N/A"
                st.markdown(f"""
**Stage 2 — Cleanup:** lowercase → remove URLs → remove punctuation → clean spaces.

**Stage 3 — Pre-processing:** Tokenization (split to words) → Stop Word Removal ({len(STOP_WORDS)} words removed).

**Stage 4 — TF-IDF:** ngram_range=(1,3) · sublinear_tf=True · min_df=2 · max_df=0.95

**Stage 5 — Random Forest:** 100 trees · class_weight='balanced' · 80/20 split (no leakage)

**Stage 6 — Evaluation:**

| Metric | Value | Formula | What it means |
|--------|-------|---------|---------------|
| Accuracy   | {acc*100:.1f}%  | (TP+TN) / all | نسبة التنبؤات الصحيحة |
| Precision  | {prec*100:.1f}% | TP / (TP+FP)  | من المتنبأ بضعفهم، كم منهم ضعيف فعلاً؟ |
| Recall     | {rec*100:.1f}%  | TP / (TP+FN)  | من الضعاف فعلاً، كم اكتشفهم النموذج؟ |
| F1-Score   | {f1*100:.1f}%   | 2·P·R/(P+R)   | التوازن بين Precision وRecall |
| CV Acc.    | {cv*100:.1f}%   | 5-fold mean   | ثبات النموذج — مانه overfitting |
| AUC-ROC    | {auc_formula}   | Area under ROC curve (OvR) | جودة النموذج بغض النظر عن العتبة — 1.0 = مثالي |

**SHAP vs Feature Importance:**
- Feature Importance → يخبرك *كم* مقرر مهم (magnitude فقط)
- SHAP → يخبرك *كم* و*في أي اتجاه* (علامة عالية = يرفع/يخفض احتمال الرسوب)
                """)

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
                    ax3.text(j, i, str(cm_m[i,j]), ha='center', va='center',
                             fontsize=13, fontweight='bold', color=clr)
            ax3.set_xticks(range(len(lbls))); ax3.set_xticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_yticks(range(len(lbls))); ax3.set_yticklabels(lbls, color='#94a3b8', fontsize=9)
            ax3.set_xlabel('Predicted', color='#94a3b8')
            ax3.set_ylabel('Actual',    color='#94a3b8')
            ax3.spines[:].set_color('#1e3a5f')
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            # ── Feature Importance vs SHAP ──
            fi_col, shap_col = st.columns(2)

            with fi_col:
                st.markdown("**Feature Importance (Stage 5)**")
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
                st.markdown("**SHAP Values (Stage 6) — stronger explanation**")
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

                    # مقارنة RF vs SHAP
                    st.markdown("**RF Importance vs SHAP — top 5**")
                    compare_df = pd.DataFrame({
                        'Course':     imps.head(5).index,
                        'RF Imp.':    imps.head(5).values.round(4),
                        'SHAP':       [shap_vals.get(c, 0) for c in imps.head(5).index],
                        'Agreement':  ['✅' if abs(imps.rank(ascending=False)[c] - shap_vals.rank(ascending=False).get(c,99)) <= 2
                                       else '⚠️' for c in imps.head(5).index]
                    })
                    st.dataframe(compare_df, use_container_width=True, hide_index=True)
                else:
                    st.info("SHAP computation skipped (small dataset or error).")

            # ── Save to session ──
            st.session_state.update({
                'avgs':avgs,'weak_c':weak_c,'mid_c':mid_c,'good_c':good_c,
                'counts':counts,'n_stu':len(df),'n_tr':len(X_tr),'n_te':len(X_te),
                'met':{'acc':acc,'prec':prec,'rec':rec,'f1':f1,'cv':cv,
                       'auc': auc_val},
                'imps':imps,'shap_vals':shap_vals,
                'fr':fail_rate,'wt':weak_t,'at':avg_t,'cc':course_cols
            })

            # ══════════════════════════════════════════
            # INDEPENDENT AI REPORT — STUDENT DATA ONLY
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Student Performance Only")
            st.markdown('<div class="box-purple">This report is generated from student data alone. No job market data required. Each course gets specific recommendations based on its fail rate and RF importance.</div>', unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar to generate the AI report.")
            else:
                if st.button("📋 Generate Student Performance Report", key="b1_report"):
                    with st.spinner("Generating per-course recommendations..."):

                        weak_detail = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fail_rate.get(c,0):.1f}%, rf_importance={imps.get(c,0):.3f}"
                            for c, v in weak_c.items()
                        ]) or "None"

                        mid_detail = "\n".join([
                            f"- {c}: avg={v:.1f}, fail_rate={fail_rate.get(c,0):.1f}%"
                            for c, v in mid_c.items()
                        ]) or "None"

                        top5_imp = "\n".join([f"- {c}: {v:.3f}" for c, v in imps.head(5).items()])

                        prompt_student = f"""You are a senior academic consultant writing a formal report for the Head of a Computer Science Department.

=== STUDENT DATA ({len(df)} students) ===
Random Forest: Train={len(X_tr)} | Test={len(X_te)}
Accuracy={acc*100:.1f}% | Precision={prec*100:.1f}% | Recall={rec*100:.1f}% | F1={f1*100:.1f}% | CV={cv*100:.1f}%
Thresholds: Weak<{weak_t} | Average {weak_t}–{avg_t} | Excellent>{avg_t}
Distribution: Weak={counts.get('Weak',0)} | Average={counts.get('Average',0)} | Excellent={counts.get('Excellent',0)}

WEAK COURSES:
{weak_detail}

MEDIUM COURSES:
{mid_detail}

TOP 5 MOST INFLUENTIAL (RF Feature Importance):
{top5_imp}

ALL COURSES IN DATASET:
{', '.join(course_cols)}

=== REPORT STRUCTURE ===
Write a formal report with EXACTLY these sections:

## 1. Executive Summary
2-3 sentences: key student performance numbers and main concerns.

## 2. Per-Course Analysis & Recommendations
For EACH weak and medium course, a dedicated subsection:

### [Exact Course Name]
- **Diagnosis:** why students struggle (based on fail rate and RF importance)
- **3 Concrete Teaching Improvements:** specific to THIS course topic
- **Learning Resources:**
  * 2-3 real YouTube channels/playlists for THIS specific topic
  * 1-2 practice platforms with specific resource names
  * 1 recommended textbook
- **Priority:** Critical / High / Medium

## 3. Priority Action Plan
Numbered from most urgent. Timeline: Next semester / 1 year / 2-3 years.

Use exact course names from the data. Be specific, not generic."""

                        result = call_groq(groq_key, prompt_student, max_tokens=2500)
                        st.markdown(result)

                        # Download
                        report_txt = (
                            f"CS DEPARTMENT — STUDENT PERFORMANCE REPORT\n{'='*60}\n\n"
                            f"Students={len(df)} | Train={len(X_tr)} | Test={len(X_te)}\n"
                            f"Accuracy={acc*100:.1f}% | F1={f1*100:.1f}%\n\n"
                            f"WEAK COURSES:\n{weak_detail}\n\n"
                            f"{'='*60}\nAI RECOMMENDATIONS\n{'='*60}\n\n{result}"
                        )
                        st.download_button(
                            "📥 Download Student Report",
                            data=report_txt,
                            file_name="student_performance_report.txt",
                            mime="text/plain"
                        )
    else:
        st.info("📂 Upload any student grades CSV to begin.")


# ╔══════════════════════════════════════════════╗
# ║  TAB 2 — JOB MARKET NLP + OWN REPORT       ║
# ╚══════════════════════════════════════════════╝
with tab2:
    st.markdown('<p class="sec">02 · Job Market — TF-IDF NLP + Independent AI Report</p>', unsafe_allow_html=True)
    st.markdown('<span class="badge-independent">✦ INDEPENDENT — No other tab required</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box-blue">
    <b>NLP pipeline:</b> Tokenise → TF-IDF matrix → weighted skill scores → job coverage % → ranked skill table.<br>
    The AI report is generated from job market data alone — no student data needed.
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

        if st.button("🔍 Run NLP Analysis", key="b2"):
            with st.spinner("Running TF-IDF NLP..."):

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

                raw_texts   = jobs_df[desc_col].dropna().tolist()
                # Stage 2: Enhanced Cleanup
                cleaned_texts = [clean_text(t) for t in raw_texts]
                # Stage 3: Pre-processing (stop words + tokenization)
                clean_texts   = [preprocess_text(t) for t in raw_texts]

                # Stage 4: TF-IDF محسّن (بناءً على المحاضرة)
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 3),   # يفهم "machine learning" كوحدة
                    min_df=2,             # تجاهل مهارة بوظيفة وحدة
                    max_df=0.95,          # تجاهل كلمات في كل الوظائف
                    sublinear_tf=True,    # log scaling — يقلل هيمنة الكلمات المتكررة
                )
                tfidf_matrix  = vectorizer.fit_transform(clean_texts)
                feature_names = vectorizer.get_feature_names_out()
                tfidf_sums    = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                tfidf_dict    = dict(zip(feature_names, tfidf_sums))

                all_text = ' '.join(clean_texts)
                skill_results = {}
                for cat, skills in TAXONOMY.items():
                    for skill in skills:
                        sc    = clean_text(skill)
                        freq  = all_text.count(sc)
                        tscore= tfidf_dict.get(sc, 0.0)
                        jw    = sum(1 for t in clean_texts if sc in t)
                        cov   = jw / len(clean_texts) * 100
                        if freq > 0:
                            skill_results[skill] = {
                                'category':    cat,
                                'freq':        freq,
                                'tfidf_score': round(float(tscore), 2),
                                'job_coverage': round(cov, 1),
                                'jobs_with':   jw
                            }

                sorted_by_tfidf = sorted(
                    skill_results.items(),
                    key=lambda x: x[1]['tfidf_score'], reverse=True
                )
                top30 = sorted_by_tfidf[:30]

            # ── Pipeline stages summary ──
            with st.expander("🔬 Pipeline Stages Applied (Stage 2 → 4)"):
                st.markdown(f"""
| Stage | What happened | Detail |
|-------|--------------|--------|
| **Stage 2: Cleanup** | URLs removed · Punctuation cleaned · Whitespace normalized | `{len(raw_texts):,}` texts processed |
| **Stage 3: Pre-process** | Tokenization · Stop words removed | `{len(STOP_WORDS)}` stop words filtered |
| **Stage 4: TF-IDF** | ngram=(1,3) · sublinear_tf · min_df=2 · max_df=0.95 | Matrix: `{len(raw_texts):,}` docs |

**Why sublinear_tf=True?** Uses log(1+count) instead of raw count — prevents high-frequency words from dominating.

**Why ngram_range=(1,3)?** Captures "machine learning" and "deep learning" as single units.

**Why min_df=2?** Ignores skills appearing in only 1 job — likely typos.
                """)

            # ── Charts ──
            CAT_COLORS = {
                'Programming Languages':'#3b82f6','Web & Mobile':'#16a34a',
                'Data Science & AI':'#7c3aed','Cloud & DevOps':'#d97706',
                'Cybersecurity':'#dc2626','Databases':'#0891b2',
                'Software Engineering':'#059669','Networking':'#9333ea',
                'Emerging Tech':'#f59e0b','Custom':'#ec4899'
            }

            st.markdown("#### Top 20 Skills — TF-IDF Score")
            names  = [s[0] for s in top30[:20]]
            scores = [s[1]['tfidf_score'] for s in top30[:20]]
            covs   = [s[1]['job_coverage'] for s in top30[:20]]
            cats   = [s[1]['category'] for s in top30[:20]]
            bclrs  = [CAT_COLORS.get(c,'#64748b') for c in cats]

            fig5, ax5 = plt.subplots(figsize=(10,8))
            fig5.patch.set_facecolor('#0f172a'); ax5.set_facecolor('#0f172a')
            bars5 = ax5.barh(names, scores, color=bclrs, height=.6)
            for b,v,cov in zip(bars5, scores, covs):
                ax5.text(v+.1, b.get_y()+b.get_height()/2,
                         f'{v:.1f} | {cov:.0f}% jobs',
                         va='center', fontsize=7, color='#94a3b8')
            ax5.tick_params(colors='#94a3b8', labelsize=8)
            ax5.spines[:].set_color('#1e3a5f')
            ax5.set_xlabel('TF-IDF Score', color='#94a3b8')
            handles = [mpatches.Patch(color=v, label=k)
                       for k,v in CAT_COLORS.items() if k in cats]
            ax5.legend(handles=handles, fontsize=7, facecolor='#0f172a',
                       labelcolor='#94a3b8', loc='lower right')
            plt.tight_layout(); st.pyplot(fig5); plt.close()

            # Stats
            s1,s2,s3,s4 = st.columns(4)
            for col,(l,v) in zip([s1,s2,s3,s4],[
                ("Total Jobs",    len(jobs_df)),
                ("Skills Found",  len(skill_results)),
                ("Top Skill",     top30[0][0] if top30 else '—'),
                ("Top TF-IDF",    f"{top30[0][1]['tfidf_score']:.1f}" if top30 else 0)
            ]):
                col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.2rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

            # Full table
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
                'n_jobs': len(jobs_df), 'sorted_skills': sorted_by_tfidf
            })

            # ══════════════════════════════════════════
            # INDEPENDENT AI REPORT — JOB MARKET ONLY
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🤖 AI Report — Job Market Analysis Only")
            st.markdown('<div class="box-purple">This report analyses what the job market demands and recommends curriculum additions — based on job data alone, no student grades needed.</div>', unsafe_allow_html=True)

            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar to generate the AI report.")
            else:
                if st.button("📋 Generate Job Market Report", key="b2_report"):
                    with st.spinner("Generating market-driven curriculum recommendations..."):

                        top20_skills = "\n".join([
                            f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, coverage={d['job_coverage']:.1f}% of jobs"
                            for s, d in top30[:20]
                        ])

                        cat_summary = "\n".join([
                            f"- {cat}: total demand score {total:.1f}"
                            for cat, total in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
                        ])

                        prompt_jobs = f"""You are a senior academic consultant writing a formal report for the Head of a Computer Science Department, focused on job market alignment.

=== JOB MARKET DATA ({len(jobs_df)} listings, TF-IDF NLP analysis) ===
TOP 20 SKILLS BY TF-IDF SCORE:
{top20_skills}

DEMAND BY CATEGORY:
{cat_summary}

=== REPORT STRUCTURE ===
Write a formal report with EXACTLY these sections:

## 1. Market Overview
Key findings from the job market analysis. What skills dominate? What is the trend?

## 2. Top Skills Analysis
For the top 10 skills by TF-IDF score:
### [Skill Name] — TF-IDF: X.X | Coverage: X%
- What this skill is used for in industry
- Why it's in demand
- How it should be taught (theory vs practical balance)
- Recommended learning resources: 2 YouTube channels + 1 platform + 1 textbook

## 3. Recommended New Courses
4–5 new courses a CS department should add, each:
- **Course name**
- **Skills covered** (with TF-IDF scores)
- **Market evidence:** coverage % across job postings
- **Suggested year/semester**
- **Priority:** High / Medium / Low

## 4. Curriculum Gaps
Skills with high TF-IDF that are typically missing from traditional CS curricula.
Rank by urgency.

## 5. Action Plan
Numbered, most urgent first. Timeline for each.

Be specific. Use actual skill names and scores from the data."""

                        result_jobs = call_groq(groq_key, prompt_jobs, max_tokens=2500)
                        st.markdown(result_jobs)

                        report_txt = (
                            f"CS DEPARTMENT — JOB MARKET REPORT\n{'='*60}\n\n"
                            f"Jobs Analyzed: {len(jobs_df)}\n\n"
                            f"TOP SKILLS (TF-IDF):\n{top20_skills}\n\n"
                            f"{'='*60}\nAI RECOMMENDATIONS\n{'='*60}\n\n{result_jobs}"
                        )
                        st.download_button(
                            "📥 Download Job Market Report",
                            data=report_txt,
                            file_name="job_market_report.txt",
                            mime="text/plain"
                        )
    else:
        st.info("📂 Upload a job listings CSV to begin.")


# ╔══════════════════════════════════════════════╗
# ║  TAB 3 — COMBINED REPORT (optional)         ║
# ╚══════════════════════════════════════════════╝
with tab3:
    st.markdown('<p class="sec">03 · Combined Institutional Report — Student + Job Market</p>', unsafe_allow_html=True)

    has_s = 'weak_c' in st.session_state
    has_j = 'top30'  in st.session_state

    # Status badges
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
        st.info("Complete both Tab 1 and Tab 2 to unlock the combined report. Each tab also generates its own independent report.")
        st.stop()

    # ── unpack session state ──
    wc   = st.session_state['weak_c']
    mc   = st.session_state['mid_c']
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

    # ── shared strings (used by multiple sub-tabs) ──
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
        f"- {s} ({d['category']}): TF-IDF={d['tfidf_score']:.1f}, coverage={d['job_coverage']:.1f}%"
        for s,d in top30[:20]
    ])

    # ── Summary KPI bar (always visible) ──
    g1,g2,g3,g4,g5 = st.columns(5)
    auc_display = f"{met['auc']*100:.1f}%" if met.get('auc') else "N/A"
    for col,(l,v) in zip([g1,g2,g3,g4,g5],[
        ("Students",     ns),
        ("Train/Test",   f"{ntr}/{nte}"),
        ("Weak Courses", len(wc)),
        ("Jobs Analyzed",nj),
        ("F1 / AUC",     f"{met['f1']*100:.1f}% / {auc_display}"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.1rem">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════
    # SUB-TABS — كل قسم لحاله
    # ════════════════════════════════════════════
    r1, r2, r3, r4, r5 = st.tabs([
        "📊 Overview & Metrics",
        "🔴 Course Recommendations",
        "💼 Market Gap Analysis",
        "📚 New Courses to Add",
        "🚀 Action Plan",
    ])

    # ────────────────────────────────────────────
    # SUB-TAB 1 — Overview & Metrics
    # ────────────────────────────────────────────
    with r1:
        st.markdown('<p class="sec">Overview — Student Data + Market Summary</p>', unsafe_allow_html=True)

        # Student level distribution
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🎓 Student Distribution")
            for level, color in [('Weak','box-red'),('Average','box-amber'),('Excellent','box-green')]:
                n = cnt.get(level, 0)
                pct = n/ns*100 if ns else 0
                st.markdown(f'<div class="{color}"><b>{level}</b> — {n} students ({pct:.0f}%)</div>', unsafe_allow_html=True)

            st.markdown("#### 🔴 Weak Courses")
            if len(wc):
                for c,v in wc.items():
                    st.markdown(f'<div class="box-red"><b>{c}</b> — Avg:{v:.1f} · Fail:{fr.get(c,0):.1f}% · RF Imp:{imps.get(c,0):.3f}</div>', unsafe_allow_html=True)
            else:
                st.success("No weak courses.")

        with col_r:
            st.markdown("#### 💼 Top 10 Market Skills")
            t_df = pd.DataFrame([
                (s, d['category'], d['tfidf_score'], f"{d['job_coverage']:.1f}%")
                for s,d in top30[:10]
            ], columns=['Skill','Category','TF-IDF','Coverage'])
            st.dataframe(t_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📐 How Metrics Were Calculated")
        st.markdown(f"""
| Metric | Value | Formula | Meaning |
|--------|-------|---------|---------|
| Accuracy   | {met['acc']*100:.1f}%  | (TP+TN) / all         | Overall correct predictions |
| Precision  | {met['prec']*100:.1f}% | TP / (TP+FP)          | Of predicted weak, how many truly weak |
| Recall     | {met['rec']*100:.1f}%  | TP / (TP+FN)          | Of all weak, how many did we catch |
| F1-Score   | {met['f1']*100:.1f}%   | 2·P·R / (P+R)         | Balance between precision & recall |
| CV Acc.    | {met['cv']*100:.1f}%   | 5-fold mean           | Stability — not overfitting |
| AUC-ROC    | {auc_display}          | Area under ROC (OvR)  | Quality regardless of threshold |

**Train/Test:** 80% ({ntr} students) train · 20% ({nte} students) test — classification applied after split (no leakage).

**TF-IDF:** ngram=(1,3) · sublinear_tf=True · min_df=2 · max_df=0.95 — stop words removed before vectorization.
        """)

    # ────────────────────────────────────────────
    # SUB-TAB 2 — Per-Course Recommendations
    # ────────────────────────────────────────────
    with r2:
        st.markdown('<p class="sec">Per-Course AI Recommendations — one section per weak course</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-blue">Each weak course gets its own dedicated AI analysis: why students struggle, teaching improvements, learning resources, and market relevance.</div>', unsafe_allow_html=True)

        if not wc.empty if hasattr(wc, 'empty') else not len(wc):
            st.success("✅ No weak courses in this dataset.")
        else:
            if not groq_key:
                st.warning("⚠️ Add Groq API Key in the sidebar.")
            else:
                if st.button("🔍 Generate Per-Course Recommendations", key="b3_courses"):
                    with st.spinner("Generating per-course analysis..."):

                        prompt_courses = f"""You are a senior academic consultant. Write per-course recommendations for a CS department head.

=== DATA ===
Students: {ns} | Thresholds: Weak<{wt} | Average {wt}–{at}

WEAK COURSES:
{weak_detail}

TOP 5 RF INFLUENTIAL COURSES:
{top5_imp}

TOP 20 MARKET SKILLS (TF-IDF):
{top20_skills}

=== INSTRUCTIONS ===
For EACH weak course below, write a dedicated section with EXACTLY this structure:

### [Exact Course Name] | Fail Rate: X% | RF Importance: X.XXX

**Why students struggle:**
2-3 specific reasons based on the course topic and fail rate.

**3 Teaching Improvements:**
1. [specific method for THIS course topic]
2. [specific method for THIS course topic]
3. [specific method for THIS course topic]

**Learning Resources:**
- YouTube: [2 real channel names focused on THIS topic with example playlist]
- Practice: [1 specific platform + specific resource name]
- Textbook: [1 well-known book for THIS exact topic]

**Market Relevance:**
Which TF-IDF skills from the job market data relate to this course? List them with their scores.

**Priority:** Critical / High / Medium

---

Be specific. Use the actual course names. Different resources for each course."""

                        result_courses = call_groq(groq_key, prompt_courses, max_tokens=2500)
                        st.session_state['report_courses'] = result_courses

                if 'report_courses' in st.session_state:
                    st.markdown(st.session_state['report_courses'])
                    st.download_button(
                        "📥 Download Course Recommendations",
                        data=st.session_state['report_courses'],
                        file_name="course_recommendations.txt",
                        mime="text/plain",
                        key="dl_courses"
                    )

    # ────────────────────────────────────────────
    # SUB-TAB 3 — Market Gap Analysis
    # ────────────────────────────────────────────
    with r3:
        st.markdown('<p class="sec">Curriculum–Market Gap — what the market wants vs what we teach</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-blue">Cross-references TF-IDF skill demand with existing courses to identify gaps — skills in high demand but absent from the curriculum.</div>', unsafe_allow_html=True)

        # Static gap table (no AI needed)
        st.markdown("#### 💼 Full Market Skills vs Curriculum Coverage")
        gap_rows = []
        for s, d in top30[:20]:
            # هل المهارة مغطاة؟ نبحث في أسماء المقررات
            covered = any(
                s.lower() in c.lower() or c.lower() in s.lower()
                for c in cc
            )
            gap_rows.append({
                'Skill':         s,
                'Category':      d['category'],
                'TF-IDF Score':  d['tfidf_score'],
                'Job Coverage %':d['job_coverage'],
                'In Curriculum': '✅ Yes' if covered else '❌ Gap',
            })
        gap_df = pd.DataFrame(gap_rows)
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

        # Gap score summary
        gaps_only = [r for r in gap_rows if r['In Curriculum'] == '❌ Gap']
        covered   = [r for r in gap_rows if r['In Curriculum'] == '✅ Yes']
        g1c, g2c = st.columns(2)
        g1c.metric("Skills with Gap",    len(gaps_only), delta_color="inverse")
        g2c.metric("Skills Covered",     len(covered))

        st.markdown("---")
        if not groq_key:
            st.warning("⚠️ Add Groq API Key for AI gap analysis.")
        else:
            if st.button("🔍 Generate Gap Analysis Report", key="b3_gap"):
                with st.spinner("Analysing curriculum-market gaps..."):

                    prompt_gap = f"""You are a senior academic consultant. Write a curriculum-market gap analysis for a CS department.

=== DATA ===
CURRENT COURSES IN CURRICULUM:
{', '.join(cc)}

TOP 20 MARKET SKILLS (TF-IDF NLP from {nj} job listings):
{top20_skills}

=== INSTRUCTIONS ===
Write a formal gap analysis with EXACTLY these sections:

## Gap Analysis: Curriculum vs Market Demand

### Skills in High Demand NOT Covered by Existing Courses
List each gap skill with:
- TF-IDF score and job coverage %
- Why it matters for graduates
- Which existing course could absorb it (if any) vs needs a new course

### Skills Partially Covered (needs strengthening)
List skills that exist in courses but at insufficient depth.

### Skills Well Covered ✅
Brief list of skills the curriculum already handles well.

### Gap Score Summary
Estimated % of top-20 market skills covered vs not covered.

Be specific. Use actual course names and skill names from the data."""

                    result_gap = call_groq(groq_key, prompt_gap, max_tokens=2000)
                    st.session_state['report_gap'] = result_gap

            if 'report_gap' in st.session_state:
                st.markdown(st.session_state['report_gap'])
                st.download_button(
                    "📥 Download Gap Analysis",
                    data=st.session_state['report_gap'],
                    file_name="gap_analysis.txt",
                    mime="text/plain",
                    key="dl_gap"
                )

    # ────────────────────────────────────────────
    # SUB-TAB 4 — New Courses to Add
    # ────────────────────────────────────────────
    with r4:
        st.markdown('<p class="sec">New Courses — recommended additions based on market gap</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-blue">Based on TF-IDF skill demand and curriculum gaps, the AI recommends specific new courses to add — with market evidence for each.</div>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
        else:
            if st.button("🔍 Generate New Course Recommendations", key="b3_newcourses"):
                with st.spinner("Designing new course recommendations..."):

                    prompt_new = f"""You are a senior academic consultant recommending new courses for a CS department.

=== DATA ===
CURRENT COURSES:
{', '.join(cc)}

TOP 20 MARKET SKILLS (TF-IDF from {nj} jobs):
{top20_skills}

WEAK COURSES IN CURRENT CURRICULUM:
{weak_detail}

=== INSTRUCTIONS ===
Recommend exactly 5 new courses with EXACTLY this structure for each:

---
## Course [N]: [Course Name]

**Skills Covered:**
List 4-6 skills from the TF-IDF data this course would teach, with their scores.

**Market Evidence:**
- TF-IDF total demand score for these skills
- Job coverage % (how many job postings require these skills)
- Growing / Stable / Declining trend

**Why This Course Is Needed:**
2-3 sentences linking the gap in current curriculum to market demand.

**Suggested Placement:** Year X, Semester Y (Required / Elective)

**Prerequisites:** [existing courses from the curriculum]

**Priority:** High / Medium / Low
---

Base recommendations on actual TF-IDF scores. Do not recommend courses already in the curriculum."""

                    result_new = call_groq(groq_key, prompt_new, max_tokens=2000)
                    st.session_state['report_new'] = result_new

            if 'report_new' in st.session_state:
                st.markdown(st.session_state['report_new'])
                st.download_button(
                    "📥 Download New Course Recommendations",
                    data=st.session_state['report_new'],
                    file_name="new_courses.txt",
                    mime="text/plain",
                    key="dl_new"
                )

    # ────────────────────────────────────────────
    # SUB-TAB 5 — Action Plan
    # ────────────────────────────────────────────
    with r5:
        st.markdown('<p class="sec">Priority Action Plan — what to do first, second, third</p>', unsafe_allow_html=True)
        st.markdown('<div class="box-blue">A numbered action plan synthesising all findings: course fixes, new additions, and curriculum restructuring — with timelines.</div>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Add Groq API Key in the sidebar.")
        else:
            if st.button("🚀 Generate Full Action Plan", key="b3_action"):
                with st.spinner("Synthesising action plan..."):

                    # جمع كل التقارير السابقة كـ context إذا موجودة
                    prev_reports = ""
                    if 'report_courses' in st.session_state:
                        prev_reports += f"\n\nCOURSE RECOMMENDATIONS SUMMARY:\n{st.session_state['report_courses'][:800]}"
                    if 'report_gap' in st.session_state:
                        prev_reports += f"\n\nGAP ANALYSIS SUMMARY:\n{st.session_state['report_gap'][:800]}"
                    if 'report_new' in st.session_state:
                        prev_reports += f"\n\nNEW COURSES SUMMARY:\n{st.session_state['report_new'][:800]}"

                    prompt_action = f"""You are a senior academic consultant. Write a priority action plan for a CS department head.

=== DATA ===
Students: {ns} | F1: {met['f1']*100:.1f}% | Jobs analysed: {nj}
Weak courses ({len(wc)}): {', '.join(wc.index.tolist() if hasattr(wc,'index') else list(wc.keys()))}
Top market skills: {', '.join([s for s,_ in top30[:10]])}
Current courses: {', '.join(cc)}
{prev_reports}

=== INSTRUCTIONS ===
Write a formal priority action plan with EXACTLY this structure:

## Executive Summary
3-4 sentences: key numbers and most critical finding.

## Priority Action Plan

### 🔴 IMMEDIATE (Next Semester)
Numbered list of 3-4 actions. For each:
- **Action:** what exactly to do
- **Target:** which course or curriculum area
- **Expected Impact:** measurable outcome
- **Owner:** Department Head / Course Instructor / Curriculum Committee

### 🟡 SHORT-TERM (Within 1 Year)
Numbered list of 3-4 actions. Same format.

### 🟢 MEDIUM-TERM (1–3 Years)
Numbered list of 2-3 actions. Same format.

## Success Metrics
How to measure if the actions worked. Include:
- Student performance targets (fail rate reduction %)
- Curriculum coverage target (% of top-20 skills covered)
- Timeline for first review

Be specific. Reference actual course names and skill scores."""

                    result_action = call_groq(groq_key, prompt_action, max_tokens=2000)
                    st.session_state['report_action'] = result_action

            if 'report_action' in st.session_state:
                st.markdown(st.session_state['report_action'])

                # Download كل التقارير مجتمعة
                full_report = (
                    f"CS DEPARTMENT — FULL INSTITUTIONAL REPORT\n{'='*60}\n\n"
                    f"Students={ns} | Jobs={nj} | F1={met['f1']*100:.1f}%\n\n"
                    f"WEAK COURSES:\n{weak_detail}\n\n"
                    f"TOP SKILLS:\n{top20_skills}\n\n"
                    f"{'='*60}\n1. COURSE RECOMMENDATIONS\n{'='*60}\n"
                    f"{st.session_state.get('report_courses','Not generated')}\n\n"
                    f"{'='*60}\n2. GAP ANALYSIS\n{'='*60}\n"
                    f"{st.session_state.get('report_gap','Not generated')}\n\n"
                    f"{'='*60}\n3. NEW COURSES\n{'='*60}\n"
                    f"{st.session_state.get('report_new','Not generated')}\n\n"
                    f"{'='*60}\n4. ACTION PLAN\n{'='*60}\n"
                    f"{st.session_state.get('report_action','Not generated')}"
                )
                st.download_button(
                    "📥 Download Full Report (All Sections)",
                    data=full_report,
                    file_name="full_institutional_report.txt",
                    mime="text/plain",
                    key="dl_full"
                )
