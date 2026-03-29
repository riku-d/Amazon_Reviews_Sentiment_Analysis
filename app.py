"""
Amazon Product Reviews — Sentiment Analysis App
================================================
Run Amazon_Reviews_SVM.ipynb ONCE to generate Models/.
App loads instantly — no re-training on startup.

Folder structure:
  Models/
    model_svm.pkl
    tfidf_vectorizer.pkl
    scaler.pkl
    metrics.json
    data_processed.csv
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re, json, pickle, os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── NLTK ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    nltk.download("stopwords",     quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    sw = set(stopwords.words("english")) - {
        "not","no","never","nor","without","against","hardly","barely"
    }
    return sw, PorterStemmer(), SentimentIntensityAnalyzer()

STOPWORDS_NLP, STEMMER, VADER = init_nlp()
PALETTE = ["#dc3545","#28a745","#146EB4","#FF9900","#232F3E"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews · SVM",
    page_icon="https://cdn.iconscout.com/icon/free/png-512/free-amazon-icon-svg-download-png-1912058.png?f=webp&w=512",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }

.banner{
    background:linear-gradient(135deg,#0d1b2a 0%,#1b3a5c 55%,#0d1b2a 100%);
    border-radius:18px; padding:2rem 2.5rem; margin-bottom:1.5rem;
    border:1px solid #1e4976; box-shadow:0 8px 32px rgba(0,0,0,0.45);
}
.banner h1{ margin:0; font-size:2.1rem; font-weight:800;
    background:linear-gradient(90deg,#fff 0%,#f5a623 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.banner p{ margin:0.4rem 0 0; color:#8aaec8; font-size:0.9rem; }

.kpi{ background:#0d1b2a; border:1px solid #1e4976; border-radius:14px;
      padding:1.2rem 1rem; text-align:center; color:white; margin:0.2rem;
      box-shadow:0 2px 12px rgba(0,0,0,0.3); }
.kpi .v{ font-size:2rem; font-weight:700; color:#f5a623; }
.kpi .l{ font-size:0.7rem; color:#8aaec8; text-transform:uppercase;
         letter-spacing:1px; margin-top:0.3rem; }

.badge{ background:linear-gradient(135deg,#1b3a5c,#0d1b2a);
        border:1px solid #1e4976; border-radius:14px;
        padding:1.2rem 1.4rem; color:white; margin:0.3rem 0;
        box-shadow:0 4px 16px rgba(0,0,0,0.3); }
.badge h3{ margin:0 0 0.4rem; font-size:0.78rem; color:#f5a623;
           text-transform:uppercase; letter-spacing:1px; }
.badge .v{ font-size:1.9rem; font-weight:700; }

.pos-pill{ background:#d4edda; color:#155724; padding:0.5rem 1.6rem;
           border-radius:30px; font-weight:700; font-size:1.1rem; display:inline-block; }
.neg-pill{ background:#f8d7da; color:#721c24; padding:0.5rem 1.6rem;
           border-radius:30px; font-weight:700; font-size:1.1rem; display:inline-block; }

.info-box{ background:linear-gradient(135deg,#0d1b2a 0%,#1b3a5c 55%,#0d1b2a 100%);border-left:4px solid #146EB4;
           border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0; }
.warn-box{ background:#fff3cd; border-left:4px solid #ffc107;
           border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0; }
.fix-box{  background:#d4edda; border-left:4px solid #28a745;
           border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0; }
.mono{ font-family:'DM Mono',monospace; font-size:0.82rem;
       background:linear-gradient(135deg,#0d1b2a,#1b3a5c); color:#f0c070;
       border-left:4px solid #f5a623; border-radius:8px; padding:0.7rem 1rem; }

.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{ background:#1b3a5c; color:#c9e6ff;
    border-radius:8px 8px 0 0; padding:8px 20px; font-weight:500; transition:all 0.2s; }
.stTabs [data-baseweb="tab"]:hover{ background:#254d78; color:#fff; }
.stTabs [aria-selected="true"]{ background:#0d1b2a !important; color:#f5a623 !important;
    font-weight:700; box-shadow:0 3px 10px rgba(0,0,0,0.5); }
div[data-testid="stExpander"]{ border:1px solid #1e4976; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODELS_DIR = "Models"

@st.cache_resource
def load_models():
    needed = ["model_svm.pkl","tfidf_vectorizer.pkl","scaler.pkl",
              "metrics.json","data_processed.csv"]
    missing = [f for f in needed if not os.path.exists(f"{MODELS_DIR}/{f}")]
    if missing:
        return None,None,None,None,None,missing
    mdl   = pickle.load(open(f"{MODELS_DIR}/model_svm.pkl","rb"))
    tfidf = pickle.load(open(f"{MODELS_DIR}/tfidf_vectorizer.pkl","rb"))
    scl   = pickle.load(open(f"{MODELS_DIR}/scaler.pkl","rb"))
    with open(f"{MODELS_DIR}/metrics.json") as f:
        mets = json.load(f)
    dframe = pd.read_csv(f"{MODELS_DIR}/data_processed.csv")
    return mdl,tfidf,scl,mets,dframe,[]

model,tfidf,scaler,metrics,df,missing_files = load_models()

# ── NLP helpers ───────────────────────────────────────────────────────────────
def expand_contractions(text, contractions):
    text = text.lower()
    for k,v in contractions.items():
        text = text.replace(k,v)
    return text

def build_tokens(text, contractions):
    text = expand_contractions(text, contractions)
    text = re.sub("[^a-zA-Z]"," ",text)
    words = text.lower().split()
    tokens = []
    negate = False
    for word in words:
        if word in {"not","no","never","nor","without"}:
            negate = True; continue
        stem = STEMMER.stem(word)
        if word not in STOPWORDS_NLP:
            tokens.append(("not_"+stem) if negate else stem)
        negate = False
    return tokens

def run_predict(text, contractions, negative_phrases, threshold=0.40):
    toks     = build_tokens(text, contractions)
    tf_vec   = tfidf.transform([" ".join(toks)]).toarray()
    sc       = VADER.polarity_scores(text)
    n_ph     = sum(1 for p in negative_phrases if p in text.lower())
    extra    = np.array([[sc["compound"],sc["pos"],sc["neg"],sc["neu"],n_ph]])
    combined = np.hstack([tf_vec,extra])
    scaled   = scaler.transform(combined)
    proba    = model.predict_proba(scaled)[0]
    classes  = list(model.classes_)
    # classes are [1,2] → index of class 2 (Positive)
    pos_idx  = classes.index(2)
    neg_idx  = classes.index(1)
    prob_pos = proba[pos_idx]
    prob_neg = proba[neg_idx]
    pred     = 2 if prob_pos >= threshold else 1
    return pred, prob_pos, prob_neg, sc, n_ph

def make_wc(text, title, cmap):
    wc = WordCloud(background_color="white",max_words=80,width=700,height=380,colormap=cmap)
    wc.generate(text)
    fig,ax = plt.subplots(figsize=(7,3.6))
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    ax.set_title(title,fontsize=12,fontweight="bold")
    plt.tight_layout()
    return fig

def top_words(texts, stop, n=15):
    words = " ".join(texts).lower().split()
    words = [w for w in words if w.isalpha() and w not in stop and len(w)>2]
    return Counter(words).most_common(n)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/amazon (1).svg", width=150)
    st.markdown("**Sentiment Analysis**")
    st.markdown("---")
    if model:
        # Safe values
        bp = metrics["best_params"]
        kernel_val = bp.get("kernel", "Linear (LinearSVC)")

        st.markdown(
            f'<div class="mono">kernel = <b>{kernel_val}</b><br>'
            f'C      = <b>{bp["C"]}</b><br>'
            f'SMOTE  = <b>✅</b> · VADER = <b>✅</b><br>'
            f'Phrases= <b>✅</b></div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        di = metrics["dataset_info"]
        st.markdown(f"**Train size:** `{di['train_size']:,}`")
        st.markdown(f"**Test size:** `{di['test_size']:,}`")
        st.markdown(f"**Test Accuracy:** `{metrics['test_accuracy']*100:.2f}%`")
        st.markdown(f"**ROC-AUC:** `{metrics['roc_auc']:.4f}`")
        rpt = metrics["classification_report"]
        neg_rec = rpt["Negative"]["recall"]
        st.markdown("**Negative Recall:**")
        st.progress(neg_rec, text=f"{neg_rec*100:.1f}%")
        st.markdown("---")
        st.markdown("[🔗 Kaggle Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)")
    else:
        st.error("Run notebook first to generate Models/")

# ── Banner ────────────────────────────────────────────────────────────────────
import base64

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image("assets/amazon (1).svg")

st.markdown(f"""
<div class="banner" style="display:flex; align-items:center; gap:105px;">
  
  <img src="data:image/svg+xml;base64,{img_base64}"
       width="370"
       style="border-radius:10px; flex-shrink:0;">

  <div>
        <h1>Amazon Product Reviews — Sentiment Analysis</h1>
        <p>SVM · TF-IDF Trigrams · SMOTE · VADER · GridSearchCV</p>
  </div>

</div>
""", unsafe_allow_html=True)

if missing_files:
    st.error(f"**Missing files in `Models/`:** {', '.join(missing_files)}")
    st.markdown("""
    **Setup steps:**
    1. Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
    2. Place both CSV files in the same folder as this app
    3. Open **`Amazon_Reviews_SVM.ipynb`** and run all cells
    4. Refresh this page ⚡
    ```
    Models/
    ├── model_svm.pkl
    ├── tfidf_vectorizer.pkl
    ├── scaler.pkl
    ├── metrics.json
    └── data_processed.csv
    ```
    """)
    st.stop()

# Unpack saved config
contractions     = metrics.get("contractions", {})
negative_phrases = metrics.get("negative_phrases", [])
threshold        = metrics.get("prediction_threshold", 0.40)
di               = metrics["dataset_info"]
rpt              = metrics["classification_report"]
bp               = metrics["best_params"]
cb               = di

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊  EDA",
    "⚙️  Pipeline",
    "🤖  Model & Tuning",
    "🔍  Predict",
    "📈  Metrics",
])

# ══════════════════════════════════════════════════════
# TAB 1 · EDA
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis — Amazon Product Reviews")

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (f"{di['train_size']+di['test_size']:,}", "Total Samples Used"),
        (f"{di['pos_count']:,}",                   "Positive (Train)"),
        (f"{di['neg_count']:,}",                   "Negative (Train)"),
        ("1=Neg · 2=Pos",                          "Label Mapping"),
    ]
    for col,(val,lbl) in zip([c1,c2,c3,c4],kpis):
        col.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Dataset Preview")
    cols_show = [c for c in ["sentiment","title","review","review_length","word_count"] if c in df.columns]
    st.dataframe(df[cols_show].head(12), width="stretch")

    st.markdown("---")
    st.markdown("### Sentiment Distribution")
    l,r = st.columns(2)
    with l:
        sent_counts = df["sentiment"].value_counts().sort_index()
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=["Negative (1)","Positive (2)"],
            y=sent_counts.values,
            marker_color=["#dc3545","#28a745"],
            text=sent_counts.values, textposition="outside",
        ))
        fig_sent.update_layout(title="Sentiment Count", template="simple_white",
                               yaxis_title="Count",
                               yaxis_range=[0, sent_counts.max()*1.15])
        st.plotly_chart(fig_sent, width="stretch")

    with r:
        fig_pie = px.pie(
            values=sent_counts.values,
            names=["Negative","Positive"],
            color_discrete_sequence=["#dc3545","#28a745"],
            title="Sentiment Distribution (%)", hole=0.38
        )
        st.plotly_chart(fig_pie, width="stretch")

    st.markdown("---")
    st.markdown("### Review Length Analysis")
    if "review_length" in df.columns:
        c1,c2 = st.columns(2)
        with c1:
            fig_rl = px.histogram(
                df, x="review_length",
                color=df["sentiment"].map({1:"Negative",2:"Positive"}),
                barmode="overlay", nbins=60,
                color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                title="Review Length by Sentiment",
                labels={"review_length":"Characters","color":"Sentiment"}
            )
            fig_rl.update_layout(template="simple_white")
            st.plotly_chart(fig_rl, width="stretch")

        with c2:
            if "word_count" in df.columns:
                fig_wc = px.box(
                    df, x=df["sentiment"].map({1:"Negative",2:"Positive"}),
                    y="word_count",
                    color=df["sentiment"].map({1:"Negative",2:"Positive"}),
                    color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                    title="Word Count Distribution",
                    labels={"x":"Sentiment","word_count":"Word Count"}
                )
                fig_wc.update_layout(template="simple_white", showlegend=False,
                                     xaxis_title="Sentiment")
                st.plotly_chart(fig_wc, width="stretch")

        # Mean length comparison
        mean_rl = df.groupby("sentiment")["review_length"].mean()
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Bar(
            x=["Negative","Positive"],
            y=mean_rl.values,
            marker_color=["#dc3545","#28a745"],
            text=[f"{v:.0f}" for v in mean_rl.values],
            textposition="outside",
        ))
        fig_ml.update_layout(title="Mean Review Length by Sentiment",
                             template="simple_white", yaxis_title="Characters")
        st.plotly_chart(fig_ml, width="stretch")

    st.markdown("---")
    st.markdown("### Top Words by Sentiment")
    stop = set(stopwords.words("english"))
    c1,c2 = st.columns(2)
    for col, sent_val, colour, title in [
        (c1, 1, "#dc3545", "Top 15 Words — Negative"),
        (c2, 2, "#28a745", "Top 15 Words — Positive"),
    ]:
        with col:
            subset = df[df["sentiment"]==sent_val]["review"].astype(str)
            words  = top_words(subset, stop, n=15)
            if words:
                wds, cts = zip(*words)
                fig_tw = go.Figure(go.Bar(
                    x=list(cts)[::-1], y=list(wds)[::-1],
                    orientation="h", marker_color=colour,
                ))
                fig_tw.update_layout(title=title, template="simple_white",
                                     xaxis_title="Frequency", height=420)
                st.plotly_chart(fig_tw, width="stretch")

    st.markdown("---")
    st.markdown("### Word Clouds")
    wc1,wc2 = st.columns(2)
    neg_text_ = " ".join(df[df["sentiment"]==1]["review"].astype(str))
    pos_text_ = " ".join(df[df["sentiment"]==2]["review"].astype(str))
    with wc1: st.pyplot(make_wc(neg_text_,"Negative Reviews","Reds"),width="stretch")
    with wc2: st.pyplot(make_wc(pos_text_,"Positive Reviews","Greens"),width="stretch")

    st.markdown("---")
    st.markdown("### Title Length Analysis")
    if "title_length" in df.columns:
        fig_tl = px.histogram(
            df, x="title_length",
            color=df["sentiment"].map({1:"Negative",2:"Positive"}),
            barmode="overlay", nbins=50,
            color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
            title="Title Length by Sentiment",
        )
        fig_tl.update_layout(template="simple_white")
        st.plotly_chart(fig_tl, width="stretch")

    st.markdown("---")
    st.markdown("### Correlation Heatmap")
    num_cols = [c for c in ["review_length","word_count","title_length","sentiment"] if c in df.columns]
    if len(num_cols) >= 2:
        fig_corr, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Feature Correlation Matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr, width="stretch")


# ══════════════════════════════════════════════════════
# TAB 2 · Pipeline
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("⚙️ Preprocessing Pipeline")

    st.markdown("""
    <div class="info-box">
    <b>Dataset:</b> Amazon Product Reviews — columns: <code>sentiment</code> (1=Neg, 2=Pos),
    <code>title</code> (product title), <code>review</code> (review body).
    Title and review are <b>concatenated</b> to maximise the text signal before preprocessing.
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣  Load & Sample",         "20k train / 5k test samples from Kaggle CSVs (no header row)"),
        ("2️⃣  Null Handling",          "title nulls → 'No Title' · review nulls → empty string"),
        ("3️⃣  Combine Fields",         "`full_text = title + ' ' + review` — richer signal than review alone"),
        ("4️⃣  Contraction Expansion",  "20 contractions expanded: won't→will not, can't→can not, etc."),
        ("5️⃣  Negation Prefixing",     '"not good" → token `not_good` · preserves negation through stopword removal'),
        ("6️⃣  Regex + Lowercase",      "Strip non-alpha, lowercase everything"),
        ("7️⃣  Stopword Removal",       f"{len(STOPWORDS_NLP)} stopwords removed — KEEPS: not/no/never/nor/without"),
        ("8️⃣  Porter Stemming",        "disappointed→disappoint · working→work · breaking→break"),
        ("9️⃣  TF-IDF (trigrams)",      "max_features=5000, ngram_range=(1,3), sublinear_tf=True, min_df=2"),
        ("🔟  VADER Lexicon",          "4 numeric features: compound/pos/neg/neu — catches implicit sentiment"),
        ("1️⃣1️⃣  Phrase Blacklist",      f"{len(negative_phrases)} negative phrases: 'waste of money', 'stopped working'..."),
        ("1️⃣2️⃣  MinMaxScaler",          "Scale all 5005 features to [0,1] for SVM's distance geometry"),
        ("1️⃣3️⃣  SMOTE",                f"Applied if imbalance ratio > 1.5 — balances training set synthetically"),
    ]
    for title,desc in steps:
        with st.expander(title):
            st.markdown(desc)

    st.markdown("---")
    st.markdown("### 🧠 SVM Theory")
    col_l, col_r = st.columns([1.2,1])
    with col_l:
        st.markdown("""
**SVM for Text Classification:**

| Concept | Meaning |
|---|---|
| **Hyperplane** | `w·x + b = 0` — decision boundary |
| **Support Vectors** | Closest training points — define the margin |
| **Margin** | `2/‖w‖` — SVM maximises this |
| **C** | Regularisation: low→wide margin, high→narrow |
| **Kernel trick** | Non-linear boundaries via K(xᵢ,xⱼ) |
| **class_weight='balanced'** | Corrects for class imbalance |
| **F1 macro scoring** | Both classes penalised equally in grid search |

**Why title+review combined?**
> Titles like *"Terrible product"* or *"Love this!"* carry strong sentiment.
> Concatenating them gives TF-IDF and VADER richer signal.

**Why trigrams?**
> Captures *"does not work"*, *"stopped working after"*
> as single discriminative features.
        """)
    with col_r:
        # SVM 2D illustration
        from sklearn.svm import SVC as _SVC
        np.random.seed(42)
        n = 45
        Xp = np.random.randn(n,2)+[2.2,2.2]
        Xn = np.random.randn(n,2)+[-2.2,-2.2]
        Xt = np.vstack([Xp,Xn])
        yt = np.array([1]*n+[0]*n)
        toy = _SVC(kernel="linear",C=1); toy.fit(Xt,yt)
        xx,yy = np.meshgrid(np.linspace(-6,6,250),np.linspace(-6,6,250))
        Z = toy.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        fig_sv,ax = plt.subplots(figsize=(5.5,5))
        ax.contourf(xx,yy,Z,levels=[-4,-1,0,1,4],
                    colors=["#f8d7da","#fff9e6","#d4edda"],alpha=0.5)
        ax.contour(xx,yy,Z,levels=[-1,0,1],
                   linestyles=["--","-","--"],
                   colors=["#dc3545","#232F3E","#28a745"],
                   linewidths=[1.8,2.8,1.8])
        ax.scatter(Xp[:,0],Xp[:,1],c="#28a745",edgecolors="white",s=55,label="Positive",zorder=3)
        ax.scatter(Xn[:,0],Xn[:,1],c="#dc3545",edgecolors="white",s=55,label="Negative",zorder=3)
        sv = toy.support_vectors_
        ax.scatter(sv[:,0],sv[:,1],s=190,facecolors="none",
                   edgecolors="#232F3E",linewidths=2.2,label="Support Vectors",zorder=5)
        ax.set_title("SVM — Optimal Hyperplane",fontweight="bold",fontsize=11)
        ax.legend(fontsize=9,loc="upper left")
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.set_xlim(-6,6); ax.set_ylim(-6,6)
        plt.tight_layout()
        st.pyplot(fig_sv, width="stretch")

    st.markdown("---")
    st.markdown("### VADER on Amazon-style Reviews")
    test_reviews = [
        "Waste of money, broke after two days",
        "Does not work as advertised",
        "Absolutely love this product, works perfectly!",
        "Returned it immediately, terrible quality",
        "Best purchase I have made in years",
        "Cheap and flimsy, fell apart immediately",
    ]
    vader_rows = []
    for txt in test_reviews:
        sc   = VADER.polarity_scores(txt)
        n_ph = sum(1 for p in negative_phrases if p in txt.lower())
        vader_rows.append({
            "Review":       txt,
            "neg":          round(sc["neg"],3),
            "pos":          round(sc["pos"],3),
            "compound":     round(sc["compound"],3),
            "Phrase hits":  n_ph,
            "Signal":       "🔴 Neg" if sc["compound"]<0 or n_ph>0 else "🟢 Pos",
        })
    st.dataframe(pd.DataFrame(vader_rows), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════
# TAB 3 · Model & Tuning
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 GridSearchCV & Model Analysis")
    kernel_val = bp.get("kernel", "Linear (LinearSVC)")
    b1,b2= st.columns(2)
    with b1:
        st.markdown(f'<div class="badge"><h3>Kernel</h3><div class="v">{kernel_val}</div></div>', unsafe_allow_html=True)

    with b2:
        st.markdown(f'<div class="badge"><h3>C</h3><div class="v">{bp["C"]}</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="mono">GridSearchCV: 3 C values × 3-fold CV = <b>9 fits</b> | '
        f'scoring = <b>f1_macro</b><br>'
        f'SMOTE training set: <b>{cb["smote_neg"]:,} neg + {cb["smote_pos"]:,} pos</b></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📊 Performance Summary")
    perf_df = pd.DataFrame([
        {"Metric":"Train Accuracy",  "Value":f"{metrics['train_accuracy']*100:.2f}%"},
        {"Metric":"Test Accuracy",   "Value":f"{metrics['test_accuracy']*100:.2f}%"},
        {"Metric":"ROC-AUC",         "Value":f"{metrics['roc_auc']:.4f}"},
        {"Metric":"Avg Precision",   "Value":f"{metrics['avg_precision']:.4f}"},
        {"Metric":"CV F1 Mean",      "Value":f"{metrics['cv_f1_mean']*100:.2f}%"},
        {"Metric":"CV F1 Std",       "Value":f"±{metrics['cv_f1_std']*100:.2f}%"},
        {"Metric":"Neg Recall",      "Value":f"{rpt['Negative']['recall']*100:.1f}%"},
        {"Metric":"Neg F1",          "Value":f"{rpt['Negative']['f1-score']*100:.1f}%"},
        {"Metric":"Threshold",       "Value":str(threshold)},
    ])
    st.dataframe(perf_df, width="stretch", hide_index=True)

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("#### Confusion Matrix")
        cm_arr = np.array(metrics["confusion_matrix"])
        fig_cm,ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm_arr,display_labels=["Negative","Positive"]).plot(
            ax=ax,colorbar=False,cmap="Blues")
        ax.set_title("Best SVM — LinearSVC", fontweight="bold")
        for lbl,(r,c_) in zip(["TN","FP","FN","TP"],[(0,0),(0,1),(1,0),(1,1)]):
            ax.text(c_+0.5,r+0.72,lbl,ha="center",fontsize=9,
                    color="white" if cm_arr[r,c_]>cm_arr.max()*0.5 else "gray",
                    transform=ax.transData)
        plt.tight_layout()
        st.pyplot(fig_cm, width="stretch")

    with col_b:
        st.markdown("#### Classification Report")
        st.dataframe(pd.DataFrame(rpt).transpose().round(3), width="stretch")

    col_c,col_d = st.columns(2)
    with col_c:
        st.markdown("#### ROC Curve")
        fpr_v = metrics["roc_fpr"]; tpr_v = metrics["roc_tpr"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_v,y=tpr_v,mode="lines",
            name=f"SVM (AUC={metrics['roc_auc']:.3f})",
            line=dict(color="#f5a623",width=2.5),
            fill="tozeroy",fillcolor="rgba(245,166,35,0.1)"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            name="Random",line=dict(color="#dc3545",dash="dash")))
        fig_roc.update_layout(template="simple_white",title="ROC Curve",
                              xaxis_title="FPR",yaxis_title="TPR")
        st.plotly_chart(fig_roc, width="stretch")

    with col_d:
        st.markdown("#### SMOTE Balance Impact")
        smote_d = pd.DataFrame({
            "Split":["Train (orig)","Train (orig)","After SMOTE","After SMOTE","Test","Test"],
            "Class":["Negative","Positive","Negative","Positive","Negative","Positive"],
            "Count":[cb["neg_count"],cb["pos_count"],
                     cb["smote_neg"],cb["smote_pos"],
                     cb["test_neg"],cb["test_pos"]]
        })
        fig_sm2 = px.bar(smote_d,x="Split",y="Count",color="Class",
                         color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                         barmode="group",title="Class Balance Across Splits")
        fig_sm2.update_layout(template="simple_white")
        st.plotly_chart(fig_sm2, width="stretch")

    st.markdown("---")
    st.markdown("### Per-Class Metric Bar Chart")
    cls_rows = []
    for cls in ["Negative","Positive"]:
        if cls in rpt:
            cls_rows.append({
                "Class":cls,
                "Precision %":round(rpt[cls]["precision"]*100,2),
                "Recall %":   round(rpt[cls]["recall"]*100,2),
                "F1-Score %": round(rpt[cls]["f1-score"]*100,2),
            })
    cls_melt = pd.DataFrame(cls_rows).melt(id_vars="Class",var_name="Metric",value_name="Score")
    fig_cls = px.bar(cls_melt,x="Metric",y="Score",color="Class",barmode="group",
                     color_discrete_map={"Negative":"#dc3545","Positive":"#28a745"},
                     title="Per-Class Metrics Comparison",text_auto=True)
    fig_cls.update_traces(texttemplate="%{y:.1f}%",textposition="outside")
    fig_cls.update_layout(template="simple_white",yaxis_range=[0,110])
    st.plotly_chart(fig_cls, width="stretch")


# ══════════════════════════════════════════════════════
# TAB 4 · Predict
# ══════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Real-Time Sentiment Prediction")
    st.markdown("**VADER + phrase detection + SVM** — catches both explicit and implicit negatives. ⚡")

    c_set,_ = st.columns([1,2])
    with c_set:
        user_threshold = st.slider(
            "Prediction threshold",
            min_value=0.20, max_value=0.60,
            value=float(threshold), step=0.05,
            help="Predict Positive only if P(positive) ≥ threshold. Lower = more sensitive to negatives."
        )

    user_rev = st.text_area(
        "Enter an Amazon product review (title + review body):",
        placeholder='"Waste of money. Stopped working after a week. Would not recommend."'
                    '\nor\n'
                    '"Amazing product! Works exactly as described, very happy with this purchase."',
        height=130
    )

    if st.button("🚀  Predict Sentiment", type="primary"):
        if not user_rev.strip():
            st.warning("Please enter a review.")
        else:
            pred,prob_pos,prob_neg,vader_sc,n_ph = run_predict(
                user_rev, contractions, negative_phrases, user_threshold)
            pill  = "pos-pill" if pred==2 else "neg-pill"
            label = "Positive ✅" if pred==2 else "Negative ❌"

            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f'<br><span class="{pill}">{label}</span>',unsafe_allow_html=True)
            with c2: st.metric("Positive confidence",f"{prob_pos*100:.1f}%")
            with c3: st.metric("Negative confidence",f"{prob_neg*100:.1f}%")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_pos*100,
                delta={"reference":user_threshold*100},
                title={"text":"Positive Sentiment Score (%)"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"#28a745" if pred==2 else "#dc3545"},
                    "steps":[
                        {"range":[0,user_threshold*100],"color":"#fde8ea"},
                        {"range":[user_threshold*100,65],"color":"#fff9e6"},
                        {"range":[65,100],"color":"#e8f5e9"},
                    ],
                    "threshold":{"line":{"color":"#232F3E","width":3},"value":user_threshold*100}
                }
            ))
            fig_g.update_layout(height=280,margin=dict(t=60,b=10,l=20,r=20))
            st.plotly_chart(fig_g, width="stretch")

            st.markdown("#### 🔬 Feature Breakdown")
            fa,fb_col,fc = st.columns(3)
            with fa:
                st.markdown("**VADER Scores**")
                st.dataframe(pd.DataFrame({
                    "Score":  ["Negative","Positive","Neutral","Compound"],
                    "Value":  [round(vader_sc["neg"],3),round(vader_sc["pos"],3),
                               round(vader_sc["neu"],3),round(vader_sc["compound"],3)]
                }),width="stretch",hide_index=True)
            with fb_col:
                st.markdown("**Phrase Hits**")
                matched = [p for p in negative_phrases if p in user_rev.lower()]
                if matched:
                    st.error(f"**{len(matched)} negative phrase(s):**\n\n" +
                             "\n".join(f"• {p}" for p in matched))
                else:
                    st.success("No negative phrases detected")
            with fc:
                st.markdown("**Tokens (preview)**")
                toks = build_tokens(user_rev, contractions)
                st.code(" · ".join(toks[:30])+("..." if len(toks)>30 else ""))

    st.markdown("---")
    st.markdown("#### 🧪 Sample Reviews")
    samples = {
        "😊 Excelent Product":     ("Absolutely love this! Best smart speaker ever. Plays music perfectly, controls all my devices, answers every question. 5 stars!",2),
        "😡 Broke quickly":        ("Terrible product. Broke after a week. Support was useless. Wasted my money. Horrible experience. Will never buy again.",1),
        "😐 Neutral / Mixed":      ("It works great for basic use but nothing special, don't buy",1),
        "👍 Good value":          ("Good product for the price. Does what it says. Happy with it.", 2),
    }
    cols = st.columns(2)
    for i,(lbl,(txt,true_lbl)) in enumerate(samples.items()):
        with cols[i%2]:
            with st.expander(lbl):
                st.write(txt)
                st.caption(f"True label: {'✅ Positive' if true_lbl==2 else '❌ Negative'}")
                if st.button("Predict this", key=lbl):
                    p,pp,pn,vs,np_ = run_predict(txt,contractions,negative_phrases,user_threshold)
                    result  = "✅ Positive" if p==2 else "❌ Negative"
                    correct = "🎯 Correct!" if p==true_lbl else "⚠️ Missed"
                    st.markdown(f"**{result}** — P(pos):{pp*100:.1f}% {correct}")
                    if vs["compound"]<0 or np_>0:
                        st.caption(f"VADER:{vs['compound']:.3f} · Phrases:{np_}")


# ══════════════════════════════════════════════════════
# TAB 5 · Metrics
# ══════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Full Metrics Dashboard")

    st.markdown("### Per-Class Metrics")
    class_data = []
    for cls in ["Negative","Positive"]:
        if cls in rpt:
            class_data.append({
                "Class":       cls,
                "Precision %": round(rpt[cls]["precision"]*100,2),
                "Recall %":    round(rpt[cls]["recall"]*100,2),
                "F1-Score %":  round(rpt[cls]["f1-score"]*100,2),
                "Support":     int(rpt[cls]["support"]),
            })
    st.dataframe(pd.DataFrame(class_data),width="stretch",hide_index=True)

    
    st.markdown("---")
    st.markdown("### Macro & Weighted Averages")
    avg_data = []
    for avg_type in ["macro avg","weighted avg"]:
        if avg_type in rpt:
            avg_data.append({
                "Average":     avg_type.title(),
                "Precision %": round(rpt[avg_type]["precision"]*100,2),
                "Recall %":    round(rpt[avg_type]["recall"]*100,2),
                "F1-Score %":  round(rpt[avg_type]["f1-score"]*100,2),
            })
    st.dataframe(pd.DataFrame(avg_data),width="stretch",hide_index=True)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Test Accuracy",  f"{metrics['test_accuracy']*100:.2f}%")
    c2.metric("ROC-AUC",        f"{metrics['roc_auc']:.4f}")
    c3.metric("Neg Recall",     f"{rpt['Negative']['recall']*100:.1f}%")
    c4.metric("Macro F1",       f"{rpt['macro avg']['f1-score']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🏁 Design Decisions")
    st.markdown(f"""
| Decision | Choice | Justification |
|---|---|---|
| **Algorithm** | SVM | Max-margin; excels in high-dim TF-IDF space |
| **Kernel** | `Linear (LinearSVC)` | Best F1-macro via 120-fit GridSearchCV |
| **C** | `{bp['C']}` | Regularisation — optimised on F1-macro |
| **Text fields** | title + review | Title carries strong sentiment signal |
| **n-grams** | trigrams (1,3) | "does not work" as one feature unit |
| **Features** | TF-IDF + VADER + phrases | VADER catches implicit negatives TF-IDF misses |
| **Imbalance** | SMOTE | Synthetic minorities → balanced training |
| **class_weight** | balanced | Equal misclassification cost per class |
| **Threshold** | {threshold} | Tuned for better negative recall |
| **Scoring** | F1-macro | Treats both classes equally in grid search |
| **Scaling** | MinMaxScaler | SVM distances require normalised inputs |
| **Negation** | Prefix `not_` | "not good" ≠ "good" without this |
""")
