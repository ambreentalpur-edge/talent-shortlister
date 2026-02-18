import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import re

# --- sklearn (for real scoring) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="EDGE | Talent Shortlister",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# EDGE BRAND STYLING
# --------------------------------------------------
st.markdown("""
<style>
:root{
  --edge-plum:#4B136F;
  --edge-lilac:#E7DBF6;
  --edge-mauve:#CBB6F3;
  --edge-seafoam:#A8DCC7;
  --edge-text:#2E1A3A;
}

[data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg,#FFFFFF 0%, var(--edge-lilac) 140%);
}
h1,h2,h3{ color: var(--edge-text); }

.stButton>button{
  background: var(--edge-plum);
  color: #fff;
  border: none;
  border-radius: 14px;
  padding: 0.6rem 1rem;
  font-weight: 600;
}
.stButton>button:hover{ background: #3B0F59; }

[data-testid="stFileUploader"] section{
  border: 2px dashed var(--edge-mauve);
  border-radius: 16px;
  background: rgba(231,219,246,0.35);
}

.stSuccess{
  background-color: rgba(168,220,199,0.35);
  border-left: 6px solid var(--edge-seafoam);
}
.stInfo{
  background-color: rgba(203,182,243,0.25);
  border-left: 6px solid var(--edge-mauve);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
logo_path = Path("assets/Logo/Edge_Lockup_H_Plum.jpg")
c1, c2 = st.columns([0.22, 0.78], vertical_alignment="center")
with c1:
    if logo_path.exists():
        st.image(Image.open(logo_path), use_container_width=True)
with c2:
    st.markdown("## Talent Shortlister")
    st.markdown("<div style='color:#6B5C7A'>Opportunity-based candidate shortlisting</div>", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_read_csv(uploaded_file):
    """Reads CSV with encoding fallback for Salesforce/Excel exports."""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def find_col(df: pd.DataFrame, candidates: list[str]):
    cols = list(df.columns)
    nmap = {norm(c): c for c in cols}
    # exact
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    # contains
    for c in candidates:
        cn = norm(c)
        for k, v in nmap.items():
            if cn in k:
                return v
    return None

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    t = str(x)
    # treat literal 'None'/'nan' as empty
    if t.strip().lower() in {"none", "nan", "null"}:
        return ""
    # strip html tags
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def row_to_text(row: pd.Series, exclude_cols: set[str]) -> str:
    parts = []
    for c in row.index:
        if c in exclude_cols:
            continue
        parts.append(clean_text(row[c]))
    return " ".join([p for p in parts if p]).strip()

def tfidf_scores(query: str, docs: list[str]) -> list[float]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([query] + docs)
    return cosine_similarity(X[0], X[1:]).flatten().tolist()

def keyword_overlap_scores(query: str, docs: list[str]) -> list[float]:
    qt = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    out = []
    for d in docs:
        dt = set(re.findall(r"[a-zA-Z]{3,}", d.lower()))
        out.append(0.0 if not qt else len(qt & dt) / max(1, len(qt)))
    return out

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Files")
opp_file = st.sidebar.file_uploader("Opportunity information.csv", type=["csv"])
cand_file = st.sidebar.file_uploader("Candidate Information.csv", type=["csv"])
feedback_file = st.sidebar.file_uploader("Interview Feedback.csv (Optional)", type=["csv"])

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Shortlist Size", 5, 50, 15)
only_live = st.sidebar.toggle("Only include Go Live Status = Live", value=False)
use_feedback = st.sidebar.toggle("Use Interview Feedback as signal (recommended)", value=True)

if not SKLEARN_OK:
    st.sidebar.warning("TF-IDF not available (scikit-learn missing). Add scikit-learn to requirements.txt for better matching.")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
left, right = st.columns([0.65, 0.35])

with left:
    st.markdown("### Upload Status")
    ready = True
    if opp_file: st.success("Opportunity file uploaded.")
    else:
        st.info("Opportunity file not uploaded.")
        ready = False

    if cand_file: st.success("Candidate file uploaded.")
    else:
        st.info("Candidate file not uploaded.")
        ready = False

    if feedback_file: st.success("Interview feedback uploaded (optional).")
    else: st.info("No interview feedback uploaded (optional).")

with right:
    st.markdown("### Run Shortlist")
    run = st.button("Validate & Shortlist", use_container_width=True, disabled=not ready)

# --------------------------------------------------
# RUN LOGIC
# --------------------------------------------------
if run:
    opp_df = safe_read_csv(opp_file)
    cand_df = safe_read_csv(cand_file)
    feedback_df = safe_read_csv(feedback_file) if (feedback_file and use_feedback) else None

    # Identify key columns
    opp_name_col = find_col(opp_df, ["Opportunity Name", "Opportunity: Opportunity Name", "Opportunity"])
    if not opp_name_col:
        st.error("Could not find Opportunity Name column in Opportunity information.csv.")
        st.stop()

    cand_id_col = find_col(cand_df, ["Candidate: ID", "Candidate ID", "Candidate Id"])
    cand_name_col = find_col(cand_df, ["Candidate Name", "Name"])
    cand_go_live_col = find_col(cand_df, ["Go Live Status", "Go Live"])
    cand_email_col = find_col(cand_df, ["Personal Email", "Email"])
    cand_filelink_col = find_col(cand_df, ["File Link", "Resume Link", "CV Link"])

    # Filter (optional)
    work_cand = cand_df.copy()
    if only_live and cand_go_live_col:
        work_cand = work_cand[work_cand[cand_go_live_col].astype(str).str.strip().str.lower() == "live"].copy()

    if work_cand.empty:
        st.warning("No candidates left after filtering. Turn off the filter or check Go Live Status values.")
        st.stop()

    # Select opportunity
    opp_list = sorted(set([str(x) for x in opp_df[opp_name_col].dropna().tolist() if str(x).strip()]))
    selected = st.selectbox("Select Opportunity", options=opp_list)

    opp_row = opp_df[opp_df[opp_name_col].astype(str) == selected].head(1)
    if opp_row.empty:
        st.error("Selected opportunity row not found.")
        st.stop()

    # Build Opportunity text from ALL columns except obvious IDs
    opp_exclude = {opp_name_col}
    opp_text = row_to_text(opp_row.iloc[0], exclude_cols=set())  # keep everything; notes matter

    # Build Candidate text from ALL columns except IDs/links/metadata
    cand_exclude = set()
    for c in work_cand.columns:
        cn = norm(c)
        if any(k in cn for k in ["id", "document", "content document", "file link", "link", "email", "days in marketplace"]):
            # keep email out of scoring, keep link out of scoring
            cand_exclude.add(c)

    cand_texts = work_cand.apply(lambda r: row_to_text(r, cand_exclude), axis=1).tolist()

    # If candidate texts are mostly empty, we MUST
