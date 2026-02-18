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
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
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
    if t.strip().lower() in {"none", "nan", "null"}:
        return ""
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
top_k = st.sidebar.slider("Shortlist Size", 4, 50, 15)
only_marketplace = st.sidebar.toggle("Only include Go Live Status = On Marketplace", value=True)
use_feedback = st.sidebar.toggle("Use Interview Feedback as signal (recommended)", value=True)

if not SKLEARN_OK:
    st.sidebar.warning("TF-IDF not available (scikit-learn missing). Add scikit-learn to requirements.txt for better matching.")

# --------------------------------------------------
# MAIN STATUS
# --------------------------------------------------
left, right = st.columns([0.65, 0.35])

ready = bool(opp_file and cand_file)

with left:
    st.markdown("### Upload Status")
    if opp_file: st.success("Opportunity file uploaded.")
    else: st.info("Opportunity file not uploaded.")
    if cand_file: st.success("Candidate file uploaded.")
    else: st.info("Candidate file not uploaded.")
    if feedback_file: st.success("Interview feedback uploaded (optional).")
    else: st.info("No interview feedback uploaded (optional).")

with right:
    st.markdown("### Run Shortlist")
    run = st.button("Generate Shortlist", use_container_width=True, disabled=not ready)

# --------------------------------------------------
# LOAD + SELECT (so Streamlit reruns don’t break)
# --------------------------------------------------
if ready:
    opp_df = safe_read_csv(opp_file)
    cand_df = safe_read_csv(cand_file)
    feedback_df = safe_read_csv(feedback_file) if (feedback_file and use_feedback) else None

    opp_name_col = find_col(opp_df, ["Opportunity Name", "Opportunity: Opportunity Name", "Opportunity"])
    if not opp_name_col:
        st.error("Could not find Opportunity Name column in Opportunity information.csv.")
        st.stop()

    cand_id_col = find_col(cand_df, ["Candidate: ID", "Candidate ID", "Candidate Id"])
    cand_name_col = find_col(cand_df, ["Candidate Name", "Name"])
    cand_go_live_col = find_col(cand_df, ["Go Live Status", "Go Live"])
    cand_email_col = find_col(cand_df, ["Personal Email", "Email"])
    cand_filelink_col = find_col(cand_df, ["File Link", "Resume Link", "CV Link"])

    # Filter eligibility
    work_cand = cand_df.copy()
    if only_marketplace and cand_go_live_col:
        work_cand = work_cand[
            work_cand[cand_go_live_col].astype(str).str.strip().str.lower().isin({"on marketplace", "marketplace"})
        ].copy()

    if work_cand.empty:
        st.warning("No candidates left after filtering. Check Go Live Status values or turn off the filter.")
        st.stop()

    opp_list = sorted(set([str(x) for x in opp_df[opp_name_col].dropna().tolist() if str(x).strip()]))
    selected = st.selectbox("Select Opportunity", options=opp_list)

    # --------------------------------------------------
    # RUN LOGIC
    # --------------------------------------------------
    if run:
        opp_row = opp_df[opp_df[opp_name_col].astype(str) == selected].head(1)
        if opp_row.empty:
            st.error("Selected opportunity row not found.")
            st.stop()

        # Opportunity text: keep everything (includes Additional Notes etc.)
        opp_text = row_to_text(opp_row.iloc[0], exclude_cols=set())

        # Candidate text: exclude ONLY true metadata columns (do NOT exclude "Industry")
        cand_exclude = set()
        for c in work_cand.columns:
            cn = norm(c)
            if any(k in cn for k in [
                "candidate: id", "candidate id",
                "content document", "contentdocument",
                "file link", "resume link", "cv link",
                "email", "days in marketplace"
            ]):
                cand_exclude.add(c)

        cand_texts = work_cand.apply(lambda r: row_to_text(r, cand_exclude), axis=1).tolist()
        empty_ratio = sum(1 for t in cand_texts if not t) / max(1, len(cand_texts))

        fb_used = False
        if feedback_df is not None:
            fb_id_col = find_col(feedback_df, ["Candidate: ID", "Candidate ID", "Candidate Id"])
            fb_email_col = find_col(feedback_df, ["Personal Email", "Email"])
            fb_text_col = find_col(feedback_df, ["Feedback", "Interview Feedback", "Comments", "Notes", "Summary", "Recommendation"])

            if not fb_text_col:
                text_cols = [c for c in feedback_df.columns if feedback_df[c].dtype == object]
                fb_text_col = max(text_cols, key=lambda c: feedback_df[c].astype(str).map(len).mean(), default=None)

            fb_map_by_id, fb_map_by_email = {}, {}

            if fb_text_col:
                if fb_id_col:
                    tmp = feedback_df[[fb_id_col, fb_text_col]].dropna()
                    tmp[fb_id_col] = tmp[fb_id_col].astype(str).str.strip()
                    fb_map_by_id = (
                        tmp.groupby(fb_id_col)[fb_text_col]
                        .apply(lambda s: " ".join(clean_text(x) for x in s.tolist()))
                        .to_dict()
                    )
                if fb_email_col:
                    tmp = feedback_df[[fb_email_col, fb_text_col]].dropna()
                    tmp[fb_email_col] = tmp[fb_email_col].astype(str).str.strip().str.lower()
                    fb_map_by_email = (
                        tmp.groupby(fb_email_col)[fb_text_col]
                        .apply(lambda s: " ".join(clean_text(x) for x in s.tolist()))
                        .to_dict()
                    )

            if fb_text_col and (fb_map_by_id or fb_map_by_email):
                new_texts = []
                for _, r in work_cand.iterrows():
                    base = row_to_text(r, cand_exclude)
                    extra = ""
                    if cand_id_col and fb_map_by_id:
                        extra = fb_map_by_id.get(str(r.get(cand_id_col, "")).strip(), "")
                    if not extra and cand_email_col and fb_map_by_email:
                        extra = fb_map_by_email.get(str(r.get(cand_email_col, "")).strip().lower(), "")
                    new_texts.append((base + " " + extra).strip())
                cand_texts = new_texts
                fb_used = True

        # Score
        if SKLEARN_OK:
            scores = tfidf_scores(opp_text, cand_texts)
            method = "TF-IDF similarity"
        else:
            scores = keyword_overlap_scores(opp_text, cand_texts)
            method = "Keyword overlap (fallback)"

        scored = work_cand.copy()
        scored["Match Score"] = scores
        scored = scored.sort_values("Match Score", ascending=False).head(top_k)

        st.success(f"Shortlist generated using {method}." + (" (Interview feedback included)" if fb_used else ""))

        show_cols = []
        for c in [cand_name_col, cand_email_col, cand_go_live_col, "Match Score", cand_filelink_col, cand_id_col]:
            if c and c in scored.columns and c not in show_cols:
                show_cols.append(c)
        if "Match Score" not in show_cols:
            show_cols.insert(0, "Match Score")

        st.dataframe(scored[show_cols], use_container_width=True)

        out_csv = scored[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Shortlist CSV",
            data=out_csv,
            file_name=f"shortlist_{re.sub(r'[^a-zA-Z0-9]+','_',selected)[:60]}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("Debug: What text was used for matching?"):
            st.markdown("**Opportunity Text Used (first 800 chars):**")
            st.write((opp_text[:800] + "…") if len(opp_text) > 800 else opp_text)
            st.markdown("**Candidate text emptiness (before feedback):**")
            st.write(f"Empty ratio: {empty_ratio:.2%}")
            st.markdown("**Example candidate text (first row, first 400 chars):**")
            example_text = cand_texts[0] if cand_texts else ""
            st.write((example_text[:400] + "…") if len(example_text) > 400 else example_text)

st.markdown("---")
st.caption("© EDGE · Internal Use Only")
