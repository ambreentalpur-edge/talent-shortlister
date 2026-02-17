import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import re

# Try to use sklearn TF-IDF if available, else fallback to keyword overlap
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
def safe_read_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Reads CSV with encoding fallback for Salesforce/Excel exports."""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")

def normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by fuzzy name match."""
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        cand_n = normalize_colname(cand)
        # exact normalized match
        if cand_n in norm_map:
            return norm_map[cand_n]
    # contains match
    for cand in candidates:
        cand_n = normalize_colname(cand)
        for n, original in norm_map.items():
            if cand_n in n:
                return original
    return None

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    t = str(x)
    # remove HTML anchors from Salesforce exports but keep visible text/URL bits
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def join_text_fields(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in row.index:
            parts.append(clean_text(row[c]))
    return " ".join([p for p in parts if p]).strip()

def keyword_overlap_score(opp_text: str, cand_texts: list[str]) -> list[float]:
    opp_tokens = set(re.findall(r"[a-zA-Z]{3,}", opp_text.lower()))
    scores = []
    for ct in cand_texts:
        c_tokens = set(re.findall(r"[a-zA-Z]{3,}", ct.lower()))
        if not opp_tokens:
            scores.append(0.0)
        else:
            scores.append(len(opp_tokens & c_tokens) / max(1, len(opp_tokens)))
    return scores

def tfidf_score(opp_text: str, cand_texts: list[str]) -> list[float]:
    docs = [opp_text] + cand_texts
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(docs)
    sims = cosine_similarity(X[0], X[1:]).flatten()
    return sims.tolist()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Files")

opp_file = st.sidebar.file_uploader("Opportunity information.csv", type=["csv"])
cand_file = st.sidebar.file_uploader("Candidate Information.csv", type=["csv"])
feedback_file = st.sidebar.file_uploader("Interview Feedback.csv (Optional)", type=["csv"])

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Shortlist Size", 5, 50, 15)
filter_live = st.sidebar.toggle("Only include Go Live Status = Live", value=True)
use_feedback = st.sidebar.toggle("Incorporate feedback text (if uploaded)", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: If opportunity details are in Additional Notes, this scorer will still use them.")

# --------------------------------------------------
# MAIN STATUS
# --------------------------------------------------
left, right = st.columns([0.65, 0.35])

with left:
    st.markdown("### Upload Status")
    ready = True
    if opp_file:
        st.success("Opportunity file uploaded.")
    else:
        st.info("Opportunity file not uploaded.")
        ready = False

    if cand_file:
        st.success("Candidate file uploaded.")
    else:
        st.info("Candidate file not uploaded.")
        ready = False

    if feedback_file:
        st.success("Interview feedback uploaded (optional).")
    else:
        st.info("No interview feedback uploaded (optional).")

with right:
    st.markdown("### Run Shortlist")
    run = st.button("Validate & Shortlist", use_container_width=True, disabled=not ready)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if run:
    try:
        opp_df = safe_read_csv(opp_file)
        cand_df = safe_read_csv(cand_file)
        feedback_df = safe_read_csv(feedback_file) if (feedback_file and use_feedback) else None

        # ---- Identify key columns (robust to your SFDC export names) ----
        opp_name_col = find_col(opp_df, ["Opportunity Name", "Opportunity: Opportunity Name", "Opportunity"])
        opp_notes_col = find_col(opp_df, ["Additional Notes", "Additional_Notes", "Notes", "Opportunity Notes"])
        opp_desc_col  = find_col(opp_df, ["Description", "Opportunity Description", "Role Description", "Job Description"])
        opp_skills_col = find_col(opp_df, ["Required Skills", "Skills", "Required skill", "Must have", "Requirements"])

        if not opp_name_col:
            st.error("Could not find an Opportunity Name column in Opportunity information.csv.")
            st.stop()

        # Candidate columns
        cand_name_col = find_col(cand_df, ["Candidate Name", "Name"])
        cand_id_col = find_col(cand_df, ["Candidate: ID", "Candidate ID", "Candidate:ID", "Candidate Id"])
        cand_go_live_col = find_col(cand_df, ["Go Live Status", "Go Live"])
        cand_bg_col = find_col(cand_df, ["Background"])
        cand_spec_col = find_col(cand_df, ["Speciality", "Specialty"])
        cand_skills_col = find_col(cand_df, ["Professional Skills", "Skills", "Professional_Skills"])
        cand_school_col = find_col(cand_df, ["School", "Education", "University"])
        cand_email_col = find_col(cand_df, ["Personal Email", "Email"])
        cand_filelink_col = find_col(cand_df, ["File Link", "Resume Link", "CV Link"])

        # Optional: filter to Live only
        work_cand = cand_df.copy()
        if filter_live and cand_go_live_col:
            work_cand = work_cand[work_cand[cand_go_live_col].astype(str).str.strip().str.lower() == "live"].copy()

        if work_cand.empty:
            st.warning("No candidates left after filtering. Turn off the Live filter or check Go Live Status values.")
            st.stop()

        # Select opportunity
        opp_names = opp_df[opp_name_col].astype(str).fillna("").tolist()
        selected = st.selectbox("Select Opportunity", options=sorted(set([o for o in opp_names if o.strip()])))
        opp_row = opp_df[opp_df[opp_name_col].astype(str) == selected].head(1)
        if opp_row.empty:
            st.error("Could not locate selected opportunity row.")
            st.stop()

        # Build opportunity text (this is where Additional Notes matters)
        opp_cols_for_text = [c for c in [opp_name_col, opp_desc_col, opp_skills_col, opp_notes_col] if c]
        opp_text = join_text_fields(opp_row.iloc[0], opp_cols_for_text)

        # Build candidate text profile
        cand_cols_for_text = [c for c in [cand_bg_col, cand_spec_col, cand_skills_col, cand_school_col] if c]
        cand_texts = work_cand.apply(lambda r: join_text_fields(r, cand_cols_for_text), axis=1).tolist()

        # Add feedback text into candidate profile (if provided)
        if feedback_df is not None and cand_id_col:
            # try to find a feedback text column
            fb_id_col = find_col(feedback_df, ["Candidate: ID", "Candidate ID", "Candidate Id"])
            fb_text_col = find_col(feedback_df, ["Feedback", "Notes", "Comments", "Interview Feedback", "Summary"])
            if fb_id_col and fb_text_col:
                fb_map = (
                    feedback_df[[fb_id_col, fb_text_col]]
                    .dropna()
                    .groupby(feedback_df[fb_id_col].astype(str))[fb_text_col]
                    .apply(lambda s: " ".join([clean_text(x) for x in s.tolist()]))
                    .to_dict()
                )
                # append feedback
                new_texts = []
                for idx, r in work_cand.iterrows():
                    cid = str(r[cand_id_col]) if cand_id_col in r.index else ""
                    new_texts.append((cand_texts[len(new_texts)] + " " + fb_map.get(cid, "")).strip())
                cand_texts = new_texts

        # Score
        if SKLEARN_OK:
            scores = tfidf_score(opp_text, cand_texts)
            method = "TF-IDF similarity"
        else:
            scores = keyword_overlap_score(opp_text, cand_texts)
            method = "Keyword overlap (fallback)"

        scored = work_cand.copy()
        scored["Match Score"] = scores

        # Sort + take top
        scored = scored.sort_values("Match Score", ascending=False).head(top_k)

        # Display
        st.success(f"Shortlist generated using {method}.")
        show_cols = []
        for c in [cand_name_col, cand_email_col, cand_go_live_col, cand_bg_col, cand_spec_col, cand_skills_col, "Match Score", cand_filelink_col, cand_id_col]:
            if c and c in scored.columns and c not in show_cols:
                show_cols.append(c)
        if "Match Score" not in show_cols:
            show_cols.insert(0, "Match Score")

        st.dataframe(scored[show_cols], use_container_width=True)

        # Download
        out_csv = scored[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Shortlist CSV",
            data=out_csv,
            file_name=f"shortlist_{re.sub(r'[^a-zA-Z0-9]+','_',selected)[:60]}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("See what text was used for matching (debug)"):
            st.markdown("**Opportunity Text Used:**")
            st.write(opp_text if opp_text else "(empty)")
            st.markdown("**Candidate fields used:**")
            st.write(cand_cols_for_text if cand_cols_for_text else "(none found)")

    except Exception as e:
        st.error(f"Error processing files: {e}")

st.markdown("---")
st.caption("© EDGE · Internal Use Only")
