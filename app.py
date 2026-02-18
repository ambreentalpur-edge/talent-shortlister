import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import re

# --- sklearn (for scoring) ---
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
# CONSTANTS
# --------------------------------------------------
VALID_COUNTRIES = {"PK", "PE", "CR"}
VALID_GENDERS = {"MALE", "FEMALE", "OTHER"}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_read_csv(uploaded_file):
    """Reads CSV with encoding fallback for Salesforce/Excel exports."""
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
    # remove BOM if present
    df.columns = [str(c).replace("ï»¿", "").strip() for c in df.columns]
    return df

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

def norm_country(x):
    if pd.isna(x): return ""
    return str(x).strip().upper()

def norm_gender(x):
    if pd.isna(x): return ""
    v = str(x).strip().lower()
    if v in {"m","male"}: return "male"
    if v in {"f","female"}: return "female"
    if v == "other": return "other"
    return v

def norm_industry_group(x):
    if pd.isna(x): return ""
    v = str(x).strip().upper()
    # your rule: Health = Medical/Dental
    if v in {"HEALTH", "MEDICAL", "DENTAL", "MEDICAL/DENTAL"}:
        return "HEALTH"
    return v

def parse_open_or_specific(raw, valid_set):
    """
    PK or PE or CR -> returns that
    PK + PE / PK & PE / both / any -> returns 'ANY'
    """
    if pd.isna(raw): 
        return ""
    s = str(raw).upper()
    s_nospace = re.sub(r"\s+", "", s)
    if any(k in s_nospace for k in ["ANY", "ALL", "BOTH"]):
        return "ANY"

    tokens = re.split(r"[,&/+\|\s]+", s)
    tokens = [t.strip().upper() for t in tokens if t.strip()]
    found = sorted(set([t for t in tokens if t in valid_set]))

    if len(found) >= 2:
        return "ANY"
    if len(found) == 1:
        return found[0]
    return ""

def background_pass(opp_background: str, cand_profile: str) -> bool:
    """
    Light strict filter:
    - take up to first 10 meaningful words (len>=4) from opp background
    - require at least one of them to appear in candidate profile
    """
    ob = clean_text(opp_background)
    if not ob:
        return True
    terms = [w for w in re.findall(r"[a-zA-Z]{4,}", ob.lower())]
    terms = terms[:10]
    if not terms:
        return True
    prof = (cand_profile or "").lower()
    return any(t in prof for t in terms)

def marketplace_pass(go_live_val: str) -> bool:
    return "marketplace" in str(go_live_val).lower()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Files")
opp_file = st.sidebar.file_uploader("Opportunity information.csv", type=["csv"])
cand_file = st.sidebar.file_uploader("Candidate Information.csv", type=["csv"])
feedback_file = st.sidebar.file_uploader("Interview Feedback.csv (Optional)", type=["csv"])

st.sidebar.markdown("---")
use_feedback = st.sidebar.toggle("Use Interview Feedback as signal (recommended)", value=True)

# business rule: always enforce marketplace eligibility
st.sidebar.info("Eligibility rule enforced: Go Live Status must be On Marketplace.")

# optional: allow user override for shortlist size
override_k = st.sidebar.toggle("Override shortlist size (otherwise 4 × placements)", value=False)
top_k = st.sidebar.slider("Shortlist Size", 4, 60, 16) if override_k else None

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
# LOAD + SELECT (keep outside button to avoid Streamlit rerun issues)
# --------------------------------------------------
if ready:
    opp_df = safe_read_csv(opp_file)
    cand_df = safe_read_csv(cand_file)
    feedback_df = safe_read_csv(feedback_file) if (feedback_file and use_feedback) else None

    # --- Identify key opportunity fields ---
    opp_name_col = find_col(opp_df, ["Opportunity: Opportunity Name", "Opportunity Name", "Opportunity"])
    opp_country_col = find_col(opp_df, ["Country Preference", "Country"])
    opp_gender_col = find_col(opp_df, ["Gender"])
    opp_industry_col = find_col(opp_df, ["Industry"])
    opp_background_col = find_col(opp_df, ["Background"])
    opp_placements_col = find_col(opp_df, ["Placements", "Number of Placements", "Hires", "No. of Hires"])

    if not opp_name_col:
        st.error("Could not find Opportunity Name column in Opportunity information.csv.")
        st.stop()

    # --- Identify candidate fields ---
    cand_id_col = find_col(cand_df, ["Candidate: ID", "Candidate ID", "Candidate Id"])
    cand_name_col = find_col(cand_df, ["Candidate Name", "Name"])
    cand_email_col = find_col(cand_df, ["Personal Email", "Email"])
    cand_go_live_col = find_col(cand_df, ["Go Live Status", "Go Live"])
    cand_country_col = find_col(cand_df, ["Country"])
    cand_gender_col = find_col(cand_df, ["Gender"])
    cand_school_col = find_col(cand_df, ["School"])  # your industry proxy
    cand_background_col = find_col(cand_df, ["Background"])
    cand_speciality_col = find_col(cand_df, ["Speciality", "Specialty"])
    cand_skills_col = find_col(cand_df, ["Professional Skills", "Skills"])
    cand_filelink_col = find_col(cand_df, ["File Link", "Resume Link", "CV Link"])

    if not cand_id_col:
        st.error("Candidate Information.csv must include Candidate: ID.")
        st.stop()
    if not cand_go_live_col:
        st.error("Candidate Information.csv must include Go Live Status.")
        st.stop()

    # --- Select opportunity ---
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

        o = opp_row.iloc[0]

        # Placements => shortlist size (4 × placements)
        placements = 1
        if opp_placements_col and clean_text(o.get(opp_placements_col)) != "":
            try:
                placements = int(float(str(o.get(opp_placements_col)).strip()))
                placements = max(1, placements)
            except Exception:
                placements = 1

        shortlist_size = top_k if override_k else 4 * placements

        # --- Parse must-have requirement values from opp ---
        opp_country_filter = parse_open_or_specific(o.get(opp_country_col, ""), VALID_COUNTRIES) if opp_country_col else ""
        opp_gender_filter = parse_open_or_specific(o.get(opp_gender_col, ""), VALID_GENDERS) if opp_gender_col else ""
        opp_industry_filter = norm_industry_group(o.get(opp_industry_col, "")) if opp_industry_col else ""
        opp_background_req = clean_text(o.get(opp_background_col, "")) if opp_background_col else ""

        # --- Build opportunity text for scoring (includes Additional Notes etc.) ---
        opp_text = row_to_text(o, exclude_cols=set())  # keep all columns; notes matter

        # --- Prepare candidate working df ---
        work = cand_df.copy()

        # Normalize candidate columns
        work["_cand_id"] = work[cand_id_col].astype(str).str.strip()
        work["_go_live"] = work[cand_go_live_col].astype(str)
        work["_country"] = work[cand_country_col].apply(norm_country) if cand_country_col else ""
        work["_gender"] = work[cand_gender_col].apply(norm_gender) if cand_gender_col else ""
        work["_industry"] = work[cand_school_col].apply(norm_industry_group) if cand_school_col else ""

        # Candidate profile text for scoring (good-to-have)
        def build_candidate_profile(r):
            parts = []
            for col in [cand_name_col, cand_school_col, cand_background_col, cand_speciality_col, cand_skills_col]:
                if col and col in work.columns:
                    parts.append(clean_text(r.get(col, "")))
            return " ".join([p for p in parts if p]).strip()

        work["_base_profile"] = work.apply(build_candidate_profile, axis=1)

        # --- Must-have filters (strict) ---
        before = len(work)

        # 1) On Marketplace
        work = work[work["_go_live"].apply(marketplace_pass)].copy()
        after_marketplace = len(work)

        # 2) Industry exact (if requirement present)
        if opp_industry_filter:
            work = work[work["_industry"].astype(str).str.upper() == opp_industry_filter.upper()].copy()
        after_industry = len(work)

        # 3) Country (if specific)
        if opp_country_filter and opp_country_filter != "ANY":
            work = work[work["_country"].astype(str).str.upper() == opp_country_filter.upper()].copy()
        after_country = len(work)

        # 4) Gender (if specific)
        if opp_gender_filter and opp_gender_filter != "ANY":
            work = work[work["_gender"].astype(str).str.lower() == opp_gender_filter.lower()].copy()
        after_gender = len(work)

        # 5) Background keyword pass (if provided)
        if opp_background_req:
            work = work[work["_base_profile"].apply(lambda t: background_pass(opp_background_req, t))].copy()
        after_background = len(work)

        # Show filter funnel
        with st.expander("Eligibility funnel (must-haves)"):
            st.write(f"Start candidates: {before}")
            st.write(f"After Marketplace: {after_marketplace}")
            st.write(f"After Industry: {after_industry}")
            st.write(f"After Country: {after_country}")
            st.write(f"After Gender: {after_gender}")
            st.write(f"After Background: {after_background}")
            st.write("Parsed requirements:")
            st.write({
                "Placements": placements,
                "Shortlist size": shortlist_size,
                "Industry filter": opp_industry_filter or "(none)",
                "Country filter": opp_country_filter or "(none)",
                "Gender filter": opp_gender_filter or "(none)",
                "Background filter": (opp_background_req[:120] + "…") if len(opp_background_req) > 120 else (opp_background_req or "(none)")
            })

        if work.empty:
            st.warning("No candidates left after must-have filtering. Most common causes: Industry mismatch, Country/Gender too specific, or Background too strict.")
            st.stop()

        # --- Append interview feedback to profile (good-to-have)
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
                # build final candidate texts for scoring
                cand_texts = []
                for _, r in work.iterrows():
                    base = r["_base_profile"]
                    extra = ""
                    if fb_map_by_id:
                        extra = fb_map_by_id.get(str(r.get(cand_id_col, "")).strip(), "")
                    if not extra and fb_map_by_email and cand_email_col and cand_email_col in work.columns:
                        extra = fb_map_by_email.get(str(r.get(cand_email_col, "")).strip().lower(), "")
                    cand_texts.append((base + " " + extra).strip())
                fb_used = True
            else:
                cand_texts = work["_base_profile"].tolist()
        else:
            cand_texts = work["_base_profile"].tolist()

        # --- Score (good-to-have) ---
        if SKLEARN_OK:
            scores = tfidf_scores(opp_text, cand_texts)
            method = "TF-IDF similarity"
        else:
            scores = keyword_overlap_scores(opp_text, cand_texts)
            method = "Keyword overlap (fallback)"

        scored = work.copy()
        scored["Match Score"] = scores
        scored = scored.sort_values("Match Score", ascending=False).head(shortlist_size)

        st.success(f"Shortlist generated using {method}." + (" (Interview feedback included)" if fb_used else ""))

        # --- Display ---
        show_cols = []
        for c in [cand_name_col, cand_email_col, cand_go_live_col, cand_school_col, cand_country_col, cand_gender_col, "Match Score", cand_filelink_col, cand_id_col]:
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

        with st.expander("Debug: Matching text preview"):
            st.markdown("**Opportunity text used (first 800 chars):**")
            st.write((opp_text[:800] + "…") if len(opp_text) > 800 else opp_text)
            st.markdown("**Example candidate text (first row, first 400 chars):**")
            ex = cand_texts[0] if cand_texts else ""
            st.write((ex[:400] + "…") if len(ex) > 400 else ex)

st.markdown("---")
st.caption("© EDGE · Internal Use Only")
