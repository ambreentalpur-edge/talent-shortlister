import streamlit as st

st.set_page_config(
    page_title="EDGE | Talent Shortlister",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
</style>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Talent Shortlister", layout="wide")
st.title("Talent Shortlister — All Opportunities (One Click)")

# ---------------- CONSTANTS ----------------
VALID_COUNTRIES = {"PK", "PE", "CR"}
VALID_GENDERS = {"MALE", "FEMALE", "OTHER"}

EXCLUDED_STAGES = {"CLOSED WON", "CLOSED LOST", "CLOSED LOST INTRO"}

ACCENT_MAP = {"clear": 1.0, "average": 0.6, "thick": 0.2}
ENERGY_MAP = {"high": 1.0, "average": 0.6, "low": 0.2}
GOOD_AVG_BAD_MAP = {"good": 1.0, "average": 0.6, "bad": 0.2}

# ---------------- HELPERS ----------------
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def norm_country(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().upper()
    return v if v in VALID_COUNTRIES else v

def norm_gender(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().lower()
    if v in ["male", "m"]:
        return "male"
    if v in ["female", "f"]:
        return "female"
    if v == "other":
        return "other"
    return v

def norm_industry_group(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip().upper()
    # Health = Medical/Dental mapping (as you specified)
    if v in {"HEALTH", "MEDICAL", "DENTAL", "MEDICAL/DENTAL"}:
        return "HEALTH"
    return v

def parse_open_or_specific(raw, valid_set):
    """
    Returns 'ANY' if open (any/all/both) OR multiple valid values listed (e.g., PK + PE, PK & PE).
    Returns the single valid value if exactly one.
    Returns NaN if none.
    """
    if pd.isna(raw):
        return np.nan

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
    return np.nan

def norm_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\+\#\s/,&-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def overlap_score(jd_text, cand_text):
    jd_terms = set(norm_text(jd_text).split())
    if not jd_terms:
        return 0.0
    cand_terms = set(norm_text(cand_text).split())
    return len(jd_terms & cand_terms) / len(jd_terms)

def map_quality(val, mapping):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    return mapping.get(v, np.nan)

def compute_quality_score(row):
    a = map_quality(row.get("Accent"), ACCENT_MAP)
    e = map_quality(row.get("Energy"), ENERGY_MAP)
    pr = map_quality(row.get("Preparedness"), GOOD_AVG_BAD_MAP)
    pi = map_quality(row.get("Pitch"), GOOD_AVG_BAD_MAP)

    vals = [v for v in [a, e, pr, pi] if pd.notna(v)]
    if vals:
        return float(sum(vals) / len(vals))
    return np.nan

def build_jd_profile(opps_df):
    # Use the known JD fields when present; otherwise just ignore missing ones
    jd_fields = [
        "Job Title", "Background", "Additional Skills", "Soft Skills",
        "Professional Background Notes", "AMS/CRM/EMR/EHR/PMS",
        "VOIP", "Insurance Portals"
    ]
    for f in jd_fields:
        if f not in opps_df.columns:
            opps_df[f] = ""

    opps_df["jd_profile"] = opps_df[jd_fields].apply(
        lambda r: " ".join([safe_str(r[c]) for c in jd_fields if safe_str(r[c])]),
        axis=1
    ).str.lower()

    return opps_df

def eligible(opp_row, cand_row):
    # Must-have: On Marketplace (robust check)
    if "marketplace" not in str(cand_row.get("Go Live Status", "")).lower():
        return False

    # Must-have: Industry group match
    if pd.notna(opp_row.get("Opportunity_Industry_Group")) and pd.notna(cand_row.get("Candidate_Industry_Group")):
        if str(opp_row["Opportunity_Industry_Group"]).upper() != str(cand_row["Candidate_Industry_Group"]).upper():
            return False

    # Must-have: Country (only if specific)
    cf = opp_row.get("Country_Filter")
    if pd.notna(cf) and str(cf).upper() != "ANY":
        if pd.isna(cand_row.get("Candidate_Country")) or str(cand_row["Candidate_Country"]).upper() != str(cf).upper():
            return False

    # Must-have: Gender (only if specific)
    gf = opp_row.get("Gender_Filter")
    if pd.notna(gf) and str(gf).upper() != "ANY":
        if pd.isna(cand_row.get("Candidate_Gender")) or str(cand_row["Candidate_Gender"]).lower() != str(gf).lower():
            return False

    # Must-have: Background (light keyword match if provided)
    bg = opp_row.get("Background")
    if pd.notna(bg) and str(bg).strip():
        bg_terms = [w for w in norm_text(bg).split() if len(w) >= 4]
        profile = str(cand_row.get("cand_profile", ""))
        if bg_terms and not any(w in profile for w in bg_terms[:10]):
            return False

    return True

def generate_unique_shortlists_all(opps_df, cands_df, pool_multiplier=12, min_pool=200):
    """
    One click: generate shortlists for all eligible opportunities.
    Enforces: a candidate can only be shortlisted for ONE opportunity.
    """

    # Order: highest placements first (reduces starvation)
    opps_order = opps_df.sort_values(["Placements", "Opportunity_ID"], ascending=[False, True]).copy()

    used_candidates = set()
    output_rows = []

    for _, o in opps_order.iterrows():
        placements = int(o.get("Placements", 1))
        need = 4 * max(1, placements)

        elig = cands_df[cands_df.apply(lambda r: eligible(o, r), axis=1)].copy()
        if elig.empty:
            continue

        # Scores
        elig["Match_Score"] = elig["cand_profile"].apply(lambda t: overlap_score(o["jd_profile"], t))

        elig["Final_Score"] = np.where(
            elig["Quality_Score"].notna(),
            0.7 * elig["Match_Score"] + 0.3 * elig["Quality_Score"],
            elig["Match_Score"]
        )

        pool_n = max(min_pool, need * pool_multiplier)
        pool = elig.sort_values("Final_Score", ascending=False).head(pool_n)

        picked = 0
        for _, r in pool.iterrows():
            cid = r.get("Candidate_ID")
            if pd.isna(cid):
                continue

            if cid in used_candidates:
                continue

            used_candidates.add(cid)
            picked += 1

            output_rows.append({
                "Opportunity_ID": o.get("Opportunity_ID", ""),
                "Opportunity_Name": o.get("Opportunity: Opportunity Name", ""),
                "Account": o.get("Account: Account Name", ""),
                "Job Title": o.get("Job Title", ""),
                "Stage": o.get("Opportunity: Stage", ""),
                "Placements": placements,
                "Rank": picked,
                "Candidate_ID": r.get("Candidate_ID", ""),
                "Candidate_Name": r.get("Candidate Name", ""),
                "Candidate_Email": r.get("Personal Email", ""),
                "Candidate_Country": r.get("Candidate_Country", ""),
                "Candidate_Gender": r.get("Candidate_Gender", ""),
                "Candidate_School": r.get("School", ""),
                "Days_in_Marketplace": r.get("Days in Marketplace", ""),
                "Match_Score": round(float(r.get("Match_Score", 0.0)), 4),
                "Quality_Score": (round(float(r.get("Quality_Score", np.nan)), 4) if pd.notna(r.get("Quality_Score", np.nan)) else ""),
                "Final_Score": round(float(r.get("Final_Score", 0.0)), 4),
                "Accent": r.get("Accent", ""),
                "Energy": r.get("Energy", ""),
                "Preparedness": r.get("Preparedness", ""),
                "Pitch": r.get("Pitch", ""),
                "Feedback_Notes": r.get("Notes", "")
            })

            if picked >= need:
                break

    return pd.DataFrame(output_rows)


# ---------------- SIDEBAR UPLOADS ----------------
st.sidebar.header("Upload CSVs")

opp_file = st.sidebar.file_uploader("1) Opportunity information.csv", type="csv")
cand_file = st.sidebar.file_uploader("2) Candidate Information.csv", type="csv")
feedback_file = st.sidebar.file_uploader("3) Interview Feedback.csv (optional but recommended)", type="csv")

if not (opp_file and cand_file):
    st.info("Upload Opportunity information.csv and Candidate Information.csv to continue.")
    st.stop()

# ---------------- LOAD ----------------
opps = pd.read_csv(opp_file, encoding="latin1")
cands = pd.read_csv(cand_file, encoding="latin1")
fb = pd.read_csv(feedback_file, encoding="latin1") if feedback_file is not None else None

# Clean weird BOM characters (e.g., ï»¿Candidate Name)
opps.columns = [c.replace("ï»¿", "").strip() for c in opps.columns]
cands.columns = [c.replace("ï»¿", "").strip() for c in cands.columns]
if fb is not None:
    fb.columns = [c.replace("ï»¿", "").strip() for c in fb.columns]

# ---------------- PREP OPPORTUNITIES ----------------
if "Opportunity: Stage" not in opps.columns:
    st.error("Opportunity file is missing 'Opportunity: Stage'. Please include it in the export.")
    st.stop()

opps["Opportunity_ID"] = opps["Form: Form Number"].astype(str).str.strip()
opps["Placements"] = pd.to_numeric(opps.get("Placements", 1), errors="coerce").fillna(1).astype(int)

opps["Stage_norm"] = opps["Opportunity: Stage"].astype(str).str.strip().str.upper()
opps_use = opps[~opps["Stage_norm"].isin(EXCLUDED_STAGES)].copy()

opps_use["Opportunity_Industry_Group"] = opps_use.get("Industry", pd.Series([np.nan] * len(opps_use))).apply(norm_industry_group)
opps_use["Country_Filter"] = opps_use.get("Country Preference", pd.Series([np.nan] * len(opps_use))).apply(
    lambda x: parse_open_or_specific(x, VALID_COUNTRIES)
)
opps_use["Gender_Filter"] = opps_use.get("Gender", pd.Series([np.nan] * len(opps_use))).apply(
    lambda x: parse_open_or_specific(x, VALID_GENDERS)
)

opps_use = build_jd_profile(opps_use)

# ---------------- PREP CANDIDATES ----------------
cands["email_norm"] = cands["Personal Email"].astype(str).str.strip().str.lower()

# Candidate_ID from your file (Candidate: ID)
cands["Candidate_ID"] = cands["Candidate: ID"].astype(str)

cands["Candidate_Country"] = cands.get("Country", pd.Series([np.nan] * len(cands))).apply(norm_country)
cands["Candidate_Gender"] = cands.get("Gender", pd.Series([np.nan] * len(cands))).apply(norm_gender)
cands["Candidate_Industry_Group"] = cands.get("School", pd.Series([np.nan] * len(cands))).apply(norm_industry_group)

# Profile text used for matching
cands["cand_profile"] = (
    cands.get("Candidate Name", pd.Series([""] * len(cands))).map(safe_str) + " " +
    cands.get("School", pd.Series([""] * len(cands))).map(safe_str) + " " +
    cands.get("Background", pd.Series([""] * len(cands))).map(safe_str) + " " +
    cands.get("Speciality", pd.Series([""] * len(cands))).map(safe_str) + " " +
    cands.get("Professional Skills", pd.Series([""] * len(cands))).map(safe_str)
).str.lower()

# ---------------- MERGE INTERVIEW FEEDBACK ----------------
if fb is not None:
    fb["email_norm"] = fb.get("Personal Email", pd.Series([""] * len(fb))).astype(str).str.strip().str.lower()
    fb["Total Score"] = pd.to_numeric(fb.get("Total Score", np.nan), errors="coerce")
    fb["Opportunity: Interview Date"] = pd.to_datetime(fb.get("Opportunity: Interview Date", np.nan), errors="coerce")

    # Keep best feedback per candidate
    fb_best = fb.sort_values(
        by=["email_norm", "Total Score", "Opportunity: Interview Date"],
        ascending=[True, False, False]
    ).drop_duplicates("email_norm", keep="first")

    fb_best["Quality_Score"] = fb_best.apply(compute_quality_score, axis=1)

    keep_cols = ["email_norm", "Accent", "Energy", "Preparedness", "Pitch", "Notes", "Quality_Score"]
    for col in keep_cols:
        if col not in fb_best.columns:
            fb_best[col] = np.nan

    cands = cands.merge(fb_best[keep_cols], on="email_norm", how="left")
else:
    cands["Quality_Score"] = np.nan
    cands["Accent"] = ""
    cands["Energy"] = ""
    cands["Preparedness"] = ""
    cands["Pitch"] = ""
    cands["Notes"] = ""

# ---------------- SUMMARY ----------------
st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Opportunities in file", len(opps))
col2.metric("Eligible opportunities (by stage)", len(opps_use))
col3.metric("Candidates in file", len(cands))
st.write("Excluded stages:", ", ".join(sorted(EXCLUDED_STAGES)))

# ---------------- RUN ----------------
if st.button("Generate shortlists for ALL eligible opportunities"):
    with st.spinner("Generating shortlists (unique candidates across all opportunities)..."):
        shortlist = generate_unique_shortlists_all(opps_use, cands)

    if shortlist.empty:
        st.warning("No shortlists generated. This usually happens if filters are too strict or required columns are missing.")
        st.stop()

    st.success(f"Generated {len(shortlist)} shortlist rows across {shortlist['Opportunity_ID'].nunique()} opportunities.")
    st.dataframe(shortlist, use_container_width=True)

    csv_bytes = shortlist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ALL shortlists (CSV)",
        data=csv_bytes,
        file_name="shortlists_all_eligible_opportunities.csv",
        mime="text/csv"
    )
