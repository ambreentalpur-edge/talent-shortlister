import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Talent Shortlister", layout="wide")
st.title("Talent Shortlister")

# ---------------- CONSTANTS ----------------
VALID_COUNTRIES = {"PK", "PE", "CR"}
VALID_GENDERS = {"MALE", "FEMALE", "OTHER"}

# ---------------- HELPERS ----------------
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def normalize_country(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    return x if x in VALID_COUNTRIES else x

def normalize_gender(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ["male", "m"]:
        return "male"
    if x in ["female", "f"]:
        return "female"
    if x == "other":
        return "other"
    return x

def normalize_industry_group(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    if x in {"HEALTH", "MEDICAL", "DENTAL", "MEDICAL/DENTAL"}:
        return "HEALTH"
    return x

def parse_open_or_specific(raw, valid_set):
    if pd.isna(raw):
        return np.nan
    s = str(raw).upper().replace(" ", "")
    if any(k in s for k in ["ANY", "ALL", "BOTH"]):
        return "ANY"
    tokens = re.split(r"[,&/+\|]+", s)
    tokens = [t for t in tokens if t in valid_set]
    if len(set(tokens)) >= 2:
        return "ANY"
    if len(tokens) == 1:
        return tokens[0]
    return np.nan

def normalize_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def overlap_score(jd_text, cand_text):
    jd_terms = set(normalize_text(jd_text).split())
    cand_terms = set(normalize_text(cand_text).split())
    if not jd_terms:
        return 0
    return len(jd_terms & cand_terms) / len(jd_terms)

def eligible(o, c):
    if str(c["Go_Live_Status"]).strip().lower() != "on marketplace":
        return False

    if o["Opportunity_Industry_Group"] != c["Candidate_Industry_Group"]:
        return False

    if o["Country_Filter"] != "ANY" and pd.notna(o["Country_Filter"]):
        if c["Candidate_Country"] != o["Country_Filter"]:
            return False

    if o["Gender_Filter"] != "ANY" and pd.notna(o["Gender_Filter"]):
        if c["Candidate_Gender"] != o["Gender_Filter"].lower():
            return False

    if pd.notna(o["Background"]) and o["Background"].strip():
        bg_words = normalize_text(o["Background"]).split()
        if not any(w in c["cand_profile"] for w in bg_words):
            return False

    return True

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload SFDC CSVs")

cand_file = st.sidebar.file_uploader("Candidates x FileConnect", type="csv")
cand_extra_file = st.sidebar.file_uploader("Candidate Attributes", type="csv")
opp_file = st.sidebar.file_uploader("Opportunities", type="csv")

if not (cand_file and cand_extra_file and opp_file):
    st.info("Upload all 3 CSV files to continue.")
    st.stop()

# ---------------- LOAD DATA ----------------
cand = pd.read_csv(cand_file, encoding="latin1")
cand_extra = pd.read_csv(cand_extra_file, encoding="latin1")
opp = pd.read_csv(opp_file, encoding="latin1")

# ---------------- MERGE CANDIDATES ----------------
cand["email_norm"] = cand["Personal_Email"].str.lower().str.strip()
cand_extra["email_norm"] = cand_extra["Personal Email"].str.lower().str.strip()

cand_extra["Days in Marketplace"] = pd.to_numeric(
    cand_extra["Days in Marketplace"], errors="coerce"
).fillna(-1)

cand_extra = cand_extra.sort_values(
    ["email_norm", "Days in Marketplace"], ascending=[True, False]
).drop_duplicates("email_norm")

merged = cand.merge(
    cand_extra.drop(columns=["School"], errors="ignore"),
    on="email_norm",
    how="left"
)

merged["Candidate_Country"] = merged["Country"].apply(normalize_country)
merged["Candidate_Gender"] = merged["Gender"].apply(normalize_gender)
merged["Candidate_Industry_Group"] = merged["School"].apply(normalize_industry_group)

merged["cand_profile"] = (
    merged["Candidate_Name"].map(safe_str) + " " +
    merged["School"].map(safe_str) + " " +
    merged.get("Background", "").map(safe_str) + " " +
    merged.get("Professional Skills", "").map(safe_str)
).str.lower()

# ---------------- PREP OPPORTUNITIES ----------------
opp["Opportunity_ID"] = opp["Form: Form Number"].astype(str)
opp["Placements"] = pd.to_numeric(opp["Placements"], errors="coerce").fillna(1).astype(int)
opp["Opportunity_Industry_Group"] = opp["Industry"].apply(normalize_industry_group)

opp["Country_Filter"] = opp["Country Preference"].apply(
    lambda x: parse_open_or_specific(x, VALID_COUNTRIES)
)
opp["Gender_Filter"] = opp["Gender"].apply(
    lambda x: parse_open_or_specific(x, VALID_GENDERS)
)

jd_fields = [
    "Job Title", "Background", "Additional Skills",
    "Soft Skills", "Professional Background Notes"
]

for f in jd_fields:
    if f not in opp.columns:
        opp[f] = ""

opp["jd_profile"] = opp[jd_fields].apply(
    lambda r: " ".join([safe_str(r[c]) for c in jd_fields]),
    axis=1
).str.lower()

# ---------------- UI ----------------
st.subheader("Select Opportunity")
label = opp["Opportunity_ID"] + " | " + opp["Job Title"]
selected = st.selectbox("Opportunity", label)

opp_row = opp[label == selected].iloc[0]

st.write("Placements:", opp_row["Placements"])

if st.button("Generate Shortlist"):
    top_n = 4 * opp_row["Placements"]

    eligible_cands = merged[merged.apply(lambda r: eligible(opp_row, r), axis=1)]

    eligible_cands["Score"] = eligible_cands["cand_profile"].apply(
        lambda t: overlap_score(opp_row["jd_profile"], t)
    )

    result = eligible_cands.sort_values("Score", ascending=False).head(top_n)

    result = result[[
        "Candidate_ID", "Candidate_Name", "Personal_Email",
        "Candidate_Country", "Candidate_Gender", "School", "Score"
    ]].reset_index(drop=True)

    result.insert(0, "Rank", range(1, len(result) + 1))

    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download Shortlist",
        result.to_csv(index=False),
        file_name=f"shortlist_{opp_row['Opportunity_ID']}.csv"
    )
