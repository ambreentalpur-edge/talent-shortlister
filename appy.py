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

    if o["Opportunity_I_]()
