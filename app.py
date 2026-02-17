import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd

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
.stButton>button:hover{
  background: #3B0F59;
}

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
# HEADER WITH SAFE LOGO LOADER
# --------------------------------------------------
logo_path = Path("assets/Logo/Edge_Lockup_H_Plum.jpg")

col1, col2 = st.columns([0.22, 0.78], vertical_alignment="center")

with col1:
    if logo_path.exists():
        logo = Image.open(logo_path)
        st.image(logo, use_container_width=True)

with col2:
    st.markdown("## Talent Shortlister")
    st.markdown(
        "<div style='color:#6B5C7A'>One-click candidate shortlisting tool</div>",
        unsafe_allow_html=True
    )

st.divider()

# --------------------------------------------------
# SAFE CSV READER (FIXES UTF-8 ERROR)
# --------------------------------------------------
def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Files")

opp_file = st.sidebar.file_uploader(
    "Opportunity information.csv",
    type=["csv"]
)

cand_file = st.sidebar.file_uploader(
    "Candidate Information.csv",
    type=["csv"]
)

feedback_file = st.sidebar.file_uploader(
    "Interview Feedback.csv (Optional)",
    type=["csv"]
)

st.sidebar.markdown("---")

top_k = st.sidebar.slider("Shortlist Size", 5, 50, 15)

# --------------------------------------------------
# MAIN CONTENT
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

    if st.button("Validate & Shortlist", use_container_width=True, disabled=not ready):

        try:
            opp_df = safe_read_csv(opp_file)
            cand_df = safe_read_csv(cand_file)

            if feedback_file:
                feedback_df = safe_read_csv(feedback_file)

            # ------------------------------------------
            # DEMO SHORTLIST LOGIC
            # Replace with your real scoring algorithm
            # ------------------------------------------
            shortlisted = cand_df.head(top_k)

            st.success("Shortlist generated successfully!")

            st.dataframe(shortlisted, use_container_width=True)

            csv = shortlisted.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Shortlist CSV",
                data=csv,
                file_name="shortlist.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing files: {e}")

st.markdown("---")
st.caption("© EDGE · Internal Use Only")
