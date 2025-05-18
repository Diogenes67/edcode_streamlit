import streamlit as st
from PIL import Image

from pathlib import Path
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
import plotly.graph_objects as go
from openai import OpenAI

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="EKoder – ED Code Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("ekoder_styles.css")

logo = Image.open("logo.png")
st.image(logo, width=150)

# Title and description  
st.markdown("<h1 style='color:#004080;font-size:48px;'>EKoder</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; color:#333;'>
<b>EKoder</b> is a clinical coding exploration tool designed for use in Australian Emergency Departments.<br><br>

This tool is intended for <b>research and educational purposes only</b>, to explore the potential use of AI in clinical coding workflows.  
It uses GPT‑4o to analyse free-text ED case notes and suggest <b>up to four ICD-10-AM principal diagnosis codes</b> that could plausibly reflect the presentation.  
These are ranked from most to least likely, with a short explanation provided for each.<br><br>

The <b>Complexity</b> column indicates the relative resource intensity typically associated with each code in an ED setting, based on historical funding weights.<br><br>

<b>Note:</b> In practice, ED funding is affected by a range of factors including patient complexity, triage category, and age.  
The ED code alone does not determine the total funding amount.<br><br>

<b>How it works:</b><br>
EKoder uses semantic similarity (via embedding-based search) to identify the most relevant codes from a shortlist of common ED diagnoses.  
It then passes those to GPT‑4o for reasoning and ranking based on your case note.  
No sidebar configuration is required — just paste or upload a case note.<br><br>

⚠️ <b>Please ensure all case notes are de-identified before uploading or pasting into this tool.</b><br>
Use of EKoder should be reviewed by your institution’s ICT, privacy, and research governance teams before integration into clinical or operational environments.<br><br>

This tool does <b>not provide medical advice</b> or guaranteed coding accuracy and is <b>not intended for patient care or formal documentation</b>.
</div>
""", unsafe_allow_html=True)


# Initialize session state variables if they don't exist
if 'results' not in st.session_state:
    st.session_state.results = None
if 'note_text' not in st.session_state:
    st.session_state.note_text = ""
if 'embedding_mode' not in st.session_state:
    st.session_state.embedding_mode = "OpenAI"  # Default to OpenAI embeddings

# === Configuration Variables ===
# Flag to switch between OpenAI-hosted embeddings and local sentence-transformers
USE_LOCAL_EMBEDDINGS = st.sidebar.radio(
    "Embedding Provider",
    options=["OpenAI", "Local"],
    index=0,
    help="Choose between OpenAI API or local embedding model"
) == "Local"

st.session_state.embedding_mode = "Local" if USE_LOCAL_EMBEDDINGS else "OpenAI"

# API Key input in sidebar
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# If API key is provided, update environment variable
#if OPENAI_API_KEY:
#    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#    client = OpenAI(api_key=OPENAI_API_KEY)
#else:
#    st.sidebar.warning("Please enter an OpenAI API key to use this application")

# File path configurations
st.sidebar.header("File Configuration")
# Use current directory as default root
default_root = os.getcwd()
ROOT = st.sidebar.text_input("Root Directory", value=default_root)
ROOT = Path(ROOT)

# File paths with file uploaders
st.sidebar.subheader("Required Files")

# For Excel file
uploaded_excel = st.sidebar.file_uploader("Upload ICD Codes Excel", type=["xlsx", "xls"], 
                                        help="Upload the FinalEDCodes_Complexity.xlsx file")
if uploaded_excel:
    EXCEL_PATH = ROOT / uploaded_excel.name
    with open(EXCEL_PATH, "wb") as f:
        f.write(uploaded_excel.getbuffer())
else:
    EXCEL_PATH = ROOT / "FinalEDCodes_Complexity.xlsx"
    if not EXCEL_PATH.exists():
        st.sidebar.warning(f"Excel file not found at {EXCEL_PATH}")

# For JSONL file
uploaded_jsonl = st.sidebar.file_uploader("Upload Few-Shot Examples", type=["jsonl"], 
                                        help="Upload the edcode_finetune examples file")
if uploaded_jsonl:
    JSONL_PATH = ROOT / uploaded_jsonl.name
    with open(JSONL_PATH, "wb") as f:
        f.write(uploaded_jsonl.getbuffer())
else:
    JSONL_PATH = ROOT / "edcode_finetune_v5_updated.jsonl"
    if not JSONL_PATH.exists():
        st.sidebar.warning(f"JSONL file not found at {JSONL_PATH}")

# Path for embedding cache
EMBEDDING_CACHE_PATH = ROOT / "ed_code_embeddings.pkl"

# Model & Dimension Definitions
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
LOCAL_MODEL_NAME = "intfloat/e5-small-v2"   # Local embedding model (if used)
CLOUD_DIM = 1536  # Dimension of OpenAI embeddings
LOCAL_DIM = 384   # Dimension of local embeddings

# Emoji lookup for funding scale visualization
funding_emojis = {
    1: "🟣", 2: "🔵", 3: "🟢",
    4: "🟡", 5: "🟠", 6: "🔴"
}

# === Utility Functions ===
def cosine(u, v):
    """Return cosine similarity between vectors u and v."""
    return np.dot(u, v) / (norm(u) * norm(v))

@st.cache_data
def get_embeddings_openai(texts):
    """Obtain embeddings from OpenAI for a list of texts."""
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        st.error(f"Error getting OpenAI embeddings: {e}")
        return None

@st.cache_data
def get_embeddings_local(texts):
    """Obtain embeddings using a local SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_MODEL_NAME, device="cpu")  # Use CPU by default, change to "cuda" or "mps" if available
        return model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    except ImportError:
        st.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Error with local embeddings: {e}")
        return None

@st.cache_data
def build_code_embeddings(descriptions, cache_path, use_local=False):
    """
    Build or load cached embeddings for the code descriptions.
    """
    expected_dim = LOCAL_DIM if use_local else CLOUD_DIM
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                embeds = pickle.load(f)
            # Determine loaded dimension
            dim = embeds.shape[1] if isinstance(embeds, np.ndarray) and embeds.ndim == 2 else (
                len(embeds[0]) if isinstance(embeds, list) and embeds else None
            )
            if dim == expected_dim:
                st.sidebar.success(f"Loaded {len(embeds)} cached embeddings ({dim}d)")
                return embeds
            else:
                st.sidebar.info(f"Cache dimension {dim} != expected {expected_dim}; regenerating...")
        except Exception as e:
            st.sidebar.warning(f"Error loading cache: {e}")

    # Generate embeddings
    with st.sidebar.status("Generating embeddings..."):
        if use_local:
            embeds = get_embeddings_local(descriptions)
        else:
            embeds = get_embeddings_openai(descriptions)

        if embeds:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(embeds, f)
                st.sidebar.success(f"Generated and cached {len(embeds)} embeddings")
            except Exception as e:
                st.sidebar.warning(f"Failed to cache embeddings: {e}")
        else:
            st.sidebar.error("Failed to generate embeddings")
            return None

    return embeds

def get_funding_emoji(code, funding_lookup):
    """Return emoji representing funding scale for a given code."""
    return funding_emojis.get(funding_lookup.get(code, 3), "🟩")

def get_top_matches(note_emb, code_embs, df, top_n=5):
    """
    Compute cosine similarity between note embedding and each code embedding.
    Return top_n codes sorted by similarity.
    """
    sims = [cosine(note_emb, e) for e in code_embs]
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    top = df.iloc[idx].copy()
    top['Similarity'] = [sims[i] for i in idx]
    return top

@st.cache_data
def load_examples(path, limit=3):
    """
    Load few-shot examples from a .jsonl file for prompt context.
    Returns a concatenated string of Casenote/Answer examples.
    """
    path = Path(path)
    if not path.exists():
        st.error(f"Example file not found: {path}")
        return ""

    ex = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit: break
                d = json.loads(line)
                ex.append(
                    f"Casenote:\n{d['messages'][0]['content']}\nAnswer:\n{d['messages'][1]['content']}"
                )
        return "\n\n---\n\n".join(ex) + "\n\n---\n\n"
    except Exception as e:
        st.error(f"Error loading examples: {e}")
        return ""

def parse_response(resp, df):
    """
    Parse the LLM response to extract ICD codes and explanations.
    Returns list of tuples: (code, term, explanation, emoji)
    """
    valid = set(df['ED Short List code'].astype(str).str.strip())
    term = dict(zip(df['ED Short List code'], df['ED Short List Term']))
    rows = []
    for line in resp.splitlines():
        m = re.match(r"\d+\.\s*([A-Z0-9\.]+)\s*[—-]\s*(.*)", line)
        if m:
            code, expl = m.groups()
            if code in valid and code != 'R69':
                rows.append((code, term[code], expl.strip('"').strip("'"), get_funding_emoji(code, funding_lookup)))
    return rows

@st.cache_data
def predict_final_codes(note, shortlist_df, fewshot, api_key):
    """
    Construct a detailed prompt including few-shot examples and the note,
    then call GPT-4o to get code suggestions.
    """
    # Initialize OpenAI client with the API key from argument
    client = OpenAI(api_key=api_key)

    options_text = "\n".join(
        f"{r['ED Short List code']} — {r['ED Short List Term']}" for _, r in shortlist_df.iterrows()
    )
    prompt = f"""
You are the head of the emergency department and an expert clinical coder.
Your rationale should help other doctors understand the pros and cons of choosing each code.

Your task is to suggest between **one and four mutually exclusive ED Short List ICD-10-AM codes** that could each plausibly serve as the **principal diagnosis**, based on the diagnostic content of the casenote.

These codes are **not** intended to build a combined clinical picture — rather, they should reflect **alternative coding options**, depending on the coder's interpretation and emphasis. **Each code must stand on its own** as a valid representation of the case presentation.

---

**How to think it through (show your work):**
1. Identify the single finding or cluster of findings that most tightly matches one code.
2. Pick that code as **#1 (best fit)** and provide a clear justification:
   - Show exactly which language in the note drives your choice
   - Highlight why it's more specific or higher-priority than the next option
3. Repeat for up to **4 total**, each time choosing the next-best fit.
4. If no highly specific match remains, choose the least-specific fallback — but **do not** use R69.
5. **Do not** list comorbidities or incidental findings unless they truly dominate the presentation.

---

**ED Code Shortlist:**
{options_text}

**Casenote:**
{note}

---

**Output Format (exactly):**
1. CODE — "<your rationale>"
2. CODE — "<your rationale>"
3. … up to 4

Please follow that structure precisely.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fewshot + prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

# === Main Application Logic ===

# Add tabs for different input methods
tab1, tab2 = st.tabs(["Text Input", "File Upload"])

with tab1:
    st.header("Enter Case Note")
    note_text = st.text_area("Type or paste the case note here:", height=300, 
                            value=st.session_state.note_text)

with tab2:
    st.header("Upload Case Note")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    if uploaded_file:
        note_text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("File contents:", note_text, height=300)

# Save note text to session state
if note_text:
    st.session_state.note_text = note_text

# Configure top_n parameter
top_n = st.sidebar.slider("Number of similar codes to consider", min_value=5, max_value=20, value=12)

# Process button
if st.button("Classify Note", type="primary", disabled=not bool(note_text and OPENAI_API_KEY)):
    if not EXCEL_PATH.exists():
        st.error(f"Excel file not found at {EXCEL_PATH}")
    elif not JSONL_PATH.exists():
        st.error(f"JSONL file not found at {JSONL_PATH}")
    else:
        with st.spinner("Loading data and building embeddings..."):
            # Load the Excel data
            raw = pd.read_excel(EXCEL_PATH)
            raw.columns = raw.columns.str.strip()
            # Rename columns for consistency with code
            raw = raw.rename(columns={
                "ED Short": "ED Short List code",
                "Diagnosis": "ED Short List Term",
                "Descriptor": "ED Short List Included conditions"
            })
            desc_list = (
                raw["ED Short List Term"] + ". " + raw["ED Short List Included conditions"].fillna("")
            )
            funding_lookup = dict(zip(
                raw["ED Short List code"], raw["Scale"].fillna(3).astype(int)
            ))

            # Build embeddings
            code_embeddings = build_code_embeddings(desc_list, EMBEDDING_CACHE_PATH, USE_LOCAL_EMBEDDINGS)
            if code_embeddings is None:
                st.error("Failed to build embeddings. Please check your configuration.")
                st.stop()

            # Load few-shot examples
            fewshot = load_examples(JSONL_PATH)
            if not fewshot:
                st.warning("No few-shot examples loaded. Results may be less accurate.")

        with st.spinner("Computing embeddings for note..."):
            # Get embeddings for the note
            if USE_LOCAL_EMBEDDINGS:
                note_emb = get_embeddings_local([note_text])[0]
            else:
                note_emb = get_embeddings_openai([note_text])[0]

            if note_emb is None:
                st.error("Failed to generate embeddings for the note.")
                st.stop()

            # Get top similar codes
            shortlist = get_top_matches(note_emb, code_embeddings, raw, top_n)

        with st.spinner("Consulting GPT-4o for diagnosis..."):
            # Query LLM
            resp = predict_final_codes(note_text, shortlist, fewshot, OPENAI_API_KEY)
            if resp is None:
                st.error("Failed to get response from GPT-4o.")
                st.stop()

            # Parse GPT response
            parsed = parse_response(resp, raw)

            # Save results to session state
            st.session_state.results = {
                "shortlist": shortlist,
                "gpt_response": resp,
                "parsed_results": parsed
            }

# Display results if available
if st.session_state.results:
    # Display shortlist
    with st.expander("Embedding Shortlist", expanded=False):
        st.dataframe(
            st.session_state.results["shortlist"][["ED Short List code", "ED Short List Term", "Similarity"]],
            use_container_width=True,
            hide_index=True
        )

    # Display GPT-4o raw response
    with st.expander("GPT-4o Raw Response", expanded=False):
        st.code(st.session_state.results["gpt_response"])

    # Display final results
    st.header("Classification Results")
    if st.session_state.results["parsed_results"]:
        results_df = pd.DataFrame(
            st.session_state.results["parsed_results"], 
            columns=["Code", "Term", "Explanation", "Emoji"]
        )

        # Create a Plotly table for better formatting
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Code", "Term", "Explanation", "Complexity"],
                fill_color='rgb(25, 25, 112)',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    results_df["Code"],
                    results_df["Term"],
                    results_df["Explanation"],
                    results_df["Emoji"]
                ],
                fill_color='rgb(240, 248, 255)',
                align='left',
                font=dict(size=13),
                height=40
            )
        )])

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=125 * len(results_df)  # Adjust height based on number of rows
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No valid codes extracted from the response.")

# Add legend outside of if/else (no indentation!)
st.markdown("""
### 🧾 Complexity Scale Legend

The **Complexity** value reflects the typical resource use associated with each diagnosis code in the Emergency Department setting, based on historical funding data.

<table style="width:100%; font-size:16px; border-collapse:collapse;">
  <thead>
    <tr>
      <th align="left">Scale</th>
      <th align="left">Funding Range (AUD)</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>🟣 1</td><td>≤ $499</td><td>Minimal complexity</td></tr>
    <tr><td>🔵 2</td><td>$500 – $699</td><td>Low complexity</td></tr>
    <tr><td>🟢 3</td><td>$700 – $899</td><td>Moderate complexity</td></tr>
    <tr><td>🟡 4</td><td>$900 – $1099</td><td>High complexity</td></tr>
    <tr><td>🟠 5</td><td>$1100 – $1449</td><td>Significant complexity</td></tr>
    <tr><td>🔴 6</td><td>≥ $1450</td><td>Very high complexity</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

# Display instructions in the sidebar
with st.sidebar.expander("Instructions", expanded=False):
    st.markdown("""
    ### How to use this app

    1. Enter your OpenAI API key in the sidebar
    2. Upload the required Excel and JSONL files or confirm their paths
    3. Enter a case note or upload a text file
    4. Click 'Classify Note' to get ICD code recommendations

    ### About the model

    This application uses:
    - OpenAI's embeddings to find similar codes
    - GPT-4o to analyze the case note and determine the most appropriate codes
    - Streamlit for the web interface

    ### Emoji legend
    - 🟣 Scale 1
    - 🔵 Scale 2
    - 🟢 Scale 3
    - 🟡 Scale 4
    - 🟠 Scale 5
    - 🔴 Scale 6
    """)

# Display requirements in the sidebar
with st.sidebar.expander("Requirements", expanded=False):
    st.code("""
    pip install streamlit pandas numpy plotly openai

    # For local embeddings:
    pip install sentence-transformers
    """)
