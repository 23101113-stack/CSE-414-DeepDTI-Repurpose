import os
import sys
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Batch

# Ensure project root is in search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import DeepDTI
from src.dataset import smiles_to_graph, seq_to_tokens

# --- Page Configuration ---
st.set_page_config(
    page_title="DeepDTI-Repurpose | In-Silico Screening",
    page_icon="🧬",
    layout="wide"
)

# --- Predefined Samples ---
SAMPLE_TARGETS = {
    "Dengue Virus NS5 Polymerase (Fragment)": (
        "GGNEGVVDKLFRKVKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFE"
        "QKKKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFEQKLKKVRGEA"
        "GTLGEVFEQKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFEQKLK"
    ),
    "Custom (Paste Below)": ""
}

SAMPLE_DRUGS = {
    "Ivermectin": "CCC(C)C1CC(CC2CC(CC=C(C(C(C=CC=C3C(O2)C(=O)OC4C3CC(C(=C4)C)O)C)O)C)O1)OC5CC(C(C(O5)C)O)OC6CC(C(C(O6)C)O)OC",
    "Ribavirin": "C1C(C(C(O1)N2C=NC(=N2)C(=O)N)O)O",
    "Favipiravir": "C1=C(N=C(C(=O)N1)C(=O)N)F",
    "Sofosbuvir": "CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3",
    "Remdesivir": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4",
    "Custom (Paste Below)": ""
}

# --- Cache Model to prevent reloading on every click ---
@st.cache_resource
def load_trained_model(weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepDTI().to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        model.eval()
        return model, device
    return None, device

# --- Sidebar ---
st.sidebar.title("⚙️ Model Controls")
model_type = st.sidebar.selectbox("Choose Trained Checkpoint:", ["DAVIS (pK_d)", "KIBA (KIBA Score)"])
weight_file = "weights/best_davis_model.pt" if "DAVIS" in model_type else "weights/best_kiba_model.pt"

model, device = load_trained_model(weight_file)

if model is None:
    st.sidebar.error(f"Checkpoint not found at: {weight_file}")
else:
    st.sidebar.success(f"Loaded: {os.path.basename(weight_file)} on {device.type.upper()}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**DeepDTI-Repurpose** uses a dual-branch neural network (GCN + Multi-scale CNN-BiLSTM + Cross-Attention) "
    "for molecular binding affinity prediction."
)

# --- Header ---
st.title("🧬 DeepDTI-Repurpose: Virtual Drug Screening Platform")
st.markdown("Predict continuous binding affinity between small molecules and target pathogen proteins.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["🎯 Single Pair Prediction", "🧪 Multi-Drug Virtual Screening"])

# ----------------- TAB 1: Single Prediction -----------------
with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("1. Small Molecule (Drug)")
        drug_choice = st.selectbox("Select Candidate Molecule:", list(SAMPLE_DRUGS.keys()))
        if drug_choice == "Custom (Paste Below)":
            smiles_input = st.text_input("Enter SMILES string:", "")
        else:
            smiles_input = st.text_input("SMILES string:", value=SAMPLE_DRUGS[drug_choice])

        # Render 2D Chemical Structure
        if smiles_input.strip():
            mol = Chem.MolFromSmiles(smiles_input.strip())
            if mol:
                img = Draw.MolToImage(mol, size=(300, 200))
                st.image(img, caption="2D Molecular Structure", use_container_width=False)
            else:
                st.warning("Invalid SMILES string.")

    with col2:
        st.subheader("2. Target Protein")
        target_choice = st.selectbox("Select Target Protein:", list(SAMPLE_TARGETS.keys()))
        if target_choice == "Custom (Paste Below)":
            seq_input = st.text_area("Paste Amino Acid Sequence (FASTA):", height=180)
        else:
            seq_input = st.text_area("Amino Acid Sequence:", value=SAMPLE_TARGETS[target_choice], height=180)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Binding Affinity", type="primary", use_container_width=True)

    if predict_btn:
        if not smiles_input.strip() or not seq_input.strip():
            st.error("Please provide both a valid SMILES string and an Amino Acid sequence.")
        elif model is None:
            st.error("Model weights are missing. Please train the model or verify the checkpoint path.")
        else:
            with st.spinner("Processing molecular graph and calculating binding affinity..."):
                graph_data = smiles_to_graph(smiles_input.strip())
                if graph_data[0] is None:
                    st.error("RDKit failed to construct graph. Please check your SMILES.")
                else:
                    x, edge_index = graph_data
                    data = Data(x=x.to(device), edge_index=edge_index.to(device))
                    batch = Batch.from_data_list([data])
                    target_tensor = seq_to_tokens(seq_input.strip(), max_len=1000).unsqueeze(0).to(device)

                    with torch.no_grad():
                        score = model(batch.x, batch.edge_index, batch.batch, target_tensor).item()

                    # Result display
                    st.success("Prediction Completed!")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        st.metric(label="Predicted Binding Affinity", value=f"{score:.4f} pKd")
                    
                    with res_col2:
                        if score >= 6.0:
                            st.info("🔥 **High Binding Affinity:** Potential nanomolar to low micromolar inhibitor.")
                        elif score >= 5.0:
                            st.warning("⚖️ **Moderate Binding Affinity:** Fair interaction; could serve as a scaffold.")
                        else:
                            st.error("❄️ **Weak Binding:** Low predicted interaction.")

# ----------------- TAB 2: Batch Virtual Screening -----------------
with tab2:
    st.subheader("Screen Candidate Drug Library against Target")
    
    screen_target_choice = st.selectbox("Choose Target for Screening:", list(SAMPLE_TARGETS.keys()), key="screen_target")
    if screen_target_choice == "Custom (Paste Below)":
        screen_seq = st.text_area("Enter Target Sequence:", height=120, key="screen_seq")
    else:
        screen_seq = SAMPLE_TARGETS[screen_target_choice]

    run_screen_btn = st.button("📊 Run Virtual Screening", type="primary")

    if run_screen_btn:
        if not screen_seq.strip():
            st.error("Target sequence cannot be empty.")
        elif model is None:
            st.error("Model checkpoint missing.")
        else:
            with st.spinner("Evaluating candidates across target binding domain..."):
                target_tensor = seq_to_tokens(screen_seq.strip(), max_len=1000).unsqueeze(0).to(device)
                screening_records = []

                for name, smiles in SAMPLE_DRUGS.items():
                    if not smiles:
                        continue
                    graph_data = smiles_to_graph(smiles)
                    if graph_data[0] is None:
                        continue
                    x, edge_index = graph_data
                    data = Data(x=x.to(device), edge_index=edge_index.to(device))
                    batch = Batch.from_data_list([data])

                    with torch.no_grad():
                        pred_score = model(batch.x, batch.edge_index, batch.batch, target_tensor).item()

                    screening_records.append({
                        "Drug Candidate": name,
                        "Predicted pKd": round(pred_score, 4),
                        "Classification": "Strong" if pred_score >= 6.0 else ("Moderate" if pred_score >= 5.0 else "Weak")
                    })

                df_screen = pd.DataFrame(screening_records).sort_values(by="Predicted pKd", ascending=False).reset_index(drop=True)

                # Show Results Table & Plot
                s_col1, s_col2 = st.columns([1, 1], gap="large")
                with s_col1:
                    st.dataframe(df_screen, use_container_width=True)
                
                with s_col2:
                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    palette = sns.color_palette("mako", len(df_screen))
                    sns.barplot(data=df_screen, y="Drug Candidate", x="Predicted pKd", palette=palette, ax=ax, edgecolor='black', linewidth=0.5)
                    ax.set_title("Ranked Candidates (Highest Affinity First)")
                    st.pyplot(fig)
