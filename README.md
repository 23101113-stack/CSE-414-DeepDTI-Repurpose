# DeepDTI-Repurpose: Dual-Branch Deep Representation Learning for Drug Repurposing

DeepDTI-Repurpose is an end-to-end deep learning framework designed to predict continuous drug-target binding affinity ($pK_d$ and KIBA scores) and prioritize FDA-approved candidate therapeutics against pathogen target proteins (e.g., Dengue virus NS5 RNA-dependent RNA polymerase).

---

## 🔬 Framework Architecture

The framework processes heterogeneous biochemical modalities via a dual-branch neural network:
* **Drug Branch (GNN):** 3-layer Graph Convolutional Network (GCN) that takes 2D molecular graphs from SMILES strings with 5 atom features (atomic symbol, degree, hybridization, implicit valence, aromaticity) followed by global mean pooling.
* **Target Branch (CNN-BiLSTM):** Multi-scale 1D-CNN (receptive kernel sizes 4, 8, 12) combined with a Bidirectional LSTM (BiLSTM) and global max-pooling to extract local motifs and long-range sequence context from protein amino acid sequences.
* **Cross-Attention Fusion:** Multi-head cross-attention layer aligning drug substructural features with protein sequence motifs.
* **Prediction Head:** Multilayer Perceptron (MLP) with Batch Normalization, LeakyReLU, and Dropout (0.2) to predict continuous interaction scores.

[![Streamlit App 1](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepdti-repurpose-app1.streamlit.app/) — **Interactive Screening Dashboard**

[![Streamlit App 2](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepdti-repurpose-app2.streamlit.app/) — **Fast Batch Virtual Screening Platform**

---

---

## 📂 Repository Structure

```text
DeepDTI-Repurpose/
├── data/
│   └── raw/
│       ├── davis_all.csv              # DAVIS benchmark dataset (30,056 interactions)
│       └── kiba_all.csv               # KIBA benchmark dataset (118,254 interactions)
├── src/
│   ├── dataset.py                     # Graph conversion, sequence tokenization & DataLoaders
│   ├── models.py                      # DeepDTI model architecture
│   ├── train_davis.py                 # DAVIS training and evaluation pipeline
│   └── train_kiba.py                  # KIBA training and evaluation pipeline
├── weights/
│   ├── best_davis_model.pt            # Checkpoint trained on DAVIS
│   └── best_kiba_model.pt             # Checkpoint trained on KIBA
├── results/
│   ├── davis_test_evaluation.png      # Ground truth vs. predicted pKd regression plot
│   ├── kiba_test_evaluation.png       # Ground truth vs. predicted KIBA score regression plot
│   ├── cross_attention_heatmap.png    # Interaction interpretability heatmap
│   ├── dengue_screening_ranked.png    # Ranked candidate repurposing bar chart
│   └── repurposing_screening_results.csv # Predicted affinities for prioritized candidates
├── evaluate_plots.py                  # Script to generate evaluation plots
├── plot_attention.py                  # Script to extract and visualize cross-attention weights
├── screening.py                       # Virtual drug repurposing inference pipeline
├── requirements.txt                   # Environment dependencies
└── README.md

```markdown
# DeepDTI-Repurpose: Dual-Branch Deep Representation Learning for Computational Drug Repurposing

DeepDTI-Repurpose is an end-to-end deep learning framework designed to predict continuous drug-target binding affinities ($pK_d$ and KIBA scores) and systematically prioritize candidate small molecules for virtual drug repurposing against viral pathogens, with a primary case study on the Dengue Virus NS5 RNA-dependent RNA polymerase (RdRp).

---

## 🔬 Framework Architecture

The framework handles heterogeneous biochemical inputs through a dual-branch representation learning architecture:

* **Drug Branch (Graph Neural Network):** Translates SMILES strings into molecular graphs with 5 explicit node features (atomic number, degree, implicit valence, hybridization, aromaticity). Processes graphs via a 3-layer Graph Convolutional Network (GCN) followed by global mean pooling to construct graph-level embedding vectors $h_d \in \mathbb{R}^{128}$.
* **Target Branch (Multi-Scale 1D-CNN + BiLSTM):** Encodes amino acid sequences via dense embeddings, captures multi-scale localized binding motifs using parallel 1D convolutions (kernel sizes 4, 8, 12), and models long-range contextual dependencies through a Bidirectional LSTM (BiLSTM). Sequence-level representation $h_t \in \mathbb{R}^{128}$ is pooled using global max-pooling.
* **Cross-Attention Fusion:** Employs multi-head cross-attention to project drug substructures and target sequence motifs into an aligned latent interaction space.
* **Prediction Head:** A fully connected Multilayer Perceptron (MLP) with Batch Normalization, LeakyReLU activation ($\alpha=0.2$), and Dropout ($p=0.2$) predicting continuous interaction metrics.

---

## 📂 Repository Structure

```text
DeepDTI-Repurpose/
├── data/
│   └── raw/
│       ├── davis_all.csv                    # DAVIS benchmark dataset (30,056 pairs)
│       └── kiba_all.csv                     # KIBA benchmark dataset (118,254 pairs)
├── src/
│   ├── dataset.py                           # Molecular graph extraction & sequence tokenization
│   ├── models.py                            # GNN + CNN-BiLSTM + Cross-Attention architecture
│   ├── train_davis.py                       # Training pipeline for DAVIS dataset
│   └── train_kiba.py                        # Training pipeline for KIBA dataset
├── weights/
│   ├── best_davis_model.pt                  # Optimal checkpoint for DAVIS (CI: 0.8224)
│   └── best_kiba_model.pt                   # Optimal checkpoint for KIBA (CI: 0.8140)
├── results/
│   ├── davis_metrics.txt                    # Final evaluation metrics on DAVIS test set
│   ├── davis_test_evaluation.png            # Ground truth vs. predicted pKd scatter plot
│   ├── kiba_test_evaluation.png             # Ground truth vs. predicted KIBA score plot
│   ├── cross_attention_heatmap.png          # Substructure-motif alignment attention map
│   ├── dengue_screening_ranked.png          # Prioritized candidate comparison bar chart
│   └── repurposing_screening_results.csv    # Full in-silico screening rank table
├── evaluate_plots.py                        # Test evaluation visualization script
├── plot_attention.py                        # Cross-attention extraction & heatmap generator
├── screening.py                             # Pathogen virtual screening pipeline
├── requirements.txt                         # Pinned Python package dependencies
└── README.md

```

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites

* Python 3.11 or 3.12 (Recommended for PyTorch Geometric & RDKit compatibility)
* NVIDIA GPU with CUDA 12.1+ support (Optional, CPU fallback supported)

### 2. Environment Setup

```powershell
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/DeepDTI-Repurpose.git
cd DeepDTI-Repurpose

# Create virtual environment
py -3.12 -m venv venv

# Activate virtual environment
.\venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate     # Linux / macOS

```

### 3. Install Dependencies

```powershell
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install torch_geometric
pip install rdkit pandas numpy scikit-learn matplotlib seaborn lifelines

```

---

## 🚀 Reproduction & Execution

### 1. Model Training & Validation

* **Train on DAVIS Benchmark:**
```powershell
python src/train_davis.py

```


* **Train on KIBA Benchmark:**
```powershell
python src/train_kiba.py

```



### 2. Evaluation & Interpretability Plots

* **Generate Test Set Regression Scatter Plots:**
```powershell
python evaluate_plots.py

```


* **Generate Cross-Attention Interaction Heatmap:**
```powershell
python plot_attention.py

```



### 3. In-Silico Screening for Dengue Virus (NS5 RdRp)

Screen candidate molecules against the Dengue virus RNA-dependent RNA polymerase target sequence:

```powershell
python screening.py

```

---

## 📊 Experimental Results

Model evaluation across both benchmark datasets using an 80/10/10 train/validation/test split:

| Dataset | Total Pairs | Metric Target | Test MSE ($\downarrow$) | Test CI ($\uparrow$) | Test Pearson $R$ ($\uparrow$) |
| --- | --- | --- | --- | --- | --- |
| **DAVIS** | 30,056 | $pK_d$ | **0.4587** | **0.8224** | **0.6817** |
| **KIBA** | 118,254 | KIBA Score | **0.4881** | **0.8140** | **0.6477** |

* **Concordance Index (CI > 0.81):** Confirms that the dual-branch framework accurately ranks compound affinities.
* **Pearson Correlation ($R > 0.64$):** Demonstrates strong positive linear association between predicted and experimental affinity values.

---

## 🧬 Virtual Screening Leads (Dengue NS5 RdRp)

Prioritized candidates evaluated against the Dengue virus NS5 RNA-dependent RNA polymerase sequence:

| Rank | Candidate Molecule | Predicted Affinity ($pK_d$) | Known Pharmacological Action | Relevance to Dengue |
| --- | --- | --- | --- | --- |
| **1** | **Ivermectin** | **6.2469** | Broad-spectrum antiparasitic | Nuclear transport inhibitor (Importin $\alpha/\beta$ target) |
| **2** | **Ribavirin** | **5.7201** | Broad-spectrum antiviral nucleoside | Established viral RdRp catalytic domain inhibitor |
| **3** | **Favipiravir** | **5.4674** | Purine nucleic acid analog | Selective RNA-dependent RNA polymerase inhibitor |
| **4** | **Doxycycline** | **5.3084** | Tetracycline class antibiotic | Reported in-vitro Dengue protease & entry inhibition |
| **5** | **Sofosbuvir** | **5.1281** | HCV NS5B polymerase inhibitor | Nucleotide prodrug targeting viral replication machinery |
| **6** | **Remdesivir** | **5.1244** | Monophosphoramidate prodrug | Delayed RNA chain termination against viral polymerases |
| **7** | **Chloroquine** | **4.8345** | 4-Aminoquinoline antimalarial | Endosomal acidification & uncoating inhibitor |

---

## 📜 Citation & References

* Davis, M. I., et al. (2011). Comprehensive analysis of kinase inhibitor selectivity. *Nature Biotechnology*, 29(11), 1046–1051.
* Tang, J., et al. (2014). Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. *Journal of Chemical Information and Modeling*, 54(3), 735–743.
* Nguyen, T., et al. (2021). GraphDTA: Predicting drug–target binding affinity with graph neural networks. *Bioinformatics*, 37(8), 1140–1147.

