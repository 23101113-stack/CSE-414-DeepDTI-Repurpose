import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# --- পেজ সেটআপ ---
st.set_page_config(page_title="Easy Drug Repurposing", layout="centered")

st.title("💊 AI Drug Repurposing Finder")
st.write("বিদ্যমান ওষুধ দিয়ে নতুন রোগের চিকিৎসা সম্ভব কিনা তা এক ক্লিকে পরীক্ষা করুন।")

st.divider()

# --- ১. ব্যাকএন্ডের রেডিমেড ডেটা (সিম্পল ডেটাবেস) ---
DISEASE_TARGETS = {
    "Dengue Fever (ডেঙ্গু)": "MLNEHAITTRVLVANVLPVHGKLLREILKDGTLDVNLAKREVKLSFEEMGEEEAELLSVGDWDLG",
    "COVID-19 (করোনা ভাইরাস)": "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQ",
    "Alzheimer's Disease (অ্যালঝেইমার)": "TQHGIRLPLRSGLGGAPLGLRLPRETDEEPEEPGRRGSFVEMVDNLRGKSGQGYYVEMTVGSPPQTLNILVDTGSSNFAVGAAPHPFLHRYYQRQLSSTYRDLRKGVYVPYTQGKWEGELGTDLVSIPHGPNVTVRANIAAITESDKFFINGSNWEGILGLAYAEIARPDDSLEPFFDSLVKQTHVPNLFSLQLCGAGFPLNQSEVLASVGGSMIIGGIDHSLYTGSLWYTPIRREWYYEVIIVRVEINGQDLKMDCKEYNYDKSIVDSGTTNLRLPKKVFEAAVKSIKAASSTEKFPDGFWLGEQLVCWQAGTTPWNIFPVISLYLMGEVTNQSFRITILPQQYLRPVEDVATSQDDCYKFAISQSSTGTVMGAVIMEGFYVVFDRARKRIGFAVSACHVHDEFRTAAVEGPFVTLDMEDCGYNIPQTDESTLMTIAYVMAAICALFMLPLCLMVCQWRCLRCLRQQHDDFADDISLLK"
}

# ড্রাগ ডেটাবেস (নাম, আসল রোগ, কেমিক্যাল SMILES, এবং ডামি অ্যাফিনিটি স্কোর)
DRUG_DATABASE = [
    {"Name": "Imatinib", "Original_Use": "Leukemia (ব্লাড ক্যান্সার)", "SMILES": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", "Score": 8.75},
    {"Name": "Niclosamide", "Original_Use": "Tapeworm Infection (কৃমিনাশক)", "SMILES": "O=C(Nc1ccc([N+](=O)[O-])cc1Cl)c2cc(Cl)ccc2O", "Score": 8.30},
    {"Name": "Sunitinib", "Original_Use": "Kidney Cancer (কিডনি ক্যান্সার)", "SMILES": "CCN(CC)CCNC(=O)c1c(C)[nH]c(\C=C2/C(=O)Nc3ccc(F)cc23)c1C", "Score": 7.95},
    {"Name": "Metformin", "Original_Use": "Type 2 Diabetes (ডায়াবেটিস)", "SMILES": "CN(C)C(=N)NC(=N)N", "Score": 5.40},
    {"Name": "Paracetamol", "Original_Use": "Fever / Pain (জ্বর ও ব্যথা)", "SMILES": "CC(=O)Nc1ccc(O)cc1", "Score": 4.10}
]

# --- ২. ইউজার ইনপুট প্যানেল ---
selected_disease = st.selectbox(
    "১. কোন রোগের জন্য ওষুধ খুঁজতে চান?",
    list(DISEASE_TARGETS.keys())
)

action_mode = st.radio(
    "২. কীভাবে দেখতে চান?",
    ["টপ ৫টি সেরা ড্রাগের তালিকা দেখুন (Virtual Screening)", "নির্দিষ্ট একটি ড্রাগ টেস্ট করুন (Single Test)"]
)

# --- ৩. ফলাফল প্রদর্শন ---
st.divider()

if action_mode == "টপ ৫টি সেরা ড্রাগের তালিকা দেখুন (Virtual Screening)":
    if st.button("🔍 রিপারপাসিং শুরু করুন", type="primary", use_container_width=True):
        st.subheader(f"📊 {selected_disease}-এর জন্য সম্ভাব্য ড্রাগসমূহ:")
        
        # টেবিল সাজানো
        df = pd.DataFrame(DRUG_DATABASE)[["Name", "Original_Use", "Score"]]
        df.columns = ["ওষুধের নাম (Drug)", "আসল রোগের নাম (Original Use)", "ম্যাচিং স্কোর (pKd Score)"]
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # সেরা ড্রাগ হাইলাইট
        top_drug = DRUG_DATABASE[0]
        st.success(f"🏆 **সেরা ড্রাগ ক্যান্ডিডেট:** {top_drug['Name']} (বর্তমান ব্যবহার: {top_drug['Original_Use']})")
        
        # ছবি
        st.write(f"**{top_drug['Name']}-এর রাসায়নিক গঠন:**")
        mol = Chem.MolFromSmiles(top_drug["SMILES"])
        st.image(Draw.MolToImage(mol, size=(380, 180)), caption=top_drug["Name"])

else:
    # সিঙ্গেল টেস্ট মোড
    drug_names = [d["Name"] for d in DRUG_DATABASE]
    chosen_drug_name = st.selectbox("ওষুধ সিলেক্ট করুন:", drug_names)
    
    if st.button("🧪 টেস্ট ফলাফল দেখুন", type="primary", use_container_width=True):
        # সিলেক্টেড ড্রাগের তথ্য
        drug_info = next(d for d in DRUG_DATABASE if d["Name"] == chosen_drug_name)
        score = drug_info["Score"]
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.metric("বাইন্ডিং স্কোর (pKd)", f"{score} / 10")
            if score >= 7.0:
                st.success("✅ **High Affinity:** ড্রাগটি এই রোগে ভালো কাজ করার জোরালো সম্ভাবনা রয়েছে!")
            elif score >= 5.0:
                st.warning("⚠️ **Moderate:** মাঝারি কার্যকারিতা দেখাতে পারে।")
            else:
                st.error("❌ **Low:** এই রোগের ক্ষেত্রে কার্যকর হওয়ার সম্ভাবনা কম।")
                
            st.write(f"**আসল ব্যবহার:** {drug_info['Original_Use']}")
            
        with col_res2:
            mol = Chem.MolFromSmiles(drug_info["SMILES"])
            st.image(Draw.MolToImage(mol, size=(300, 150)), caption=drug_info["Name"])