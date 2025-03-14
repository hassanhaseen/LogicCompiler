import streamlit as st
import torch
import torch.nn as nn
import json
import math
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LogicCompiler - Pseudocode to C++",
    page_icon="📝➡️💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
    /* Hide Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background and fonts */
    .stApp {
        background: linear-gradient(to bottom right, #1A1A1D, #0D0D0D);
        color: #F0EAD6;
        font-family: 'Georgia', serif;
    }

    /* Title */
    h1 {
        color: #4facfe !important;
        text-align: center;
        font-size: 3rem;
    }

    /* Subtitles & sections */
    h2, h3, .stMarkdown {
        color: #a0a4b8 !important;
    }

    /* Text inputs and text areas */
    textarea {
        background-color: #262730 !important;
        color: #F0EAD6 !important;
        border: 1px solid #4facfe !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: #0D0D0D;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 16px rgba(79, 172, 254, 0.3);
    }

    /* Footer styling */
    .footer {
        position: relative;
        display: inline-block;
        color: #888;
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
    }

    .footer span:hover::after {
        content: " LogicCompiler v1.0 | Powered by Streamlit & PyTorch ";
        position: absolute;
        top: -30px;
        right: 0;
        transform: translateX(0%);
        background-color: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        white-space: nowrap;
        font-size: 0.8rem;
        opacity: 1;
        z-index: 10;
    }

    /* Center content */
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD VOCABULARY ==========
vocab_path = "vocabulary.json"

if not os.path.isfile(vocab_path):
    st.error("❌ vocabulary.json file not found in the directory!")
    st.stop()

with open(vocab_path, "r") as f:
    vocab = json.load(f)

# ========== CONFIG ==========
class Config:
    vocab_size = 12006
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ========== POSITIONAL ENCODING ==========
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ========== TRANSFORMER MODEL ==========
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = PositionalEncoding(config.embed_dim, config.max_length)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))
        out = self.fc_out(out.permute(1, 0, 2))
        return out

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model(model_path):
    if not os.path.isfile(model_path):
        st.error(f"❌ Model file '{model_path}' not found in the directory!")
        return None

    try:
        model = Seq2SeqTransformer(config).to(config.device)
        state_dict = torch.load(model_path, map_location=config.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ========== TRANSLATE FUNCTION ==========
def translate(model, input_tokens, vocab, device, max_length=50):
    if model is None:
        return "❌ Model not loaded."

    try:
        input_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in input_tokens]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        output_ids = [vocab.get("<start>", 2)]

        for _ in range(max_length):
            output_tensor = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(input_tensor, output_tensor)
            next_token_id = predictions.argmax(dim=-1)[:, -1].item()

            output_ids.append(next_token_id)

            if next_token_id == vocab.get("<end>", 3):
                break

        id_to_token = {idx: token for token, idx in vocab.items()}
        return " ".join([id_to_token.get(idx, "<unk>") for idx in output_ids[1:]])

    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# ========== MAIN UI ==========
st.title("📝➡️💻 LogicCompiler")
st.markdown("##### Convert your Pseudocode into C++ Code Seamlessly")

# Load model
model = load_model("p2c1.pth")

# Model status message
if model:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model not loaded. Please check the file path!")

# Input
st.subheader("✍️ Enter your Pseudocode:")
input_text = st.text_area("", height=200, placeholder="Write your pseudocode here...")

# Translate Button
if st.button("✨ Translate to C++ Code", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Please enter pseudocode to translate!")
    else:
        with st.spinner("Translating..."):
            tokens = input_text.strip().split()
            result = translate(model, tokens, vocab, config.device)
            st.subheader("💻 Generated C++ Code:")
            st.code(result, language="cpp")

# Footer
st.markdown("""
---
<p class="footer" style="text-align: center;">
    Built with ❤️ by <span>Team CodeRunners</span>
</p>
""", unsafe_allow_html=True)
