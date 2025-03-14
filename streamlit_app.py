import re
import streamlit as st
import torch
import torch.nn as nn
import json
import math

# Set page config for better appearance
st.set_page_config(
    page_title="LogicCompiler",
    page_icon="📝➡️💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load vocabulary
try:
    with open("vocabulary.json", "r") as f:
        vocab = json.load(f)
    st.sidebar.write(f"✅ Vocabulary loaded with {len(vocab)} tokens")
except FileNotFoundError:
    vocab = {}
    st.sidebar.warning("Vocabulary file not found.")

# Transformer Configuration
class Config:
    vocab_size = 12006  # Adjust as needed
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Positional Encoding
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

# Transformer Model
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

@st.cache_resource
def load_model(model_path):
    try:
        model = Seq2SeqTransformer(config).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        st.sidebar.success(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

def translate(model, input_tokens, vocab, device, max_length=50):
    if model is None:
        return "Model not loaded."
    model.eval()
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
        return f"Translation error: {str(e)}"

# UI Header
st.markdown("""
    <h1 style='text-align: center; color: #007acc;'>LogicCompiler 📝 ➡️ 💻</h1>
    <p style='text-align: center;'>Convert pseudocode into C++ code effortlessly!</p>
    """, unsafe_allow_html=True)

# Load model
model = load_model("p2c1.pth")

# Code Input
st.markdown("### Enter your pseudocode:")
input_text = st.text_area("Pseudocode Input", height=200, placeholder="Write your pseudocode here...")

# Translate Button
if st.button("Translate to C++ Code"):
    if not input_text.strip():
        st.warning("Please enter some pseudocode to translate!")
    else:
        with st.spinner("Translating..."):
            tokens = input_text.strip().split()
            result = translate(model, tokens, vocab, config.device)
            st.markdown("### Generated C++ Code")
            st.code(result, language="cpp")

# Sidebar Info
st.sidebar.title("LogicCompiler Info")
st.sidebar.info("Version 1.0")
st.sidebar.markdown("Developed by Hassan Haseen © 2025")
