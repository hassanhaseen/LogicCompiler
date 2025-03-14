import streamlit as st
import torch
import torch.nn as nn
import json
import math
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LogicCompiler",
    page_icon="📝➡️💻",
    layout="centered"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
        /* Hide default Streamlit style elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Remove top padding */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        /* Body background */
        body {
            background-color: #0f1117;
        }

        /* Main container */
        .main-container {
            max-width: 800px;
            margin: auto;
            background-color: #1c1f26;
            padding: 3rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            color: #f5f6fa;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Header styling */
        .header h1 {
            font-size: 3rem;
            color: #4facfe;
            margin-bottom: 0.3rem;
        }

        .header p {
            font-size: 1.1rem;
            color: #a0a4b8;
        }

        /* Status messages */
        .status-message {
            text-align: center;
            margin-top: 1rem;
            color: #2ecc71;
            font-weight: bold;
        }

        .error-message {
            text-align: center;
            margin-top: 1rem;
            color: #e74c3c;
            font-weight: bold;
        }

        /* Text area */
        textarea {
            border-radius: 8px !important;
            background-color: #2b2f3a !important;
            color: #ffffff !important;
            font-size: 1rem !important;
            border: 1px solid #4facfe !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton>button:hover {
            transform: scale(1.03);
            box-shadow: 0px 4px 15px rgba(79, 172, 254, 0.4);
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #a0a4b8;
            font-size: 0.9rem;
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

# ========== MAIN CONTAINER ==========
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <div class='header'>
        <h1>LogicCompiler</h1>
        <p>Convert Pseudocode into C++ Code Seamlessly</p>
    </div>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
model = load_model("p2c1.pth")

# ========== MODEL STATUS ==========
if model:
    st.markdown("<div class='status-message'>✅ Model loaded successfully!</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='error-message'>❌ Model not loaded. Check the file!</div>", unsafe_allow_html=True)

# ========== INPUT SECTION ==========
st.subheader("📝 Enter your pseudocode:")
input_text = st.text_area("", height=200, placeholder="Write your pseudocode here...")

# ========== TRANSLATE BUTTON ==========
if st.button("Translate to C++ Code"):
    if not input_text.strip():
        st.warning("⚠️ Please enter pseudocode to translate!")
    else:
        with st.spinner("Translating..."):
            tokens = input_text.strip().split()
            result = translate(model, tokens, vocab, config.device)
            st.subheader("💻 Generated C++ Code:")
            st.code(result, language="cpp")

# ========== FOOTER ==========
st.markdown("""
    <div class='footer'>
        © 2025 Hassan Haseen - LogicCompiler v1.0
    </div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
