import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from torch_geometric.nn import GCNConv

# =========================
# Model definition
# =========================
def get_model(in_features, hidden_dim, out_classes, num_layers=3, dropout=0.15):
    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.reduce = torch.nn.Linear(in_features, hidden_dim)
            self.convs = torch.nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
            self.out = torch.nn.Linear(hidden_dim // 2, out_classes)
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
            self.bn2 = torch.nn.BatchNorm1d(hidden_dim // 2)

        def forward(self, x, edge_index, edge_weight=None):
            x = F.relu(self.bn(self.reduce(x)))
            x = F.dropout(x, self.dropout, self.training)
            for i, conv in enumerate(self.convs):
                res = x
                x = F.relu(conv(x, edge_index, edge_weight))
                if i > 0:
                    x = x + res
                x = F.dropout(x, self.dropout, self.training)
            x = F.relu(self.fc1(x))
            x = F.relu(self.bn2(self.fc2(x)))
            return self.out(x)
    return GNN()

# =========================
# Load model and vectorizer
# =========================
@st.cache_resource
def load_model_and_vectorizer(
    model_path="optimal_gcn_weights.pth",
    vectorizer_path="tfidf_vectorizer.pkl",
    hidden_dim=64,
    num_layers=5,
    dropout=0.4,
    label_map={"negative": 0, "neutral": 1, "positive": 2}
):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    model = get_model(
        in_features=len(vectorizer.get_feature_names_out()),
        hidden_dim=hidden_dim,
        out_classes=len(label_map),
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    label_decoder = {v: k for k, v in label_map.items()}
    return model, vectorizer, label_decoder

# =========================
# Predict single text
# =========================
def predict_sentiment(model, vectorizer, text, label_decoder):
    tfidf_vec = vectorizer.transform([text]).toarray()
    x = torch.FloatTensor(tfidf_vec)

    # Dummy edge self-loop
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    return label_decoder[pred_idx], float(probs[pred_idx]), probs

# =========================
# Streamlit UI
# =========================
st.title("📊 Sentiment Analysis with GCN (Single Inference)")

model, vectorizer, label_decoder = load_model_and_vectorizer()

user_input = st.text_area("Masukkan teks untuk analisis sentimen:")

if st.button("Prediksi"):
    if user_input.strip():
        label, confidence, probs = predict_sentiment(model, vectorizer, user_input, label_decoder)
        st.markdown(f"### ✅ Prediksi: **{label}**")
        st.write(f"Confidence: {confidence:.4f}")
        st.bar_chart({lbl: float(p) for lbl, p in zip(label_decoder.values(), probs)})
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")




# HOW TO RUN streamlit run app.py