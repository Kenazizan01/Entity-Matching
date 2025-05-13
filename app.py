import streamlit as st
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

st.set_page_config(page_title="Product Similarity Checker", layout="centered")

st.title("🛍️ Product Similarity Prediction")
st.write("Enter two product descriptions to check if they match.")

# Load model and tokenizer (cached to avoid reloading every time)
@st.cache_resource
def load_model_and_tokenizer():
    peft_model_id = "Kenazin/Llama-3.1-8B-peft-p-tuning-v4-8"
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# Prediction function
def predict_product_similarity(product_info_1, product_info_2, threshold):
    combined_input = f"Product 1: {product_info_1}\nProduct 2: {product_info_2}"
    inputs = tokenizer(
        combined_input,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        prob_class_1 = probs[0, 1].item()
        predicted_class = 1 if prob_class_1 >= threshold else 0

    return predicted_class

# Streamlit input
product1 = st.text_area("📝 Product 1 Description")
product2 = st.text_area("📝 Product 2 Description")
product_type = st.selectbox("📦 Product Type", ["Hardware Product", "Non-Hardware Product"])

if st.button("🔍 Predict"):
    if not product1 or not product2:
        st.warning("Please enter both product descriptions.")
    else:
        threshold = 0.97 if product_type == "Hardware Product" else 0.5
        pred_class = predict_product_similarity(product1, product2, threshold)
        label = "✅ Match" if pred_class == 1 else "❌ Not a Match"

        st.markdown(f"### Result: {label}")
