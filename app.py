import pandas as pd
import torch
import streamlit as st

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from transformer import Transformer


def predict_all_probs(
        model,
        tokenizer,
        device,
        id2label,
        abstract: str = "",
        max_length: int = 256,
) -> pd.DataFrame:
    model.eval()

    inputs = tokenizer(
        (abstract or "").strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    rows = []
    for idx, p in enumerate(probs):
        rows.append(
            {
                "topic": id2label[idx],
                "probability": float(p),
                "probability_percent": round(float(p) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def predict_top1(
        model,
        tokenizer,
        device,
        id2label,
        abstract: str = "",
        max_length: int = 256,
) -> dict:
    df = predict_all_probs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        id2label=id2label,
        abstract=abstract,
        max_length=max_length,
    )

    top1 = df.iloc[0]
    return {
        "topic": top1["topic"],
        "probability": float(top1["probability"]),
        "probability_percent": float(top1["probability_percent"]),
    }


@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Svyat-dr/article_classifier"

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="transformer_checkpoint.pt",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = Transformer(
        vocab_size=checkpoint["vocab_size"],
        max_len=checkpoint["max_len"],
        num_classes=checkpoint["num_classes"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        d_feed_forward=checkpoint["d_feed_forward"],
        dropout=checkpoint["dropout"],
        pad_token_id=checkpoint["pad_token_id"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id2label = checkpoint["id2label"]

    return tokenizer, model, device, id2label


def main():
    st.set_page_config(page_title="Paper Topic Classifier", layout="centered")
    st.title("Classification of article topics")

    st.write("Enter the article abstract.")
    st.write(
        "The model predicts one most probable topic among the following classes: "
        "physics, mathematics, computer science."
    )

    tokenizer, model, device, id2label = load_model_and_tokenizer()

    abstract = st.text_area("Abstract", height=200)

    classification_button = st.button("Classify")

    if classification_button:
        if not abstract.strip():
            st.error("Abstract is required.")
            return

        with st.spinner("Classifying the article..."):
            result = predict_top1(
                model=model,
                tokenizer=tokenizer,
                device=device,
                id2label=id2label,
                abstract=abstract,
            )

        st.success(
            f"Predicted topic: {result['topic']}"
        )


if __name__ == "__main__":
    main()
