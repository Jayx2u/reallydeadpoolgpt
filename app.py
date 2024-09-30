# dead

import torch
import tiktoken
import gradio

from supplementary import GPTModel, generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def main(start_context, max_okens):

    # Determine CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the weights
    weights = torch.load("weights.pth", map_location=torch.device(device))

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.4,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }


    model = GPTModel(GPT_CONFIG_124M).to(device)

    model.load_state_dict(weights)

    # Define tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate text
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=max_okens,
        context_size=256
    )

    return token_ids_to_text(token_ids, tokenizer)

if __name__ == "__main__":
    gr.Interface(fn=main, inputs=[gr.Textbox(label='Starting context'), gr.Number(label="Maximum output tokens")], outputs=[gr.Textbox(label="Response:")], title="ReallyDeadpoolGPT", article="GPT trying to copy Deadpool's humour but fails miserably... :/").launch()

