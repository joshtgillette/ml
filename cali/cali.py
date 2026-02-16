from transformers import T5ForConditionalGeneration


def go():
    # Load the model and tokenizer
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")
