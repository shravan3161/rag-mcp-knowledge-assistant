import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


@torch.inference_mode()
def generate_answer(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise customer support assistant. "
                "Answer directly in 2 to 3 brief sentences using ONLY the facts in the context. "
                "Do not use line breaks or bullet points. Write clean, direct prose."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    raw_answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Clean any newline escape sequences into clean readable prose
    clean_answer = " ".join(line.strip() for line in raw_answer.splitlines() if line.strip())

    return clean_answer
