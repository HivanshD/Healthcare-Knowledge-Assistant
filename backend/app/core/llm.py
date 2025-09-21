from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .config import settings

class LocalGenerator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.GEN_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        max_new_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=settings.TOP_P,
                temperature=settings.TEMPERATURE,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_answer_prompt(question: str, contexts: list[str]) -> str:
    ctx_block = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = f"""You are a helpful assistant. Use ONLY the context to answer.
If the answer cannot be found in the context, say you don't know.

Question: {question}

{ctx_block}

Answer:"""
    return prompt
