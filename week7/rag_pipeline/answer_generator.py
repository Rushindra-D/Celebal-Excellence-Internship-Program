"""
Handles Stage 7 of the pipeline: Answer Generation.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class AnswerGenerator:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_prompt(self, user_question, retrieved_chunks):

        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

        return (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not present in the context, say "
            "'I don't know based on the provided documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            "Answer:"
        )

    def generate_answer(self, user_question, retrieved_chunks):

        prompt = self.build_prompt(user_question, retrieved_chunks)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer