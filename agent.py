"""
RAG agent.

Wraps the local Qwen3-0.6B model as a LangChain LLM and combines it with the
policy retriever behind a strict anti-hallucination prompt:

  * answer only from retrieved university IT policy context,
  * if the exact fact (e.g. the IT Service Desk phone number) is not in the
    context, state that the available information does not provide it,
  * never invent phone numbers, room numbers, names, or other facts.
"""

import os
import re
from pathlib import Path
from typing import ClassVar

import torch
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_tool import build_retriever

# The SFT exercise downloads the model to ../SCSE_26_GEX3/models/qwen3-0.6b.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "SCSE_26_GEX3"
    / "models"
    / "qwen3-0.6b"
)

SYSTEM_PROMPT = (
    "You are a university IT support assistant. "
    "Answer the user's question using ONLY the information in the retrieved "
    "university IT policy context below. "
    "If the exact information requested is not present in the context, you must "
    "reply that the available university IT information does not provide it. "
    "Never invent or guess phone numbers, room numbers, names, quotas, or any "
    "other fact not stated in the context. Keep the answer short and direct."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    SYSTEM_PROMPT
    + "\n\nRetrieved policy context:\n{context}\n\n"
    + "Question: {question}\nAnswer:"
)

_THINKING_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class QwenLocalLLM(LLM):
    """LangChain LLM backed by a local Transformers Qwen3 model."""

    model_path: str = ""

    _model: ClassVar[object] = None
    _tokenizer: ClassVar[object] = None
    _device: ClassVar[str] = "cpu"

    def __init__(self, model_path: str = None, **kwargs):
        path = (
            model_path
            or os.environ.get("QWEN_MODEL_PATH")
            or str(DEFAULT_MODEL_PATH)
        )
        super().__init__(model_path=path, **kwargs)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        QwenLocalLLM._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        QwenLocalLLM._tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )
        QwenLocalLLM._model = AutoModelForCausalLM.from_pretrained(
            path, dtype="auto", local_files_only=True
        ).to(QwenLocalLLM._device)
        QwenLocalLLM._model.eval()

    @property
    def _llm_type(self) -> str:
        return "qwen3-0.6b-local"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        tokenizer = QwenLocalLLM._tokenizer
        model = QwenLocalLLM._model

        messages = [{"role": "user", "content": prompt}]
        try:  # Qwen3 chat template; disable the reasoning/thinking mode
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = _THINKING_BLOCK.sub("", text).strip()
        return text


def _format_context(docs) -> str:
    return "\n\n".join(doc.page_content.strip() for doc in docs)


def build_agent(documents):
    """Assemble the retrieval-augmented Qwen agent as a LangChain chain."""
    retriever = build_retriever(documents)
    llm = QwenLocalLLM()

    agent = (
        {
            "context": retriever | _format_context,
            "question": RunnablePassthrough(),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return agent
