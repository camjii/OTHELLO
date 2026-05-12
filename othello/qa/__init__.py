"""QA modules: load FLAN-T5 variants and run grounded / vanilla inference."""

from othello.qa.flan_t5 import FlanT5Answerer, answer_dataframe

__all__ = ["FlanT5Answerer", "answer_dataframe"]
