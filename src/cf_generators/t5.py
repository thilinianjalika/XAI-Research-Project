from .base import BaseGenerator
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import List, Dict
import nltk


class T5Generator(BaseGenerator):
    def __init__(
        self, config_path: str, root: str = None, download: bool = False
    ) -> None:
        super().__init__(config_path, root, download)
        tokenizer = T5Tokenizer.from_pretrained(self.config["model_config"])
        model = T5ForConditionalGeneration.from_pretrained(self.config["model_config"])
        state = torch.load(self.config["paths"]["model"])
        model.load_state_dict(state)
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, inp: str, variations: int) -> List[str]:
        # format input
        inp = nltk.sent_tokenize(inp)
        inp = ["contradict: " + sent for sent in inp]

        # generate
        sentence_sets = []
        for sent in inp:
            input_ids = self.tokenizer(sent, return_tensors="pt").input_ids
            label_ids = self.model.generate(
                input_ids, num_return_sequences=variations, num_beams=variations
            )
            sents = [
                self.tokenizer.decode(label_id_row, skip_special_tokens=True)
                for label_id_row in label_ids
            ]
            sentence_sets.append(sents)

        paras = []
        for sentences in zip(*sentence_sets):
            para = " ".join(sentences)
            paras.append(para)

        return paras

    def set_config(self, config: Dict) -> None:
        print("WARN: Nothing to do")
