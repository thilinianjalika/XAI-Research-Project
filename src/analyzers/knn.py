import numpy as np
from ..processors import TextVectorizer
from ..cf_generators import T5Generator, WordFlippingGenerator
from typing import Tuple, List, Dict, Union, Any
import yaml
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseAnalyzer
import scipy
import joblib


class KNNAnalyzer(BaseAnalyzer):
    _data_labels = ["negative", "positive"]

    def __init__(
        self,
        knn_path: str,
        vectorizer_path: str,
        cf_generator_config: str = None,
        cf_generator_root: str = None,
    ) -> None:
        supported_cf_gens = ("t5-cf-generator", "wf-cf-generator")
        if type(cf_generator_config) == str:
            with open(cf_generator_config) as handler:
                cf_gen_nm = yaml.load(handler, yaml.FullLoader)["name"]
        else:
            cf_gen_nm = cf_generator_config["name"]

        if cf_gen_nm not in supported_cf_gens:
            raise ValueError(
                f"Unsupported Counterfactual Generator definition. Supported generators are {supported_cf_gens}"
            )

        model = joblib.load(knn_path)
        text_vectorizer = TextVectorizer(vectorizer_path)

        if cf_gen_nm == "t5-cf-generator":
            assert (
                type(cf_generator_config) == str
            ), "'cf_generator_config' must be a path object"
            cf_generator = T5Generator(cf_generator_config, cf_generator_root)
        else:
            cf_generator = WordFlippingGenerator(cf_generator_config)

        self._model: KNeighborsClassifier = model
        self._text_vectorizer = text_vectorizer
        self._cf_generator = cf_generator
        self._report_data = {}

    def _add_labels(self, inp: Union[Dict[int, Any], np.ndarray]) -> Dict[str, Any]:
        assert len(inp) == len(self._data_labels)

        if type(inp) == dict:
            out = {self._data_labels[k]: v for (k, v) in inp.items()}
        elif type(inp) == np.ndarray:
            out = {lbl: v for (lbl, v) in zip(self._data_labels, inp)}
        else:
            raise ValueError("Unsupported data type")

        return out

    def explanation(self) -> str:
        rd = self._report_data
        if type(rd["output"]) == str:
            report = f"""
======== Analysis Report ========

Input text                  : {rd["input"]["text"]}
Input neighbor counts       : {self._add_labels(rd['input']['nb_cnts'])}
Input class probabilities   : {self._add_labels(rd['input']['probs'])}
Input class densities       : {self._add_labels(rd['input']['densi'])}
Input review class          : {self._data_labels[rd['input']['review_cls']]}

Contadictions               : {rd["output"]}"""
        else:
            contra_txts = rd["output"]["text"]
            contra_nb_cnts = [
                str(self._add_labels(cnt)) for cnt in rd["output"]["nb_cnts"]
            ]
            contra_probs = [str(self._add_labels(cnt)) for cnt in rd["output"]["probs"]]
            contra_densts = [
                str(self._add_labels(cnt)) for cnt in rd["output"]["densi"]
            ]
            tabbed_newline = "\n\t"
            report = f"""
======== Analysis Report ========

Input text                  : {rd["input"]["text"]}
Input neighbor counts       : {self._add_labels(rd['input']['nb_cnts'])}
Input class probabilities   : {self._add_labels(rd['input']['probs'])}
Input class densities       : {self._add_labels(rd['input']['densi'])}
Input review class          : {self._data_labels[rd['input']['review_cls']]}

Contradictory texts: {tabbed_newline+tabbed_newline.join(contra_txts)}
Contradictory neighbor counts: {tabbed_newline+tabbed_newline.join(contra_nb_cnts)}
Contradictory class probabilities: {tabbed_newline+tabbed_newline.join(contra_probs)}
Contradictory class densities: {tabbed_newline+tabbed_newline.join(contra_densts)}
Closest counterfactual ID: {rd['output']['matching_cf_id']}
"""

        return report

    def _get_neighbour_stat(self, vects: List[np.ndarray]) -> List[Dict[int, int]]:
        distances, indices = self._model.kneighbors(vects)
        k_nearest_labels = np.array(
            [self._model.classes_[self._model._y[idx]] for idx in indices]
        )
        counts = []
        densts = []
        for i in range(vects.shape[0]):
            count = {}
            densi = {}
            for c in self._model.classes_:
                count[c] = (k_nearest_labels[i] == c).sum()
                lbl_dists = distances[i][k_nearest_labels[i] == c]
                densi[c] = len(lbl_dists) ** 2 / lbl_dists.sum()
            counts.append(count)
            densts.append(densi)
        return counts, densts

    def _get_probs(self, counts: Tuple[Dict[int, int]]) -> np.ndarray:
        probs = np.array([list(count.values()) for count in counts])
        probs = probs / self._model.n_neighbors
        return probs

    def _get_farthest_id(self, densts: List[np.ndarray], rv_cls: int) -> int:
        cf_cls = 1 - rv_cls
        (review_densi, *contra_densts) = densts
        review_densi = review_densi[rv_cls]
        contra_densts = np.array([densi[cf_cls] for densi in contra_densts])
        diff = np.abs(contra_densts - review_densi)
        fthst_id = diff.argmin()
        return fthst_id

    def __call__(self, text: str, search_space: int) -> str:
        self._report_data["input"] = {"text": text, "search_space": search_space}

        # 1. generate contradictions
        contradictions = self._cf_generator(text, search_space)

        # 2. project to the vectorizer's vector space (X)
        vect = self._text_vectorizer([text, *contradictions])  # (1+search_space,n)

        # 3. get neighbour statistics
        counts, densts = self._get_neighbour_stat(vect)
        (review_count, *contra_counts) = counts

        # 4. get probabilities
        probs = self._get_probs(counts)
        (review_prob, *contra_probs) = probs
        review_cls = review_prob.argmax()

        if len(densts) > 1:
            # 5. get farthest opposite
            fthst_id = self._get_farthest_id(densts, review_cls)

            # 6. select counter factual
            cf = contradictions[fthst_id]

            # 7. reporting
            (review_densi, *contra_densts) = densts
            input = {
                "text": text,
                "nb_cnts": review_count,
                "probs": review_prob,
                "review_cls": review_cls,
                "densi": review_densi,
            }
            output = {
                "text": contradictions,
                "nb_cnts": contra_counts,
                "probs": contra_probs,
                "densi": contra_densts,
                "matching_cf_id": fthst_id,
            }
            self._report_data["input"] = input
            self._report_data["output"] = output

            return cf
        else:
            # no contradictions were available for the given configuration
            review_densi = densts[0]
            input = {
                "text": text,
                "nb_cnts": review_count,
                "probs": review_prob,
                "review_cls": review_cls,
                "densi": review_densi,
            }
            self._report_data["input"] = input
            self._report_data[
                "output"
            ] = "No contradictions possible for the given test case configuration"

    def set_config(self, config) -> None:
        self._cf_generator.set_config(config["generator_config"])
