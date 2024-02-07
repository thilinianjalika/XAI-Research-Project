from sklearn.svm import SVC
import numpy as np
import os
from ..datasets import IMDBDataset
from ..processors import TextVectorizer
from ..cf_generators import T5Generator, WordFlippingGenerator
import json
from typing import Union, List, Dict
from scipy.sparse import csr_matrix, vstack
import yaml
from .base import BaseAnalyzer
import joblib


class SVMDistanceAnalyzer(BaseAnalyzer):
    valid_sets = ["train", "val", "test"]

    def __init__(
        self,
        model: SVC,
        dataset: IMDBDataset,
        buffer_path: str,
        active_set: str = "test",
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.preprocessor = dataset.preprocessor
        self.input_encoder = dataset.input_encoder
        self.buffer_path = buffer_path
        self.active_set = active_set
        self._load_ds_distances()

    def _load_ds_distances(self) -> None:
        active_set = self.active_set
        assert active_set in self.valid_sets
        distances = []
        buffer = {}
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path) as handler:
                buffer = json.load(handler)
            if "svm_distances" in buffer.keys():
                if active_set in buffer["svm_distances"].keys():
                    distances = buffer["svm_distances"][active_set]
        if len(distances) == 0:
            x = getattr(self.dataset, f"x_{active_set}")
            distances = list(self.model.decision_function(x))
            if "svm_distances" in buffer.keys():
                buffer["svm_distances"][active_set] = distances
            else:
                buffer["svm_distances"] = {active_set: distances}
            with open(self.buffer_path, "w") as handler:
                json.dump(buffer, handler, indent=4)

        self.ds_distances = np.array(buffer["svm_distances"][active_set])

    def _get_distances(self, sentences: List[str]) -> np.ndarray:
        processed_texts = [self.preprocessor(sentence) for sentence in sentences]
        input_vectors = self.input_encoder.transform(processed_texts)
        distances = self.model.decision_function(input_vectors)
        return distances

    def _get_indices_from_distances(self, distances: np.ndarray) -> np.ndarray:
        n = distances.size
        ds_distances = np.vstack([self.ds_distances] * n)
        distances = distances.reshape(n, 1)
        diff = (ds_distances - distances).__abs__()
        indices = np.argmin(diff, axis=1)
        return indices

    def get_counterfactual_examples(
        self, sentences: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        if type(sentences) == str:
            sentences = [sentences]
        distances = self._get_distances(sentences)
        inverse_distances_query = -1 * distances
        indices = self._get_indices_from_distances(inverse_distances_query)
        inverse_sentences = self.dataset.get_reviews(self.active_set, indices)
        inverse_sentences = [rev["original"] for rev in inverse_sentences]
        inverse_distances = self._get_distances(inverse_sentences)
        analysis = [
            {
                "original": {"sentence": sentence, "distance": distance},
                "opposite": {
                    "sentence": inverse_sentence,
                    "distance": inverse_distance,
                },
            }
            for (sentence, inverse_sentence, distance, inverse_distance) in zip(
                sentences, inverse_sentences, distances, inverse_distances
            )
        ]
        return analysis


class SVMMirrorAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        svm_path: str,
        vectorizer_path: str,
        cf_generator_config: Union[str, dict] = None,
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

        model = joblib.load(svm_path)
        text_vectorizer = TextVectorizer(vectorizer_path)

        if cf_gen_nm == "t5-cf-generator":
            assert (
                type(cf_generator_config) == str
            ), "'cf_generator_config' must be a path object"
            cf_generator = T5Generator(cf_generator_config, cf_generator_root)
        else:
            cf_generator = WordFlippingGenerator(cf_generator_config)

        self._model = model
        self._text_vectorizer = text_vectorizer
        self._cf_generator = cf_generator
        self._report_data = {}

    def _get_plane(self) -> List[float]:
        svm = self._model
        kernel = svm.kernel
        if kernel == "rbf":
            w = svm._dual_coef_.toarray()[0]
            b = svm.intercept_[0]
            plane = [*w, b]
        else:
            raise NotImplementedError(
                "Plane extraction is only implemented for 'rbf' kernel"
            )
        return plane

    def _project_to_kernel_space(self, vect: csr_matrix) -> np.ndarray:
        """
        Applies the RBF kernel function
        Args:
            1. vect: Input vectors with shape (n, tfidf_n); where n is the number of vectors, tfidf_n is the size of a tfidf vector
            2: svm
        """
        svm = self._model
        kernel = svm.kernel
        if kernel == "rbf":
            gamma = svm._gamma  # 1/(2*(sigma**2))
            sv = svm.support_vectors_  # shape: (m, tfidf_n)
            vect = vect.toarray()

            k_arr = []
            for i in range(vect.shape[0]):
                v = vect[i]
                diff = np.array(sv - v)
                k = np.exp(-gamma * (diff**2).sum(axis=1))
                k_arr.append(k)
            k_arr = np.array(k_arr)
        else:
            raise NotImplementedError(
                "Vector projection is only implemented for 'rbf' kernel"
            )

        return k_arr

    def _project_to_kernel_space_2(self, vect: csr_matrix) -> csr_matrix:
        """
        Applies the RBF kernel function
        Args:
            1. vect: Input vectors with shape (n, tfidf_n); where n is the number of vectors, tfidf_n is the size of a tfidf vector
            2: svm
        """
        svm = self._model
        kernel = svm.kernel
        if kernel == "rbf":
            gamma = svm._gamma  # 1/(2*(sigma**2))
            sv = svm.support_vectors_  # shape: (m, tfidf_n)

            k_arr = []
            for i in range(vect.shape[0]):
                v = vect[i]
                diff = sv - vstack([v] * sv.shape[0])
                k = csr_matrix(np.exp(-gamma * diff.multiply(diff).sum(axis=1).T))
                k_arr.append(k)
            k_arr = vstack(k_arr)
        else:
            raise NotImplementedError(
                "Vector projection is only implemented for 'rbf' kernel"
            )

        return k_arr

    def _get_mirror_point(self, query_point: np.ndarray) -> List[float]:
        plane = self._get_plane()
        *w, b = plane
        w = np.array(w)
        t_0 = -(b + w.dot(query_point)) / (np.linalg.norm(w) ** 2)
        mp = query_point + 2 * t_0 * w
        return mp

    def _get_dist(self, vector1, vector2, method="cos-sim"):
        if method == "cos-sim":
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm1 * norm2)
        else:
            raise NotImplementedError(
                "Similarity checking only implemented for cosine similarity ('cos-sim')"
            )
        return similarity

    def explanation(self) -> str:
        if type(self._report_data["output"]) == str:
            report = self._report_data["output"]
        else:
            tabbed_newline = "\n\t"
            report = f"""
======== Analysis Report ========

Input text                      : {self._report_data["input"]["text"]}

Generated contradictory texts   : {tabbed_newline+tabbed_newline.join(self._report_data["output"]["generated_text"])}

Distances to the mirror point   : {tabbed_newline+tabbed_newline.join([str(dist) for dist in self._report_data["output"]["distances"]])}

Closest contradictory text      : {self._report_data["output"]["generated_text"][self._report_data["output"]["closest_id"]]}
"""
        return report

    def __call__(self, text: str, search_space: int) -> str:
        self._report_data["input"] = {"text": text, "search_space": search_space}

        # 1. generate contradictions
        contradictions = self._cf_generator(text, search_space)

        # 2. project to the vectorizer's vector space (X)
        vect = self._text_vectorizer([text, *contradictions])  # (1+search_space,n)

        # 3. project to the svm kernel's vector space (K)
        vect_ks = self._project_to_kernel_space(vect)
        (review_vect_ks, *contra_vect_ks) = vect_ks

        # 4. find the mirror point of 'review' on the kernel space (C)
        review_vect_ks_mp = self._get_mirror_point(review_vect_ks)

        if len(contra_vect_ks) > 0:
            # 5. find the closest point to C out of the contradictory points
            dists = []
            for cv in contra_vect_ks:
                dist = self._get_dist(cv, review_vect_ks_mp)
                dists.append(dist)
            i = np.argmin(dists)
            clst_txt = contradictions[i]

            # reporting
            self._report_data["output"] = {"generated_text": contradictions}
            self._report_data["output"]["distances"] = dists
            self._report_data["output"]["closest_id"] = i

            return clst_txt
        else:
            # no contradictions were available for the given configuration
            self._report_data[
                "output"
            ] = "No contradictions possible for the given test case configuration"

    def set_config(self, config) -> None:
        self._cf_generator.set_config(config["generator_config"])
