import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from ordered_set import OrderedSet
import joblib
from itertools import compress
from ..processors import TextVectorizer
from .base import BaseAnalyzer
import json
from typing import Dict, Any
import re
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.sparse import lil_matrix
from ordered_set import OrderedSet
from itertools import compress


class SEDCAnalyzer(BaseAnalyzer):
    """Class for generating evidence counterfactuals for classifiers on behavioral/text data"""

    def __init__(
        self,
        model_path,
        vectorizer_path,
        threshold_classifier,
        max_iter=100,
        max_explained=1,
        BB=True,
        max_features=30,
        time_maximum=120,
    ):
        """Init function

        Args:
            self._classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:

                def self._classifier_fn(X):
                    c=classification_model.predict_proba(X)
                    y_predicted_proba=c[:,1]
                    return y_predicted_proba

            threshold_classifier: [float] the threshold that is used for classifying
            instances as positive or not. When score or probability exceeds the
            threshold value, then the instance is predicted as positive.
            We have no default value, because it is important the user decides
            a good value for the threshold.

            feature_names: [numpy.array] contains the interpretable feature names,
            such as the words themselves in case of document classification or the names
            of visited URLs.

            max_iter: [int] maximum number of iterations in the search procedure.
            Default is set to 50.

            max_explained: [int] maximum number of EDC explanations generated.
            Default is set to 1.

            BB: [“True” or “False”]  when the algorithm is augmented with
            branch-and-bound (BB=True), one is only interested in the (set of)
            shortest explanation(s). Default is "True".

            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.

            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """

        self.threshold_classifier = np.float64(threshold_classifier)
        self.max_iter = max_iter
        self.max_explained = max_explained
        self.BB = BB
        self.max_features = max_features
        self.time_maximum = time_maximum
        self.revert = None
        self.initial_class = None

        input_encoder = joblib.load(vectorizer_path)
        feature_names = input_encoder.get_feature_names_out()
        self.feature_names = feature_names
        loaded_vocab = input_encoder.vocabulary_
        self.loaded_vocab = loaded_vocab

        model = joblib.load(model_path)
        self._model = model
        text_vectorizer = TextVectorizer(vectorizer_path)
        self._text_vectorizer = text_vectorizer

        self._report_data = {}
        self.instances = []
        self.scores = []

    def _plot_graph(self, data, x_label="X", y_label="Y", title="Graph"):
        x = range(len(data))
        plt.plot(x, data, marker="o", linestyle="-")

        # Add labels to the axes
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Add a title to the graph
        plt.title(title)

        # Show the plot
        plt.show()

    def get_instances(self):
        return self.instances

    def get_scores(self):
        return self.scores

    def _perturb_fn(self, x, inst, print_flag=0):
        """Function to perturb instance x -> Deform the array -> assign 0 to the x-th column"""
        """
        Returns perturbed instance inst
        """
        inst[:, x] = 0
        return inst

    def   _classifier_fn(self, x, negative_to_positive=0):
        """Returns the prediction probability of class 1 -> Not class 0"""
        prediction = self._model.predict_proba(x)
        # If prediction is [1] retrurn the probability of class 1 else return probability of class 0
        if negative_to_positive == 1:
            return prediction[:, 0]
        return prediction[:, 1]

    def _expand_and_prune(
        self,
        comb,
        expanded_combis,
        feature_set,
        candidates_to_expand,
        explanations_sets,
        scores_candidates_to_expand,
        instance,
        cf,
    ):
        """Function to expand "best-first" feature combination and prune explanation_candidates and candidates_to_expand"""

        comb = OrderedSet(comb)
        expanded_combis.append(comb)

        old_candidates_to_expand = [frozenset(x) for x in candidates_to_expand]
        old_candidates_to_expand = set(old_candidates_to_expand)

        feature_set_new = []
        for feature in feature_set:
            if len(comb & feature) == 0:  # set operation: intersection
                feature_set_new.append(feature)

        new_explanation_candidates = []
        for element in feature_set_new:
            union = comb | element  # set operation: union
            new_explanation_candidates.append(union)

        # Add new explanation candidates to the list of candidates to expand
        candidates_to_expand_notpruned = (
            candidates_to_expand.copy()
        )  # voeg de nieuwe combinaties toe aan combinations_to_expand
        for new_candidate in new_explanation_candidates:
            candidates_to_expand_notpruned.append(new_candidate)

        # Calculate scores of new combinations and add to scores_candidates_to_expand
        perturbed_instances = [
            self._perturb_fn(x, inst=instance.copy()) for x in new_explanation_candidates
        ]
        scores_perturbed_new = [cf(x) for x in perturbed_instances]
        scores_candidates_to_expand_notpruned = (
            scores_candidates_to_expand + scores_perturbed_new
        )
        dictionary_scores = dict(
            zip(
                [str(x) for x in candidates_to_expand_notpruned],
                scores_candidates_to_expand_notpruned,
            )
        )

        # *** Pruning step: remove all candidates to expand that have an explanation as subset ***
        candidates_to_expand_pruned_explanations = []
        for combi in candidates_to_expand_notpruned:
            pruning = 0
            for explanation in explanations_sets:
                if (explanation.issubset(combi)) or (explanation == combi):
                    pruning = pruning + 1
            if pruning == 0:
                candidates_to_expand_pruned_explanations.append(combi)

        candidates_to_expand_pruned_explanations_frozen = [
            frozenset(x) for x in candidates_to_expand_pruned_explanations
        ]
        candidates_to_expand_pruned_explanations_ = set(
            candidates_to_expand_pruned_explanations_frozen
        )

        expanded_combis_frozen = [frozenset(x) for x in expanded_combis]
        expanded_combis_ = set(expanded_combis_frozen)

        # *** Pruning step: remove all candidates to expand that are in expanded_combis ***
        candidates_to_expand_pruned = (
            candidates_to_expand_pruned_explanations_ - expanded_combis_
        )
        ind_dict = dict(
            (k, i)
            for i, k in enumerate(candidates_to_expand_pruned_explanations_frozen)
        )
        indices = [ind_dict[x] for x in candidates_to_expand_pruned]
        candidates_to_expand = [
            candidates_to_expand_pruned_explanations[i] for i in indices
        ]

        # The new explanation candidates are the ones that are NOT in the old list of candidates to expand
        new_explanation_candidates_pruned = (
            candidates_to_expand_pruned - old_candidates_to_expand
        )
        candidates_to_expand_frozen = [frozenset(x) for x in candidates_to_expand]
        ind_dict2 = dict((k, i) for i, k in enumerate(candidates_to_expand_frozen))
        indices2 = [ind_dict2[x] for x in new_explanation_candidates_pruned]
        explanation_candidates = [candidates_to_expand[i] for i in indices2]

        scores_candidates_to_expand = [
            dictionary_scores[x] for x in [str(c) for c in candidates_to_expand]
        ]
        scores_explanation_candidates = [
            dictionary_scores[x] for x in [str(c) for c in explanation_candidates]
        ]

        return (
            explanation_candidates,
            candidates_to_expand,
            expanded_combis,
            scores_candidates_to_expand,
            scores_explanation_candidates,
        )

    def __call__(self, instance, someNumber):
        """Generates evidence counterfactual explanation for the instance.
        ONLY IF THE CURRENT INSTANCE IS POSITIVE -> Limitation

        Args:
            instance: [numpy.array or sparse matrix] instance to explain

        Returns:
            None
        """

        # *** INITIALIZATION ***
        print("Start initialization...")
        tic = time.time()
        instance = lil_matrix(instance)
        iteration = 0
        nb_explanations = 0
        minimum_size_explanation = np.nan
        explanations = []
        explanations_sets = []
        explanations_score_change = []
        expanded_combis = []
        score_predicted = self._classifier_fn(instance)
        self.scores.append(score_predicted[0])
        indices_active_elements = np.nonzero(instance)[1]
        number_active_elements = len(indices_active_elements)
        indices_active_elements = indices_active_elements.reshape(
            (number_active_elements, 1)
        )

        candidates_to_expand = []
        for features in indices_active_elements:
            candidates_to_expand.append(OrderedSet(features))

        explanation_candidates = candidates_to_expand.copy()

        feature_set = [frozenset(x) for x in indices_active_elements]

        print("Initialization is complete.")
        print("\n Elapsed time %d \n" % (time.time() - tic))

        # *** WHILE LOOP ***
        while (
            (iteration < self.max_iter)
            and (nb_explanations < self.max_explained)
            and (len(candidates_to_expand) != 0)
            and (len(explanation_candidates) != 0)
            and ((time.time() - tic) < self.time_maximum)
        ):
            iteration += 1
            print("\n Iteration %d \n" % iteration)

            if iteration == 1:
                perturbed_instances = [
                    self._perturb_fn(x, inst=instance.copy()) for x in explanation_candidates
                ]
                scores_explanation_candidates = [
                    self._classifier_fn(x) for x in perturbed_instances
                ]
                scores_candidates_to_expand = scores_explanation_candidates.copy()

            scores_perturbed_new_combinations = [
                x[0] for x in scores_explanation_candidates
            ]

            # ***CHECK IF THERE ARE EXPLANATIONS***
            explanations += list(
                compress(
                    explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            nb_explanations += len(
                list(
                    compress(
                        explanation_candidates,
                        scores_perturbed_new_combinations < self.threshold_classifier,
                    )
                )
            )
            explanations_sets += list(
                compress(
                    explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            explanations_sets = [set(x) for x in explanations_sets]
            explanations_score_change += list(
                compress(
                    scores_explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )

            # Adjust max_length
            if self.BB == True:
                if len(explanations) != 0:
                    lengths = []
                    for explanation in explanations:
                        lengths.append(len(explanation))
                    lengths = np.array(lengths)
                    max_length = lengths.min()
                else:
                    max_length = number_active_elements
            else:
                max_length = number_active_elements

            # Eliminate combinations from candidates_to_expand ("best-first" candidates) that can not be expanded
            # Pruning based on Branch & Bound=True, max. features allowed and number of active features
            candidates_to_expand_updated = []
            scores_candidates_to_expand_updated = []
            for j, combination in enumerate(candidates_to_expand):
                if (
                    (len(combination) < number_active_elements)
                    and (len(combination) < max_length)
                    and (len(combination) < self.max_features)
                ):
                    candidates_to_expand_updated.append(combination)
                    scores_candidates_to_expand_updated.append(
                        scores_candidates_to_expand[j]
                    )

            # *** IF LOOP ***
            if (len(candidates_to_expand_updated) == 0) or (
                nb_explanations >= self.max_explained
            ):
                print("Stop iterations...")
                explanation_candidates = []  # stop algorithm

            elif len(candidates_to_expand_updated) != 0:
                explanation_candidates = []
                it = 0
                indices = []

                scores_candidates_to_expand2 = []
                for score in scores_candidates_to_expand_updated:
                    if score[0] < self.threshold_classifier:
                        scores_candidates_to_expand2.append(2 * score_predicted)
                    else:
                        scores_candidates_to_expand2.append(score)

                # *** WHILE LOOP ***
                while (
                    (len(explanation_candidates) == 0)
                    and (it < len(scores_candidates_to_expand2))
                    and ((time.time() - tic) < self.time_maximum)
                ):
                    print("While loop iteration %d" % it)

                    if it != 0:
                        for index in indices:
                            scores_candidates_to_expand2[index] = 2 * score_predicted

                    index_combi_max = np.argmax(
                        score_predicted - scores_candidates_to_expand2
                    )
                    indices.append(index_combi_max)
                    expanded_combis.append(
                        candidates_to_expand_updated[index_combi_max]
                    )
                    self.scores.append(scores_candidates_to_expand2[index_combi_max][0])
                    comb_to_expand = candidates_to_expand_updated[index_combi_max]
                    func = self._expand_and_prune(
                        comb_to_expand,
                        expanded_combis,
                        feature_set,
                        candidates_to_expand_updated,
                        explanations_sets,
                        scores_candidates_to_expand_updated,
                        instance,
                        self._classifier_fn,
                    )
                    explanation_candidates = func[0]
                    candidates_to_expand = func[1]
                    expanded_combis = func[2]
                    scores_candidates_to_expand = func[3]
                    scores_explanation_candidates = func[4]

                    it += 1

            print("\n Elapsed time %d \n" % (time.time() - tic))

        # *** FINAL PART OF ALGORITHM ***
        print("Iterations are done.")

        explanation_set = []
        explanation_feature_names = []
        for i in range(len(explanations)):
            explanation_feature_names = []
            for features in explanations[i]:
                explanation_feature_names.append(self.feature_names[features])
            explanation_set.append(explanation_feature_names)

        if len(explanations) != 0:
            lengths_explanation = []
            for explanation in explanations:
                l = len(explanation)
                lengths_explanation.append(l)
            minimum_size_explanation = np.min(lengths_explanation)

        number_explanations = len(explanations)
        if np.size(explanations_score_change) > 1:
            inds = np.argsort(explanations_score_change, axis=0)
            inds = np.fliplr([inds])[0]
            inds_2 = []
            for i in range(np.size(inds)):
                inds_2.append(inds[i][0])
            explanation_set_adjusted = []
            for i in range(np.size(inds)):
                j = inds_2[i]
                explanation_set_adjusted.append(explanation_set[j])
            explanations_score_change_adjusted = []
            for i in range(np.size(inds)):
                j = inds_2[i]
                explanations_score_change_adjusted.append(explanations_score_change[j])
            explanation_set = explanation_set_adjusted
            explanations_score_change = explanations_score_change_adjusted

        time_elapsed = time.time() - tic
        print("\n Total elapsed time %d \n" % time_elapsed)
        self.scores.append(explanations_score_change[0 : self.max_explained][0])

        return {
            "explanation set": explanation_set[0 : self.max_explained],
            "number active elements": number_active_elements,
            "number explanations found": number_explanations,
            "size smallest explanation": minimum_size_explanation,
            "time elapsed": time_elapsed,
            "differences score": explanations_score_change[0 : self.max_explained],
            "iterations": iteration,
        }

    def explanation(self) -> str:
        return json.dumps(self._report_data, indent=4)

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Config must contain the following keys:
        {
            "threshold_classifier": float,
            "max_iter": int,
            "time_maximum": int,
        }
        """

        self.threshold_classifier = np.float64(config["threshold_classifier"])
        self.max_iter = config["max_iter"]
        self.time_maximum = config["time_maximum"]
