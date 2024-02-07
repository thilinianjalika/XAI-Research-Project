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


class RFAnalyzer(BaseAnalyzer):
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
            classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:

                def classifier_fn(X):
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

    def _print_ref_instance(self, ref_inst):
        printable_array = []
        indices_active_elements = np.nonzero(ref_inst)[1]
        for item in indices_active_elements:
            printable_array.append(".." + self.feature_names[item] + "..")
        print(printable_array)

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

    def _replace_fn(self, x, y, inst, print_flag=0):
        """Function to perturb instance x -> Deform the array -> assign 0 to the x-th column"""
        """
        Returns perturbed instance inst
        """
        new_inst = inst.copy()
        try:
            temp_x = inst[:, x]
            temp_y = inst[:, y]
            new_inst[:, x] = temp_y
            new_inst[:, y] = temp_x
        except:
            new_inst[:, x] = 0
        return new_inst

    def _classifier_fn(self, x, negative_to_positive=0):
        """Returns the prediction probability of class 1 -> Not class 0"""
        prediction = self._model.predict_proba(x)
        # If prediction is [1] retrurn the probability of class 1 else return probability of class 0
        if negative_to_positive == 1:
            return prediction[:, 0]
        return prediction[:, 1]

    def _print_instance(self, pert_inst, ref_inst):
        """Function to print the perturbed instance"""
        """
        Returns perturbed instance inst
        """
        feature_names = self.feature_names

        indices_active_elements_ref = np.nonzero(ref_inst)[1]
        indices_active_elements_pert = np.nonzero(pert_inst)[1]
        ref_set = set(indices_active_elements_ref)
        pert_set = set(indices_active_elements_pert)
        # elements in ref_set but not in pert_set
        removed_word_indices = ref_set - pert_set
        # elements in pert_set but not in ref_set
        added_word_indices = pert_set - ref_set
        printable_array = []
        for item in indices_active_elements_ref:
            printable_array.append(".." + feature_names[item] + "..")
        # Change formatting of removed words
        for item in removed_word_indices:
            printable_array[
                printable_array.index(".." + feature_names[item] + "..")
            ] = ("--" + feature_names[item] + "--")
        # change formatting of added words
        for item in added_word_indices:
            printable_array.append("++" + feature_names[item] + "++")
        printable_array.append("  --> class 1 Score = ")
        printable_array.append(self._classifier_fn(pert_inst)[0])
        print(printable_array)
        return printable_array

    def _conditional_replace_fn(self, x, y, inst, print_flag=0):
        for i in range(len(x)):
            if isinstance(y[i], str):
                inst[:, x[i]] = 0
            else:
                temp_x = inst[:, x[i]]
                temp_y = inst[:, y[i]]
                inst[:, x[i]] = temp_y
                inst[:, y[i]] = temp_x
        return inst

    def _get_featues_importances(self, instance):
        """Get feature importances with the sign of the change in prediction probability for a given instance.
        Uses the gini impurity in the RF model.
        Fast calculation as values are calculated during training period.
        reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_

        Args:
            antonyms_indices: indices of antonyms in the feature vector
            model: trained model with feature_importances_

        Returns:
            tuple of features and their indices in the feature vector
        """
        feature_importance = self._model.feature_importances_
        initial_score = self._model.predict_proba(instance)[0][1]
        print("initial_score ", initial_score)
        indices_active_elements = np.array(np.nonzero(instance)[1]).reshape(
            len(np.nonzero(instance)[1]), 1
        )
        feature_set = [frozenset(x) for x in indices_active_elements]
        candidates_to_expand = []
        for features in indices_active_elements:
            candidates_to_expand.append(OrderedSet(features))
        explanation_candidates = candidates_to_expand.copy()
        perturbed_instances = [
            self._perturb_fn(x, inst=instance.copy()) for x in explanation_candidates
        ]
        scores_explanation_candidates = [
            self._classifier_fn(x) for x in perturbed_instances
        ]
        sign_change = [
            1 if (initial_score - x[0]) > 0 else -1
            for x in scores_explanation_candidates
        ]

        current_index = 0
        sign_changed_importances = []
        for element in indices_active_elements:
            sign_changed_importances.append(
                {
                    str(self.feature_names[element[0]]): sign_change[current_index]
                    * feature_importance[element[0]]
                }
            )
            print(
                element[0],
                " ",
                feature_importance[element[0]],
                sign_change[current_index],
                scores_explanation_candidates[current_index],
            )
            feature_importance[element[0]] = (
                sign_change[current_index] * feature_importance[element[0]]
            )
            print(element[0], " ", feature_importance[element[0]])
            current_index += 1
        print("sign_changed_importances ", sign_changed_importances)
        self._report_data["feature_importances"] = sign_changed_importances
        # if sign change is 0, feature_importance value set to -value
        # feature_importance = [x if x > 0 else -x for x in feature_importance]
        return feature_importance

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
        feature_names,
        revert=0,
    ):
        """Function to expand "best-first" feature combination and prune explanation_candidates and candidates_to_expand"""

        comb = OrderedSet(comb)
        expanded_combis.append(comb)

        print("\n\n")

        old_candidates_to_expand = [frozenset(x) for x in candidates_to_expand]
        old_candidates_to_expand = set(old_candidates_to_expand)

        feature_set_new = []
        ## If the feature is not in the current combination -> add it to a new list
        for feature in feature_set:
            if len(comb & feature) == 0:  # set operation: intersection
                feature_set_new.append(
                    feature
                )  # If the feature is not in the current combination to remove from the instance

        # Add each element in the new set -> which were initially not present -> to the accepted combination -> create new combinations -> (EXPANSION)
        new_explanation_candidates = []
        for element in feature_set_new:
            union = comb | element  # set operation: union
            new_explanation_candidates.append(
                union
            )  # Create new combinations to remove from the instance

        # Add new explanation candidates to the list of candidates to expand
        candidates_to_expand_notpruned = candidates_to_expand.copy()
        for new_candidate in new_explanation_candidates:
            candidates_to_expand_notpruned.append(new_candidate)

        # Calculate scores of new combinations and add to scores_candidates_to_expand
        # perturb each new candidate and get the score for each.
        perturbed_instances = [
            self._perturb_fn(x, inst=instance.copy())
            for x in new_explanation_candidates
        ]
        for instance_p in perturbed_instances:
            self._print_instance(instance_p, instance)
        scores_perturbed_new = [cf(x, revert) for x in perturbed_instances]
        ## Append the newly created score array to the passes existing array
        scores_candidates_to_expand_notpruned = (
            scores_candidates_to_expand + scores_perturbed_new
        )
        # create a dictionary of scores dictionary where the
        # keys are string representations of the candidates from candidates_to_expand_notpruned, and the
        # values are the corresponding scores from scores_candidates_to_expand_notpruned
        dictionary_scores = dict(
            zip(
                [str(x) for x in candidates_to_expand_notpruned],
                scores_candidates_to_expand_notpruned,
            )
        )

        # *** Pruning step: remove all candidates to expand that have an explanation as subset ***
        candidates_to_expand_pruned_explanations = []
        # take one combination from candidates
        for combi in candidates_to_expand_notpruned:
            pruning = 0
            for (
                explanation
            ) in (
                explanations_sets
            ):  # if an explanation is present as a subser in combi, does not add it to the to be expanded list -> because solution with a smaller size exists
                if (explanation.issubset(combi)) or (explanation == combi):
                    pruning = pruning + 1
            if (
                pruning == 0
            ):  # If it is not a superset of a present explanation -> add it to the list
                candidates_to_expand_pruned_explanations.append(combi)
        # Each element is frozen as a set
        candidates_to_expand_pruned_explanations_frozen = [
            frozenset(x) for x in candidates_to_expand_pruned_explanations
        ]
        # But the total set f frozen sets are not frozen
        candidates_to_expand_pruned_explanations_ = set(
            candidates_to_expand_pruned_explanations_frozen
        )

        expanded_combis_frozen = [frozenset(x) for x in expanded_combis]
        expanded_combis_ = set(expanded_combis_frozen)

        # *** Pruning step: remove all candidates to expand that are in expanded_combis *** -> Same as above
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

        # Get scores of the new candidates and explanations.
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

    def __call__(self, text, someNumber):
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

        input_string = "Given a vector"
        if isinstance(text, str):
            input_string = str(text)
            text = self._text_vectorizer(text)
        text = lil_matrix(text)

        print("initial sentence is ... ")
        print(text.get_shape())
        self._print_ref_instance(text)

        iteration = 0
        nb_explanations = 0
        minimum_size_explanation = np.nan
        explanations = []
        explanations_sets = []
        explanations_score_change = []
        expanded_combis = []

        score_predicted = self._classifier_fn(text)  ## Returns Prediction Prob
        # Intial class is 1 is score is greater than threshold
        if score_predicted > self.threshold_classifier:
            self.initial_class = [1]
        else:
            self.initial_class = [0]
            self.revert = 1
        print(
            "score_predicted  ",
            score_predicted,
            "  initial_class  ",
            self.initial_class,
        )
        self.scores.append(score_predicted[0])

        self._report_data["input"] = {
            "text": input_string,
            "score for positive": score_predicted[0],
            "initial class": self.initial_class[0],
        }

        importances = self._get_featues_importances(text)
        features = []
        for ind in range(len(importances)):
            if importances[ind] != 0:
                features.append(
                    {
                        "feature": ind,
                        "word": self.feature_names[ind],
                        "importance": importances[ind],
                    }
                )
        sorted_data_in = sorted(features, key=lambda x: x["importance"], reverse=True)
        inverse_sorted_data_in = sorted(features, key=lambda x: x["importance"])

        print("sorted_data_in ", sorted_data_in)
        if self.revert == 1:
            sorted_data_in = inverse_sorted_data_in

        indices_active_elements = np.nonzero(text)[
            1
        ]  ## -> Gets non zero elements in the instance as an array [x, y, z]
        sorted_indices = sorted(
            indices_active_elements, key=lambda x: importances[x], reverse=True
        )
        indices_active_elements = np.array(sorted_indices)
        number_active_elements = len(indices_active_elements)
        indices_active_elements = indices_active_elements.reshape(
            (number_active_elements, 1)
        )  ## -> Reshape to get a predictable

        candidates_to_expand = (
            []
        )  # -> These combinations are further expanded -> These are the elements to be removed from the sentence
        for features in indices_active_elements:
            candidates_to_expand.append(OrderedSet(features))
        print("candidates_to_expand ", candidates_to_expand)
        ## > Gets an array with each element in reshaped incides as an ordered set -> [OrderedSet([430]), OrderedSet([588]), OrderedSet([595])]

        explanation_candidates = candidates_to_expand.copy()
        print("explanation_candidates ", explanation_candidates)
        ## Gets a copy of the above array -> Initially

        feature_set = [
            frozenset(x) for x in indices_active_elements
        ]  ## Immutable -> can be used as keys in dictionary
        ## Used features in the current x-reference -> incides of the words in the review.

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
            ## Stop if maximum iterations exceeded
            #  number of explanations generated is greater than the maximum explanations
            #  There are no candidates to expand
            #  There are no explanation candidates -> Used to force stop while loop below
            #  Or maximum allowed time exceeded
            iteration += 1
            print("\n Iteration %d \n" % iteration)

            if iteration == 1:
                print("Run in first iteration -> perturbation done \n")
                # Print the word in each index in the explanation candidates
                # for item in explanation_candidates:
                #     print([self.feature_names[x] for x in item])
                print("explanation_candidates \n", explanation_candidates, "\n")
                perturbed_instances = [
                    self._perturb_fn(x, inst=text.copy())
                    for x in explanation_candidates
                ]

                for instance_p in perturbed_instances:
                    self._print_instance(instance_p, text)

                scores_explanation_candidates = [
                    self._classifier_fn(x, self.revert) for x in perturbed_instances
                ]
                # Get predictions for each perturbed instance where one or more elements are removed from the initial instance
                # It is in form of [[x], [y], [z]]
                print(
                    "scores_explanation_candidates \n",
                    scores_explanation_candidates,
                    "\n",
                )
                scores_candidates_to_expand = scores_explanation_candidates.copy()

            scores_perturbed_new_combinations = [
                x[0] for x in scores_explanation_candidates
            ]
            # Therefore get it to the shape [x, y, z] by getting the [0] th element of each element array
            # print(
            #     "scores_perturbed_new_combinations ", scores_perturbed_new_combinations
            # )

            # ***CHECK IF THERE ARE EXPLANATIONS***
            new_explanations = list(
                compress(
                    explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            # Get explanation candidates where their probability is less than the threshold classifier -> Positive becomes negative
            # print("New Explanations \n", new_explanations)
            explanations += list(
                compress(
                    explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            # print("\n explanations, explanations_score_change", explanations)
            nb_explanations += len(
                list(
                    compress(
                        explanation_candidates,
                        scores_perturbed_new_combinations < self.threshold_classifier,
                    )
                )
            )  # Update number of explanations which pass the required threshold
            explanations_sets += list(
                compress(
                    explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            explanations_sets = [
                set(x) for x in explanations_sets
            ]  # Convert each array to a set -> to get the words
            explanations_score_change += list(
                compress(
                    scores_explanation_candidates,
                    scores_perturbed_new_combinations < self.threshold_classifier,
                )
            )
            # print('explanations_score_change', explanations_score_change)

            # Adjust max_length
            if self.BB == True:
                if len(explanations) != 0:
                    lengths = []  # Record length of each explanation found
                    for explanation in explanations:
                        lengths.append(len(explanation))
                    lengths = np.array(lengths)
                    max_length = lengths.min()
                    # Get minimum length of the found explanations as max length -> Do not search for explanations with longer length
                else:
                    max_length = number_active_elements  # Else can find maximum length equal to number of words in instance
            else:
                max_length = number_active_elements
            print("\n-------------Max length updated to - ", max_length)

            # Eliminate combinations from candidates_to_expand ("best-first" candidates) that can not be expanded
            # Pruning based on Branch & Bound=True, max. features allowed and number of active features
            candidates_to_expand_updated = []
            scores_candidates_to_expand_updated = (
                []
            )  # enumerate -> Find count of || to list one after another
            for j, combination in enumerate(candidates_to_expand):
                if (
                    (len(combination) < number_active_elements)
                    and (len(combination) < max_length)
                    and (len(combination) < self.max_features)
                ):
                    # Combination length should be less than the words in the input and max length of the required explanation and required maximum features
                    candidates_to_expand_updated.append(
                        combination
                    )  # If the combination matches, it is further expanded
                    scores_candidates_to_expand_updated.append(
                        scores_candidates_to_expand[j]
                    )
                    # Add the prediction score to the new array
                    # get the score from the scores_candidates_to_expand using the current index

            print(
                "\nlen(candidates_to_expand_updated)",
                len(candidates_to_expand_updated),
                " 0 ",
            )
            print(
                "\nnb_explanations",
                nb_explanations,
                " >= self.max_explained ",
                self.max_explained,
            )

            # *** IF LOOP ***
            # expanding the candidates to update will exceed the max length set in the earlier loop
            if (len(candidates_to_expand_updated) == 0) or (
                nb_explanations >= self.max_explained
            ):
                ## If the number of explanations exceeded the required number
                ## or no candidates
                ## no explanations present

                print("nb_explanations Stop iterations...")
                explanation_candidates = []  # stop algorithm
                ## Found all the candidates
                print(
                    "scores_candidates_to_expand_updated  ",
                    scores_candidates_to_expand_updated,
                )
                # print("candidates_to_expand_updated   ", candidates_to_expand_updated)

            elif len(candidates_to_expand_updated) != 0:
                ## If there are possible candidates

                explanation_candidates = []
                it = 0  # Iteration of the while loop
                indices = []

                scores_candidates_to_expand2 = []
                for score in scores_candidates_to_expand_updated:
                    if score[0] < self.threshold_classifier:
                        scores_candidates_to_expand2.append(2 * score_predicted)
                    else:
                        scores_candidates_to_expand2.append(score)
                # update candidate scores if they have score less than threshold -> To expand them further
                shap_candidates_to_expand2 = []
                for candidate in candidates_to_expand_updated:
                    importancess = 0
                    for word in candidate:
                        # find word in feature column in sorted_data
                        for ind in range(len(sorted_data_in)):
                            if sorted_data_in[ind]["feature"] == word:
                                importancess += sorted_data_in[ind]["importance"]
                                break
                    shap_candidates_to_expand2.append(importancess)

                # print(
                #     "\n scores_candidates_to_expand2 before loop",
                #     scores_candidates_to_expand2,
                # )

                # *** WHILE LOOP ***
                while (
                    (len(explanation_candidates) == 0)
                    and (it < len(scores_candidates_to_expand2))
                    and ((time.time() - tic) < self.time_maximum)
                ):
                    # Stop if candidates are found or looped through more than there are candidates or maximum time reached

                    print("While loop iteration %d" % it)

                    if it != 0:  # Because indices are not there in the first iteration
                        for index in indices:
                            scores_candidates_to_expand2[index] = 2 * score_predicted

                    # print(
                    #     "\n scores_candidates_to_expand2 after loop",
                    #     scores_candidates_to_expand2,
                    # )
                    # print("\n indices", indices)

                    # do elementwise subtraction between score_predicted and scores_candidates_to_expand2
                    subtractionList = []
                    for x, y in zip(score_predicted, scores_candidates_to_expand2):
                        print("\n x, y", x - y)
                        subtractionList.append(x - y)

                    # Do element wise subtraction between the prediction score of the x_ref and every element of the scores_candidates_to_expand2
                    index_combi_max = np.argmax(subtractionList)
                    index_importance_max = np.argmax(shap_candidates_to_expand2)
                    index_importance_min = np.argmin(shap_candidates_to_expand2)

                    print(
                        "subtrac max ",
                        index_combi_max,
                        " index_shap_max ",
                        index_importance_max,
                    )
                    if iteration < 3:
                        print("---------USING IMPORTANCE----------")
                        if self.revert == 1:
                            index_combi_max = index_importance_min
                        else:
                            index_combi_max = index_importance_max
                        # Get the index of the maximum value -> Expand it
                    else:
                        print("++++++++USING DIFFERENCE+++++++++")
                    print(
                        "\n index_combi_max",
                        candidates_to_expand_updated[np.argmax(subtractionList)],
                        "\n index_importance_max",
                        candidates_to_expand_updated[index_importance_max],
                        "\n using combination",
                        candidates_to_expand_updated[index_combi_max],
                    )
                    for element in candidates_to_expand_updated[index_combi_max]:
                        print(self.feature_names[element])
                    indices.append(index_combi_max)
                    expanded_combis.append(
                        candidates_to_expand_updated[index_combi_max]
                    )
                    # Add this combination to already expanded combinations as it will be expanded next by expand and prune function
                    
                    comb_to_expand = candidates_to_expand_updated[index_combi_max]
                    self.scores.append(
                        scores_candidates_to_expand_updated[index_combi_max][0]
                    )
                    # Expand the found combination with highest difference
                    func = self._expand_and_prune(
                        comb_to_expand,
                        expanded_combis,
                        feature_set,
                        candidates_to_expand_updated,
                        explanations_sets,
                        scores_candidates_to_expand_updated,
                        text,
                        self._classifier_fn,
                        self.feature_names,
                        self.revert,
                    )
                    """Returns:
                        - explanation_candidates: combinations of features that are explanation
                        candidates to be checked in the next iteration
                        - candidates_to_expand: combinations of features that are candidates to
                        expanded in next iterations or candidates for "best-first"
                        - expanded_combis: [list] list of combinations of features that are already
                        expanded as "best-first"
                        - scores_candidates_to_expand: scores after perturbation for the candidate
                        combinations of features to be expanded
                        - scores_explanation_candidates: scores after perturbation of explanation candidates"""
                    explanation_candidates = func[0]
                    candidates_to_expand = func[1]
                    expanded_combis = func[2]
                    scores_candidates_to_expand = func[3]
                    scores_explanation_candidates = func[4]

                    it += 1

                print(
                    "\n\n\niteration - ", iteration, " self.max_iter - ", self.max_iter
                )
                print(
                    "\n\nlen(candidates_to_expand) - ",
                    len(candidates_to_expand),
                    " != 0 ",
                )
                self.instances.append(len(candidates_to_expand))
                print(
                    "\n\nlen(explanation_candidates) - ",
                    len(explanation_candidates),
                    " !=0 ",
                )
                print(
                    "\n\n(time.time() - tic) - ",
                    (time.time() - tic),
                    " self.time_maximum - ",
                    self.time_maximum,
                )
            print("\n Elapsed time %d \n" % (time.time() - tic))

        # *** FINAL PART OF ALGORITHM ***
        print("Iterations are done.")
        #print("instances ", self.instances)
        # self._plot_graph(
        #     self.instances, "iterations", "candidates", "iteration-candidates graph"
        # )
        # self._plot_graph(self.scores, "iterations", "scores", "iteration-scores graph")
        explanation_set = []
        explanation_feature_names = []
        index_of_min_length_explanation = -1
        for i in range(len(explanations)):
            explanation_feature_names = []
            for features in explanations[i]:
                explanation_feature_names.append(self.feature_names[features])
            explanation_set.append(explanation_feature_names)

        if len(explanations) == 0:
            try:
                print("Finding the best alternative")
                removed_words = []

                best_explanation = np.argmin(scores_candidates_to_expand)
                best_explanation_score = scores_candidates_to_expand[best_explanation]
                best_explanation_comb = candidates_to_expand[best_explanation]
                removed_words = [self.feature_names[x] for x in best_explanation_comb]
                best_sentence = self._perturb_fn(
                    best_explanation_comb, inst=text.copy()
                )
                best_positive_score = self._classifier_fn(best_sentence)

                if best_positive_score > self.threshold_classifier:
                    final_class = [1]
                else:
                    final_class = [0]

                final_string = "Given a vector"
                if input_string != "Given a vector":
                    words = input_string.split()
                    final_words = []
                    for word in words:
                        if word.lower() in removed_words:
                            final_words.append("-" * len(word))
                        else:
                            final_words.append(word)

                    # Join the words back into a final string
                    final_string = " ".join(final_words)
                self._report_data["output"] = {
                    "Removed_words": removed_words,
                    "Error": "Couldn't reverse label within the time limit",
                    "final_text": final_string,
                    "final score for positive": best_positive_score[0],
                    "final class": final_class[0],
                }
                self._report_data["process"] = {
                    "explanation set": None,
                    "number active elements": number_active_elements,
                    "number explanations found": 0,
                    "size smallest explanation": 0,
                    "time elapsed": time.time() - tic,
                    "differences score": 0,
                    "iterations": iteration,
                    "final_sentence": None,
                }
                return
            except:
                print("Solution absolutely cannot be found")
                self._report_data["output"] = {
                    "Removed_words": None,
                    "Error": "Sentence out of limits or too short",
                    "final_text": None,
                    "final score for positive": None,
                    "final class": None,
                }
                self._report_data["process"] = {
                    "explanation set": None,
                    "number active elements": number_active_elements,
                    "number explanations found": 0,
                    "size smallest explanation": 0,
                    "time elapsed": time.time() - tic,
                    "differences score": 0,
                    "iterations": iteration,
                    "final_sentence": None,
                }
                return

        if len(explanations) != 0:
            lengths_explanation = []
            for explanation in explanations:
                l = len(explanation)
                lengths_explanation.append(l)
            minimum_size_explanation = np.min(lengths_explanation)
            index_of_min_length_explanation = np.argmin(lengths_explanation)

        final_sentence = text.copy()
        print("len explanations", len(explanations))
        if len(explanations) != 0:
            final_sentence = self._perturb_fn(
                explanations[index_of_min_length_explanation],
                text.copy(),
            )

        final_prob = self._classifier_fn(final_sentence)
        print("final_prob", final_prob)
        self.scores.append(final_prob[0])

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

        removed_words = [
            item
            for sublist in explanation_set[0 : self.max_explained]
            for item in sublist
        ]

        print(
            "If we remove the words ",
            removed_words,
            "From the review, the prediction will be reversed",
        )

        final_string = "Given a vector"
        if input_string != "Given a vector":
            words = input_string.split()
            final_words = []
            for word in words:
                if word.lower() in removed_words:
                    final_words.append("-" * len(word))
                else:
                    final_words.append(word)

            # Join the words back into a final string
            final_string = " ".join(final_words)

        word_pattern = r"\b\w+\b"

        # Split the input string into words while preserving punctuation
        words_with_punctuation = re.findall(word_pattern, final_string)

        final_words = []
        for word in words_with_punctuation:
            if word.lower() in removed_words:
                final_words.append("-" * len(word))
            else:
                final_words.append(word)

        final_string = " ".join(final_words)

        if final_prob > self.threshold_classifier:
            final_class = [1]
        else:
            final_class = [0]

        self._report_data["output"] = {
            "Removed_words": removed_words,
            "final_text": final_string,
            "final score for positive": final_prob[0],
            "final class": final_class[0],
        }

        print(self._report_data["input"])
        print(self._report_data["output"])

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
