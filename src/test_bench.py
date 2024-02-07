from .analyzers import KNNAnalyzer, SVMAnalyzer, RFAnalyzer, LRAnalyzer
import os
from typing import Dict, List
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse._csr import csr_matrix


class TestBench:
    valid_analyzers = ["knn", "svm", "lr", "rf"]

    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        analyzer_name: str,
        **kwargs,
    ) -> None:
        # validate
        if analyzer_name not in self.valid_analyzers:
            raise ValueError(
                f"Unsupported analyzer definition. Supported analyzer names are {self.valid_analyzers}"
            )

        if analyzer_name == "knn":
            analyzer = KNNAnalyzer(
                knn_path=model_path,
                vectorizer_path=vectorizer_path,
                **kwargs,
            )
        elif analyzer_name == "svm":
            analyzer = SVMAnalyzer(
                svm_path=model_path,
                vectorizer_path=vectorizer_path,
                **kwargs,
            )
        elif analyzer_name == "rf":
            analyzer = RFAnalyzer(
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                **kwargs,
            )
        elif analyzer_name == "lr":
            analyzer = LRAnalyzer(
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                **kwargs,
            )

        self._analyzer = analyzer
        self._analyzer_name = analyzer_name

    def __call__(
        self,
        configurations: Dict,
        text: str,
        variations: int = None,
        log_dir: str = None,
    ) -> List[str]:
        reports = []
        for i, config in enumerate(configurations):
            config_name = config["name"]

            self._analyzer.set_config(config)
            self._analyzer(text, variations)
            report = self._analyzer.explanation()

            if log_dir is not None:
                config_log_dir = os.path.join(log_dir, f"{i+1}-config-{config_name}")
                os.makedirs(config_log_dir, exist_ok=True)
                with open(os.path.join(config_log_dir, "config.yaml"), "w") as handler:
                    yaml.dump(config, handler)
                with open(os.path.join(config_log_dir, "report.txt"), "w") as handler:
                    handler.write(report)

            report = f"==== Configuration {config_name} ({i+1}) ====\n" + report
            reports.append(report)

        return reports

    def evaluate(self, x: csr_matrix, y: csr_matrix, save_dir: str = None) -> None:
        model = self._analyzer._model
        model_name = self._analyzer_name.upper()

        prob = model.predict_proba(x)
        pred = prob.argmax(axis=1)
        report = metrics.classification_report(y, pred)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        plot_names = ["Confusion Matrix", "Label Correlogram"]
        for i, name in enumerate(plot_names):
            axis = ax[i]
            axis.set_title(name)

            if i == 0:
                # confusion matrix
                confusion_matrix = metrics.confusion_matrix(y, pred)
                cm_display = metrics.ConfusionMatrixDisplay(
                    confusion_matrix=confusion_matrix,
                    display_labels=["Negative", "Positive"],
                )
                cm_display.plot(ax=axis)

            else:
                # label correlogram
                correlation_matrix = pd.DataFrame(prob).corr().to_numpy()
                cm_display = metrics.ConfusionMatrixDisplay(
                    confusion_matrix=correlation_matrix,
                    display_labels=["Negative", "Positive"],
                )
                cm_display.plot(ax=axis, cmap="coolwarm")

        plt.tight_layout()

        report = f"       ---- Classification report for {model_name} ----\n{report}\n"
        if save_dir is None:
            print(report)
            plt.show()
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "evaluation.txt"), "w") as handler:
                handler.write(report)
            plt.savefig(os.path.join(save_dir, "evaluation.jpg"))
            plt.close()
