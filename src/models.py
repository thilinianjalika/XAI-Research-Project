from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import wget
import os, shutil
import yaml
import joblib
from typing import List, Tuple, Dict

try:
    from .processors import Preprocessor, LUTLabelEncoder
except ImportError:
    from processors import Preprocessor, LUTLabelEncoder


class BaseAnalysisModel:
    def fit(self, x, y):
        self.model.fit(x, y)

    def save(self, save_path: str):
        joblib.dump(self.model, save_path)


class RFModel(BaseAnalysisModel):
    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            bootstrap=False,
            ccp_alpha=0.0,
            class_weight=None,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            max_leaf_nodes=None,
            max_samples=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=4,
            min_samples_split=5,
            min_weight_fraction_leaf=0.0,
            n_estimators=100,
            n_jobs=None,
            oob_score=False,
            random_state=None,
            verbose=0,
            warm_start=False,
        )


class SVCModel(BaseAnalysisModel):
    def __init__(self) -> None:
        self.model = SVC(
            C=10,
            break_ties=False,
            cache_size=200,
            class_weight=None,
            coef0=0.0,
            decision_function_shape="ovr",
            degree=3,
            gamma=1,
            kernel="rbf",
            max_iter=-1,
            probability=True,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False,
        )


class LRModel(BaseAnalysisModel):
    def __init__(self) -> None:
        self.model = LogisticRegression(
            C=4.281332398719396,
            class_weight=None,
            dual=False,
            fit_intercept=True,
            intercept_scaling=1,
            l1_ratio=None,
            max_iter=100,
            multi_class="auto",
            n_jobs=None,
            penalty="l2",
            random_state=None,
            solver="lbfgs",
            tol=0.0001,
            verbose=0,
            warm_start=False,
        )


class KNNModel(BaseAnalysisModel):
    def __init__(self) -> None:
        self.model = KNeighborsClassifier(
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
            n_neighbors=90,
            p=2,
            weights="distance",
        )


class AnalysisModelWrapper:
    def __init__(self, config: Dict, model) -> None:
        self.model = model
        self.preprocessor = Preprocessor()
        input_encoder_name = config["encoders"]["input_encoder_name"]
        input_encoder_path = config["paths"][input_encoder_name]
        self.input_encoder = joblib.load(input_encoder_path)
        self.output_decoder = LUTLabelEncoder(config["encoders"]["output_labels"])

    def __call__(self, txt_lst: List[str]) -> Tuple[List[float], List[str]]:
        if type(txt_lst) == str:
            txt_lst = [txt_lst]

        txt_lst = [self.preprocessor(txt) for txt in txt_lst]
        input_arr = self.input_encoder.transform(txt_lst)
        prob = self.model.predict_proba(input_arr)
        scores = prob[:, 1].tolist()
        pred = prob.argmax(axis=1)
        output = self.output_decoder.inverse_tranform(pred)
        return scores, output


class DownloadableModel:
    def __init__(
        self, config_path: str, root: str = None, download: bool = False
    ) -> None:
        with open(config_path) as handler:
            config = yaml.load(handler, Loader=yaml.FullLoader)
        self.name = config["name"]
        if root is None:
            self.root = os.path.abspath(os.path.split(config_path)[0])
        else:
            self.root = os.path.abspath(root)
        self.source_url = config["source_url"]
        self.config = config

        self._parse_paths()
        if not self._validate():
            if download:
                self._download_n_extract()
            else:
                raise FileNotFoundError(
                    f"Files not found in {self.root}. Use 'download=True' to download from source"
                )

    def _parse_paths(self) -> None:
        for k, v in self.config["paths"].items():
            self.config["paths"][k] = os.path.join(self.root, v)

    def _validate(self) -> bool:
        paths = self.config["paths"].values()
        paths_exist = [os.path.exists(path) for path in paths]
        valid = all(paths_exist)
        return valid

    def _download_n_extract(self) -> List[str]:
        os.makedirs(self.root, exist_ok=True)
        download_path = os.path.join(self.root, os.path.split(self.source_url)[1])
        print(f"Downloading from source ({self.source_url}) to {download_path}")
        download_path = wget.download(self.source_url, download_path)
        shutil.unpack_archive(download_path, self.root)


class AnalysisModels(DownloadableModel):
    def __init__(
        self, config_path: str, root: str = None, download: bool = False
    ) -> None:
        super().__init__(config_path, root, download)
        self._load_models()

    def _load_models(self) -> None:
        for name in self.config["models"].keys():
            path = self.config["paths"][name]
            model = joblib.load(path)
            model = AnalysisModelWrapper(self.config, model)
            setattr(self, name, model)

    def __str__(self) -> str:
        return f"A collection of pretrained sklearn models.\nContains the models {list(self.config['models'].keys())}"
