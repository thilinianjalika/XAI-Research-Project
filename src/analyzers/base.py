from typing import Any


class BaseAnalyzer:
    _model = None
    _text_vectorizer = None
    _cf_generator = None
    _report_data = None

    def explanation(self) -> str:
        raise NotImplementedError("Method not implemented yet.")

    def __call__(self, text: str, search_space: int) -> str:
        raise NotImplementedError("Method not implemented yet.")

    def set_config(self, config) -> None:
        raise NotImplementedError("Method not implemented yet.")
