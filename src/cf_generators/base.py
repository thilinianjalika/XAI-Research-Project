from typing import List, Dict, Any
from ..models import DownloadableModel


class BaseGenerator(DownloadableModel):
    def __call__(self, inp: str, variations: int = 4) -> List[str]:
        raise NotImplementedError("Method not implemented yet.")

    def set_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError("Method not implemented yet.")
