from typing import Callable, Optional

import numpy as np
from nnsight import LanguageModel


def save_logits(model: LanguageModel) -> np.ndarray:
    """Intervention that saves the logits."""
    return model.lm_head.output[0][-1].detach().cpu().numpy()


def intervention_factory(
    intervention: Optional[str],
) -> Callable[[LanguageModel], np.ndarray]:
    if intervention is None:
        return None
    elif intervention == "save_logits":
        return save_logits
    else:
        raise ValueError(f"Intervention {intervention} not supported")
