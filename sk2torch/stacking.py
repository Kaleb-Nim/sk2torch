from typing import List, Type

import torch
import torch.jit
import torch.nn as nn
from sklearn.ensemble import StackingClassifier

from .util import fill_unsupported


class TorchStackingClassifier(nn.Module):
    def __init__(
        self,
        passthrough: bool,
        estimators: List[nn.Module],
        stack_methods: List[str],
        final_estimator: nn.Module,
        classes: torch.Tensor,
    ):
        super().__init__()
        self.passthrough = passthrough
        self.estimators = nn.ModuleList(estimators)
        for model in self.estimators:
            fill_unsupported(model, "predict_proba", "decision_function", "predict")
        self.stack_methods = stack_methods
        self.final_estimator = final_estimator
        self.register_buffer("classes", classes)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [StackingClassifier]

    @classmethod
    def wrap(cls, obj: StackingClassifier) -> "TorchStackingClassifier":
        from .wrap import wrap

        return cls(
            passthrough=obj.passthrough,
            estimators=[wrap(x) for x in obj.estimators_],
            stack_methods=obj.stack_method_,
            final_estimator=wrap(obj.final_estimator_),
            classes=torch.from_numpy(obj.classes_),
        )

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, estimator in enumerate(self.estimators):
            method = self.stack_methods[i]
            if method == "predict":
                out = estimator.predict(x)
            elif method == "predict_proba":
                out = estimator.predict_proba(x)
                if out.shape[1] == 2:
                    out = out[:, 1:]
            else:
                assert method == "decision_function"
                out = estimator.decision_function(x)
            if len(out.shape) == 1:
                out = out[:, None]
            outputs.append(out)
        if self.passthrough:
            outputs.append(x.view(len(x), -1))
        return torch.cat(outputs, dim=-1)