from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, MutableMapping, Union

from src.models.cnet import ClassifierModel
from src.models.unet import UNetModel
from src.models.cond_unet import CondUNetModel



ModelFactory = Callable[..., Any]


class ModelName(str, Enum):
    CNET = "cnet"
    UNET = "unet"
    COND_UNET = "cond_unet"


@dataclass(frozen=True)
class ModelConfig:
    factory: ModelFactory
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    required_kwargs: tuple[str, ...] = ()


def _cnet_factory(**kwargs: Any) -> Any:
    return ClassifierModel(**kwargs)


def _unet_factory(**kwargs: Any) -> Any:
    return UNetModel(**kwargs)

def _cond_unet_factory(**kwargs: Any) -> Any:
    return CondUNetModel(**kwargs)


_MODEL_REGISTRY: MutableMapping[ModelName, ModelConfig] = {
    ModelName.CNET: ModelConfig(factory=_cnet_factory),
    ModelName.UNET: ModelConfig(factory=_unet_factory, required_kwargs=("marginal_prob_std",)),
    ModelName.COND_UNET: ModelConfig(factory=_cond_unet_factory),
}


def _resolve_model_name(name: Union[str, ModelName]) -> ModelName:
    if isinstance(name, ModelName):
        return name
    normalized = str(name).lower()
    for candidate in ModelName:
        if candidate.value == normalized:
            return candidate
    available = ", ".join(item.value for item in ModelName)
    raise ValueError(f"Model {name!r} is not recognized. Available: {available}.")


def get_model_by_name(network_name: Union[str, ModelName], **kwargs: Any) -> Any:
    model_name = _resolve_model_name(network_name)
    config = _MODEL_REGISTRY[model_name]
    kwargs.pop("name", None)  # Remove name if present

    params: Dict[str, Any] = dict(config.default_kwargs)
    params.update(kwargs)

    missing_args = [arg for arg in config.required_kwargs if arg not in params]
    if missing_args:
        missing = ", ".join(missing_args)
        raise ValueError(f"Model {model_name.value!r} requires the following arguments: {missing}.")

    return config.factory(**params)

