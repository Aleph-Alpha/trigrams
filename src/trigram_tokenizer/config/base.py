import json
import yaml  # type: ignore
from typing import Dict, Any, Union, Optional
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Extra


def overwrite_recursive(d: Dict, d_new: Dict):
    for k, v in list(d_new.items()):
        if isinstance(v, dict):
            if k not in d:
                d[k] = dict()
            overwrite_recursive(d[k], d_new[k])
        else:
            d[k] = v


class BaseConfig(BaseModel):
    """
    Base config class providing general settings for non-mutability and json serialization options
    """

    class Config:
        extra = Extra.forbid  # forbid unknown fields on instantiation
        frozen = True  # make all fields immutable

    def __post_init__(self):
        pass

    def as_dict(self) -> Dict[Any, Any]:
        """
        return a json-serializable dict of self
        """
        self_dict: Dict[Any, Any] = dict()

        for k, v in dict(self).items():
            assert isinstance(k, str)
            if isinstance(v, BaseConfig):
                self_dict[k] = v.as_dict()
            elif isinstance(v, Path):
                self_dict[k] = str(v)
            elif isinstance(v, list) and all([isinstance(i, Path) for i in v]):
                self_dict[k] = [str(i) for i in v]
            elif isinstance(v, list) and all([isinstance(i, BaseConfig) for i in v]):
                self_dict[k] = [i.as_dict() for i in v]
            elif isinstance(v, Enum):
                self_dict[k] = v.value
            elif isinstance(v, list) and all([isinstance(i, Enum) for i in v]):
                self_dict[k] = [i.value for i in v]
            else:
                self_dict[k] = v

        return self_dict

    @classmethod
    def from_dict(cls, d: dict, overwrite_values: Optional[Dict] = None):
        if overwrite_values is not None:
            overwrite_recursive(d, overwrite_values)

        return cls(**d)

    def as_str(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def from_str(cls, s: str):
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_yaml(
        cls, yml_filename: Union[str, Path], overwrite_values: Optional[Dict] = None
    ):
        with open(yml_filename) as conf_file:
            config_dict = yaml.load(conf_file, Loader=yaml.FullLoader)

        if overwrite_values is not None:
            overwrite_recursive(config_dict, overwrite_values)
        return cls.from_dict(config_dict)

    def save(self, out_file: Path, indent=4):
        json.dump(self.as_dict(), open(out_file, "w", encoding="UTF-8"), indent=indent)

    @classmethod
    def get_template_str(cls, indent=4, level=1):
        # get schema
        schema = cls.schema()
        fields = cls.__fields__

        # save out
        result = ""
        result += " " * (level - 1) * indent + "{\n"
        result += " " * level * indent + f"# {schema['title']}\n"
        result += " " * level * indent + f"# {schema['description']}\n"

        for field_index, (field_name, schema_definition) in enumerate(
            schema["properties"].items()
        ):
            field_definition = fields[field_name]
            is_last = field_index == len(schema["properties"]) - 1

            # description and field name
            result += " " * level * indent + f"\n"
            result += " " * level * indent
            result += f"# {field_definition.field_info.description}\n"
            result += " " * level * indent
            result += f'"{field_name}": '

            # field value
            if isinstance(field_definition.default, BaseConfig):
                result += field_definition.default.get_template_str(
                    indent=indent, level=level + 1
                )
            else:
                if isinstance(field_definition.default, Enum):
                    result += f"{json.dumps(field_definition.default.value)}"
                else:
                    result += f"{json.dumps(field_definition.default)}"

            # comma in the end
            if not is_last:
                result += ","

            # finalize
            result += "\n"

        result += " " * (level - 1) * indent + "}\n"

        return result

    @classmethod
    def save_template(cls, out_file: Path, indent=4):
        # convert to yaml str
        result = cls.get_template_str(indent=indent)

        # save out
        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(result)
