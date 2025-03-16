from dataclasses import dataclass, asdict
from src.constants import EMPTY_DATA
import re


def recursive_to_str(entity):
    if isinstance(entity, list):
        out = ''
        for e in entity:
            out += f'- {recursive_to_str(e)}\n'
        return out.strip()
    elif isinstance(entity, dict):
        out = ''
        for k, v in entity.items():
            out += f'{k}: {recursive_to_str(v)}\n'
        return out.strip()
    else:
        out = str(entity)
        out = re.sub(r'\n+', '\n', out)  # many \n is wait of space
        return out


def is_entity_empty(entity):
    # recursively check if entry is == EMPTY_DATA
    if isinstance(entity, list):
        if len(entity) == 0:
            return True
        for e in entity:
            if not is_entity_empty(e):
                return False
    elif isinstance(entity, dict):
        if len(entity) == 0:
            return True
        for k, v in entity.items():
            if not is_entity_empty(v):
                return False
    elif isinstance(entity, str):
        if entity != EMPTY_DATA:
            return False
    elif isinstance(entity, (int, float, bool)):
        return False
    elif isinstance(entity, BaseEntity):
        for k, v in entity.dict().items():
            if not is_entity_empty(v):
                return False
    else:
        raise Exception(f'Unknown type {type(entity)} with {entity=}')
    return True


@dataclass
class BaseEntity:
    def __post_init__(self):
        # make sure none of the fields are empty
        for k, v in self.__dict__.items():
            if v is None:
                raise Exception(f'{k} is None')
        return

    def __str__(self):
        to_string = ''
        for k, v in self.__dict__.items():
            if v == EMPTY_DATA:
                continue
            elif is_entity_empty(v):
                continue
            elif isinstance(v, (list, dict)):
                to_string += f'{k}:\n{recursive_to_str(v)}\n'
            else:
                to_string += f'{k}: {recursive_to_str(v)}\n'
        return to_string.strip()

    def dict(self):
        return {k: v for k, v in asdict(self).items()}