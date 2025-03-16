from typing import Any


def _format_inner_data(data: Any):
    assert data is not None # data should not be None
    formatted_str = ""
    if isinstance(data, dict):
        for k, v in data.items():
            formatted_str += f"{k}: {_format_inner_data(v)}; "
        return formatted_str.strip()
    elif isinstance(data, list):
        inner_data = [f"{_format_inner_data(v)}" for v in data]
        formatted_str += ", ".join(inner_data)
        return formatted_str.strip()
    elif isinstance(data, str):
        return data
    elif isinstance(data, (int, float)):
        return str(data)
    else:
        raise ValueError(f"Unknown data type: {type(data)}, {data}")


def dict_to_text(dict_data: dict):
    non_nested_data = {}
    nested_data = {}
    for k, v in dict_data.items():
        if isinstance(v, (str, int, float, bool)):
            non_nested_data[k] = str(v)
        else:
            nested_data[k] = v
    formatted_sections = [f"## {k}\n{_format_inner_data(v)}".strip() for k, v in nested_data.items()]
    formatted_flat_sections = [f"{k}: {v}" for k, v in non_nested_data.items()]

    formatted_sections.insert(0, "## Basic Info\n" + "\n".join(formatted_flat_sections))
    return "\n\n".join(formatted_sections).strip()


def dict_to_sectional_text(dict_data: dict):
    non_nested_data = {}
    nested_data = {}
    for k, v in dict_data.items():
        if isinstance(v, (str, int, float, bool)):
            non_nested_data[k] = str(v)
        else:
            nested_data[k] = v
    formatted_sections = {k: _format_inner_data(v) for k, v in nested_data.items()}
    formatted_flat_sections = [f"{k}: {v}" for k, v in non_nested_data.items()]
    formatted_sections["Basic Info"] = "\n".join(formatted_flat_sections)
    return formatted_sections