from typing import Dict, List, Union


PREPROCESSING = Dict[str, Dict[str, Union[str, int, float, List[Union[int, float]]]]]


def make_indent_string(string: str, num_indent: int = 2) -> str:
    indent = " " * num_indent
    return f"{indent}{string}"
