from base64 import b64decode
from typing import Any

from pydantic import BaseModel, field_validator, GetCoreSchemaHandler, ValidationInfo
from pydantic_core import CoreSchema, core_schema
from numpy import frombuffer, ndarray, uint8
from numpy.typing import NDArray

import re


class NoiseData:
    def __init__(self, data: NDArray):
        self.__data = data

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate_ndarray,
            core_schema.any_schema()
        )
    
    @classmethod
    def validate_ndarray(cls, value: NDArray) -> NoiseData:
        if not isinstance(value, ndarray):
            raise TypeError("Wrong type")
        
        return cls(value)

    def __getitem__(self, key) -> NDArray:
        return self.__data[key]

    def to_bytes(self) -> bytes:
        return self.__data.tobytes()


class Image(BaseModel):
    height:     int
    width:      int
    data:       NoiseData
    format:     str
    mipmaps:    bool

    @staticmethod
    def __parese_data(string: str, width: int, height: int) -> NDArray:
        data =  re.search(r'PackedByteArray\("([^"]+)"\)', string)

        assert data is not None

        assert len(data.groups()) != 0

        byte_array_string = data.group(1)

        bytes_data = b64decode(byte_array_string)

        return frombuffer(bytes_data, dtype=uint8).reshape((width, height))

    @field_validator('data', mode='before')
    @classmethod
    def convert_str_to_bytes(cls, data: str | NoiseData, info: ValidationInfo):
        assert type(data) in (str, NoiseData)

        width:  int | None = info.data.get("width")
        height: int | None = info.data.get("height")

        if width is None or height is None:
            raise ValueError("width and height shouldn't be None")
        
        if isinstance(data, str):
            return cls.__parese_data(data, width, height)
        
        return data

    def get(self, index1: int, index2: int) -> float:
        assert 0 <= index1 < self.height
        assert 0 <= index2 < self.width

        return self.data[index1][index2]


def __validate_json(string: str) -> str:
    pattern = r'("data"\s*:\s*)(PackedByteArr[^,\s}\n]*)'
    
    def replace_with_quoted(match: re.Match) -> str:
        prefix: str = match.group(1)
        value: str = match.group(2)
        
        value = value.replace('"', '\\"')

        return f'{prefix}"{value}"'
    
    return re.sub(pattern, replace_with_quoted, string)



def parse(filename: str, expected_size: int = 256) -> list[Image]:
    with open(filename, "r") as file:
        data = file.read()
    
    sub_strings = re.findall(r'data\s*=\s*(\{.*?\})', data, re.DOTALL)

    assert len(sub_strings) == expected_size

    result = []

    for sub_string in sub_strings:
        valid_json: str = __validate_json(sub_string)

        result.append(
            Image.model_validate_json(valid_json)
        )
    
    return result
