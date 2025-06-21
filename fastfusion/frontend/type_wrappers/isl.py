from typing import Any, TypeVar, Generic, Type

import islpy as isl

from pydantic import BaseModel, ValidationError
from pydantic_core import core_schema

class ISLStr:
    def __init__(self, isl_type: Type):
        self.isl_type = isl_type

    def __get_pydantic_core_schema__(self, _source, _handler):
        def validate(value: Any):
            if not isinstance(value, str):
                raise TypeError('Value must be a string')
            try:
                return self.isl_type(value)
            except Exception as e:
                raise ValueError(f'Invalid input for {self.isl_type.__name__}: {e}')
        
        return core_schema.no_info_plain_validator_function(validate)

ISLMap = ISLStr(isl.Map)
ISLSet = ISLStr(isl.Set)

class MyModel(BaseModel):
    region: ISLSet
    mapping: ISLMap

# Example
try:
    m = MyModel(
        region="[n] -> { S[i] : 0 <= i < n }",
        mapping="{ S[i] -> T[j] : i = j }"
    )
    print("region:", m.region)
    print("mapping:", m.mapping)
except ValidationError as e:
    print(e)