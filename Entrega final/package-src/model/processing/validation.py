from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    bedrooms: Optional[float]
    bathrooms: Optional[float]
    sqft_living: Optional[int]
    sqft_lot: Optional[int]
    floors: Optional[float]
    waterfront: Optional[int]
    view: Optional[int]
    condition: Optional[int]
    sqft_above: Optional[int]
    sqft_basement: Optional[int]
    yr_built: Optional[int]
    yr_renovated: Optional[int]

    # Dummy variables
    city_Algona: Optional[bool]
    city_Auburn: Optional[bool]
    city_Beaux_Arts_Village: Optional[bool]
    city_Bellevue: Optional[bool]
    city_Black_Diamond: Optional[bool]
    city_Bothell: Optional[bool]
    city_Burien: Optional[bool]
    city_Carnation: Optional[bool]
    city_Clyde_Hill: Optional[bool]
    city_Covington: Optional[bool]
    city_Des_Moines: Optional[bool]
    city_Duvall: Optional[bool]
    city_Enumclaw: Optional[bool]
    city_Fall_City: Optional[bool]
    city_Federal_Way: Optional[bool]
    city_Issaquah: Optional[bool]
    city_Kenmore: Optional[bool]
    city_Kent: Optional[bool]
    city_Kirkland: Optional[bool]
    city_Lake_Forest_Park: Optional[bool]
    city_Maple_Valley: Optional[bool]
    city_Medina: Optional[bool]
    city_Mercer_Island: Optional[bool]
    city_Milton: Optional[bool]
    city_Newcastle: Optional[bool]
    city_Normandy_Park: Optional[bool]
    city_North_Bend: Optional[bool]
    city_Pacific: Optional[bool]
    city_Preston: Optional[bool]
    city_Ravensdale: Optional[bool]
    city_Redmond: Optional[bool]
    city_Renton: Optional[bool]
    city_Sammamish: Optional[bool]
    city_SeaTac: Optional[bool]
    city_Seattle: Optional[bool]
    city_Shoreline: Optional[bool]
    city_Skykomish: Optional[bool]
    city_Snoqualmie: Optional[bool]
    city_Snoqualmie_Pass: Optional[bool]
    city_Tukwila: Optional[bool]
    city_Vashon: Optional[bool]
    city_Woodinville: Optional[bool]
    city_Yarrow_Point: Optional[bool]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
