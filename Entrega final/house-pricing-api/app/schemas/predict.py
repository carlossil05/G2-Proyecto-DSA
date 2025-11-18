from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples. Debe predecir 13.635679994241398 en el ejemplo
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "bedrooms": 3.0,
                        "bathrooms": 2.5,
                        "sqft_living": 3660,
                        "sqft_lot": 39478,
                        "floors": 2.0,
                        "waterfront": 0,
                        "view": 2,
                        "condition": 4,
                        "sqft_above": 3260,
                        "sqft_basement": 400,
                        "yr_built": 1989,
                        "yr_renovated": False,
                        "city_Algona": False,
                        "city_Auburn": False,
                        "city_Beaux_Arts_Village": False,
                        "city_Bellevue": False,
                        "city_Black_Diamond": False,
                        "city_Bothell": False,
                        "city_Burien": False,
                        "city_Carnation": False,
                        "city_Clyde_Hill": False,
                        "city_Covington": False,
                        "city_Des_Moines": False,
                        "city_Duvall": False,
                        "city_Enumclaw": True,
                        "city_Fall_City": False,
                        "city_Federal_Way": False,
                        "city_Issaquah": False,
                        "city_Kenmore": False,
                        "city_Kent": False,
                        "city_Kirkland": False,
                        "city_Lake_Forest_Park": False,
                        "city_Maple_Valley": False,
                        "city_Medina": False,
                        "city_Mercer_Island": False,
                        "city_Milton": False,
                        "city_Newcastle": False,
                        "city_Normandy_Park": False,
                        "city_North_Bend": False,
                        "city_Pacific": False,
                        "city_Preston": False,
                        "city_Ravensdale": False,
                        "city_Redmond": False,
                        "city_Renton": False,
                        "city_Sammamish": False,
                        "city_SeaTac": False,
                        "city_Seattle": False,
                        "city_Shoreline": False,
                        "city_Skykomish": False,
                        "city_Snoqualmie": False,
                        "city_Snoqualmie_Pass": False,
                        "city_Tukwila": False,
                        "city_Vashon": False,
                        "city_Woodinville": False,
                        "city_Yarrow_Point": False
                    }
                ]
            }
        }
