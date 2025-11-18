import math

import numpy 

from model.predict import make_prediction


def test_make_prediction(sample_input_data):

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")

    # Must return a list
    assert isinstance(predictions, list)

    # Must match the number of input rows
    assert len(predictions) == len(sample_input_data)

    # Should have no validation errors
    assert result.get("errors") is None

    # First prediction should be a float
    assert isinstance(predictions[0], (float, np.floating))

    # Predictions should be finite numbers
    assert math.isfinite(predictions[0])
    assert not math.isnan(predictions[0])
