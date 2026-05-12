# Pytest unit testing
import pytest
import joblib
import numpy as np

# fixture - load model once
@pytest.fixture
def model():
    return joblib.load("anilist_model.pkl")

# Test 1 - check model loaded
def test_model_loaded(model):
    assert model is not None

# Test 2 - Check prediction shape
def test_prediction_shape(model):
    sample_data = np.array([
        [8.5, 12] 
    ])
    prediction = model.predict(sample_data)
    assert len(prediction) == 1

# Test 3 - Known prediction
def test_known_prediction(model):
    sample_data = np.array([[9.0, 24]])
    prediction = model.predict(sample_data)
    
    assert isinstance(prediction[0], str)

# Test 4 - Probabilities sum to 1
def test_probability_sum(model):
    sample_data = np.array([
        [7.0, 12]
    ])
    probabilities = model.predict_proba(sample_data)
    total_prob = probabilities[0].sum()
    assert total_prob == pytest.approx(1.0, abs=1e-6)

# Test 5 - Parameterized Testing
@pytest.mark.parametrize(
    "sample_data, expected_type",
    [
        ([[8.0, 12]], str),
        ([[5.0, 1]], str),
    ]
)
def test_multiple_predictions(model, sample_data, expected_type):
    prediction = model.predict(sample_data)
    assert isinstance(prediction[0], expected_type)
