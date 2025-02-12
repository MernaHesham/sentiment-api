from httpx import ASGITransport, AsyncClient
from app import app
import pytest

@pytest.mark.anyio
async def test_predict_endpoint():
    # Clear any existing cache
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with a short text
        response = await ac.post("/predict", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

@pytest.mark.anyio
async def test_predict_quantized_endpoint():
    # Clear any existing cache
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with a short text
        response = await ac.post("/predict_quantized", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

@pytest.mark.anyio
async def test_predict_endpoint_with_invalid_input():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with invalid input (missing text field)
        response = await ac.post("/predict", json={})
        assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.anyio
async def test_predict_quantized_endpoint_with_invalid_input():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with invalid input (missing text field)
        response = await ac.post("/predict_quantized", json={})
        assert response.status_code == 422  # Unprocessable Entity