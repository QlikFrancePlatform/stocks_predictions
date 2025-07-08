import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app import app, download_stock_data, get_close_data, prepare_lstm_data, build_lstm_model, make_predictions

client = TestClient(app)

# Mock data for testing
def create_mock_stock_data():
    """Create mock stock data for testing"""
    dates = pd.date_range(start='2012-01-01', end='2019-12-17', freq='D')
    close_prices = np.random.uniform(100, 200, len(dates))
    
    df = pd.DataFrame({
        'Close': close_prices,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02,
        'Low': close_prices * 0.98,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df

@pytest.fixture
def mock_stock_data():
    return create_mock_stock_data()

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Financial API" in response.text

def test_getdata():
    """Test the getdata endpoint"""
    response = client.get("/getdata")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

@patch('app.yf.download')
def test_download_stock_data_success(mock_download, mock_stock_data):
    """Test successful stock data download"""
    mock_download.return_value = mock_stock_data
    
    result = download_stock_data(["AAPL"])
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "Close" in result.columns

@patch('app.yf.download')
def test_download_stock_data_empty(mock_download):
    """Test stock data download with empty result"""
    mock_download.return_value = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Aucune donnée téléchargée"):
        download_stock_data(["INVALID"])

def test_get_close_data_success(mock_stock_data):
    """Test extracting close data from DataFrame"""
    result = get_close_data(mock_stock_data, ["AAPL"])
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "Close" in result.columns

def test_get_close_data_no_close_column():
    """Test extracting close data when no Close column exists"""
    df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107]
    })
    
    with pytest.raises(ValueError, match="Aucune colonne 'Close' trouvée"):
        get_close_data(df, ["AAPL"])

def test_prepare_lstm_data(mock_stock_data):
    """Test LSTM data preparation"""
    close_data = get_close_data(mock_stock_data, ["AAPL"])
    
    x_train, y_train, scaler, training_data_len, scaled_data = prepare_lstm_data(close_data)
    
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(scaler, object)  # MinMaxScaler
    assert training_data_len > 0
    assert isinstance(scaled_data, np.ndarray)

def test_build_lstm_model():
    """Test LSTM model building"""
    input_shape = (60, 1)
    model = build_lstm_model(input_shape)
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')

def test_make_predictions(mock_stock_data):
    """Test making predictions"""
    close_data = get_close_data(mock_stock_data, ["AAPL"])
    x_train, y_train, scaler, training_data_len, scaled_data = prepare_lstm_data(close_data)
    model = build_lstm_model((x_train.shape[1], 1))
    
    # Train the model briefly
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
    
    predictions = make_predictions(model, scaler, scaled_data, training_data_len)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) > 0

@patch('app.download_stock_data')
def test_createdata_success(mock_download, mock_stock_data):
    """Test successful createdata endpoint"""
    mock_download.return_value = mock_stock_data
    
    data = {"stocks": "AAPL"}
    response = client.post("/createdata", json=data)
    
    assert response.status_code == 200
    assert "data" in response.text or "AAPL" in response.text

@patch('app.download_stock_data')
def test_createdata_error(mock_download):
    """Test createdata endpoint with error"""
    mock_download.side_effect = ValueError("Test error")
    
    data = {"stocks": "INVALID"}
    response = client.post("/createdata", json=data)
    
    assert response.status_code == 400
    assert "Test error" in response.text

@patch('app.download_stock_data')
@patch('app.get_close_data')
@patch('app.prepare_lstm_data')
@patch('app.build_lstm_model')
def test_predict_success(mock_build_model, mock_prepare_data, mock_get_close, mock_download, mock_stock_data):
    """Test successful predict endpoint"""
    # Setup mocks
    mock_download.return_value = mock_stock_data
    mock_get_close.return_value = get_close_data(mock_stock_data, ["AAPL"])
    
    # Mock LSTM data preparation
    close_data = get_close_data(mock_stock_data, ["AAPL"])
    x_train, y_train, scaler, training_data_len, scaled_data = prepare_lstm_data(close_data)
    mock_prepare_data.return_value = (x_train, y_train, scaler, training_data_len, scaled_data)
    
    # Mock model
    model = build_lstm_model((x_train.shape[1], 1))
    mock_build_model.return_value = model
    
    # Mock model.fit to avoid actual training
    with patch.object(model, 'fit') as mock_fit:
        mock_fit.return_value = None
        
        data = {"stocks": "AAPL"}
        response = client.post("/predict", json=data)
        
        assert response.status_code == 200
        assert "Predictions" in response.text or "Close" in response.text

@patch('app.download_stock_data')
def test_predict_error(mock_download):
    """Test predict endpoint with error"""
    mock_download.side_effect = ValueError("Test error")
    
    data = {"stocks": "INVALID"}
    response = client.post("/predict", json=data)
    
    assert response.status_code == 400
    assert "Test error" in response.text

def test_predictstocks_model():
    """Test PredictStocks Pydantic model"""
    from app import PredictStocks
    
    # Test valid data
    data = PredictStocks(stocks="AAPL,MSFT")
    assert data.stocks == "AAPL,MSFT"
    
    # Test with single stock
    data = PredictStocks(stocks="AAPL")
    assert data.stocks == "AAPL"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 