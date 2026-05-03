"""Tests for TibbleClient."""

from unittest.mock import MagicMock
import pytest

from datashield import DSSession
from datashield_tidyverse import TibbleClient


@pytest.fixture
def mock_dssession():
    """Create a mock DSSession for testing."""
    session = MagicMock(spec=DSSession)
    session.id = "test_session"
    return session


def test_tibble_client_init(mock_dssession):
    """Test that TibbleClient can be instantiated."""
    client = TibbleClient(mock_dssession)
    assert client.dssession == mock_dssession


def test_as_tibble_from_dataframe(mock_dssession):
    """Test converting a dataframe to a tibble."""
    client = TibbleClient(mock_dssession)

    client.as_tibble(df_name="mtcars", newobj="mtcars_tibble")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    # Check that newobj and call expression are correct
    assert call_args[0][0] == "mtcars_tibble"
    call_expr = call_args[0][1]
    assert "asTibbleDS" in call_expr
    assert "mtcars" in call_expr
