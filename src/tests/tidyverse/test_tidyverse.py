"""Tests for TidyverseClient."""

from unittest.mock import MagicMock
import pytest

from datashield import DSSession
from datashield_tidyverse import TidyverseClient


@pytest.fixture
def mock_dssession():
    """Create a mock DSSession for testing."""
    session = MagicMock(spec=DSSession)
    session.id = "test_session"
    return session


def test_select_creates_tidyverse_client(mock_dssession):
    """Test that TidyverseClient can be instantiated."""
    client = TidyverseClient(mock_dssession)
    assert client.dssession == mock_dssession


def test_select_basic_columns(mock_dssession):
    """Test selecting basic columns from a dataframe."""
    client = TidyverseClient(mock_dssession)

    client.select(df_name="mtcars", tidy_expr="mpg, cyl", newobj="subset")

    # Verify that assign was called with the correct arguments
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    # Check that newobj and call expression are correct
    assert call_args[0][0] == "subset"
    call_expr = call_args[0][1]
    assert "selectDS" in call_expr
    assert "mtcars" in call_expr


def test_select_with_helpers(mock_dssession):
    """Test selecting columns with tidy select helpers."""
    client = TidyverseClient(mock_dssession)

    client.select(df_name="mtcars", tidy_expr="starts_with('m'), ends_with('t')", newobj="subset")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    # Verify the expression includes the helper functions
    call_expr = call_args[0][1]
    assert "selectDS" in call_expr
    assert "mtcars" in call_expr


def test_filter_basic(mock_dssession):
    """Test filtering with a simple condition."""
    client = TidyverseClient(mock_dssession)

    client.filter(df_name="mtcars", tidy_expr="mpg > 20", newobj="filtered")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    # Check that newobj and call expression are correct
    assert call_args[0][0] == "filtered"
    call_expr = call_args[0][1]
    assert "filterDS" in call_expr
    assert "mtcars" in call_expr


def test_filter_complex_condition(mock_dssession):
    """Test filtering with complex logical conditions."""
    client = TidyverseClient(mock_dssession)

    client.filter(df_name="mtcars", tidy_expr="mpg > 20 & cyl == 4", newobj="filtered")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "filterDS" in call_expr
    assert "mtcars" in call_expr


def test_mutate_new_column(mock_dssession):
    """Test creating a new column with mutate."""
    client = TidyverseClient(mock_dssession)

    client.mutate(df_name="mtcars", tidy_expr="mpg_squared = mpg ^ 2", newobj="mutated")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    # Check that newobj and call expression are correct
    assert call_args[0][0] == "mutated"
    call_expr = call_args[0][1]
    assert "mutateDS" in call_expr
    assert "mtcars" in call_expr


def test_mutate_modify_column(mock_dssession):
    """Test modifying an existing column with mutate."""
    client = TidyverseClient(mock_dssession)

    client.mutate(df_name="mtcars", tidy_expr="mpg = mpg * 1.6", newobj="converted")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "mutateDS" in call_expr
    assert "mtcars" in call_expr


def test_mutate_multiple_columns(mock_dssession):
    """Test creating multiple columns in one mutate call."""
    client = TidyverseClient(mock_dssession)

    client.mutate(df_name="mtcars", tidy_expr="mpg_kml = mpg * 0.425, hp_kw = hp * 0.746", newobj="multi_mutate")

    # Verify assign was called
    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "mutateDS" in call_expr


def test_arrange_single_column(mock_dssession):
    """Test arranging by a single column."""
    client = TidyverseClient(mock_dssession)

    client.arrange(df_name="mtcars", tidy_expr="mpg", newobj="sorted")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "sorted"
    call_expr = call_args[0][1]
    assert "arrangeDS" in call_expr
    assert "mtcars" in call_expr


def test_arrange_multiple_columns(mock_dssession):
    """Test arranging by multiple columns with desc()."""
    client = TidyverseClient(mock_dssession)

    client.arrange(df_name="mtcars", tidy_expr="desc(mpg), cyl", newobj="sorted")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "arrangeDS" in call_expr


def test_rename_single_column(mock_dssession):
    """Test renaming a single column."""
    client = TidyverseClient(mock_dssession)

    client.rename(df_name="mtcars", tidy_expr="miles_per_gallon = mpg", newobj="renamed")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "renamed"
    call_expr = call_args[0][1]
    assert "renameDS" in call_expr
    assert "mtcars" in call_expr


def test_rename_multiple_columns(mock_dssession):
    """Test renaming multiple columns."""
    client = TidyverseClient(mock_dssession)

    client.rename(df_name="mtcars", tidy_expr="miles_per_gallon = mpg, cylinders = cyl", newobj="renamed")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "renameDS" in call_expr


def test_slice_basic(mock_dssession):
    """Test slicing rows by position."""
    client = TidyverseClient(mock_dssession)

    client.slice(df_name="mtcars", tidy_expr="1, 5, 10", newobj="sliced")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "sliced"
    call_expr = call_args[0][1]
    assert "sliceDS" in call_expr
    assert "mtcars" in call_expr


def test_slice_range(mock_dssession):
    """Test slicing with a range."""
    client = TidyverseClient(mock_dssession)

    client.slice(df_name="mtcars", tidy_expr="1:10", newobj="sliced")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "sliceDS" in call_expr


def test_group_by_single_column(mock_dssession):
    """Test grouping by a single column."""
    client = TidyverseClient(mock_dssession)

    client.group_by(df_name="mtcars", tidy_expr="cyl", newobj="grouped")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "grouped"
    call_expr = call_args[0][1]
    assert "groupByDS" in call_expr
    assert "mtcars" in call_expr


def test_group_by_multiple_columns(mock_dssession):
    """Test grouping by multiple columns."""
    client = TidyverseClient(mock_dssession)

    client.group_by(df_name="mtcars", tidy_expr="cyl, gear", newobj="grouped")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "groupByDS" in call_expr


def test_ungroup(mock_dssession):
    """Test ungrouping a grouped dataframe."""
    client = TidyverseClient(mock_dssession)

    client.ungroup(df_name="grouped_mtcars", newobj="ungrouped")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "ungrouped"
    call_expr = call_args[0][1]
    assert "ungroupDS" in call_expr
    assert "grouped_mtcars" in call_expr


def test_group_keys(mock_dssession):
    """Test getting group keys from a grouped dataframe."""
    client = TidyverseClient(mock_dssession)

    client.group_keys(df_name="grouped_mtcars", newobj="keys")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "keys"
    call_expr = call_args[0][1]
    assert "groupKeysDS" in call_expr
    assert "grouped_mtcars" in call_expr


def test_distinct_all_columns(mock_dssession):
    """Test getting distinct rows across all columns."""
    client = TidyverseClient(mock_dssession)

    client.distinct(df_name="mtcars", tidy_expr=None, newobj="unique")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "unique"
    call_expr = call_args[0][1]
    assert "distinctDS" in call_expr
    assert "mtcars" in call_expr


def test_distinct_specific_columns(mock_dssession):
    """Test getting distinct rows for specific columns."""
    client = TidyverseClient(mock_dssession)

    client.distinct(df_name="mtcars", tidy_expr="cyl, gear", newobj="unique")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "distinctDS" in call_expr


def test_bind_rows(mock_dssession):
    """Test binding rows from multiple dataframes."""
    client = TidyverseClient(mock_dssession)

    client.bind_rows(df_names=["df1", "df2", "df3"], newobj="combined")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "combined"
    call_expr = call_args[0][1]
    assert "bindRowsDS" in call_expr
    assert "df1" in call_expr
    assert "df2" in call_expr
    assert "df3" in call_expr


def test_bind_cols(mock_dssession):
    """Test binding columns from multiple dataframes."""
    client = TidyverseClient(mock_dssession)

    client.bind_cols(df_names=["df1", "df2"], newobj="combined")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "combined"
    call_expr = call_args[0][1]
    assert "bindColsDS" in call_expr
    assert "df1" in call_expr
    assert "df2" in call_expr


def test_if_else_basic(mock_dssession):
    """Test basic if_else conditional."""
    client = TidyverseClient(mock_dssession)

    client.if_else(condition="mpg > 20", true_value="'high'", false_value="'low'", newobj="mpg_category")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "mpg_category"
    call_expr = call_args[0][1]
    assert "ifElseDS" in call_expr


def test_if_else_with_missing(mock_dssession):
    """Test if_else with missing value handling."""
    client = TidyverseClient(mock_dssession)

    client.if_else(condition="mpg > 20", true_value="1", false_value="0", missing_value="NA", newobj="mpg_flag")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    call_expr = call_args[0][1]
    assert "ifElseDS" in call_expr


def test_case_when_basic(mock_dssession):
    """Test basic case_when with multiple conditions."""
    client = TidyverseClient(mock_dssession)

    client.case_when(cases="mpg > 25 ~ 'excellent', mpg > 20 ~ 'good', TRUE ~ 'average'", newobj="mpg_rating")

    mock_dssession.assign_expr.assert_called_once()
    call_args = mock_dssession.assign_expr.call_args

    assert call_args[0][0] == "mpg_rating"
    call_expr = call_args[0][1]
    assert "caseWhenDS" in call_expr
