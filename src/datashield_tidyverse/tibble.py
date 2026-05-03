"""TibbleClient for tibble operations."""

import logging
from datashield import DSSession
from .utils import make_serverside_call

logger = logging.getLogger(__name__)


class TibbleClient:
    """Client for tibble operations on DataSHIELD sessions.

    This client provides Python methods that correspond to tibble functions
    in the dsTidyverseClient R package.
    """

    def __init__(self, dssession: DSSession):
        """Initialize the TibbleClient.

        Args:
            dssession: The DataSHIELD session to use for operations
        """
        self.dssession = dssession

    def as_tibble(self, df_name: str, newobj: str) -> None:
        """Convert a data frame to a tibble.

        DataSHIELD Python implementation of tibble::as_tibble.

        Args:
            df_name: Name of server-side data frame
            newobj: Name for new server-side tibble

        Example:
            >>> client.as_tibble(
            ...     df_name="mtcars",
            ...     newobj="mtcars_tibble"
            ... )
        """
        call_expr = make_serverside_call("asTibbleDS", None, [df_name])

        logger.info(f"[{self.dssession.id}] Executing as_tibble: {call_expr}")
        self.dssession.assign_expr(newobj, call_expr)
