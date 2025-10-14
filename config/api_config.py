import os


class TiingoConfig:
    """Configuration for Tiingo API."""

    def __init__(self):
        self.api_key = os.getenv('TIINGO_API_KEY')
        self.base_url = 'https://api.tiingo.com'

        if not self.api_key:
            raise ValueError(
                "TIINGO_API_KEY environment variable not set. "
                "Please set it before using the Tiingo API."
            )
