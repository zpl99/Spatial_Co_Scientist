import os
from datetime import datetime
from datetime import timezone as tz
from typing import Any
from zoneinfo import ZoneInfo


def current_time(timezone: str = None) -> str:
    """Get the current time in ISO 8601 format.

    This tool returns the current date and time in ISO 8601 format (e.g., 2023-04-15T14:32:16.123456+00:00)
    for the specified timezone. If no timezone is provided, the value from the DEFAULT_TIMEZONE
    environment variable is used (default is 'UTC' if not set).

    :param timezone: The timezone to use (e.g., 'UTC', 'US/Pacific', 'Europe/London', 'Asia/Tokyo'). Defaults to environment variable DEFAULT_TIMEZONE ('UTC' if not set).
    :return: The current time in ISO 8601 format.

    Examples:
        >>> current_time()  # Returns current time in default timezone (from DEFAULT_TIMEZONE or UTC)
        'Current time in UTC is: 2023-04-15T14:32:16.123456+00:00'

        >>> current_time(timezone="US/Pacific")  # Returns current time in Pacific timezone
        'Current time in US/Pacific is: 2023-04-15T07:32:16.123456-07:00'
    """
    # Get environment variables at runtime
    default_timezone = os.getenv("DEFAULT_TIMEZONE", "UTC")

    # Use provided timezone or fall back to default
    timezone = timezone or default_timezone

    try:
        if timezone.upper() == "UTC":
            timezone_obj: Any = tz.utc
        else:
            timezone_obj = ZoneInfo(timezone)

        iso_format = datetime.now(timezone_obj).isoformat()
        return f"Current time in {timezone} is: {iso_format}"
    except Exception as e:
        return f"Error getting current time: {str(e)}"
