"""NYSE session calendar used to detect genuinely missing trading days.

The pandas federal holiday calendar is not interchangeable with the exchange
calendar: the NYSE closes on Good Friday but trades on Columbus Day and
Veterans Day. The rules below reproduce the regular NYSE schedule, and the
ad-hoc closure list covers the unscheduled full-day closures that fall inside
the sample period.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

# Full-day closures that are not part of the recurring schedule.
AD_HOC_CLOSURES: List[str] = [
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # National day of mourning, George H. W. Bush
    "2025-01-09",  # National day of mourning, Jimmy Carter
]


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Recurring NYSE holiday rules."""

    rules = [
        # A Saturday New Year's Day is not rolled back to the preceding Friday,
        # unlike every other exchange holiday: 31 December 2021 was a session.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2022-06-20",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def trading_sessions(start, end) -> pd.DatetimeIndex:
    """Return the expected NYSE sessions between two dates, inclusive."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    weekdays = pd.bdate_range(start=start, end=end)
    holidays = NYSEHolidayCalendar().holidays(start=start, end=end)
    ad_hoc = pd.DatetimeIndex([pd.Timestamp(d) for d in AD_HOC_CLOSURES])

    closed = holidays.union(ad_hoc)
    return weekdays.difference(closed)
