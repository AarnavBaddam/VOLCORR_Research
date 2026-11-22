"""NYSE trading calendar: regular full-day closures plus special closures listed below.
Hand-coded rules (no calendar dependency). Verify against NYSE's published calendar before the freeze."""
from datetime import date, timedelta
from functools import lru_cache

SPECIAL_CLOSURES = {date(2018, 12, 5),   # national day of mourning (President George H. W. Bush)
                    date(2025, 1, 9)}    # national day of mourning (President Carter)


def easter(y):
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(y, month, day + 1)


def nth_weekday(y, month, weekday, n):
    """n-th given weekday of a month (Monday = 0)."""
    first = date(y, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(y, month, weekday):
    nxt = date(y + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def observed(d):
    """Fixed-date holiday as NYSE observes it: Saturday -> Friday before, Sunday -> Monday after."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(y):
    h = set()
    ny = date(y, 1, 1)
    if ny.weekday() == 6:
        h.add(date(y, 1, 2))
    elif ny.weekday() != 5:  # NYSE stays open on Friday 31 Dec when 1 Jan is a Saturday
        h.add(ny)
    h.add(nth_weekday(y, 1, 0, 3))                 # Martin Luther King Jr. Day
    h.add(nth_weekday(y, 2, 0, 3))                 # Washington's Birthday
    h.add(easter(y) - timedelta(days=2))           # Good Friday
    h.add(last_weekday(y, 5, 0))                   # Memorial Day
    if y >= 2022:
        h.add(observed(date(y, 6, 19)))            # Juneteenth
    h.add(observed(date(y, 7, 4)))                 # Independence Day
    h.add(nth_weekday(y, 9, 0, 1))                 # Labor Day
    h.add(nth_weekday(y, 11, 3, 4))                # Thanksgiving
    h.add(observed(date(y, 12, 25)))               # Christmas
    return frozenset(h | {d for d in SPECIAL_CLOSURES if d.year == y})


def is_trading_day(d):
    return d.weekday() < 5 and d not in holidays(d.year)


def next_trading_day_after(d):
    """First trading day strictly after calendar date d."""
    x = d + timedelta(days=1)
    while not is_trading_day(x):
        x += timedelta(days=1)
    return x


def add_trading_days(d, n):
    """The n-th trading day after d (d itself is not counted)."""
    x = d
    for _ in range(n):
        x = next_trading_day_after(x)
    return x


def trading_days(start, end):
    """Every trading day from start to end inclusive, as dates."""
    out, x = [], start
    while x <= end:
        if is_trading_day(x):
            out.append(x)
        x += timedelta(days=1)
    return out


def last_trading_day_of_year(y):
    x = date(y, 12, 31)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x
