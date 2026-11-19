"""Research agent with web search capabilities."""

import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.config import DEFAULT_RESEARCH_AGENT_PROMPT, DEFAULT_RESEARCH_MODEL
from src.llm import get_gateway_chat_model


@tool
def get_timezone_info(timezone: str, date: str = "") -> str:
    """Deterministic IANA timezone lookup (no web search).

    Args:
        timezone: IANA zone name, e.g. "Europe/Berlin" or "America/New_York".
        date: Optional anchor date YYYY-MM-DD (noon local time); defaults to now.

    Returns the local date/time, UTC offset and DST status on that date.
    """
    try:
        zone = ZoneInfo(timezone)
        if date:
            when = datetime.strptime(date, "%Y-%m-%d").replace(hour=12, tzinfo=zone)
        else:
            when = datetime.now(zone)
    except Exception as e:
        return f"Error: {e}. Use an IANA zone name and YYYY-MM-DD date."
    offset = when.utcoffset()
    hours = offset.total_seconds() / 3600 if offset else 0
    return (
        f"zone={timezone}; local={when.isoformat()}; "
        f"utc_offset={hours:+g}h ({when.strftime('%z')}); "
        f"abbr={when.tzname()}; dst={'yes' if when.dst() else 'no'}"
    )


@tool
def get_historical_rainfall(city: str, start_year: int, end_year: int) -> str:
    """Official-style historical rainfall lookup via Open-Meteo (no web scraping).

    Args:
        city: City name, e.g. "Seattle" or "London, GB".
        start_year: First calendar year (inclusive).
        end_year: Last calendar year (inclusive); must be a completed or current year.

    Returns resolved location/coordinates, date range, and total precipitation per
    year in millimeters (mm) plus the inches equivalent, so places compare directly.
    """
    try:
        name, _, country = (p.strip() for p in (city + ",").split(",", 2))
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": name, "count": 5, "language": "en"}
        )
        with urlopen(geo_url, timeout=20) as resp:
            results = json.load(resp).get("results") or []
        country = country.strip(", ").upper()
        if country:
            results = [
                r for r in results if r.get("country_code", "").upper() == country
            ] or results
        if not results:
            return f"Error: could not geocode '{city}'."
        place = results[0]
        today = datetime.now(ZoneInfo("UTC")).date()
        if start_year > end_year or start_year > today.year:
            return (
                f"Error: invalid year range {start_year}-{end_year} "
                f"(current year is {today.year})."
            )
        end = min(datetime(end_year, 12, 31).date(), today)
        start = datetime(start_year, 1, 1).date()
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "UTC",
            }
        )
        with urlopen(url, timeout=30) as resp:
            daily = json.load(resp)["daily"]
    except Exception as e:
        return f"Error: {e}. Check city name and years."
    totals: dict[str, float] = {}
    days: dict[str, int] = {}
    for day, value in zip(daily["time"], daily["precipitation_sum"], strict=False):
        if value is not None:
            totals[day[:4]] = totals.get(day[:4], 0.0) + value
            days[day[:4]] = days.get(day[:4], 0) + 1
    lines = []
    for y, mm in sorted(totals.items()):
        year = int(y)
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        note = "" if days[y] == (366 if leap else 365) else f" PARTIAL: {days[y]} days"
        lines.append(f"{y}: {mm:.1f} mm ({mm / 25.4:.2f} in){note}")
    return (
        f"location={place['name']}, {place.get('country_code', '')} "
        f"(lat={place['latitude']}, lon={place['longitude']}); "
        f"range={start.isoformat()}..{end.isoformat()}; source=Open-Meteo archive; "
        "metric=daily precipitation_sum totals, units mm (inches given too)\n"
        + "\n".join(lines)
    )


def get_research_agent(
    system_prompt: str | None = None, model: str = DEFAULT_RESEARCH_MODEL
):
    """Create research agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use.

    This agent is specialized for:
    - Web searches and finding information online
    - Looking up current events
    - Researching topics
    - Gathering data from the internet
    """
    # Use provided prompt or fall back to default
    prompt = (
        system_prompt if system_prompt is not None else DEFAULT_RESEARCH_AGENT_PROMPT
    )

    web_search = TavilySearch(max_results=3)

    return create_agent(
        model=get_gateway_chat_model(model),
        tools=[web_search, get_timezone_info, get_historical_rainfall],
        system_prompt=prompt,
        name="ResearchAgent",
    )
