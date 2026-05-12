# drik_parser.py

from bs4 import BeautifulSoup
from datetime import datetime
import re


def clean_text(element):
    if not element:
        return ""
    text = element.get_text(" ", strip=True)
    return text.replace("ⓘ", "").strip()


def extract_date(soup):
    date_el = soup.select_one(".dpPageShortTitle")
    if not date_el:
        return None

    raw_date = clean_text(date_el)

    try:
        parsed = datetime.strptime(raw_date, "%B %d, %Y")
        return parsed.strftime("%d/%m/%Y")
    except ValueError:
        return raw_date


def parse_grid_card(card_wrapper):
    result = {}
    rows = card_wrapper.select(".dpTableRow")

    if not rows:
        return result

    last_keys = {}

    for row in rows:
        cells = row.select(".dpTableCell")

        for col_index, cell in enumerate(cells):
            classes = cell.get("class", [])
            text = clean_text(cell)

            if not text:
                continue

            if "dpTableKey" in classes:
                key = text
                value = ""

                if col_index + 1 < len(cells):
                    next_cell = cells[col_index + 1]
                    if "dpTableValue" in next_cell.get("class", []):
                        value = clean_text(next_cell)

                result[key] = value
                last_keys[col_index] = key

            elif "dpTableValue" in classes:
                key = last_keys.get(col_index - 1)

                if key:
                    existing = result.get(key)

                    if isinstance(existing, list):
                        if text not in existing:
                            existing.append(text)
                    else:
                        if existing != text:
                            result[key] = [existing, text]

    return result


def parse_strength_card(card_wrapper):
    groups = []

    strength_groups = card_wrapper.select(".dpStrengthGroup")

    for group in strength_groups:
        title_el = group.select_one(".dpMuhurtaTitle")
        title = clean_text(title_el)

        rashi_list = [
            clean_text(span)
            for span in group.select(".dpRashiCell")
        ]

        notes = [
            clean_text(note)
            for note in group.select(".dpSmallText")
        ]

        group_data = {
            "title": title,
            "for": rashi_list
        }

        if notes:
            group_data["notes"] = notes

        groups.append(group_data)

    return {"groups": groups}


def parse_muhurta_card(card_wrapper):
    result = {}

    sections = card_wrapper.select(".dpTableCell.dpTableValue")

    for section in sections:
        title_el = section.select_one(".dpTitle")
        if not title_el:
            continue

        title = clean_text(title_el)
        title = title.replace(" for the day", "").strip()

        entries = [
            clean_text(cell)
            for cell in section.select(".dpPanchangMuhurtaCell")
        ]

        result[title] = entries

    return result


def extract_cards(html, date_string=None):
    soup = BeautifulSoup(html, "html.parser")
    page_date = date_string if date_string else extract_date(soup)
    cards_output = {}

    # ❌ FIX: Use '*=' to catch classes containing 'CardWrapper' globally
    all_wrappers = soup.select("div[class*='CardWrapper']")

    for wrapper in all_wrappers:
        title_el = wrapper.select_one(".dpTableCardTitle")
        if not title_el:
            continue

        title = clean_text(title_el)

        if wrapper.select(".dpStrengthGroup"):
            cards_output[title] = parse_strength_card(wrapper)

        elif wrapper.select(".dpPanchangMuhurtaCell"):
            cards_output[title] = parse_muhurta_card(wrapper)

        elif wrapper.select(".dpTableKey"):
            cards_output[title] = parse_grid_card(wrapper)

    return {
        "date": page_date,
        "cards": cards_output
    }


def extract_tamil_panchang(html, date_string=None):
    soup = BeautifulSoup(html, "html.parser")
    page_date = date_string if date_string else extract_date(soup)
    cards_output = {}

    all_wrappers = soup.select("div[class*='CardWrapper']")

    for wrapper in all_wrappers:
        title_el = wrapper.select_one(".dpTableCardTitle")
        if not title_el:
            continue

        title = clean_text(title_el)

        if wrapper.select(".dpStrengthGroup"):
            cards_output[title] = parse_strength_card(wrapper)

        elif wrapper.select(".dpPanchangMuhurtaCell"):
            cards_output[title] = parse_muhurta_card(wrapper)

        elif wrapper.select(".dpTableKey"):
            cards_output[title] = parse_grid_card(wrapper)

    return {
        "date": page_date,
        "cards": cards_output
    }


def extract_planetary_table(html, date_string=None):
    """
    Extracts planetary positions by iterating through row elements 
    to ensure RA, Declination, and Speed are mapped correctly.
    """
    soup = BeautifulSoup(html, "html.parser")
    page_date = date_string if date_string else extract_date(soup)
    planets_data = {}

    # Target all data rows in the table
    rows = soup.select(".dpPlanetTable .dpPlanetCardContent")
    
    for row in rows:
        # Skip the header row
        if row.select_one(".dpTitleCell"):
            continue

        cells = row.find_all("div", recursive=False)
        if len(cells) < 10:
            continue

        # Extract name and clean it (removes 🔥 or ↺ symbols)
        name = clean_text(cells[0])
        
        # Mapping based on the exact HTML order provided:
        # [0] Planet | [1] Longitude | [2] Nakshatra | [3] Padam | [4] Lord 
        # [5] Full Degree | [6] Latitude | [7] Speed | [8] RA | [9] Declination
        p_data = {
            "longitude": clean_text(cells[1]),
            "nakshatra": clean_text(cells[2]),
            "padam": int(clean_text(cells[3])) if clean_text(cells[3]).isdigit() else None,
            "nakshatra_lord_sub_lord": clean_text(cells[4]),
            "full_degree": float(clean_text(cells[5])) if clean_text(cells[5]) else None,
            "latitude": clean_text(cells[6]),
            "speed_deg_per_day": float(clean_text(cells[7])) if clean_text(cells[7]) else None,
            "right_ascension": float(clean_text(cells[8])) if clean_text(cells[8]) else None,
            "declination": float(clean_text(cells[9])) if clean_text(cells[9]) else None
        }

        planets_data[name] = p_data

    return {
        "date": page_date,
        "planetary_positions": planets_data if planets_data else {"error": "No data found"}
    }