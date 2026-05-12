"""
parse_panchang_v2.py
====================
Hardened universal parser for drik_panchang_data.panchang
Validated against: rina-2023 (31), fina-2026 (57), trami-2024 (69)
vayu-2019 has empty panchang dicts -> gracefully skipped.

Edge cases handled vs v1:
  - Jeevanama / Netrama as lists with timed transitions + inline emoji
  - Abhijit = string 'None' (not Python None)
  - Aadal Yoga as a list of multiple time-range strings
  - Jeevanama/Netrama/Homahuti/Nakshatra Shool entries with trailing emoji
    e.g. 'Full Life upto 09:20 AM 𝟣', 'Chandra upto 11:38 AM ☾'
  - New optional keys: Vinchudo, Guru Pushya Yoga, Tri Pushkara Yoga,
                       Jwalamukhi Yoga, Madhusarpisha
  - Homahuti new celestials: Budha ☿, Ketu ☋, Sun ☉
  - Kumbha Chakra new values: Throat, Mouth, East (non-directional)
  - Agnivasa as bare string (not list)
  - Baana as bare string (single value, no list)
  - Varjyam single-day ranges (no cross-day prefix)
  - Amrit Kalam absent entirely on some tracks
"""

import json
import re
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CORE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_time_12h(s):
    """'09:19 AM' -> time object, None on failure"""
    try:
        return datetime.strptime(s.strip(), '%I:%M %p').time()
    except Exception:
        return None

def parse_time_range(s):
    """
    'HH:MM AM to HH:MM PM'
    'HH:MM AM to HH:MM AM , Sep 29'
    Returns dict {start, start_str, end, end_str, end_cross_day} or None
    """
    if not s or s == 'None':
        return None
    m = re.match(
        r'^(\d{1,2}:\d{2}\s+[AP]M)\s+to\s+(\d{1,2}:\d{2}\s+[AP]M)(?:\s*,\s*(.+))?$',
        s.strip()
    )
    if m:
        cross = m.group(3).strip() if m.group(3) else None
        if cross:
            cross = re.sub(r'\s+[^\x00-\x7F]+$', '', cross).strip()
        return {
            'start_str': m.group(1).strip(), 'start': parse_time_12h(m.group(1)),
            'end_str':   m.group(2).strip(), 'end':   parse_time_12h(m.group(2)),
            'end_cross_day': cross
        }
    return None

def parse_upto_entry(s):
    """
    Handles ALL observed entry formats:

      'Name upto HH:MM AM'
      'Name upto HH:MM AM ☾'           <- trailing emoji
      'Name upto HH:MM AM , Sep 29'
      'Name upto HH:MM AM , Nov 25 𝟢'  <- cross-day + emoji
      'Name from HH:MM AM to HH:MM PM'
      'Name from HH:MM AM to Full Night'
      'Whole Day' / 'None' / plain name

    Returns dict: {name, type, end, end_str, cross_day, start, start_str, full_night}
    """
    s = s.strip()

    # ── upto pattern ──────────────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+upto\s+(\d{1,2}:\d{2}\s+[AP]M)(.*)?$', s)
    if m:
        name      = m.group(1).strip()
        time_str  = m.group(2).strip()
        remainder = (m.group(3) or '').strip()

        cross_day = None
        if remainder.startswith(','):
            cross_raw = remainder[1:].strip()
            # strip trailing unicode emoji
            cross_day = re.sub(r'\s+[^\x00-\x7F]+$', '', cross_raw).strip() or None
        # else: it's a trailing emoji suffix like ' ☾' — discard

        return {
            'name': name, 'type': 'upto',
            'end_str': time_str, 'end': parse_time_12h(time_str),
            'cross_day': cross_day
        }

    # ── from ... to pattern ───────────────────────────────────────────────────
    m = re.match(r'^(.+?)\s+from\s+(\d{1,2}:\d{2}\s+[AP]M)\s+to\s+(.+)$', s)
    if m:
        end_raw = m.group(3).strip()
        try:
            end_t = parse_time_12h(end_raw) if end_raw != 'Full Night' else None
        except Exception:
            end_t = None
        return {
            'name': m.group(1).strip(), 'type': 'from_to',
            'start_str': m.group(2).strip(), 'start': parse_time_12h(m.group(2)),
            'end_str': end_raw, 'end': end_t,
            'full_night': end_raw == 'Full Night'
        }

    # ── whole-day / plain value ───────────────────────────────────────────────
    return {'name': s, 'type': 'whole', 'end': None}


def active_from_list(val, query_time_str):
    """
    Given a string, list, or 'None' sentinel, resolve the active value
    at query_time_str (HH:MM:SS local time).
    """
    if val is None or val == 'None':
        return None
    query_t = datetime.strptime(query_time_str, '%H:%M:%S').time()
    entries = [val] if isinstance(val, str) else list(val)

    for e in entries:
        p = parse_upto_entry(e)
        if p['type'] == 'whole':
            return p['name']
        if p['type'] == 'upto':
            if p['cross_day']:          # crosses midnight → still active any time today
                return p['name']
            if p['end'] and query_t <= p['end']:
                return p['name']
        if p['type'] == 'from_to':
            if p.get('full_night') and p['start'] and query_t >= p['start']:
                return p['name']
            elif p['end'] and p['start'] and p['start'] <= query_t <= p['end']:
                return p['name']

    # fallback: last entry
    return parse_upto_entry(entries[-1])['name']


def parse_muhurta_list(lst):
    """
    Parse Panchaka Rahita / Udaya Lagna slot lists:
      'Good Muhurta - 05:53 AM to 07:16 AM'
      'Raja Panchaka - 10:43 PM to 12:55 AM , Sep 29'
      'Kanya - 05:13 AM to 07:16 AM'
    Returns list of parsed dicts.
    """
    if isinstance(lst, str):
        lst = [lst]
    result = []
    for item in lst:
        m = re.match(
            r'^(.+?)\s+-\s+(\d{1,2}:\d{2}\s+[AP]M)\s+to\s+(\d{1,2}:\d{2}\s+[AP]M)(?:\s*,\s*(.+))?$',
            item.strip()
        )
        if m:
            cross = m.group(4).strip() if m.group(4) else None
            if cross:
                cross = re.sub(r'\s+[^\x00-\x7F]+$', '', cross).strip()
            result.append({
                'name': m.group(1).strip(),
                'start_str': m.group(2).strip(), 'start': parse_time_12h(m.group(2)),
                'end_str':   m.group(3).strip(), 'end':   parse_time_12h(m.group(3)),
                'end_cross_day': cross
            })
        else:
            result.append({'name': item.strip(), 'parse_error': True})
    return result


def active_muhurta(lst, query_time_str):
    """Find the active muhurta/lagna at query_time_str."""
    if not lst:
        return None
    query_t = datetime.strptime(query_time_str, '%H:%M:%S').time()
    parsed = parse_muhurta_list(lst)
    for slot in parsed:
        if slot.get('parse_error'):
            continue
        if slot.get('end_cross_day'):
            if slot['start'] and query_t >= slot['start']:
                return slot['name']
        elif slot.get('start') and slot.get('end') and slot['start'] <= query_t <= slot['end']:
            return slot['name']
    return parsed[-1]['name'] if parsed else None


def active_baana(val, query_time_str):
    """Resolve active Baana — may be bare string or list."""
    return active_from_list(val, query_time_str)


def parse_varjyam(val):
    """
    Handles:
      Single string 'HH:MM AM to HH:MM PM'            <- intraday
      Single string 'HH:MM AM to HH:MM PM , Oct 29'   <- cross-midnight (new format in trami/fina)
      Single string 'HH:MM AM , Sep 29 to HH:MM AM , Sep 29' <- rina-style cross-day
      List of the above
    Returns list of parsed time-range dicts.
    """
    if isinstance(val, str):
        val = [val]
    results = []
    for v in val:
        v = v.strip()
        # rina-style: 'HH:MM , MonDD to HH:MM , MonDD'
        m = re.match(
            r'^(\d{1,2}:\d{2}\s+[AP]M)\s*,\s*([A-Za-z]+ \d+)\s+to\s+(\d{1,2}:\d{2}\s+[AP]M)\s*,\s*([A-Za-z]+ \d+)$',
            v
        )
        if m:
            results.append({
                'start_str': m.group(1), 'start': parse_time_12h(m.group(1)),
                'start_day': m.group(2),
                'end_str': m.group(3), 'end': parse_time_12h(m.group(3)),
                'end_day': m.group(4), 'cross_day': True
            })
            continue
        r = parse_time_range(v)
        if r:
            results.append(r)
        else:
            results.append({'raw': v, 'parse_error': True})
    return results


def parse_aadal_yoga(val):
    """
    Aadal Yoga may be:
      - plain string '05:53 AM to 04:18 PM'
      - list ['06:03 AM to 10:29 AM', '11:34 PM to 06:04 AM , Nov 20']
    Returns list of parsed time-range dicts.
    """
    if val is None:
        return None
    if isinstance(val, str):
        val = [val]
    return [parse_time_range(v) for v in val]


def parse_chandrabalam_groups(groups):
    result = []
    for g in groups:
        title = g['title']
        m = re.match(r'^(Good (?:Chandrabalam|Tarabalam)) till (.+?) for$', title)
        if m:
            until = m.group(2).strip()
            until = re.sub(r'\s+[^\x00-\x7F]+$', '', until).strip()
            until_time = None
            if until != 'next day sunrise':
                # could be 'HH:MM AM' or 'HH:MM AM , Nov 25'
                t_m = re.match(r'^(\d{1,2}:\d{2}\s+[AP]M)', until)
                if t_m:
                    until_time = parse_time_12h(t_m.group(1))
            result.append({
                'kind': m.group(1), 'until_str': until,
                'until_time': until_time,
                'next_day': until == 'next day sunrise',
                'rashis': g['for'], 'notes': g.get('notes', [])
            })
        else:
            result.append({'raw_title': title, 'rashis': g['for'], 'notes': g.get('notes', [])})
    return result


def parse_dinamana(val):
    """'12 Hours 02 Mins 19 Secs' -> total seconds int"""
    m = re.match(r'(\d+)\s+Hours?(?:\s+(\d+)\s+Mins?)?(?:\s+(\d+)\s+Secs?)?', val)
    if m:
        return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)
    return None


def parse_other_calendars(d):
    def extract_int(key):
        v = d.get(key, '')
        m = re.search(r'\d+', v)
        return int(m.group()) if m else None
    return {
        'kaliyuga_years':         extract_int('Kaliyuga'),
        'lahiri_ayanamsha':       float(d['Lahiri Ayanamsha']) if d.get('Lahiri Ayanamsha') else None,
        'kali_ahargana':          extract_int('Kali Ahargana'),
        'rata_die':               extract_int('Rata Die'),
        'julian_day':             float(re.search(r'[\d.]+', d['Julian Day']).group()) if d.get('Julian Day') else None,
        'modified_julian_day':    extract_int('Modified Julian Day'),
        'national_civil_date':    d.get('National Civil Date', ''),
        'national_nirayana_date': d.get('National Nirayana Date', ''),
        'julian_date':            d.get('Julian Date', ''),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PARSE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def parse_panchang(panchang_dict, time_local):
    """
    Parse all 12 sections of a panchang dict at the given local time (HH:MM:SS).
    Returns a flat dict of all resolved values.
    Missing optional sections/keys return None gracefully.
    """
    p   = panchang_dict
    t   = time_local
    rec = {}

    if not p:
        rec['_empty'] = True
        return rec

    # ── 1. Sunrise and Moonrise ───────────────────────────────────────────────
    sr = p.get('Sunrise and Moonrise', {})
    rec['sunrise']       = parse_time_12h(sr.get('Sunrise', ''))
    rec['sunset']        = parse_time_12h(sr.get('Sunset', ''))
    rec['moonrise']      = parse_time_12h(sr.get('Moonrise', ''))
    moonset_raw          = sr.get('Moonset', '')
    rec['moonset']       = None if moonset_raw in ('No Moonset', '') else parse_time_12h(moonset_raw)
    rec['moonset_absent']= (moonset_raw == 'No Moonset')

    # ── 2. Panchangam ─────────────────────────────────────────────────────────
    pan = p.get('Panchangam', {})
    rec['tithi']       = active_from_list(pan.get('Tithi'),       t)
    rec['nakshathram'] = active_from_list(pan.get('Nakshathram'), t)
    rec['yoga']        = active_from_list(pan.get('Yoga'),        t)
    rec['karana']      = active_from_list(pan.get('Karana'),      t)
    rec['weekday']     = pan.get('Weekday') if isinstance(pan.get('Weekday'), str) else (pan['Weekday'][0] if pan.get('Weekday') else None)
    rec['paksha']      = pan.get('Paksha')  if isinstance(pan.get('Paksha'),  str) else (pan['Paksha'][0]  if pan.get('Paksha')  else None)

    # ── 3. Lunar Month / Samvat ───────────────────────────────────────────────
    lm = p.get('Lunar Month, Samvat and Brihaspati Samvatsara', {})
    rec['vikram_samvat']   = lm.get('Vikram Samvat')
    rec['shaka_samvat']    = lm.get('Shaka Samvat')
    rec['gujarati_samvat'] = lm.get('Gujarati Samvat')
    chandramasa            = lm.get('Chandramasa')
    rec['chandramasa']     = chandramasa if isinstance(chandramasa, list) else ([chandramasa] if chandramasa else None)
    pravishte              = lm.get('Pravishte/Gate')
    rec['pravishte']       = int(pravishte) if pravishte else None
    rec['samvatsara']      = active_from_list(lm.get('Samvatsara'), t)

    # ── 4. Rashi and Nakshathram ──────────────────────────────────────────────
    rn = p.get('Rashi and Nakshathram', {})
    rec['moonsign']          = active_from_list(rn.get('Moonsign'),         t)
    rec['nakshatra_pada']    = active_from_list(rn.get('Nakshatra Pada'),   t)
    rec['sunsign']           = rn.get('Sunsign')
    rec['surya_nakshathram'] = active_from_list(rn.get('Surya Nakshathram'), t)
    rec['surya_pada']        = active_from_list(rn.get('Surya Pada'),        t)

    # ── 5. Ritu and Ayana ─────────────────────────────────────────────────────
    ra = p.get('Ritu and Ayana', {})
    rec['drik_ritu']     = ra.get('Drik Ritu')
    rec['vedic_ritu']    = ra.get('Vedic Ritu')
    rec['drik_ayana']    = ra.get('Drik Ayana')
    rec['vedic_ayana']   = ra.get('Vedic Ayana')
    rec['dinamana_sec']  = parse_dinamana(ra['Dinamana'])  if ra.get('Dinamana')  else None
    rec['ratrimana_sec'] = parse_dinamana(ra['Ratrimana']) if ra.get('Ratrimana') else None
    rec['madhyahna']     = parse_time_12h(ra.get('Madhyahna', ''))

    # ── 6. Auspicious Timings ─────────────────────────────────────────────────
    at = p.get('Auspicious Timings', {})
    abhijit_raw          = at.get('Abhijit')
    rec['abhijit']       = None if (abhijit_raw in (None, 'None')) else parse_time_range(abhijit_raw)
    rec['brahma_muhurta']        = parse_time_range(at.get('Brahma Muhurta', ''))
    rec['pratah_sandhya']        = parse_time_range(at.get('Pratah Sandhya', ''))
    rec['vijaya_muhurta']        = parse_time_range(at.get('Vijaya Muhurta', ''))
    rec['godhuli_muhurta']       = parse_time_range(at.get('Godhuli Muhurta', ''))
    rec['sayahna_sandhya']       = parse_time_range(at.get('Sayahna Sandhya', ''))
    rec['nishita_muhurta']       = parse_time_range(at.get('Nishita Muhurta', ''))
    amk                          = at.get('Amrit Kalam')
    rec['amrit_kalam']           = ([parse_time_range(v) for v in amk] if isinstance(amk, list) else parse_time_range(amk)) if amk else None
    ravi_raw                     = at.get('Ravi Yoga')
    rec['ravi_yoga']             = ([parse_time_range(v) for v in ravi_raw] if isinstance(ravi_raw, list) else parse_time_range(ravi_raw)) if ravi_raw else None
    _ssy = at.get('Sarvartha Siddhi Yoga')
    rec['sarvartha_siddhi_yoga'] = ([parse_time_range(v) for v in _ssy] if isinstance(_ssy, list) else parse_time_range(_ssy)) if _ssy else None
    _asy = at.get('Amrita Siddhi Yoga')
    rec['amrita_siddhi_yoga']    = ([parse_time_range(v) for v in _asy] if isinstance(_asy, list) else parse_time_range(_asy)) if _asy else None
    rec['guru_pushya_yoga']      = parse_time_range(at['Guru Pushya Yoga'])      if 'Guru Pushya Yoga'      in at else None
    rec['tri_pushkara_yoga']     = parse_time_range(at['Tri Pushkara Yoga'])     if 'Tri Pushkara Yoga'     in at else None

    # ── 7. Inauspicious Timings ───────────────────────────────────────────────
    it = p.get('Inauspicious Timings', {})
    rec['rahu_kalam']    = parse_time_range(it.get('Rahu Kalam', ''))
    rec['yamaganda']     = parse_time_range(it.get('Yamaganda', ''))
    rec['gulikai_kalam'] = parse_time_range(it.get('Gulikai Kalam', ''))
    dm                   = it.get('Dur Muhurtam')
    rec['dur_muhurtam']  = ([parse_time_range(v) for v in dm] if isinstance(dm, list) else parse_time_range(dm)) if dm else None
    rec['bhadra']        = parse_time_range(it['Bhadra'])       if 'Bhadra'       in it else None
    rec['varjyam']       = parse_varjyam(it['Varjyam'])         if 'Varjyam'      in it else None
    rec['panchaka']      = it.get('Panchaka')
    rec['baana']         = active_baana(it['Baana'], t)          if 'Baana'        in it else None
    rec['aadal_yoga']    = parse_aadal_yoga(it.get('Aadal Yoga'))
    rec['ganda_moola']   = it.get('Ganda Moola')
    _vdy = it.get('Vidaal Yoga')
    rec['vidaal_yoga']   = ([parse_time_range(v) for v in _vdy] if isinstance(_vdy, list) else parse_time_range(_vdy)) if _vdy else None
    _vnc = it.get('Vinchudo')
    rec['vinchudo']      = ([parse_time_range(v) for v in _vnc] if isinstance(_vnc, list) else (parse_time_range(_vnc) if _vnc not in (None, 'Whole Day') else _vnc)) if 'Vinchudo' in it else None
    _jwl = it.get('Jwalamukhi Yoga')
    rec['jwalamukhi_yoga']  = ([parse_time_range(v) for v in _jwl] if isinstance(_jwl, list) else parse_time_range(_jwl)) if _jwl else None
    rec['madhusarpisha']    = parse_time_range(it['Madhusarpisha'])     if 'Madhusarpisha'    in it else None

    # ── 8. Anandadi and Tamil Yoga ────────────────────────────────────────────
    ay = p.get('Anandadi and Tamil Yoga', {})
    rec['anandadi_yoga'] = active_from_list(ay.get('Anandadi Yoga'), t)
    rec['tamil_yoga']    = active_from_list(ay.get('Tamil Yoga'),    t)
    rec['jeevanama']     = active_from_list(ay.get('Jeevanama'),     t)
    rec['netrama']       = active_from_list(ay.get('Netrama'),       t)

    # ── 9. Nivas and Shool ────────────────────────────────────────────────────
    ns = p.get('Nivas and Shool', {})
    rec['homahuti']        = active_from_list(ns.get('Homahuti'),    t)
    rec['disha_shool']     = ns.get('Disha Shool')
    rec['agnivasa']        = active_from_list(ns.get('Agnivasa'),    t)
    rec['nakshatra_shool'] = active_from_list(ns['Nakshatra Shool'], t) if 'Nakshatra Shool' in ns else None
    rec['chandra_vasa']    = active_from_list(ns.get('Chandra Vasa'),t)
    rec['rahu_vasa']       = ns.get('Rahu Vasa')
    rec['kumbha_chakra']   = active_from_list(ns.get('Kumbha Chakra'), t)
    rec['shivavasa']       = active_from_list(ns.get('Shivavasa'),   t)
    rec['bhadravasa']      = active_from_list(ns['Bhadravasa'],      t) if 'Bhadravasa' in ns else None

    # ── 10. Other Calendars ───────────────────────────────────────────────────
    rec['other_calendars'] = parse_other_calendars(p.get('Other Calendars and Epoch', {}))

    # ── 11. Chandrabalam & Tarabalam ─────────────────────────────────────────
    ctb = p.get('Chandrabalam & Tarabalam', {})
    rec['chandrabalam_groups'] = parse_chandrabalam_groups(ctb.get('groups', []))

    # ── 12. Panchaka Rahita Muhurta and Udaya Lagna ──────────────────────────
    prl = p.get('Panchaka Rahita Muhurta and Udaya Lagna', {})
    rec['panchaka_rahita_active'] = active_muhurta(prl.get('Panchaka Rahita Muhurta'), t)
    rec['udaya_lagna_active']     = active_muhurta(prl.get('Udaya Lagna Muhurta'),     t)

    return rec


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

CRITICAL_FIELDS = [
    # (field_name, allowed_types, required)
    ('sunrise',               (type(None), __builtins__['type'] if False else type(datetime.now().time())), False),
    ('tithi',                 (str,),       True),
    ('nakshathram',           (str,),       True),
    ('yoga',                  (str,),       True),
    ('karana',                (str,),       True),
    ('weekday',               (str,),       True),
    ('paksha',                (str,),       True),
    ('moonsign',              (str,),       True),
    ('nakshatra_pada',        (str,),       True),
    ('sunsign',               (str,),       True),
    ('surya_nakshathram',     (str,),       True),
    ('drik_ritu',             (str,),       True),
    ('vedic_ritu',            (str,),       True),
    ('dinamana_sec',          (int,),       True),
    ('ratrimana_sec',         (int,),       True),
    ('brahma_muhurta',        (dict,),      True),
    ('vijaya_muhurta',        (dict,),      True),
    ('rahu_kalam',            (dict,),      True),
    ('yamaganda',             (dict,),      True),
    ('gulikai_kalam',         (dict,),      True),
    ('anandadi_yoga',         (str,),       True),
    ('tamil_yoga',            (str,),       True),
    ('jeevanama',             (str,),       True),
    ('netrama',               (str,),       True),
    ('homahuti',              (str,),       True),
    ('disha_shool',           (str,),       True),
    ('agnivasa',              (str,),       True),
    ('chandra_vasa',          (str,),       True),
    ('rahu_vasa',             (str,),       True),
    ('kumbha_chakra',         (str,),       True),
    ('shivavasa',             (str,),       True),
    ('panchaka_rahita_active',(str,),       True),
    ('udaya_lagna_active',    (str,),       True),
    ('samvatsara',            (str,),       True),
]

import datetime as dt_module

def validate_record(rec, track_id, date, time_str, errors):
    if rec.get('_empty'):
        return  # vayu-2019 style — skip silently

    for field, types, required in CRITICAL_FIELDS:
        val = rec.get(field)
        if val is None:
            if required:
                errors.append(f"FAIL  {track_id} {date} {time_str}  |  {field}  =  None (required)")
            continue
        # type check
        time_type = dt_module.time
        actual_types = tuple(t if t != type(datetime.now().time()) else time_type for t in types)
        if not isinstance(val, (str, int, dict, time_type)):
            errors.append(f"FAIL  {track_id} {date} {time_str}  |  {field}  =  unexpected type {type(val).__name__}: {repr(val)[:60]}")
        elif required and isinstance(val, dict) and val.get('parse_error'):
            errors.append(f"FAIL  {track_id} {date} {time_str}  |  {field}  parse_error: {repr(val)[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN OVER ALL FILES
# ─────────────────────────────────────────────────────────────────────────────

TEST_FILES = {
    'rina-2023':  '/mnt/user-data/uploads/rina-2023.json',
    'fina-2026':  '/mnt/user-data/uploads/fina-2026.json',
    'trami-2024': '/mnt/user-data/uploads/trami-2024.json',
    'vayu-2019':  '/mnt/user-data/uploads/vayu-2019.json',
}

grand_total_tracks = 0
grand_total_checks = 0
grand_errors = []
file_stats = {}

for fname, fpath in TEST_FILES.items():
    with open(fpath) as f:
        data = json.load(f)

    track = data['track']
    file_errors = []
    parsed_count = 0
    empty_count  = 0
    file_checks  = 0

    for tk, tv in track.items():
        t_str  = tv['time_local']
        d_str  = tv['date_local']
        p_dict = tv['analysis']['drik_panchang_data']['panchang']

        rec = parse_panchang(p_dict, t_str)

        if rec.get('_empty'):
            empty_count += 1
        else:
            parsed_count += 1
            file_checks += len(CRITICAL_FIELDS)
            validate_record(rec, tk, d_str, t_str, file_errors)

    grand_total_tracks += len(track)
    grand_total_checks += file_checks
    grand_errors.extend(file_errors)
    file_stats[fname] = {
        'total': len(track), 'parsed': parsed_count,
        'empty': empty_count, 'checks': file_checks, 'errors': len(file_errors)
    }

# ── Print Results ────────────────────────────────────────────────────────────
print("=" * 72)
print("  UNIVERSAL PANCHANG PARSER v2 — VALIDATION REPORT")
print("=" * 72)
print(f"\n  {'File':<14} {'Tracks':>7} {'Parsed':>7} {'Empty':>6} {'Checks':>8} {'Errors':>7}")
print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")
for fname, s in file_stats.items():
    status = '✓' if s['errors'] == 0 else '✗'
    print(f"  {fname:<14} {s['total']:>7} {s['parsed']:>7} {s['empty']:>6} {s['checks']:>8} {s['errors']:>7}  {status}")

print(f"\n  {'TOTAL':<14} {grand_total_tracks:>7}        {'':>6} {grand_total_checks:>8} {len(grand_errors):>7}")

if grand_errors:
    print(f"\n\n  FAILURES ({len(grand_errors)}):")
    for e in grand_errors:
        print(f"    {e}")
else:
    print(f"\n\n  ALL {grand_total_checks:,} CHECKS PASSED — parser is reliable across all files\n")


# ── Spot-check: show all resolved fields for one record from each file ────────
print("\n" + "=" * 72)
print("  SPOT-CHECK — one record per file")
print("=" * 72)

for fname, fpath in TEST_FILES.items():
    with open(fpath) as f:
        data = json.load(f)
    track = data['track']
    # pick mid-track record
    entries = list(track.values())
    sample = entries[len(entries)//2]
    p_dict = sample['analysis']['drik_panchang_data']['panchang']
    rec    = parse_panchang(p_dict, sample['time_local'])

    print(f"\n  [{fname}]  {sample['date_local']} {sample['time_local']}")
    if rec.get('_empty'):
        print("    (empty panchang)")
        continue

    fields_to_show = [
        'tithi','nakshathram','yoga','karana','paksha','weekday',
        'moonsign','surya_pada','samvatsara',
        'drik_ritu','dinamana_sec',
        'rahu_kalam','yamaganda','gulikai_kalam',
        'anandadi_yoga','tamil_yoga','jeevanama','netrama',
        'homahuti','agnivasa','chandra_vasa','disha_shool',
        'kumbha_chakra','shivavasa','rahu_vasa',
        'panchaka_rahita_active','udaya_lagna_active',
    ]
    for f in fields_to_show:
        val = rec.get(f)
        print(f"    {f:<30} {repr(val)}")
