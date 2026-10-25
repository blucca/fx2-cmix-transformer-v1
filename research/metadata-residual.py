#!/usr/bin/env python3
"""Reversible, cold-state numeric rewrite of complete PHDA9-WRT header records.

Research-only: preserve all bytes except old revision IDs (<2^24) and new
revision timestamps. @ marks a signed decimal residual. Each call starts
with empty maps and a relative page coordinate of zero; no external tables.
Only canonical username/user-ID or IP records are accepted, including the
optional preserved \xee\xcf header line before the revision marker.
"""
import bisect
import datetime
import json
import resource
import time
from pathlib import Path
import re
import sys

THRESHOLD = 1 << 24
EPOCH = datetime.date(2001, 1, 1)
RECORD = re.compile(
    rb"N([^\n]+)\n(?:\xee\xcfN[^\n]*\n)?\xdf\x99N\n\xdf\x86N([^\n]+)\n\xdf\xcdN([^\n]+)\n"
    rb"(\xdf\x8fN\n(?:\xdf\x87N[^\n]+\n\xdf\x86N([0-9]+)\n|ipN[^\n]+\n)/\xdf\x8fN\n)"
)


def decimal(raw):
    value = int(raw)
    if str(value).encode() != raw:
        raise ValueError("noncanonical decimal")
    return value


def format_time(seconds):
    if not 0 <= seconds < (datetime.date(2101, 1, 1) - EPOCH).days * 86400:
        raise ValueError("timestamp outside 2001..2100")
    day = EPOCH + datetime.timedelta(days=seconds // 86400)
    return ("%02d%dJ%d" % (day.year - 2001,
            (day.month - 1) * 31 + day.day - 1, seconds % 86400)).encode()


def parse_time(raw):
    day_raw, sec_raw = raw.split(b"J")
    if len(day_raw) < 3 or not day_raw[:2].isdigit():
        raise ValueError("timestamp needs a two-digit year and a day")
    d, sec = decimal(day_raw[2:]), decimal(sec_raw)
    if d < 0 or not 0 <= sec < 86400:
        raise ValueError("invalid timestamp day or seconds")
    day = datetime.date(int(day_raw[:2]) + 2001, d // 31 + 1, d % 31 + 1)
    seconds = (day - EPOCH).days * 86400 + sec
    if format_time(seconds) != raw:
        raise ValueError("noncanonical timestamp")
    return seconds


def round_div(n, d):
    """Integer nearest, exact halves away from zero; d is positive."""
    if d <= 0:
        raise ValueError("nonpositive interpolation denominator")
    q, rem = divmod(abs(n), d)
    q += 2 * rem >= d
    return -q if n < 0 else q


def old_predict(keys, values, page):
    j = bisect.bisect_left(keys, page)
    if not keys:
        return 0, j
    k = min((i for i in (j - 1, j) if 0 <= i < len(keys)),
            key=lambda i: (abs(keys[i] - page), keys[i]))
    return values[k] + page - keys[k], j


def clock_predict(keys, values, revision):
    j = bisect.bisect_left(keys, revision)
    if not keys:
        return 0, j
    if j == 0:
        a, b = 0, min(1, len(keys) - 1)
    elif j == len(keys):
        a, b = max(0, j - 2), j - 1
    else:
        a, b = j - 1, j
    if a == b:
        return values[a], j
    return values[a] + round_div((revision - keys[a]) * (values[b] - values[a]),
                                 keys[b] - keys[a]), j


def rewrite(data, decode=False):
    """Return bytes; fail on an incomplete or noncanonical header record."""
    if not data:
        raise ValueError("empty header")
    old_keys, old_values, new_keys, new_values = [], [], [], []
    seen_pages, seen_revisions = set(), set()
    page, cursor = 0, 0
    out = bytearray()
    while cursor < len(data):
        m = RECORD.match(data, cursor)
        if m is None:
            raise ValueError("invalid or incomplete record at byte %d" % cursor)
        delta = decimal(m[1])
        if not -(2**31 - 1) <= delta <= 2**31 - 1:
            raise ValueError("page delta outside signed 31-bit magnitude")
        page += delta
        if page in seen_pages:
            raise ValueError("duplicate page key")
        if m[5] is not None and not 0 <= decimal(m[5]) <= 2**31 - 1:
            raise ValueError("invalid contributor ID")
        r_raw, t_raw = m[2], m[3]
        is_old = r_raw.startswith(b"@") if decode else decimal(r_raw) < THRESHOLD
        if is_old:
            pred, j = old_predict(old_keys, old_values, page)
            revision = pred + decimal(r_raw[1:]) if decode else decimal(r_raw)
            if not 0 < revision < THRESHOLD:
                raise ValueError("old revision outside branch range")
            parse_time(t_raw)
            replacement = str(revision).encode() if decode else b"@" + str(revision - pred).encode()
            field = 2
        else:
            revision = decimal(r_raw)
            if not THRESHOLD <= revision <= 2**31 - 1:
                raise ValueError("new revision outside branch range")
            pred, j = clock_predict(new_keys, new_values, revision)
            if decode:
                if not t_raw.startswith(b"@"):
                    raise ValueError("new timestamp needs residual marker")
                seconds = pred + decimal(t_raw[1:])
                replacement = format_time(seconds)
            else:
                seconds = parse_time(t_raw)
                replacement = b"@" + str(seconds - pred).encode()
            field = 3
        if revision in seen_revisions:
            raise ValueError("duplicate revision key")
        seen_pages.add(page)
        seen_revisions.add(revision)
        # The current target becomes available only after its prediction.
        if is_old:
            old_keys.insert(j, page)
            old_values.insert(j, revision)
        else:
            new_keys.insert(j, revision)
            new_values.insert(j, seconds)
        out.extend(data[cursor:m.start(field)])
        out.extend(replacement)
        out.extend(data[m.end(field):m.end()])
        cursor = m.end()
    return bytes(out)


def self_test():
    def record(delta, revision, seconds):
        return (b"N" + str(delta).encode() + b"\n\xdf\x99N\n\xdf\x86N" +
                str(revision).encode() + b"\n\xdf\xcdN" + format_time(seconds) +
                b"\n\xdf\x8fN\n\xdf\x87N@test\n\xdf\x86N1\n/\xdf\x8fN\n")

    # Negative relative page coordinates, two branches, late inserts, ties.
    rows = [(10, 17000000, 100), (-20, 1000, 50), (4, 1005, 90),
            (25, 17000002, 103), (-13, 17000001, 102), (-14, 1002, 40),
            (15, 17000003, 104), (-14, 1001, 41)]
    source = b"".join(record(*row) for row in rows)
    coded = rewrite(source)
    assert rewrite(coded, decode=True) == source
    extra = source.replace(b"\n\xdf\x99N", b"\n\xee\xcfNpreserved\n\xdf\x99N", 1)
    assert rewrite(rewrite(extra), decode=True) == extra
    fields = [(m[2], m[3]) for m in RECORD.finditer(coded)]
    assert [fields[i][1] for i in (0, 3, 4, 6)] == [b"@100", b"@3", b"@0", b"@0"]
    assert [fields[i][0] for i in (1, 2, 5, 7)] == [b"@1000", b"@1", b"@0", b"@-2"]
    shifted = record(rows[0][0] + 100, *rows[0][1:]) + b"".join(record(*r) for r in rows[1:])
    assert [(m[2], m[3]) for m in RECORD.finditer(rewrite(shifted))] == fields
    assert [round_div(n, 2) for n in (3, -3, 1, -1, 0)] == [2, -2, 1, -1, 0]
    assert old_predict([0, 10], [100, 300], 5) == (105, 1)
    assert clock_predict([10, 12], [100, 103], 9)[0] == 98
    assert clock_predict([10, 12], [100, 103], 13)[0] == 105
    invalid = [b"", source[:-1], source.replace(b"N17000000", b"N017000000", 1),
               source.replace(b"\xdf\x99N", b"\xdf\x98N", 1),
               record(1, 17000000, 100) + record(0, 17000001, 101),
               record(1, 17000000, 100) + record(1, 17000000, 101),
               source.replace(format_time(100), b"0059J0", 1)]
    for bad in invalid:
        try:
            rewrite(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid header accepted")
    try:
        rewrite(coded.replace(b"@0\n", b"@-0\n", 1), decode=True)
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical residual accepted")
    print("metadata-residual self-test passed")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--self-test"]:
            self_test()
        elif len(sys.argv) == 4 and sys.argv[1] in ("encode", "decode"):
            start = time.process_time()
            data = Path(sys.argv[2]).read_bytes()
            output = rewrite(data, decode=sys.argv[1] == "decode")
            Path(sys.argv[3]).write_bytes(output)
            print(json.dumps({"mode": sys.argv[1], "input_bytes": len(data),
                              "output_bytes": len(output), "cpu_seconds": time.process_time() - start,
                              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
        else:
            raise ValueError("usage: metadata-residual.py encode|decode INPUT OUTPUT | --self-test")
    except (ValueError, OSError) as exc:
        sys.exit("metadata-residual: " + str(exc))
