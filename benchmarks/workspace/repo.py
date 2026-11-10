"""The shared workspace: a small Python package (12 modules x 4 functions) with functions left unimplemented (NotImplementedError) and one test file per
module. Each function has a reference solution (used only to validate the tests at build time and as the ideal-agent oracle)."""
from __future__ import annotations

# module -> [(func, signature, docstring, reference_body, tests)]
FUNCS: dict[str, list] = {
    "strings": [
        ("slugify", "text: str", "Lowercase, replace runs of non-alphanumeric characters with single '-', strip leading/trailing '-'.",
         '''import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")''',
         ['assert slugify("Hello, World!") == "hello-world"', 'assert slugify("  A  B  ") == "a-b"', 'assert slugify("x--y__z") == "x-y-z"']),
        ("is_palindrome", "text: str", "True if the letters and digits of `text`, ignoring case and all other characters, read the same backwards.",
         '''s = "".join(c.lower() for c in text if c.isalnum())
    return s == s[::-1]''',
         ['assert is_palindrome("A man, a plan, a canal: Panama")', 'assert not is_palindrome("hello")', 'assert is_palindrome("")']),
        ("camel_to_snake", "name: str", "Convert CamelCase/lowerCamel to snake_case: insert '_' before each uppercase letter (except a leading one) and lowercase everything.",
         '''import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()''',
         ['assert camel_to_snake("CamelCase") == "camel_case"', 'assert camel_to_snake("getHTTPResponse") == "get_h_t_t_p_response"', 'assert camel_to_snake("simple") == "simple"']),
        ("truncate", "text: str, width: int", "Return `text` unchanged if len(text) <= width, else the first width-3 characters followed by '...'. width is at least 3.",
         '''return text if len(text) <= width else text[: width - 3] + "..."''',
         ['assert truncate("hello world", 8) == "hello..."', 'assert truncate("hi", 8) == "hi"', 'assert truncate("abcdef", 6) == "abcdef"']),
    ],
    "numbers": [
        ("median", "values: list", "Median of a non-empty list of numbers (mean of the two middle values for even length); returns a float.",
         '''s = sorted(values)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2''',
         ['assert median([3, 1, 2]) == 2.0', 'assert median([4, 1, 3, 2]) == 2.5', 'assert median([7]) == 7.0']),
        ("clamp", "x, lo, hi", "Return x limited to the closed interval [lo, hi].",
         '''return lo if x < lo else hi if x > hi else x''',
         ['assert clamp(5, 0, 3) == 3', 'assert clamp(-1, 0, 3) == 0', 'assert clamp(2, 0, 3) == 2']),
        ("is_prime", "n: int", "True if n is a prime number (n < 2 is not prime).",
         '''if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True''',
         ['assert is_prime(2) and is_prime(13)', 'assert not is_prime(1) and not is_prime(15)', 'assert is_prime(97)']),
        ("gcd_list", "values: list", "Greatest common divisor of a non-empty list of positive integers.",
         '''import math
    g = values[0]
    for v in values[1:]:
        g = math.gcd(g, v)
    return g''',
         ['assert gcd_list([12, 18, 24]) == 6', 'assert gcd_list([7, 13]) == 1', 'assert gcd_list([10]) == 10']),
    ],
    "dates": [
        ("is_leap_year", "year: int", "Gregorian leap year rule: divisible by 4, except centuries unless divisible by 400.",
         '''return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)''',
         ['assert is_leap_year(2024) and is_leap_year(2000)', 'assert not is_leap_year(1900) and not is_leap_year(2023)']),
        ("days_between", "a: str, b: str", "Number of days from ISO date a to ISO date b (b - a; may be negative). Use the datetime module.",
         '''from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days''',
         ['assert days_between("2024-01-01", "2024-01-31") == 30', 'assert days_between("2024-03-01", "2024-02-28") == -2']),
        ("add_days", "d: str, n: int", "ISO date string n days after ISO date d (n may be negative).",
         '''from datetime import date, timedelta
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()''',
         ['assert add_days("2024-02-28", 2) == "2024-03-01"', 'assert add_days("2024-01-01", -1) == "2023-12-31"']),
        ("weekday_name", "d: str", "English weekday name ('Monday'..'Sunday') of ISO date d.",
         '''from datetime import date
    return date.fromisoformat(d).strftime("%A")''',
         ['assert weekday_name("2024-05-19") == "Sunday"', 'assert weekday_name("2024-01-01") == "Monday"']),
    ],
    "seqs": [
        ("chunk", "items: list, size: int", "Split items into consecutive lists of length size (the last may be shorter). size >= 1.",
         '''return [items[i:i + size] for i in range(0, len(items), size)]''',
         ['assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]', 'assert chunk([], 3) == []']),
        ("flatten", "nested: list", "Flatten one level: [[1, 2], [3]] -> [1, 2, 3].",
         '''return [x for sub in nested for x in sub]''',
         ['assert flatten([[1, 2], [3], []]) == [1, 2, 3]', 'assert flatten([]) == []']),
        ("dedupe_keep_order", "items: list", "Remove duplicates keeping the first occurrence order.",
         '''seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out''',
         ['assert dedupe_keep_order([3, 1, 3, 2, 1]) == [3, 1, 2]', 'assert dedupe_keep_order([]) == []']),
        ("running_max", "values: list", "List where element i is the maximum of values[:i+1].",
         '''out = []
    m = None
    for v in values:
        m = v if m is None or v > m else m
        out.append(m)
    return out''',
         ['assert running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]', 'assert running_max([]) == []']),
    ],
    "paths": [
        ("file_extension", "path: str", "Extension after the last '.' of the final path component, lowercased, without the dot; '' if none (dotfiles like '.bashrc' have no extension).",
         '''name = path.rsplit("/", 1)[-1]
    if "." not in name[1:]:
        return ""
    return name.rsplit(".", 1)[-1].lower()''',
         ['assert file_extension("a/b/report.PDF") == "pdf"', 'assert file_extension("README") == ""', 'assert file_extension(".bashrc") == ""']),
        ("parent_dir", "path: str", "Parent directory of a '/'-separated path ('' when there is none); no trailing slash.",
         '''return path.rsplit("/", 1)[0] if "/" in path else ""''',
         ['assert parent_dir("a/b/c.txt") == "a/b"', 'assert parent_dir("c.txt") == ""']),
        ("normalize_path", "path: str", "Collapse repeated '/' and resolve '.' and '..' components (never above the root of a relative path); no trailing slash.",
         '''parts = []
    for p in path.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/".join(parts)''',
         ['assert normalize_path("a//b/./c/../d") == "a/b/d"', 'assert normalize_path("./x/") == "x"', 'assert normalize_path("../a") == "a"']),
        ("join_paths", "*parts", "Join non-empty parts with single '/' separators, stripping leading/trailing '/' of each part.",
         '''return "/".join(p.strip("/") for p in parts if p and p.strip("/"))''',
         ['assert join_paths("a/", "/b", "c") == "a/b/c"', 'assert join_paths("a", "", "b") == "a/b"']),
    ],
    "text": [
        ("word_count", "text: str", "Number of whitespace-separated words.",
         '''return len(text.split())''',
         ['assert word_count("a b  c") == 3', 'assert word_count("") == 0']),
        ("most_common_word", "text: str", "Most frequent lowercase word (split on whitespace, punctuation .,!? stripped); ties -> the one that appears first.",
         '''words = [w.strip(".,!?").lower() for w in text.split()]
    words = [w for w in words if w]
    best, cnt = None, 0
    for w in words:
        c = words.count(w)
        if c > cnt:
            best, cnt = w, c
    return best''',
         ['assert most_common_word("the cat and the dog!") == "the"', 'assert most_common_word("a b b a") == "a"']),
        ("wrap_text", "text: str, width: int", "Greedy word wrap: list of lines, each at most width characters when a word fits; a word longer than width gets its own line.",
         '''lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines''',
         ['assert wrap_text("the quick brown fox", 9) == ["the quick", "brown fox"]', 'assert wrap_text("", 5) == []', 'assert wrap_text("abcdefghij k", 5) == ["abcdefghij", "k"]']),
        ("count_vowels", "text: str", "Number of vowels (aeiou, either case) in text.",
         '''return sum(1 for c in text.lower() if c in "aeiou")''',
         ['assert count_vowels("Hello World") == 3', 'assert count_vowels("xyz") == 0']),
    ],
    "intervals": [
        ("merge_intervals", "intervals: list", "Merge overlapping or touching closed intervals given as (start, end) tuples; return a sorted list of tuples.",
         '''out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out''',
         ['assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]', 'assert merge_intervals([(5, 6), (1, 2), (2, 3)]) == [(1, 3), (5, 6)]', 'assert merge_intervals([]) == []']),
        ("total_length", "intervals: list", "Total length covered by the closed intervals (start, end) (overlapping parts counted once).",
         '''covered, last_end = 0, None
    for s, e in sorted(intervals):
        if last_end is None or s > last_end:
            covered += e - s
            last_end = e
        elif e > last_end:
            covered += e - last_end
            last_end = e
    return covered''',
         ['assert total_length([(1, 3), (2, 6)]) == 5', 'assert total_length([(0, 1), (2, 3)]) == 2', 'assert total_length([]) == 0']),
        ("overlap", "a: tuple, b: tuple", "Length of the overlap of two closed intervals (start, end); 0 if they do not overlap.",
         '''return max(0, min(a[1], b[1]) - max(a[0], b[0]))''',
         ['assert overlap((1, 5), (3, 8)) == 2', 'assert overlap((1, 2), (3, 4)) == 0', 'assert overlap((0, 10), (2, 3)) == 1']),
        ("contains_point", "intervals: list, x", "True if x lies inside any of the closed intervals (start, end).",
         '''return any(s <= x <= e for s, e in intervals)''',
         ['assert contains_point([(1, 3), (5, 7)], 6)', 'assert not contains_point([(1, 3), (5, 7)], 4)', 'assert contains_point([(2, 2)], 2)']),
    ],
    "colors": [
        ("hex_to_rgb", "color: str", "'#rrggbb' (case-insensitive, leading '#' optional) -> (r, g, b) tuple of ints.",
         '''c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))''',
         ['assert hex_to_rgb("#FF8000") == (255, 128, 0)', 'assert hex_to_rgb("000000") == (0, 0, 0)']),
        ("rgb_to_hex", "r: int, g: int, b: int", "(r, g, b) -> lowercase '#rrggbb'.",
         '''return "#%02x%02x%02x" % (r, g, b)''',
         ['assert rgb_to_hex(255, 128, 0) == "#ff8000"', 'assert rgb_to_hex(0, 0, 0) == "#000000"']),
        ("luminance", "r: int, g: int, b: int", "Perceived luminance 0.299*r + 0.587*g + 0.114*b, rounded to 1 decimal.",
         '''return round(0.299 * r + 0.587 * g + 0.114 * b, 1)''',
         ['assert luminance(255, 255, 255) == 255.0', 'assert luminance(0, 0, 0) == 0.0', 'assert luminance(0, 0, 255) == 29.1']),
        ("invert_hex", "color: str", "Inverted color of '#rrggbb' (each channel becomes 255 - value), lowercase with '#'.",
         '''c = color.lstrip("#")
    return "#" + "".join("%02x" % (255 - int(c[i:i + 2], 16)) for i in (0, 2, 4))''',
         ['assert invert_hex("#ffffff") == "#000000"', 'assert invert_hex("#FF8000") == "#007fff"']),
    ],
    "money": [
        ("format_cents", "cents: int", "Integer cents -> '$d.dd' with thousands separators in the dollar part; negative amounts as '-$d.dd'.",
         '''sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"''',
         ['assert format_cents(123456) == "$1,234.56"', 'assert format_cents(-5) == "-$0.05"', 'assert format_cents(0) == "$0.00"']),
        ("parse_amount", "text: str", "Parse an amount like '$1,234.56', '1234.5' or '-$3' into integer cents (at most 2 decimals are given).",
         '''t = text.strip().replace(",", "").replace("$", "")
    sign = -1 if t.startswith("-") else 1
    t = t.lstrip("-+")
    whole, _, frac = t.partition(".")
    frac = (frac + "00")[:2]
    return sign * (int(whole or 0) * 100 + int(frac))''',
         ['assert parse_amount("$1,234.56") == 123456', 'assert parse_amount("1234.5") == 123450', 'assert parse_amount("-$3") == -300']),
        ("apply_discount", "cents: int, percent: float", "Price after a percentage discount, as integer cents rounded to the nearest cent (half up).",
         '''import math
    return int(math.floor(cents * (100 - percent) / 100 + 0.5))''',
         ['assert apply_discount(10000, 15) == 8500', 'assert apply_discount(999, 10) == 899', 'assert apply_discount(100, 0) == 100']),
        ("split_bill", "cents: int, n: int", "Split cents among n people as evenly as possible: a list of n integer shares whose sum is cents; the first shares get the extra cents.",
         '''q, r = divmod(cents, n)
    return [q + 1 if i < r else q for i in range(n)]''',
         ['assert split_bill(100, 3) == [34, 33, 33]', 'assert split_bill(10, 2) == [5, 5]', 'assert sum(split_bill(1001, 7)) == 1001']),
    ],
    "matrix": [
        ("transpose", "m: list", "Transpose a non-empty rectangular matrix given as a list of row lists.",
         '''return [list(r) for r in zip(*m)]''',
         ['assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]', 'assert transpose([[1]]) == [[1]]']),
        ("identity", "n: int", "n x n identity matrix as nested lists of ints.",
         '''return [[1 if i == j else 0 for j in range(n)] for i in range(n)]''',
         ['assert identity(2) == [[1, 0], [0, 1]]', 'assert identity(0) == []']),
        ("matmul", "a: list, b: list", "Matrix product of a (p x q) and b (q x r), both as nested lists.",
         '''return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]''',
         ['assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]', 'assert matmul([[1, 0]], [[2], [3]]) == [[2]]']),
        ("trace", "m: list", "Sum of the main diagonal of a square matrix (0 for an empty matrix).",
         '''return sum(m[i][i] for i in range(len(m)))''',
         ['assert trace([[1, 2], [3, 4]]) == 5', 'assert trace([]) == 0']),
    ],
    "validators": [
        ("is_valid_email", "s: str", "True if s has exactly one '@', a non-empty local part, a domain with at least two non-empty '.'-separated labels, and no whitespace.",
         '''if s.count("@") != 1 or any(c.isspace() for c in s):
        return False
    local, domain = s.split("@")
    labels = domain.split(".")
    return bool(local) and len(labels) >= 2 and all(labels)''',
         ['assert is_valid_email("a.b@example.com")', 'assert not is_valid_email("no-at.example.com")', 'assert not is_valid_email("x@domain")', 'assert not is_valid_email("x@a..b")']),
        ("luhn_check", "digits: str", "Luhn checksum validity of a non-empty string of decimal digits.",
         '''total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0''',
         ['assert luhn_check("79927398713")', 'assert not luhn_check("79927398710")', 'assert luhn_check("0")']),
        ("is_ipv4", "s: str", "True if s is a dotted-quad IPv4 address: four decimal numbers 0-255 without leading zeros (except '0' itself).",
         '''parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or (len(p) > 1 and p[0] == "0") or int(p) > 255:
            return False
    return True''',
         ['assert is_ipv4("192.168.0.1")', 'assert not is_ipv4("256.1.1.1")', 'assert not is_ipv4("01.2.3.4")', 'assert not is_ipv4("1.2.3")']),
        ("is_hex_color", "s: str", "True if s is '#' followed by exactly 6 hexadecimal digits (either case).",
         '''return len(s) == 7 and s[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in s[1:])''',
         ['assert is_hex_color("#a1B2c3")', 'assert not is_hex_color("a1b2c3")', 'assert not is_hex_color("#12345g")']),
    ],
    "stats": [
        ("mean", "values: list", "Arithmetic mean of a non-empty list, as a float.",
         '''return sum(values) / len(values)''',
         ['assert mean([1, 2, 3, 4]) == 2.5', 'assert mean([5]) == 5.0']),
        ("variance", "values: list", "Population variance (mean of squared deviations from the mean) of a non-empty list, as a float.",
         '''m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)''',
         ['assert variance([2, 4, 4, 4, 5, 5, 7, 9]) == 4.0', 'assert variance([3]) == 0.0']),
        ("mode", "values: list", "Most frequent value of a non-empty list; ties -> the value that appears first.",
         '''best, cnt = None, 0
    for v in values:
        c = values.count(v)
        if c > cnt:
            best, cnt = v, c
    return best''',
         ['assert mode([1, 2, 2, 3, 3]) == 2', 'assert mode([7]) == 7', 'assert mode(["a", "b", "b"]) == "b"']),
        ("percent_change", "old, new", "Percentage change from old to new, rounded to 1 decimal (old is non-zero).",
         '''return round((new - old) / old * 100, 1)''',
         ['assert percent_change(50, 75) == 50.0', 'assert percent_change(200, 150) == -25.0', 'assert percent_change(3, 4) == 33.3']),
    ],
}

BASE_VERSION = "0.4.0"


def module_source(mod: str, implemented: set | None = None) -> str:
    """Source of pkg/<mod>.py; functions in `implemented` get their reference body, the others raise NotImplementedError."""
    implemented = implemented or set()
    parts = [f'"""{mod} utilities."""\n']
    for func, sig, doc, body, _ in FUNCS[mod]:
        if func in implemented:
            parts.append(f"def {func}({sig}):\n    \"\"\"{doc}\"\"\"\n    {body}\n\n")
        else:
            parts.append(f"def {func}({sig}):\n    \"\"\"{doc}\"\"\"\n    raise NotImplementedError  # TODO\n\n")
    return "\n".join(parts).rstrip() + "\n"


def test_source(mod: str) -> str:
    lines = [f"from pkg.{mod} import " + ", ".join(f for f, *_ in FUNCS[mod]), ""]
    for func, _, _, _, tests in FUNCS[mod]:
        lines.append(f"\ndef test_{func}():")
        for t in tests:
            lines.append(f"    {t}")
    return "\n".join(lines) + "\n"


def pristine_files() -> dict:
    files = {"pkg/__init__.py": f'"""Shared utility package."""\n__version__ = "{BASE_VERSION}"\n',
             "CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n## 0.4.0\n- initial package skeleton\n",
             "README.md": "# pkg\nUtility functions. Run tests with the run_tests tool (one test file per module under tests/).\n"}
    for mod in FUNCS:
        files[f"pkg/{mod}.py"] = module_source(mod)
        files[f"tests/test_{mod}.py"] = test_source(mod)
    return files


def bump_patch(version: str) -> str:
    a, b, c = version.split(".")
    return f"{a}.{b}.{int(c) + 1}"
