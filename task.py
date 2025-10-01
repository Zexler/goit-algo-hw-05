from __future__ import annotations

import timeit
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


SearchFunction = Callable[[str, str], int]


def boyer_moore_search(text: str, pattern: str) -> int:
    """Return the index of the first occurrence of pattern in text or -1."""
    m = len(pattern)
    n = len(text)
    if m == 0:
        return 0
    if m > n:
        return -1

    skip: Dict[str, int] = {char: idx for idx, char in enumerate(pattern)}
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            return i
        mismatch_char = text[i + j]
        i += max(1, j - skip.get(mismatch_char, -1))
    return -1


def _build_kmp_lps(pattern: str) -> List[int]:
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def knuth_morris_pratt_search(text: str, pattern: str) -> int:
    """Return the index of the first occurrence of pattern in text or -1."""
    m = len(pattern)
    n = len(text)
    if m == 0:
        return 0
    if m > n:
        return -1

    lps = _build_kmp_lps(pattern)
    i = 0  # index for text
    j = 0  # index for pattern
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                return i - j
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


def rabin_karp_search(text: str, pattern: str, *, base: int = 256, modulus: int = 1_000_000_007) -> int:
    """Return the index of the first occurrence of pattern in text or -1."""
    m = len(pattern)
    n = len(text)
    if m == 0:
        return 0
    if m > n:
        return -1

    high_order_multiplier = pow(base, m - 1, modulus)
    pattern_hash = 0
    window_hash = 0

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % modulus
        window_hash = (window_hash * base + ord(text[i])) % modulus

    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            if text[i : i + m] == pattern:
                return i
        if i < n - m:
            window_hash = (window_hash - ord(text[i]) * high_order_multiplier) % modulus
            window_hash = (window_hash * base + ord(text[i + m])) % modulus
            window_hash %= modulus
    return -1


def read_texts() -> Dict[str, str]:
    base_dir = Path(__file__).resolve().parent
    paths = {
        "Стаття 1": base_dir / "стаття-1.txt",
        "Стаття 2": base_dir / "стаття-2.txt",
    }
    texts = {}
    for label, path in paths.items():
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "windows-1251"):
            try:
                texts[label] = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            texts[label] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def choose_patterns(texts: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    selected: Dict[str, Tuple[str, str]] = {}
    for idx, (label, text) in enumerate(texts.items(), start=1):
        length = max(1, min(80, max(len(text) // 25, 1)))
        start = min(len(text) // 3, max(0, len(text) - length - 1))
        existing = text[start : start + length] or text

        fake_base = f"вигаданий_рядок_{idx}_"
        fake = fake_base
        while fake in text:
            fake += "X"

        selected[label] = (existing, fake)
    return selected


def measure_algorithm(algorithm: SearchFunction, text: str, pattern: str, *, iterations: int = 20) -> float:
    total_time = timeit.timeit(lambda: algorithm(text, pattern), number=iterations)
    return total_time / iterations


def render_table(title: str, rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return title
    col_widths: List[int] = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    divider = "  " + "-+-".join("-" * width for width in col_widths)
    formatted_rows: List[str] = []
    for idx, row in enumerate(rows):
        padded = [cell.ljust(col_widths[col]) for col, cell in enumerate(row)]
        formatted_rows.append("  " + " | ".join(padded))
        if idx == 0:
            formatted_rows.append(divider)
    return "\n".join([title, *formatted_rows])


def main() -> None:
    algorithms: Dict[str, SearchFunction] = {
        "Бойера-Мура": boyer_moore_search,
        "Кнута-Морріса-Пратта": knuth_morris_pratt_search,
        "Рабіна-Карпа": rabin_karp_search,
    }

    texts = read_texts()
    pattern_choices = choose_patterns(texts)

    per_text_totals: Dict[str, Dict[str, float]] = {
        label: {name: 0.0 for name in algorithms} for label in texts
    }
    overall_totals: Dict[str, float] = {name: 0.0 for name in algorithms}

    sections: List[str] = []
    for label, text in texts.items():
        existing, fake = pattern_choices[label]
        rows: List[List[str]] = [["Підрядок", *algorithms.keys()]]
        for pattern_label, pattern in (("Існуючий", existing), ("Вигаданий", fake)):
            row = [f"{pattern_label} ({len(pattern)})"]
            for algo_name, algo_fn in algorithms.items():
                avg_time = measure_algorithm(algo_fn, text, pattern)
                per_text_totals[label][algo_name] += avg_time
                overall_totals[algo_name] += avg_time
                row.append(f"{avg_time * 1_000_000:.2f}")
            rows.append(row)
        sections.append(render_table(f"Текст: {label}", rows))
        best = min(per_text_totals[label], key=per_text_totals[label].get)
        sections.append(f"  Найшвидший алгоритм для тексту: {best}")

    summary_rows: List[List[str]] = [["Алгоритм", "Сумарний середній час, мкс"]]
    for name in sorted(overall_totals, key=overall_totals.get):
        summary_rows.append([name, f"{overall_totals[name] * 1_000_000:.2f}"])
    sections.append(render_table("Загальний підсумок", summary_rows))
    fastest_overall = min(overall_totals, key=overall_totals.get)
    sections.append(f"Найшвидший алгоритм загалом: {fastest_overall}")

    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
