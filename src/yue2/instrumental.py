"""Instrumental songs: silence the vocal voice of a planned score.

The token model follows the planned score far more closely than the style tags, so tags such as
"instrumental, no vocals" alone still produce a singer. Replacing every note of the Vocal voice with
rests (keeping its chord symbols and the instrumental voice untouched) removes the sung melody
while leaving the harmony and arrangement in place.

Planned scores name their voices ``Vocal`` and ``Ins``; a supplied score (a transcription or a
pasted file) may name its single voice anything, so that voice is silenced when there is no
``Vocal``. A multi-voice score with no ``Vocal`` voice is left unchanged.
"""
from __future__ import annotations
import re

MARKERS = "[Intro]\n[Verse]\n[Chorus]\n[Outro]"
TAGS = "Instrumental, no vocals, no singing"


def structure_only(lyrics: str) -> str:
    """Keep only the section markers of a lyric sheet, or a default structure if it has none."""
    lines = [line.strip() for line in lyrics.splitlines() if re.fullmatch(r"\s*\[[^\]]+\]\s*", line)]
    return "\n".join(lines) if lines else MARKERS


def instrumental_tags(style: str) -> str:
    return style if "instrumental" in style.lower() else f"{TAGS}, {style}"


def voice_names(abc: str) -> list[str]:
    """The voices of a score in order of first appearance (``V:`` headers and body lines)."""
    names = []
    for line in abc.splitlines():
        m = re.match(r"V:\s*(\S+)", line)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


def silenced_voice(abc: str) -> str | None:
    """The voice ``silence_vocals`` silences: ``Vocal`` if the score has one, the only voice of a
    single-voice score, else None (nothing can be silenced safely)."""
    names = voice_names(abc)
    if "Vocal" in names:
        return "Vocal"
    return names[0] if len(names) == 1 else None


def silence_vocals(abc: str, voice_name: str | None = None) -> str:
    """Every bar of ``voice_name`` (default: ``silenced_voice(abc)``) becomes its chord symbols
    followed by a whole-bar rest. The score is returned unchanged if there is no such voice."""
    if voice_name is None:
        voice_name = silenced_voice(abc)
    if voice_name is None or voice_name not in voice_names(abc):
        return abc
    meter, unit = (4, 4), 8
    out, voice = [], None
    for line in abc.splitlines():
        if line.startswith("M:"):
            m = re.match(r"M:\s*(\d+)/(\d+)", line)
            if m:
                meter = (int(m.group(1)), int(m.group(2)))
        elif line.startswith("L:"):
            m = re.match(r"L:\s*1/(\d+)", line)
            if m:
                unit = int(m.group(1))
        if re.match(r"V:\s*\S+", line):
            if "clef" not in line and "name=" not in line:
                voice = line.split(":", 1)[1].strip().split()[0]
            out.append(line); continue
        if line[:2] in ("X:", "T:", "M:", "L:", "Q:", "K:") or line.startswith("%"):
            out.append(line); continue
        if voice == voice_name and "|" in line:
            bar_units = meter[0] * unit // meter[1]
            bars = []
            for bar in line.split("|"):
                if not bar.strip():
                    continue
                if re.fullmatch(r"\s*Z\d*\s*", bar):
                    bars.append(bar.strip()); continue
                bars.append("".join(re.findall(r'"[^"]*"', bar)) + f"z{bar_units}")
            out.append("|".join(bars) + "|")
        else:
            out.append(line)
    return "\n".join(out) + "\n"
