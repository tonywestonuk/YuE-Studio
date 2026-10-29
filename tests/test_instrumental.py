"""Instrumental songs: lyrics/tags preparation and silencing the vocal voice of a score (pure Python)."""
import importlib.util, json
from pathlib import Path

from yue2.instrumental import (MARKERS, TAGS, instrumental_tags, silence_vocals, silenced_voice, structure_only,
                               voice_names)
from yue2.protocol import SongRequest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=86\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:Em\n')
PLANNED = HEADER + 'V: Vocal\nz24z4"Em"z4|"Em"z32|\nz8B2d6d4"C"B2d6B2d2-|d6z24z2|\nV: Ins\nZ2|"Em"E8G8B8e8|\n'
SINGLE = 'X:1\nT:\nM:4/4\nL:1/16\nK:G\nV: Melody clef=treble\nV: Melody\nG4A4B4c4|d8d8|\n'
TWO_UNNAMED = 'X:1\nM:3/4\nL:1/8\nK:C\nV: Lead\nV: Pad\nV: Lead\nC2E2G2|\nV: Pad\nC6|\n'


def test_structure_only_keeps_markers_or_defaults():
    assert structure_only("[Verse]\nla la\n\n[Chorus]\nmore words\n") == "[Verse]\n[Chorus]"
    assert structure_only("no markers here") == MARKERS


def test_instrumental_tags_prefix_once():
    assert instrumental_tags("pop, piano") == f"{TAGS}, pop, piano"
    assert instrumental_tags("Instrumental jazz") == "Instrumental jazz"


def test_voice_names_in_order_of_appearance():
    assert voice_names(PLANNED) == ["Vocal", "Ins"]
    assert voice_names(SINGLE) == ["Melody"]
    assert voice_names("X:1\nK:C\nC4|\n") == []


def test_silence_vocals_on_a_planned_score():
    out = silence_vocals(PLANNED)
    lines = out.splitlines()
    assert silenced_voice(PLANNED) == "Vocal"
    assert lines[:8] == HEADER.splitlines()                      # header untouched
    assert lines[8] == "V: Vocal"
    assert lines[9] == '"Em"z32|"Em"z32|'                        # chords kept, notes become whole-bar rests (4/4 at 1/32)
    assert lines[10] == '"C"z32|z32|'
    assert lines[11] == "V: Ins"
    assert lines[12] == 'Z2|"Em"E8G8B8e8|'                       # instrumental voice untouched
    assert out.endswith("\n")


def test_silence_vocals_falls_back_to_the_only_voice():
    assert silenced_voice(SINGLE) == "Melody"
    out = silence_vocals(SINGLE)
    assert out.splitlines()[-1] == "z16|z16|"                    # 4/4 at 1/16
    assert "G4A4" not in out


def test_silence_vocals_leaves_a_multi_voice_score_without_vocal_unchanged():
    assert silenced_voice(TWO_UNNAMED) is None
    assert silence_vocals(TWO_UNNAMED) == TWO_UNNAMED
    assert silence_vocals(TWO_UNNAMED, "Nope") == TWO_UNNAMED   # an explicit voice the score lacks
    assert silence_vocals(TWO_UNNAMED, "Lead").splitlines()[-3] == "z6|"   # 3/4 at 1/8, explicitly chosen


def _worker():
    """tools/yue2_worker.py imports without the model or torch (checked: 0.06 s, stdlib + yue2.locks only)."""
    spec = importlib.util.spec_from_file_location("yue2_worker", ROOT / "tools" / "yue2_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logs(capsys):
    return [json.loads(line)["message"] for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_worker_silences_a_supplied_score(capsys):
    worker = _worker()
    request = SongRequest(style="Instrumental, pop", lyrics="[Verse]", abc=PLANNED, abc_open=False)
    out = worker.silenced_request(request, request.abc, "supplied")
    assert out.abc == silence_vocals(PLANNED) and out.abc_open is False
    assert out.style == request.style and out.lyrics == request.lyrics
    assert _logs(capsys) == ["Instrumental: vocal voice silenced in the supplied score"]


def test_worker_replaces_a_hummed_opening_by_the_planned_score(capsys):
    worker = _worker()
    request = SongRequest(style="pop", lyrics="[Verse]", abc="V: Vocal\nB4|", abc_open=True)
    out = worker.silenced_request(request, PLANNED, "planned")
    assert out.abc == silence_vocals(PLANNED) and out.abc_open is False
    assert _logs(capsys) == ["Instrumental: vocal voice silenced in the planned score"]


def test_worker_names_a_fallback_voice_and_warns_when_none(capsys):
    worker = _worker()
    request = SongRequest(style="pop", lyrics="[Verse]", abc=SINGLE)
    assert worker.silenced_request(request, SINGLE, "supplied").abc == silence_vocals(SINGLE)
    assert worker.silenced_request(request, TWO_UNNAMED, "supplied") is None
    assert _logs(capsys) == ["Instrumental: voice 'Melody' silenced in the supplied score (it has no Vocal voice)",
                             "Instrumental: the supplied score has no Vocal voice; vocals could not be silenced"]
