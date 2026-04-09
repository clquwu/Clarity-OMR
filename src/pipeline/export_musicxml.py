#!/usr/bin/env python3
"""Stage D serialization from assembled token streams to MusicXML."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.pipeline.assemble_score import (
    AssembledScore,
    AssembledStaff,
    AssembledSystem,
    StaffLocation,
)


DURATION_QUARTER_LENGTH = {
    "_whole": 4.0,
    "_half": 2.0,
    "_quarter": 1.0,
    "_eighth": 0.5,
    "_sixteenth": 0.25,
    "_thirty_second": 0.125,
    "_sixty_fourth": 0.0625,
}

_XML_NAMESPACE_SCHEMA = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://www.w3.org/XML/1998/namespace"
           xmlns:xml="http://www.w3.org/XML/1998/namespace"
           elementFormDefault="qualified"
           attributeFormDefault="unqualified">
  <xs:attribute name="lang" type="xs:language"/>
</xs:schema>
"""

_XLINK_NAMESPACE_SCHEMA = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://www.w3.org/1999/xlink"
           xmlns:xlink="http://www.w3.org/1999/xlink"
           elementFormDefault="qualified"
           attributeFormDefault="unqualified">
  <xs:simpleType name="typeType">
    <xs:restriction base="xs:token">
      <xs:enumeration value="simple"/>
    </xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="showType">
    <xs:restriction base="xs:token">
      <xs:enumeration value="new"/>
      <xs:enumeration value="replace"/>
      <xs:enumeration value="embed"/>
      <xs:enumeration value="other"/>
      <xs:enumeration value="none"/>
    </xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="actuateType">
    <xs:restriction base="xs:token">
      <xs:enumeration value="onLoad"/>
      <xs:enumeration value="onRequest"/>
      <xs:enumeration value="other"/>
      <xs:enumeration value="none"/>
    </xs:restriction>
  </xs:simpleType>
  <xs:attribute name="href" type="xs:anyURI"/>
  <xs:attribute name="type" type="xlink:typeType"/>
  <xs:attribute name="role" type="xs:anyURI"/>
  <xs:attribute name="title" type="xs:string"/>
  <xs:attribute name="show" type="xlink:showType"/>
  <xs:attribute name="actuate" type="xlink:actuateType"/>
</xs:schema>
"""


def _require_music21():
    try:
        from music21 import chord, clef, converter, key, meter, note, stream
    except ImportError as exc:
        raise RuntimeError("music21 is required for MusicXML export. Install with: pip install music21") from exc
    return chord, clef, converter, key, meter, note, stream


def _require_music21_notation():
    try:
        from music21 import articulations, dynamics, expressions, spanner, tie
    except ImportError as exc:
        raise RuntimeError("music21 is required for notation export support.") from exc
    return articulations, dynamics, expressions, spanner, tie


def _require_lxml_etree():
    try:
        from lxml import etree
    except ImportError as exc:
        raise RuntimeError("lxml is required for MusicXML schema validation. Install with: pip install lxml") from exc
    return etree


def _resolve_musicxml_schema_path() -> Path:
    try:
        import music21
    except ImportError as exc:
        raise RuntimeError("music21 is required for MusicXML schema validation.") from exc
    schema_path = Path(music21.__file__).resolve().parent / "musicxml" / "musicxml.xsd"
    if not schema_path.exists():
        raise RuntimeError(f"MusicXML schema file not found at expected path: {schema_path}")
    return schema_path


def _get_musicxml_schema():
    etree = _require_lxml_etree()
    schema_path = _resolve_musicxml_schema_path()
    cache = getattr(_get_musicxml_schema, "_cache", None)
    if cache is not None and cache[0] == schema_path:
        return cache[1]

    class _SchemaResolver(etree.Resolver):
        def resolve(self, url, pubid, context):  # type: ignore[override]
            if url.endswith("/xml.xsd"):
                return self.resolve_string(_XML_NAMESPACE_SCHEMA, context)
            if url.endswith("/xlink.xsd"):
                return self.resolve_string(_XLINK_NAMESPACE_SCHEMA, context)
            return None

    parser = etree.XMLParser(no_network=True, resolve_entities=False)
    parser.resolvers.add(_SchemaResolver())
    schema_doc = etree.parse(str(schema_path), parser)
    schema = etree.XMLSchema(schema_doc)
    setattr(_get_musicxml_schema, "_cache", (schema_path, schema))
    return schema


def _validate_musicxml_schema_file(musicxml_path: Path) -> Dict[str, object]:
    etree = _require_lxml_etree()
    schema = _get_musicxml_schema()
    parser = etree.XMLParser(no_network=True, resolve_entities=False)
    document = etree.parse(str(musicxml_path), parser)
    is_valid = bool(schema.validate(document))
    errors = [f"line {item.line}: {item.message}" for item in schema.error_log] if not is_valid else []
    return {
        "schema_valid": is_valid,
        "schema_error_count": len(errors),
        "schema_errors_preview": errors[:10],
    }


def _parse_pitch_token(token: str) -> str:
    if not token.startswith("note-"):
        raise ValueError(f"Expected note token, got '{token}'")
    return token[len("note-") :]


def _parse_grace_pitch_token(token: str) -> str:
    if not token.startswith("gracenote-"):
        raise ValueError(f"Expected gracenote token, got '{token}'")
    return token[len("gracenote-") :]


def _decode_duration(tokens: Sequence[str], start_index: int) -> Tuple[float, int]:
    index = start_index
    while index < len(tokens) and tokens[index].startswith("<tuplet_"):
        # Tuplet predictions are currently unreliable in the OMR output and
        # produce false triplet timing in MusicXML. Consume the tokens so the
        # stream stays aligned, but ignore their timing effect in export.
        index += 1

    if index >= len(tokens):
        raise ValueError("Duration token missing.")
    base_token = tokens[index]
    if base_token not in DURATION_QUARTER_LENGTH:
        raise ValueError(f"Unsupported duration token '{base_token}'")
    quarter_length = DURATION_QUARTER_LENGTH[base_token]
    index += 1

    if index < len(tokens) and tokens[index] == "_dot":
        quarter_length *= 1.5
        index += 1
    elif index < len(tokens) and tokens[index] == "_double_dot":
        quarter_length *= 1.75
        index += 1
    return quarter_length, index


def _parse_clef(clef_token: str):
    _, clef, _, _, _, _, _ = _require_music21()
    mapping = {
        "clef-G2": clef.TrebleClef,
        "clef-F4": clef.BassClef,
        "clef-C3": clef.AltoClef,
        "clef-C4": clef.TenorClef,
        "clef-C1": clef.SopranoClef,
        "clef-G2_8vb": clef.Treble8vbClef,
        "clef-G2_8va": clef.Treble8vaClef,
    }
    clef_class = mapping.get(clef_token)
    return clef_class() if clef_class is not None else None


def _parse_key_signature(token: str):
    _, _, _, key, _, _, _ = _require_music21()
    value = token[len("keySignature-") :]
    if value == "none":
        return None
    if value.endswith("M"):
        tonic = value[:-1]
        mode = "major"
    elif value.endswith("m"):
        tonic = value[:-1]
        mode = "minor"
    else:
        tonic = value
        mode = "major"
    return key.Key(tonic, mode)


def _parse_time_signature(token: str):
    _, _, _, _, meter, _, _ = _require_music21()
    value = token[len("timeSignature-") :]
    if value == "C":
        value = "4/4"
    elif value == "C/":
        value = "2/2"
    if value == "other":
        return None
    return meter.TimeSignature(value)


def _append_event_to_voice(voice, event) -> None:
    voice.append(event)


def append_tokens_to_part(part, tokens: Sequence[str]) -> None:
    chord, _, _, _, _, note, stream = _require_music21()
    articulations, dynamics, expressions, spanner, tie = _require_music21_notation()
    try:
        from music21 import bar as m21_bar
    except ImportError as exc:
        raise RuntimeError("music21 is required for barline serialization support.") from exc

    articulation_factories = {
        "staccato": getattr(articulations, "Staccato", None),
        "accent": getattr(articulations, "Accent", None),
        "tenuto": getattr(articulations, "Tenuto", None),
        "marcato": getattr(articulations, "StrongAccent", None),
        "staccatissimo": getattr(articulations, "Staccatissimo", None),
        "portato": getattr(articulations, "DetachedLegato", None),
        "sforzando": getattr(articulations, "StrongAccent", None),
        "snap_pizz": getattr(articulations, "SnapPizzicato", None),
    }
    expression_factories = {
        "fermata": getattr(expressions, "Fermata", None),
        "trill": getattr(expressions, "Trill", None),
        "mordent": getattr(expressions, "Mordent", None),
        "turn": getattr(expressions, "Turn", None),
    }
    articulation_tokens = set(articulation_factories.keys()) | set(expression_factories.keys())

    measure_number = len(list(part.getElementsByClass(stream.Measure))) + 1
    pending_clef = None
    pending_key = None
    pending_time = None
    current_measure = None
    active_voice = 1
    voices: Dict[int, object] = {}
    pending_articulation_tokens: List[str] = []
    pending_dynamic_tokens: List[str] = []
    pending_tempo_tokens: List[str] = []
    pending_tie_start = False
    pending_tie_end = False
    pending_slur_starts = 0
    pending_slur_ends = 0
    open_slur_events: List[object] = []
    created_spanners: List[object] = []

    def _apply_articulations_and_expressions(event) -> None:
        nonlocal pending_articulation_tokens
        if not pending_articulation_tokens:
            return
        for notation_token in pending_articulation_tokens:
            articulation_factory = articulation_factories.get(notation_token)
            if articulation_factory is not None:
                try:
                    event.articulations.append(articulation_factory())
                    continue
                except (AttributeError, TypeError, ValueError):
                    pass
            expression_factory = expression_factories.get(notation_token)
            if expression_factory is not None:
                try:
                    event.expressions.append(expression_factory())
                    continue
                except (AttributeError, TypeError, ValueError):
                    pass
            event.expressions.append(expressions.TextExpression(notation_token.replace("_", " ")))
        pending_articulation_tokens = []

    def _apply_tie(event) -> None:
        nonlocal pending_tie_start, pending_tie_end
        if not (pending_tie_start or pending_tie_end):
            return
        tie_type = "continue" if (pending_tie_start and pending_tie_end) else ("start" if pending_tie_start else "stop")
        if hasattr(event, "tie"):
            event.tie = tie.Tie(tie_type)
        if hasattr(event, "notes"):
            for chord_note in event.notes:
                chord_note.tie = tie.Tie(tie_type)
        pending_tie_start = False
        pending_tie_end = False

    def _apply_slur_links(event) -> None:
        nonlocal pending_slur_starts, pending_slur_ends
        if pending_slur_starts > 0:
            for _ in range(pending_slur_starts):
                open_slur_events.append(event)
            pending_slur_starts = 0
        if pending_slur_ends > 0:
            for _ in range(pending_slur_ends):
                if open_slur_events:
                    start_event = open_slur_events.pop()
                    created_spanners.append(spanner.Slur(start_event, event))
            pending_slur_ends = 0

    def _append_tempo_mark(target_measure, tempo_text: str) -> None:
        clean_text = tempo_text.replace("_", " ").strip()
        if not clean_text:
            return
        try:
            from music21 import tempo as m21_tempo

            target_measure.insert(0, m21_tempo.MetronomeMark(text=clean_text))
        except Exception:
            target_measure.insert(0, expressions.TextExpression(clean_text))

    def _decode_duration_or_skip(start_index: int) -> Optional[Tuple[float, int]]:
        try:
            return _decode_duration(tokens, start_index)
        except ValueError:
            return None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "<measure_start>":
            current_measure = stream.Measure(number=measure_number)
            measure_number += 1
            voices = {1: stream.Voice(id="voice_1")}
            active_voice = 1
            if pending_clef is not None:
                current_measure.insert(0, pending_clef)
                pending_clef = None
            if pending_key is not None:
                current_measure.insert(0, pending_key)
                pending_key = None
            if pending_time is not None:
                current_measure.insert(0, pending_time)
                pending_time = None
            if pending_tempo_tokens:
                for tempo_text in pending_tempo_tokens:
                    _append_tempo_mark(current_measure, tempo_text)
                pending_tempo_tokens = []
            idx += 1
            continue

        if token == "<measure_end>":
            if current_measure is None:
                raise ValueError("Encountered <measure_end> without active measure.")
            if len(voices) == 1 and 1 in voices:
                for element in voices[1]:
                    current_measure.append(element)
            else:
                for voice_id in sorted(voices):
                    current_measure.insert(0, voices[voice_id])
            part.append(current_measure)
            current_measure = None
            idx += 1
            continue

        if token.startswith("<voice_"):
            raw = token.strip("<>").split("_")[-1]
            active_voice = int(raw)
            voices.setdefault(active_voice, stream.Voice(id=f"voice_{active_voice}"))
            idx += 1
            continue

        if token.startswith("clef-"):
            pending_clef = _parse_clef(token)
            idx += 1
            continue

        if token.startswith("keySignature-"):
            pending_key = _parse_key_signature(token)
            idx += 1
            continue

        if token.startswith("timeSignature-"):
            pending_time = _parse_time_signature(token)
            idx += 1
            continue

        if token.startswith("dynamic-"):
            pending_dynamic_tokens.append(token[len("dynamic-") :])
            idx += 1
            continue

        if token.startswith("tempo-"):
            tempo_text = token[len("tempo-") :]
            if current_measure is None:
                pending_tempo_tokens.append(tempo_text)
            else:
                _append_tempo_mark(current_measure, tempo_text)
            idx += 1
            continue

        if token in articulation_tokens:
            pending_articulation_tokens.append(token)
            idx += 1
            continue

        if token == "tie_start":
            pending_tie_start = True
            idx += 1
            continue
        if token == "tie_end":
            pending_tie_end = True
            idx += 1
            continue
        if token == "slur_start":
            pending_slur_starts += 1
            idx += 1
            continue
        if token == "slur_end":
            pending_slur_ends += 1
            idx += 1
            continue
        if token in {"cresc_start", "cresc_end", "decresc_start", "decresc_end"}:
            if current_measure is not None:
                current_voice = voices.setdefault(active_voice, stream.Voice(id=f"voice_{active_voice}"))
                current_voice.append(expressions.TextExpression(token.replace("_", " ")))
            idx += 1
            continue
        if token in {"barline", "double_barline", "final_barline", "repeat_start", "repeat_end", "repeat_both"}:
            if current_measure is not None:
                if token == "barline":
                    current_measure.rightBarline = m21_bar.Barline("regular")
                elif token == "double_barline":
                    current_measure.rightBarline = m21_bar.Barline("double")
                elif token == "final_barline":
                    current_measure.rightBarline = m21_bar.Barline("final")
                elif token == "repeat_start":
                    current_measure.leftBarline = m21_bar.Repeat(direction="start")
                elif token == "repeat_end":
                    current_measure.rightBarline = m21_bar.Repeat(direction="end")
                elif token == "repeat_both":
                    current_measure.leftBarline = m21_bar.Repeat(direction="start")
                    current_measure.rightBarline = m21_bar.Repeat(direction="end")
            idx += 1
            continue

        if token in {"<bos>", "<staff_start>", "<staff_end>", "<eos>"}:
            idx += 1
            continue

        if current_measure is None:
            idx += 1
            continue

        current_voice = voices.setdefault(active_voice, stream.Voice(id=f"voice_{active_voice}"))
        if pending_dynamic_tokens:
            for dynamic_mark in pending_dynamic_tokens:
                try:
                    current_voice.append(dynamics.Dynamic(dynamic_mark))
                except (TypeError, ValueError):
                    current_voice.append(expressions.TextExpression(f"dynamic-{dynamic_mark}"))
            pending_dynamic_tokens = []

        if token == "rest":
            duration_result = _decode_duration_or_skip(idx + 1)
            if duration_result is None:
                idx += 1
                continue
            duration_q, next_idx = duration_result
            _append_event_to_voice(current_voice, note.Rest(quarterLength=duration_q))
            idx = next_idx
            continue

        if token == "<chord_start>":
            chord_pitches: List[str] = []
            idx += 1
            while idx < len(tokens) and tokens[idx] != "<chord_end>":
                if tokens[idx].startswith("note-"):
                    chord_pitches.append(_parse_pitch_token(tokens[idx]))
                idx += 1
            if idx >= len(tokens) or tokens[idx] != "<chord_end>":
                # Skip malformed chord spans instead of aborting full export.
                idx += 1
                continue
            duration_result = _decode_duration_or_skip(idx + 1)
            if duration_result is None:
                idx += 1
                continue
            duration_q, next_idx = duration_result
            if chord_pitches:
                chord_event = chord.Chord(chord_pitches, quarterLength=duration_q)
                _apply_articulations_and_expressions(chord_event)
                _apply_tie(chord_event)
                _apply_slur_links(chord_event)
                _append_event_to_voice(current_voice, chord_event)
            else:
                # Fallback for invalid chord content (for example, rest inside chord span).
                rest_event = note.Rest(quarterLength=duration_q)
                _apply_tie(rest_event)
                _append_event_to_voice(current_voice, rest_event)
            idx = next_idx
            continue

        if token.startswith("note-"):
            pitch = _parse_pitch_token(token)
            duration_result = _decode_duration_or_skip(idx + 1)
            if duration_result is None:
                idx += 1
                continue
            duration_q, next_idx = duration_result
            note_event = note.Note(pitch, quarterLength=duration_q)
            _apply_articulations_and_expressions(note_event)
            _apply_tie(note_event)
            _apply_slur_links(note_event)
            _append_event_to_voice(current_voice, note_event)
            idx = next_idx
            continue

        if token.startswith("gracenote-"):
            pitch = _parse_grace_pitch_token(token)
            duration_result = _decode_duration_or_skip(idx + 1)
            if duration_result is None:
                idx += 1
                continue
            duration_q, next_idx = duration_result
            grace_event = note.Note(pitch, quarterLength=duration_q).getGrace()
            _apply_articulations_and_expressions(grace_event)
            _apply_tie(grace_event)
            _apply_slur_links(grace_event)
            _append_event_to_voice(current_voice, grace_event)
            idx = next_idx
            continue

        idx += 1

    for slur in created_spanners:
        part.insert(0, slur)


def assembled_score_to_music21(score: AssembledScore):
    _, _, _, _, _, _, stream = _require_music21()
    music_score = stream.Score(id="omr_score")
    parts = {label: stream.Part(id=label) for label in score.part_order}

    systems = sorted(score.systems, key=lambda item: (item.page_index, item.system_index))
    for system in systems:
        for staff in system.staves:
            part = parts.setdefault(staff.part_label, stream.Part(id=staff.part_label))
            append_tokens_to_part(part, staff.tokens)

    for label in score.part_order:
        music_score.append(parts[label])
    return music_score


def _get_bpm(score) -> float:
    """Return the first MetronomeMark BPM from a score, or 120.0 as default."""
    try:
        from music21 import tempo as m21_tempo
    except ImportError:
        return 120.0
    for elem in score.recurse():
        if isinstance(elem, m21_tempo.MetronomeMark):
            bpm = elem.getQuarterBPM()
            if bpm and bpm > 0:
                return float(bpm)
    return 120.0


def _extract_note_events(score, bpm: float = 120.0) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
    """Extract note-level events from a music21 score for mir_eval comparison.

    Returns ``(intervals, pitches_hz)`` where *intervals* is an Nx2 array of
    ``[onset, offset]`` in **seconds** (converted from quarter-note offsets
    using *bpm*) and *pitches_hz* is an N-array of frequencies in Hz.
    """
    import numpy as np

    chord_cls, _, _, _, _, note_cls, stream_cls = _require_music21()
    qn_to_sec = 60.0 / bpm

    onsets: List[float] = []
    offsets: List[float] = []
    pitches: List[float] = []

    for part in score.parts:
        for element in part.recurse().notesAndRests:
            if isinstance(element, note_cls.Rest):
                continue
            abs_offset = float(element.getOffsetInHierarchy(score)) * qn_to_sec
            dur = float(element.duration.quarterLength) * qn_to_sec
            if dur <= 0:
                dur = 0.001  # grace notes: give a tiny duration so mir_eval won't reject
            if isinstance(element, note_cls.Note):
                onsets.append(abs_offset)
                offsets.append(abs_offset + dur)
                pitches.append(float(element.pitch.frequency))
            elif isinstance(element, chord_cls.Chord):
                for chord_note in element.notes:
                    onsets.append(abs_offset)
                    offsets.append(abs_offset + dur)
                    pitches.append(float(chord_note.pitch.frequency))

    if not onsets:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)

    intervals = np.column_stack([onsets, offsets]).astype(np.float64)
    return intervals, np.array(pitches, dtype=np.float64)


def _compute_musical_similarity(
    *,
    predicted_score,
    reference_musicxml_path: Path,
) -> Optional[Dict[str, float]]:
    """Compare two scores at the note level using *mir_eval* transcription metrics.

    Returns precision, recall, F1 and average overlap ratio (onset-only and
    onset+offset) or ``None`` when dependencies are missing or the reference
    file does not exist.
    """
    try:
        import mir_eval  # noqa: F401 – also validates availability
        import numpy as np  # noqa: F811
    except ImportError:
        return None

    if not reference_musicxml_path.exists():
        return None

    _, _, converter, _, _, _, _ = _require_music21()
    try:
        ref_score = converter.parse(str(reference_musicxml_path))
    except Exception:
        return None

    # Use reference BPM to convert both scores to seconds
    ref_bpm = _get_bpm(ref_score)
    ref_intervals, ref_pitches = _extract_note_events(ref_score, bpm=ref_bpm)
    est_intervals, est_pitches = _extract_note_events(predicted_score, bpm=ref_bpm)

    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        return {
            "musical_precision": 0.0,
            "musical_recall": 0.0,
            "musical_f1": 0.0,
            "musical_overlap": 0.0,
            "musical_onset_precision": 0.0,
            "musical_onset_recall": 0.0,
            "musical_onset_f1": 0.0,
            "musical_samples": 1,
        }

    # onset + offset matching (full note accuracy)
    prec, rec, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,  # 50 ms (mir_eval standard)
        pitch_tolerance=50.0,  # cents
    )

    # onset-only matching (ignores note duration differences)
    oprec, orec, of1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,
        pitch_tolerance=50.0,
        offset_ratio=None,  # disable offset matching
    )

    return {
        "musical_precision": float(prec),
        "musical_recall": float(rec),
        "musical_f1": float(f1),
        "musical_overlap": float(overlap),
        "musical_onset_precision": float(oprec),
        "musical_onset_recall": float(orec),
        "musical_onset_f1": float(of1),
        "musical_samples": 1,
    }


def _strip_rest_measure_attr(xml_path: Path) -> None:
    """Remove ``measure`` attribute from ``<rest>`` elements.

    music21 emits ``<rest measure="yes"/>`` for whole-measure rests, but the
    MusicXML XSD bundled with music21 defines the ``rest`` element with the
    ``display-step-octave`` complex type which declares **no** attributes.
    Stripping the attribute keeps the musical semantics intact while passing
    XSD validation.
    """
    import re

    raw = xml_path.read_bytes()
    text = raw.decode("utf-8")
    fixed = re.sub(r"(<rest)\s+measure\s*=\s*\"[^\"]*\"", r"\1", text)
    if fixed is not text:
        xml_path.write_bytes(fixed.encode("utf-8"))


def _write_musicxml_safe(music_score, output_path: Path) -> None:
    """Write MusicXML, working around music21 voice consistency bugs.

    music21's makeRests/makeTies can crash with KeyError when a voice
    (e.g. voice_3) exists in one measure but not the next. We catch this
    and fall back to writing without makeNotation.
    """
    try:
        music_score.write("musicxml", fp=str(output_path))
    except KeyError:
        # Fallback: export using GeneralObjectExporter with makeNotation=False
        from music21.musicxml.m21ToXml import GeneralObjectExporter
        exporter = GeneralObjectExporter(music_score)
        exporter.makeNotation = False
        xml_bytes = exporter.parse()
        output_path.write_bytes(xml_bytes)
    # Strip invalid measure attribute from <rest> elements (music21 bug).
    _strip_rest_measure_attr(output_path)


def validate_musicxml_roundtrip(
    score_obj,
    reference_musicxml_path: Optional[Path] = None,
) -> Dict[str, object]:
    chord, _, converter, _, _, note, stream = _require_music21()
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        _write_musicxml_safe(score_obj, temp_path)
        schema_validation = _validate_musicxml_schema_file(temp_path)
        reparsed = converter.parse(str(temp_path))

        validation = {
            "parts": len(reparsed.parts),
            "measure_mismatches": 0,
            "unmatched_tie_starts": 0,
            "unmatched_tie_stops": 0,
            "key_signature_mismatches": 0,
            "time_signature_mismatches": 0,
            **schema_validation,
        }
        measure_signature_observations: Dict[int, Dict[str, List[object]]] = {}
        for part in reparsed.parts:
            current_key_signature: Optional[int] = None
            current_time_signature = None
            open_ties: Dict[str, int] = {}
            for measure_index, measure in enumerate(part.getElementsByClass(stream.Measure), start=1):
                if measure.keySignature is not None:
                    current_key_signature = int(measure.keySignature.sharps or 0)
                if measure.timeSignature is not None:
                    current_time_signature = measure.timeSignature
                observation = measure_signature_observations.setdefault(measure_index, {"key": [], "time": []})
                observation["key"].append(current_key_signature)
                observation["time"].append(
                    current_time_signature.ratioString if current_time_signature is not None else None
                )
                if current_time_signature is not None:
                    expected = float(current_time_signature.barDuration.quarterLength)
                    actual = float(sum(element.duration.quarterLength for element in measure.notesAndRests))
                    if abs(expected - actual) > 0.1:
                        validation["measure_mismatches"] += 1
                for element in measure.notes:
                    if isinstance(element, note.Note):
                        note_elements = [element]
                    elif isinstance(element, chord.Chord):
                        note_elements = list(element.notes)
                    else:
                        continue
                    for tied_note in note_elements:
                        tie_obj = tied_note.tie
                        if tie_obj is None:
                            continue
                        pitch_key = tied_note.pitch.nameWithOctave
                        if tie_obj.type == "start":
                            open_ties[pitch_key] = open_ties.get(pitch_key, 0) + 1
                        elif tie_obj.type == "continue":
                            if open_ties.get(pitch_key, 0) <= 0:
                                validation["unmatched_tie_stops"] += 1
                        elif tie_obj.type == "stop":
                            if open_ties.get(pitch_key, 0) <= 0:
                                validation["unmatched_tie_stops"] += 1
                            else:
                                open_ties[pitch_key] -= 1
            validation["unmatched_tie_starts"] += sum(value for value in open_ties.values() if value > 0)
        for observation in measure_signature_observations.values():
            key_values = {value for value in observation["key"] if value is not None}
            time_values = {value for value in observation["time"] if value is not None}
            if len(key_values) > 1:
                validation["key_signature_mismatches"] += 1
            if len(time_values) > 1:
                validation["time_signature_mismatches"] += 1
        if reference_musicxml_path is not None:
            roundtrip = _compute_musical_similarity(
                predicted_score=score_obj,
                reference_musicxml_path=reference_musicxml_path,
            )
            if roundtrip is not None:
                validation.update(roundtrip)
        return validation
    finally:
        temp_path.unlink(missing_ok=True)


def write_musicxml(
    score: AssembledScore,
    output_path: Path,
    reference_musicxml_path: Optional[Path] = None,
) -> Dict[str, object]:
    music_score = assembled_score_to_music21(score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_musicxml_safe(music_score, output_path)
    validation = validate_musicxml_roundtrip(music_score, reference_musicxml_path=reference_musicxml_path)
    if not bool(validation.get("schema_valid", False)):
        preview = validation.get("schema_errors_preview") or []
        detail = preview[0] if preview else "unknown schema validation error"
        raise ValueError(f"Generated MusicXML failed XSD validation: {detail}")
    validation["output_path"] = str(output_path)
    return validation


def write_musicxml_per_system(
    score: AssembledScore,
    output_stem: Path,
) -> List[Dict[str, object]]:
    """Export one MusicXML file per system.

    Files are written as ``<output_stem>_systems/page_0001_system_01.musicxml``, etc.
    Returns a list of validation dicts (one per system) in page/system order.
    """
    folder = output_stem.parent / f"{output_stem.stem}_systems"
    systems = sorted(score.systems, key=lambda s: (s.page_index, s.system_index))
    results: List[Dict[str, object]] = []
    system_counter_per_page: Dict[int, int] = {}
    for system in systems:
        system_score = AssembledScore(systems=[system], part_order=list(score.part_order))
        page_num = system.page_index + 1
        sys_num = system_counter_per_page[system.page_index] = system_counter_per_page.get(system.page_index, 0) + 1
        system_path = folder / f"page_{page_num:04d}_system_{sys_num:02d}.musicxml"
        try:
            validation = write_musicxml(system_score, system_path)
        except (ValueError, KeyError) as exc:
            music_score = assembled_score_to_music21(system_score)
            system_path.parent.mkdir(parents=True, exist_ok=True)
            _write_musicxml_safe(music_score, system_path)
            validation = {"schema_valid": False, "best_effort": True, "warning": str(exc), "output_path": str(system_path)}
        results.append(validation)
    return results


def write_musicxml_per_page(
    score: AssembledScore,
    output_stem: Path,
) -> List[Dict[str, object]]:
    """Export one MusicXML file per page.

    Files are written as ``<output_stem>_pages/page_0001.musicxml``, etc.
    Returns a list of validation dicts (one per page) in page order.
    """
    from collections import defaultdict

    folder = output_stem.parent / f"{output_stem.stem}_pages"
    systems_by_page: Dict[int, List] = defaultdict(list)
    for system in score.systems:
        systems_by_page[system.page_index].append(system)

    results: List[Dict[str, object]] = []
    for page_index in sorted(systems_by_page):
        page_systems = sorted(systems_by_page[page_index], key=lambda s: s.system_index)
        page_score = AssembledScore(systems=page_systems, part_order=list(score.part_order))
        page_path = folder / f"page_{page_index + 1:04d}.musicxml"
        try:
            validation = write_musicxml(page_score, page_path)
        except (ValueError, KeyError) as exc:
            music_score = assembled_score_to_music21(page_score)
            page_path.parent.mkdir(parents=True, exist_ok=True)
            _write_musicxml_safe(music_score, page_path)
            validation = {"schema_valid": False, "best_effort": True, "warning": str(exc), "output_path": str(page_path)}
        results.append(validation)
    return results


def load_assembled_score(path: Path) -> AssembledScore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    systems: List[AssembledSystem] = []
    for system_data in payload.get("systems", []):
        staves: List[AssembledStaff] = []
        for staff_data in system_data.get("staves", []):
            location = staff_data.get("location", {})
            staves.append(
                AssembledStaff(
                    sample_id=staff_data["sample_id"],
                    tokens=list(staff_data["tokens"]),
                    part_label=staff_data["part_label"],
                    measure_count=int(staff_data["measure_count"]),
                    clef=staff_data.get("clef"),
                    key_signature=staff_data.get("key_signature"),
                    time_signature=staff_data.get("time_signature"),
                    location=StaffLocation(
                        page_index=int(location.get("page_index", 0)),
                        y_top=float(location.get("y_top", 0.0)),
                        y_bottom=float(location.get("y_bottom", 0.0)),
                        x_left=float(location.get("x_left", 0.0)),
                        x_right=float(location.get("x_right", 0.0)),
                    ),
                )
            )

        systems.append(
            AssembledSystem(
                page_index=int(system_data["page_index"]),
                system_index=int(system_data["system_index"]),
                staves=staves,
                canonical_measure_count=int(system_data.get("canonical_measure_count", 0)),
                canonical_key_signature=system_data.get("canonical_key_signature"),
                canonical_time_signature=system_data.get("canonical_time_signature"),
            )
        )
    return AssembledScore(systems=systems, part_order=list(payload.get("part_order", [])))
