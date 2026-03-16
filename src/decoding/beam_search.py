#!/usr/bin/env python3
"""Constrained beam search decoding for OMR token generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.decoding.grammar_fsa import GrammarFSA, GrammarState
from src.tokenizer.vocab import OMRVocabulary, build_default_vocabulary


StepFn = Callable[[Sequence[str]], Dict[str, float]]
StepFnWithState = Callable[[Sequence[str], object | None], Tuple[Dict[str, float], object | None]]
PenaltyFn = Callable[[Sequence[str], str], float]


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 5
    max_steps: int = 512
    length_penalty_alpha: float = 0.0
    eos_token: str = "<eos>"


@dataclass
class BeamHypothesis:
    tokens: List[str]
    score: float
    grammar: GrammarFSA
    state: object | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.tokens) and self.tokens[-1] == "<eos>"


def _clone_grammar(grammar: GrammarFSA) -> GrammarFSA:
    cloned = GrammarFSA(grammar.vocab)
    cloned.state = replace(
        grammar.state,
        voice_beat_positions=dict(grammar.state.voice_beat_positions),
    )
    return cloned


def _parse_note_token(token: str) -> Optional[Tuple[str, int, str]]:
    match = re.fullmatch(r"note-([A-G])([#b]{0,2})(\d)", token)
    if not match:
        return None
    pitch_class, accidental, octave = match.groups()
    return pitch_class, int(octave), accidental


def pitch_range_penalty(prefix: Sequence[str], candidate: str) -> float:
    parsed = _parse_note_token(candidate)
    if parsed is None:
        return 0.0
    pitch_class, octave, _ = parsed

    active_clef = None
    for token in reversed(prefix):
        if token.startswith("clef-"):
            active_clef = token
            break

    if active_clef == "clef-F4":
        if octave > 5 or (octave == 5 and pitch_class not in {"C", "D"}):
            return 5.0
    if active_clef == "clef-G2":
        if octave < 3:
            return 5.0
    return 0.0


def accidental_consistency_penalty(prefix: Sequence[str], candidate: str) -> float:
    parsed = _parse_note_token(candidate)
    if parsed is None:
        return 0.0
    pitch_class, _, accidental = parsed
    if not accidental:
        return 0.0

    active_measure_tokens: List[str] = []
    for token in reversed(prefix):
        active_measure_tokens.append(token)
        if token == "<measure_start>":
            break
    active_measure_tokens.reverse()

    accidental_map: Dict[str, str] = {}
    for token in active_measure_tokens:
        note_info = _parse_note_token(token)
        if note_info is None:
            continue
        current_pitch, _, current_accidental = note_info
        if current_accidental:
            accidental_map[current_pitch] = current_accidental

    previous = accidental_map.get(pitch_class)
    if previous is None:
        return 0.0
    if previous != accidental:
        return 3.0
    return 0.0


def measure_balance_penalty(
    prefix: Sequence[str],
    candidate: str,
    grammar: Optional[GrammarFSA] = None,
    penalty_weight: float = 2.5,
) -> float:
    """Penalise <measure_end> when the accumulated beats don't match the time signature."""
    if candidate != "<measure_end>" or grammar is None:
        return 0.0
    state = grammar.state
    if not state.measure_open:
        return 0.0
    # In multi-voice measures beat tracking drifts; skip penalty.
    if state.voice_switches_in_measure > 0:
        return 0.0
    expected = state.beats_per_measure
    actual = state.beat_position
    diff = abs(expected - actual)
    if diff <= 1e-6:
        return 0.0
    return penalty_weight * diff


def cv_note_count_penalty(
    prefix: Sequence[str],
    candidate: str,
    *,
    cv_note_count: Optional[int] = None,
    tolerance: int = 2,
    penalty_weight: float = 3.0,
) -> float:
    """Penalize note emissions when the running count significantly exceeds the CV prior.

    If the CV detected N noteheads, penalize note tokens once the beam has
    already emitted N + tolerance notes.  This prevents over-generation
    without being too rigid (the tolerance allows for CV under-detection).
    """
    if cv_note_count is None:
        return 0.0
    if not candidate.startswith("note-"):
        return 0.0

    # Count notes already in the prefix
    current_notes = sum(1 for t in prefix if t.startswith("note-"))
    excess = current_notes - (cv_note_count + tolerance)
    if excess <= 0:
        return 0.0
    return penalty_weight * min(excess, 5)


# Semitone distance table for pitch comparison
_PITCH_CLASS_SEMITONE = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def _note_to_semitone(token: str) -> Optional[int]:
    """Convert a note token like 'note-C#4' to an absolute semitone number."""
    parsed = _parse_note_token(token)
    if parsed is None:
        return None
    pitch_class, octave, accidental = parsed
    base = _PITCH_CLASS_SEMITONE.get(pitch_class)
    if base is None:
        return None
    semitone = octave * 12 + base
    for ch in accidental:
        if ch == "#":
            semitone += 1
        elif ch == "b":
            semitone -= 1
    return semitone


# Expected semitone ranges per clef family (generous bounds)
_CLEF_SEMITONE_RANGE = {
    "clef-G2": (48, 96),   # C4–C8 (treble)
    "clef-F4": (24, 60),   # C2–C5 (bass)
    "clef-C3": (36, 72),   # C3–C6 (alto)
    "clef-C4": (36, 72),   # C3–C6 (tenor)
}


def _cv_pitches_plausible(cv_pitches: Sequence[str], prefix: Sequence[str]) -> bool:
    """Check if CV pitch estimates are plausible for the actual clef in the prefix.

    If the beam has already established a clef, verify that at least 30% of
    CV pitches fall within the expected range. If not, the CV clef estimation
    was wrong and the pitch prior should be disabled.
    """
    # Find the clef from the beam's prefix tokens
    active_clef = None
    for token in prefix:
        if token.startswith("clef-"):
            active_clef = token
            break
    if active_clef is None:
        return True  # No clef yet, can't check

    expected_range = _CLEF_SEMITONE_RANGE.get(active_clef)
    if expected_range is None:
        return True  # Unknown clef, allow

    low, high = expected_range
    in_range = 0
    total = 0
    for pitch in cv_pitches:
        if pitch is None:
            continue
        semi = _note_to_semitone(pitch)
        if semi is None:
            continue
        total += 1
        if low <= semi <= high:
            in_range += 1

    if total == 0:
        return True
    # If fewer than 30% of CV pitches are in the expected range, they're bad
    return (in_range / total) >= 0.30


def cv_pitch_prior_penalty(
    prefix: Sequence[str],
    candidate: str,
    *,
    cv_pitches: Optional[Sequence[str]] = None,
    penalty_weight: float = 1.5,
    octave_penalty_weight: float = 3.0,
) -> float:
    """Soft penalty when the candidate note disagrees with the CV pitch prior.

    CV provides an ordered list of estimated pitches (left-to-right). We align
    the beam's n-th note emission with the CV's n-th pitch. Since CV can't
    detect accidentals, we compare diatonic pitch class and octave only:

    - Same pitch class + same octave: no penalty (exact match)
    - Same pitch class + different octave: moderate penalty (octave error)
    - Adjacent pitch class (1 diatonic step): small penalty (CV rounding)
    - Distant pitch class: larger penalty (likely wrong note)

    The penalty is intentionally soft — Stage-B knows accidentals and musical
    context that CV doesn't, so CV only nudges, never blocks.

    Self-disabling: if the CV pitches don't match the beam's clef range
    (indicating wrong CV clef estimation), the penalty is skipped entirely.
    """
    if cv_pitches is None or not cv_pitches:
        return 0.0
    if not candidate.startswith("note-"):
        return 0.0

    # Plausibility gate: if CV pitches are out of range for the beam's clef,
    # the CV clef estimation was wrong — don't penalize.
    if not _cv_pitches_plausible(cv_pitches, prefix):
        return 0.0

    # Count how many notes the beam has already emitted
    note_index = sum(1 for t in prefix if t.startswith("note-"))

    # If we've gone past the CV list, no penalty (CV may have missed some)
    if note_index >= len(cv_pitches):
        return 0.0

    cv_pitch = cv_pitches[note_index]
    if cv_pitch is None:
        return 0.0

    # Compare semitones
    candidate_semi = _note_to_semitone(candidate)
    cv_semi = _note_to_semitone(cv_pitch)
    if candidate_semi is None or cv_semi is None:
        return 0.0

    diff = abs(candidate_semi - cv_semi)

    if diff == 0:
        return 0.0
    if diff <= 1:
        # Off by a semitone — likely an accidental difference. No penalty.
        return 0.0
    if diff == 2:
        # One diatonic step off — could be CV position rounding. Small penalty.
        return penalty_weight * 0.3
    if diff <= 4:
        # A third off — moderate disagreement
        return penalty_weight * 0.6
    if diff == 12:
        # Octave error — CV likely right about pitch class, wrong about octave
        return octave_penalty_weight
    if diff > 12:
        # More than an octave off — strong disagreement
        return octave_penalty_weight * 1.5
    # 5-11 semitones off — significant disagreement
    return penalty_weight


def make_cv_penalty_fn(
    cv_note_count: Optional[int] = None,
    cv_pitches: Optional[Sequence[str]] = None,
    tolerance: int = 2,
    penalty_weight: float = 3.0,
    pitch_prior_weight: float = 1.5,
    pitch_octave_weight: float = 3.0,
) -> PenaltyFn:
    """Build a soft penalty function that includes CV priors (count + pitch)."""
    def _penalty(prefix: Sequence[str], candidate: str) -> float:
        base = pitch_range_penalty(prefix, candidate) + accidental_consistency_penalty(prefix, candidate)
        cv_count = cv_note_count_penalty(
            prefix, candidate,
            cv_note_count=cv_note_count,
            tolerance=tolerance,
            penalty_weight=penalty_weight,
        )
        cv_pitch = cv_pitch_prior_penalty(
            prefix, candidate,
            cv_pitches=cv_pitches,
            penalty_weight=pitch_prior_weight,
            octave_penalty_weight=pitch_octave_weight,
        )
        return base + cv_count + cv_pitch
    return _penalty


def load_penalty_config(checkpoint_path) -> Optional[Dict[str, object]]:
    """Load a tuned penalty config JSON if it exists next to the checkpoint.

    Looks for <checkpoint_stem>_penalty_config.json. Returns the best_config
    dict or None if no config file is found.
    """
    import json
    from pathlib import Path

    cp = Path(str(checkpoint_path))
    config_path = cp.parent / f"{cp.stem}_penalty_config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("best_config")
    except Exception:
        return None


def make_cv_penalty_fn_from_config(
    config: Dict[str, object],
    cv_note_count: Optional[int] = None,
    cv_pitches: Optional[Sequence[str]] = None,
) -> PenaltyFn:
    """Build a penalty function using a tuned config dict."""
    return make_cv_penalty_fn(
        cv_note_count=cv_note_count,
        cv_pitches=cv_pitches,
        tolerance=int(config.get("cv_count_tolerance", 2)),
        penalty_weight=float(config.get("cv_count_weight", 3.0)),
        pitch_prior_weight=float(config.get("cv_pitch_weight", 1.5)),
        pitch_octave_weight=float(config.get("cv_pitch_octave_weight", 3.0)),
    )


def default_soft_penalty(prefix: Sequence[str], candidate: str) -> float:
    return pitch_range_penalty(prefix, candidate) + accidental_consistency_penalty(prefix, candidate)


def _apply_length_penalty(score: float, length: int, alpha: float) -> float:
    if alpha <= 0.0:
        return score
    normalizer = ((5.0 + length) / 6.0) ** alpha
    return score / normalizer


def constrained_beam_search(
    step_fn: StepFn,
    vocabulary: Optional[OMRVocabulary] = None,
    config: Optional[BeamSearchConfig] = None,
    soft_penalty_fn: Optional[PenaltyFn] = None,
    prefix_tokens: Optional[Sequence[str]] = None,
) -> List[BeamHypothesis]:
    def _wrapped_step_fn(prefix: Sequence[str], _: object | None) -> Tuple[Dict[str, float], object | None]:
        return step_fn(prefix), None

    return constrained_beam_search_with_state(
        step_fn=_wrapped_step_fn,
        vocabulary=vocabulary,
        config=config,
        soft_penalty_fn=soft_penalty_fn,
        prefix_tokens=prefix_tokens,
    )


def constrained_beam_search_with_state(
    step_fn: StepFnWithState,
    vocabulary: Optional[OMRVocabulary] = None,
    config: Optional[BeamSearchConfig] = None,
    soft_penalty_fn: Optional[PenaltyFn] = None,
    prefix_tokens: Optional[Sequence[str]] = None,
) -> List[BeamHypothesis]:
    vocab = vocabulary or build_default_vocabulary()
    search_config = config or BeamSearchConfig()
    penalty_fn = soft_penalty_fn or default_soft_penalty

    prefix = list(prefix_tokens) if prefix_tokens is not None else ["<bos>"]
    grammar = GrammarFSA(vocab)
    grammar.validate_sequence(prefix)
    beams = [BeamHypothesis(tokens=prefix, score=0.0, grammar=grammar, state=None)]

    for _ in range(search_config.max_steps):
        expanded: List[BeamHypothesis] = []
        all_complete = True

        for beam in beams:
            if beam.is_complete:
                expanded.append(beam)
                continue
            all_complete = False

            logits, beam_state = step_fn(beam.tokens, beam.state)
            valid_tokens = beam.grammar.valid_next_tokens()
            if not valid_tokens:
                continue

            candidates = []
            score_tokens = getattr(logits, "score_tokens", None)
            if callable(score_tokens):
                scored_tokens = score_tokens(valid_tokens)
                for token, token_score in scored_tokens.items():
                    penalty = penalty_fn(beam.tokens, token)
                    candidates.append((token, float(token_score) - penalty))
            else:
                for token in valid_tokens:
                    if token not in logits:
                        continue
                    penalty = penalty_fn(beam.tokens, token)
                    candidates.append((token, float(logits[token]) - penalty))
            if not candidates:
                continue

            candidates.sort(key=lambda item: item[1], reverse=True)
            for token, adjusted_score in candidates[: search_config.beam_width]:
                next_grammar = _clone_grammar(beam.grammar)
                next_grammar.step(token, strict=True)
                expanded.append(
                    BeamHypothesis(
                        tokens=[*beam.tokens, token],
                        score=beam.score + adjusted_score,
                        grammar=next_grammar,
                        state=beam_state,
                    )
                )

        if not expanded:
            break
        if all_complete:
            beams = expanded
            break

        expanded.sort(
            key=lambda beam: _apply_length_penalty(
                score=beam.score,
                length=len(beam.tokens),
                alpha=search_config.length_penalty_alpha,
            ),
            reverse=True,
        )
        beams = expanded[: search_config.beam_width]

    beams.sort(
        key=lambda beam: _apply_length_penalty(
            score=beam.score,
            length=len(beam.tokens),
            alpha=search_config.length_penalty_alpha,
        ),
        reverse=True,
    )
    return beams


def greedy_from_logits(logits_by_step: Sequence[Dict[str, float]]) -> List[str]:
    if not logits_by_step:
        return ["<bos>", "<eos>"]
    vocab = build_default_vocabulary()
    step_index = {"value": 0}

    def _step_fn(_: Sequence[str]) -> Dict[str, float]:
        idx = min(step_index["value"], len(logits_by_step) - 1)
        step_index["value"] += 1
        return logits_by_step[idx]

    result = constrained_beam_search(
        step_fn=_step_fn,
        vocabulary=vocab,
        config=BeamSearchConfig(beam_width=1, max_steps=len(logits_by_step)),
    )
    return result[0].tokens if result else ["<bos>", "<eos>"]
