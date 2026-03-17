#!/usr/bin/env python3
"""Constrained beam search decoding for OMR token generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.decoding.grammar_fsa import GrammarFSA
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
