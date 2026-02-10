#!/usr/bin/env python3
from typing import Dict

ORACLE_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Transcript", "DB Definitions", and "Target JSON".

Rules:
- Compare candidates from DB before deciding.
- In C, list candidate intents: include the target label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types). If a slot has 2+ plausible values, include value candidates as slot_type(value1|value2). If only one value, omit parentheses. If no slots, write (none).
- R must list only rejection reasons for candidates in C (except the accepted label). Do not write adoption reasons.
- The accepted label is the remaining one and should match J.
- Cite specific evidence from transcript.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.
- J must exactly match the provided Target JSON (verbatim). Do not alter it.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}
- Target JSON: {gold_json}

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1(value1|value2) | slot_type2 | slot_type3
R: label1!reason1; label2!reason2; ...
J: [Target JSON]
"""

INFER_TEXT_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Transcript" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- In C, list candidate intents: include the predicted label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types). If a slot has 2+ plausible values, include value candidates as slot_type(value1|value2). If only one value, omit parentheses. If no slots, write (none).
- R must list only rejection reasons for candidates in C (except the accepted label). Do not write adoption reasons.
- The accepted label is the remaining one and should match J.
- Cite specific evidence from transcript.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.
- J must be a single-line valid JSON object (no markdown, no code fence).
- J must use this schema exactly:
  {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
- If there are no entities, use: {{"scenario": "<string>", "action": "<string>", "entities": []}}
- Use double quotes for all JSON keys/strings.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1(value1|value2) | slot_type2 | slot_type3
R: label1!reason1; label2!reason2; ...
J: {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
"""

INFER_AUDIO_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Audio" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- In C, list candidate intents: include the predicted label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types). If a slot has 2+ plausible values, include value candidates as slot_type(value1|value2). If only one value, omit parentheses. If no slots, write (none).
- R must list only rejection reasons for candidates in C (except the accepted label). Do not write adoption reasons.
- The accepted label is the remaining one and should match J.
- Cite specific evidence from audio.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.
- J must be a single-line valid JSON object (no markdown, no code fence).
- J must use this schema exactly:
  {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
- If there are no entities, use: {{"scenario": "<string>", "action": "<string>", "entities": []}}
- Use double quotes for all JSON keys/strings.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Audio: <AUDIO>

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1(value1|value2) | slot_type2 | slot_type3
R: label1!reason1; label2!reason2; ...
J: {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
"""

INFER_TEXT_TEMPLATE_NO_COT = """System: SLU Logic Analyst. Infer the intent and slots using "Transcript" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- Consider plausible similar/inclusive alternatives and ambiguous slot values internally before finalizing J.
- Cite specific evidence from transcript in your internal reasoning.
- List candidates in the exact order they appear in DB Definitions when making internal comparisons.
- Output exactly 1 line (J) and nothing else.
- No conversational filler.
- J must be a single-line valid JSON object (no markdown, no code fence).
- J must use this schema exactly:
  {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
- If there are no entities, use: {{"scenario": "<string>", "action": "<string>", "entities": []}}
- Use double quotes for all JSON keys/strings.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}

Output Format:
J: {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
"""

INFER_AUDIO_TEMPLATE_NO_COT = """System: SLU Logic Analyst. Infer the intent and slots using "Audio" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- Consider plausible similar/inclusive alternatives and ambiguous slot values internally before finalizing J.
- Cite specific evidence from audio in your internal reasoning.
- List candidates in the exact order they appear in DB Definitions when making internal comparisons.
- Output exactly 1 line (J) and nothing else.
- No conversational filler.
- J must be a single-line valid JSON object (no markdown, no code fence).
- J must use this schema exactly:
  {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
- If there are no entities, use: {{"scenario": "<string>", "action": "<string>", "entities": []}}
- Use double quotes for all JSON keys/strings.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Audio: <AUDIO>

Output Format:
J: {{"scenario": "<string>", "action": "<string>", "entities": [{{"type": "<entity_type>", "filler": "<entity_value>"}}, ...]}}
"""


def render_oracle_prompt(db_definitions: str, gold_text: str, gold_json: str) -> str:
    return ORACLE_TEMPLATE.format(
        db_definitions=db_definitions,
        gold_text=gold_text,
        gold_json=gold_json,
    )


def render_infer_text_prompt(db_definitions: str, gold_text: str) -> str:
    return INFER_TEXT_TEMPLATE.format(
        db_definitions=db_definitions,
        gold_text=gold_text,
    )


def render_infer_audio_prompt(db_definitions: str) -> str:
    return INFER_AUDIO_TEMPLATE.format(db_definitions=db_definitions)


def render_infer_text_prompt_no_cot(db_definitions: str, gold_text: str) -> str:
    return INFER_TEXT_TEMPLATE_NO_COT.format(
        db_definitions=db_definitions,
        gold_text=gold_text,
    )


def render_infer_audio_prompt_no_cot(db_definitions: str) -> str:
    return INFER_AUDIO_TEMPLATE_NO_COT.format(db_definitions=db_definitions)
