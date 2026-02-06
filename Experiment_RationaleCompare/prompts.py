#!/usr/bin/env python3
from typing import Dict

ORACLE_TEMPLATE = """System: SLU Logic Analyst. Rationalize the "Target JSON" using "Transcript" and "DB Definitions".

Rules:
- Contrast similar/inclusive labels and ambiguous slot extractions (e.g., general_greet vs general_quirky, email_querycontact vs email_query).
- In C, list candidate intents: include the target label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types): include competing slot types that could apply; if none, write (none).
- R must mention slot rationale (why a slot value is supported or rejected).
- Cite DB rules vs transcript evidence.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- J must exactly match the provided Target JSON (verbatim). Do not alter it.
- No conversational filler.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}
- Target JSON: {gold_json}

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1 | slot_type2 | slot_type3
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Target JSON]
"""

INFER_TEXT_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Transcript" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- In C, list candidate intents: include the predicted label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types): include competing slot types that could apply; if none, write (none).
- R must mention slot rationale (why a slot value is supported or rejected).
- Cite specific evidence from transcript.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1 | slot_type2 | slot_type3
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Final JSON]
"""

INFER_AUDIO_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Audio" and "DB Definitions".

Rules:
- Compare candidates from DB before deciding.
- In C, list candidate intents: include the predicted label itself and all plausible similar/inclusive alternatives from DB.
- When plausible, include both a broad/umbrella intent and a more specific intent from the same domain family.
- In C, list candidate slots (slot types): include competing slot types that could apply; if none, write (none).
- R must mention slot rationale (why a slot value is supported or rejected).
- Cite specific evidence from audio.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Audio: <AUDIO>

Output Format:
C: Intent candidates: intent1 | intent2 | intent3 | intent4; Slot candidates: slot_type1 | slot_type2 | slot_type3
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Final JSON]
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
