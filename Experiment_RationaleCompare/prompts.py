#!/usr/bin/env python3
from typing import Dict

ORACLE_TEMPLATE = """System: SLU Logic Analyst. Rationalize the "Target JSON" using "Transcript" and "DB Definitions".

Output Format:
C: [Competing Intents|Separated by |]; [Competing Slots:Value1|Value2]
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Target JSON]

Rules:
- Contrast similar/inclusive labels and ambiguous slot extractions (e.g., specific vs broader intent like alarm_set vs alarm_query).
- C must include at least 3 competing intents and 2 competing slot values.
- R must mention slot rationale (why a slot value is supported or rejected).
- Cite DB rules vs transcript evidence.
- Output exactly 3 lines (C, R, J) and nothing else.
- List candidates in the exact order they appear in DB Definitions.
- No conversational filler.

---
[DB Definitions]
{db_definitions}

[Input Data]
- Transcript: {gold_text}
- Target JSON: {gold_json}
"""

INFER_TEXT_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Transcript" and "DB Definitions".

Output Format:
C: [Potential Intents|Separated by |]; [Potential Slots:Value1|Value2]
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Final JSON]

Rules:
- Compare candidates from DB before deciding.
- C must include at least 3 competing intents and 2 competing slot values.
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
"""

INFER_AUDIO_TEMPLATE = """System: SLU Logic Analyst. Infer the intent and slots using "Audio" and "DB Definitions".

Output Format:
C: [Potential Intents|Separated by |]; [Potential Slots:Value1|Value2]
R: [Label]![Rejection Reason]; [Label]*[Adoption Reason]
J: [Final JSON]

Rules:
- Compare candidates from DB before deciding.
- C must include at least 3 competing intents and 2 competing slot values.
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
