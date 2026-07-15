"""
Fixed, human-authored messages and information.
"""

import re
from typing import List

# Prompt templates

_STRUCTURED_SYSTEM = """
You are a calm, warm assistant that helps people work through everyday emotional states.
You have been given structured content — a framework, a practice, and an inquiry — that
is specifically suited to what this person is experiencing.

Your response must:
- Open with the mirror pattern (a reflective acknowledgement of what they said)
- Introduce the framework briefly in plain language
- Present the practice as clear, numbered steps in a way that is genuine, helpful, and above all, safe
- Close with the inquiry question, as an invitation rather than a task
- Be warm, grounded, and never clinical or prescriptive
- Be no longer than 250 words

Do not invent content. Only use what you are given.
""".strip()

_REFLECTIVE_SYSTEM = """
You are a calm, grounded assistant drawing on wisdom traditions to support someone's
reflection. You have been given relevant passages from one or more sources.

Your response must:
- Respond to the person's message directly and warmly
- Weave in at least one of the passages, making clear which source it comes from
- Respond in a way that is genuine, helpful, and above all, safe
- Keep the tone open, not instructive — you are offering a perspective, not an answer
- End with a gentle open question or invitation to stay with the thought
- Be no longer than 250 words
- Cite sources by their label (e.g. "The Tao Te Ching says..." or "From the ACT guide...")

Do not quote text that was not in the passages provided. Do not invent quotes.
""".strip()


# response templates

_SAFETY_MESSAGE = """
What you've shared sounds really important for further discussion, and I want to make sure you get the right support right now — support that goes beyond what I'm able to offer here.

Please reach out to one of the following:

• **Samaritans** — call **116 123**, free, any time, day or night.
• **Shout (Crisis Text Line)** — text **SHOUT to 85258**, free, 24/7.
• **NHS 111** — call **111** or visit **111.nhs.uk** for urgent mental health support.
• **Mind Infoline** — **0300 123 3393**, Monday to Friday, 9am–6pm.

You don't have to explain everything or have the right words.
Just reaching out is enough. These services are there for exactly this.

If you feel you or someone else is in immediate danger, please call **999** or go to your nearest A&E.
""".strip()

_DEFAULT_WATCH_MESSAGE = """
It sounds like things feel quite heavy at the moment.
If at any point you feel you need more support than this,
please know that Samaritans (116 123) are available around the clock.
"""

# resources

_CRISIS_RESOURCES = [
    {"name": "Samaritans", "contact": "116 123", "available": "24/7, free"},
    {"name": "Crisis Text Line (UK)", "contact": "Text SHOUT to 85258", "available": "24/7, free"},
    {"name": "NHS 111", "contact": "111 or 111.nhs.uk", "available": "24/7"},
    {"name": "Mind Infoline", "contact": "0300 123 3393", "available": "Mon–Fri 9am–6pm"},
]


# Keyword heuristics
# Patterns that are strong signals of crisis or inability to cope.
# Conservative: aim for high recall
_ESCALATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(can'?t|cannot)\s+(go\s+on|cope|continue|do\s+this)\b", re.I),
    re.compile(r"\bno\s+way\s+(forward|out)\b", re.I),
    re.compile(r"\b(end\s+it|end\s+my\s+life|kill\s+myself|hurt\s+myself)\b", re.I),
    re.compile(r"\b(suicid|self[- ]harm)\w*\b", re.I),
    re.compile(r"\bdon'?t\s+(want\s+to\s+be\s+here|feel\s+able\s+to\s+cope)\b", re.I),
    re.compile(r"\b(crisis|breakdown|desperate|hopeless)\b", re.I),
    re.compile(r"\bmore\s+than\s+i\s+can\s+handle\b", re.I),
    re.compile(r"\bneed\s+to\s+talk\s+to\s+someone\b", re.I),
]

_WATCH_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(exhausted|drained|overwhelmed|burned?\s*out)\b", re.I),
    re.compile(r"\b(not\s+sleeping|can'?t\s+sleep)\b", re.I),
    re.compile(r"\b(losing\s+it|falling\s+apart|breaking\s+down)\b", re.I),
]

_REFLECTIVE_PATTERNS = [
    re.compile(r"\b(wonder|wondering|meaning|purpose|what\s+matters|values?|life|soul)\b", re.I),
    re.compile(r"\b(lately|recently)\b.*\b(think|thought|reflect|question)\b", re.I),
    re.compile(r"\b(spiralling|spiral|thought\s+loop|challenge\s+(my|the)\s+thought)\b", re.I),
    re.compile(r"\b(cbt|act|acceptance|defusion|mindful)\b", re.I),
]

# Remote source configuration
_REMOTE_SOURCES = [
    {
        "id": "tao_te_ching",
        "label": "Tao Te Ching (Legge)",
        "url": "https://sacred-texts.com/tao/taote.htm",
        "type": "html",
        "tradition": "Taoist",
    },
    {
        "id": "bhagavad_gita",
        "label": "The Bhagavad Gita (Arnold)",
        "url": "https://www.gutenberg.org/files/2388/2388-h/2388-h.htm",
        "type": "html",
        "tradition": "Hindu",
    },
    {
        "id": "nhs_cbt",
        "label": "NHS Every Mind Matters — CBT Techniques",
        "url": "https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/",
        "type": "html",
        "tradition": "CBT",
    },
    {
        "id": "act_manual",
        "label": "Valued Living — ACT Self-Help Manual (Ford)",
        "url": "https://contextualscience.org/sites/default/files/Self-Help%20ACT%20Manual%20for%20Anxiety%20&%20Depression.pdf",
        "type": "pdf",
        "tradition": "ACT",
    },
]

# Local source configuration
_LOCAL_SOURCES = [
    {
        "id": "tao_te_ching",
        "label": "Tao Te Ching (Legge)",
        "filename": "Tao Te Ching - Translated by J. Legge _ Internet Sacred Text Archive.html",
        "type": "html",
        "tradition": "Taoist",
    },
    {
        "id": "bhagavad_gita",
        "label": "The Bhagavad Gita (Arnold)",
        "filename": "The Project Gutenberg E-text of The Bhagavad-Gita.html",
        "type": "html",
        "tradition": "Hindu",
    },
    {
        "id": "nhs_cbt",
        "label": "NHS Every Mind Matters — CBT Techniques",
        "filename": "Online self-help CBT techniques - Every Mind Matters - NHS.html",
        "type": "html",
        "tradition": "CBT",
    },
    {
        "id": "act_manual",
        "label": "Valued Living — ACT Self-Help Manual (Ford)",
        "filename": "Self-Help ACT Manual for Anxiety & Depression.pdf",
        "type": "pdf",
        "tradition": "ACT",
    },
]
