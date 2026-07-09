STATE_EXTRACT_PROMPT = """Analyze history and the latest message.
Output JSON with two keys:
1. 'intent': exactly one of ['refuse', 'compare', 'refine', 'evaluate']
   - refuse: out of scope / legal / general hiring advice.
   - compare: difference between assessments.
   - refine: change constraints.
   - evaluate: anything else (vague or specific intent to recommend).
2. 'slots': dictionary with these keys when present:
   - role: target job or hiring role.
   - seniority: target level.
   - skills: list of required skills or technologies.
   - languages: list of requested assessment languages.
   - duration: requested time limit or duration constraint.
   - remote: yes/no if remote delivery is requested.
   - adaptive: yes/no if adaptive testing is requested.
   Extract ONLY new or updated slots.

History:
{history}"""

INFO_SUFFICIENCY_PROMPT = """Analyze extracted slots.
Determine if we have enough context to recommend an assessment.
Enough context means a role or specific assessment/skill family is present.
Output JSON with one boolean key: 'missing_slots'.

Slots: {slots}"""

QUERY_FORM_PROMPT = """Create a search query for SHL catalog.
Use ONLY the active constraints. No fluff.
NEVER include Pre-packaged Job Solutions. Filter them out.
Include role, seniority, skills, languages, duration, remote, and adaptive constraints when present.

Slots: {slots}"""

CONFIDENCE_PROMPT = """Evaluate search results against slots.
Output JSON with boolean key 'confidence_ok'.
True if at least one result strongly matches.

Slots: {slots}
Candidates: {candidates}"""

HANDLE_REFUSAL_PROMPT = """Politely refuse the request.
Explain you only recommend SHL assessments.

Message: {message}"""

HANDLE_COMPARE_PROMPT = """Compare the specific assessments.
Use ONLY retrieved items. DO NOT invent items.

Message: {message}
Candidates: {candidates}"""

ASK_QUESTION_PROMPT = """Ask a clarifying question.
We are missing critical slots (like role or seniority).

History: {history}
Slots: {slots}"""

GROUNDED_GEN_PROMPT = """Introduce the assessment shortlist.
Connect them to user constraints.
- ONLY mention assessment names explicitly provided in the Candidates list. NEVER invent names.
- If the user mentions a competitor or outside product, you MUST explicitly state that you only recommend SHL assessments before providing any SHL equivalents.
- Do NOT generate markdown formatting inside the JSON fields.

Slots: {slots}
Candidates: {candidates}"""
