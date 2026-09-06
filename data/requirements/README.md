# Atomised regulatory requirements

Compliance gap detection is only worth anything if the requirements are real. A
gap like *"OISD-STD-105 requires gas testing to be repeated if a hot-work permit
is suspended for more than two hours; SOP-4412 revision 3 does not mention
re-testing after suspension"* lands with anyone who has run a plant. A gap like
*"your document does not mention safety"* does not.

That cuts both ways: a requirement the system asserts must be traceable, or the
finding built on it is worthless. So every requirement row carries provenance,
and the API reports the breakdown on every compliance response.

## The `text_status` field — read this before adding anything

| Value | Meaning | May be used to assert a regulatory position? |
|---|---|---|
| `verbatim` | The obligation text is reproduced exactly from a publicly available source, and `provenance_note` gives that source and the retrieval date. | Yes |
| `paraphrase_for_demo` | The obligation is a plain-English restatement written for this project, based on the published *structure* of the standard. The clause number identifies where such an obligation sits; the wording is not the standard's. | **No** |

**Everything currently shipped in `atomised_requirements.json` is
`paraphrase_for_demo`.** Nothing in this repository reproduces the text of a
paywalled standard, and nothing here should be relied on for an actual audit.

The distinction is enforced end to end: `requirements.text_status` is `NOT NULL`,
`POST /api/v1/compliance` returns `requirement_provenance` counting rows by
status, and the compliance dashboard renders the count so a viewer can see at a
glance how much of a coverage figure rests on paraphrase.

## Adding verbatim requirements

Several of the instruments named in the brief publish freely. Where you can
obtain the text legally, replace the paraphrase and set:

```json
{
  "text_status": "verbatim",
  "provenance_note": "Factories Act 1948, Section 21(1)(iv). Retrieved from <url> on <date>. Public domain (Government of India gazette text)."
}
```

Do not set `verbatim` from memory, from a summary, or from a secondary source
that quotes the standard. If you cannot cite where the exact words came from, it
is a paraphrase.

## Schema

Loaded by `scripts/load_requirements.py` into the `requirements` table.

| Field | Type | Notes |
|---|---|---|
| `req_id` | string | Stable identifier, e.g. `OISD-105-4.2-c`. Sub-letters denote the atomisation: one clause usually contains several independently testable obligations. |
| `source_standard` | string | e.g. `OISD-STD-105`, `Factories Act 1948` |
| `clause` | string | e.g. `4.2` |
| `obligation_text` | string | One obligation, independently testable |
| `modality` | enum | `shall` \| `should` \| `may` — these carry different legal weight and must never be flattened |
| `applies_to_class` | string \| null | Equipment class code the obligation attaches to |
| `frequency_months` | int \| null | For recurring obligations; drives evidence-staleness detection |
| `limit_json` | object \| null | Structured quantitative limits, e.g. `{"max_hours": 8}` |
| `trigger_text` | string \| null | The condition that makes the obligation bite |
| `testable_by` | enum | `permit_record_field` \| `procedure_text` \| `graph_state` \| `evidence_document` — determines which check applies |
| `effective_from` | date \| null | |
| `text_status` | enum | See above |
| `provenance_note` | string | Mandatory. Where the text came from. |

## Why atomisation matters

Regulations are written as prose paragraphs containing several obligations,
conditions and exceptions. A paragraph cannot be matched to a control. One
sentence about hot-work permits typically decomposes into four independently
testable requirements — validity duration, gas test before commencement, gas test
at an interval, and gas test after a suspension. The fourth is the one plants
most often miss, and it is invisible unless the paragraph was decomposed.
