# Real source corpus — what goes here and where to get it

This tree is for **real industrial documents**. It ships empty, and that is
deliberate: nothing in this repository fabricates a document and calls it real.
`.gitignore` excludes the contents, because most industrial documents are either
licensed or contain personal data.

The synthetic corpus under `data/synthetic/generated/` exists so the pipeline can
be exercised end to end today. It is labelled `synthetic_test_data` everywhere it
appears — in the database, in the API, and as a badge on every dashboard value
derived from it. Adding real documents here raises the ceiling on every metric the
evaluation harness reports; synthetic-only data caps it.

## Layout

```
data/corpus/
├── pid/           Piping & instrumentation diagrams
├── work_orders/   CMMS exports (CSV) or individual work order PDFs
├── sops/          Standard operating procedures, operating instructions
├── inspections/   UT/RT/MPI reports, thickness surveys
└── incidents/     Incident and near-miss investigation reports
```

These are the five classes the blueprint recommends, and between them they cover
structured, semi-structured, unstructured, tabular and visual sources — so
"heterogeneous ingestion" is a claim the corpus can actually support.

## Where to obtain each class legally

| Class | Source | Licence note |
|---|---|---|
| **P&ID** | Engineering course materials, university teaching sites, vendor documentation and open textbooks publish complete sample P&IDs. One good sheet is enough — depth on one drawing beats ten mediocre ones. | Check the page's own terms; many teaching materials are CC-licensed. Record it. |
| **Work orders** | Anonymised samples from an industry contact are worth more than a thousand synthetic rows: the abbreviations alone will change your parser. Otherwise, use the synthetic generator. | If from a plant, confirm in writing that anonymised release is permitted. |
| **SOPs** | Public-sector and utility operators publish operating procedures; equipment OEMs publish installation, operation and maintenance manuals containing procedural sections. | OEM manuals are usually free to download but not free to redistribute. Keep them local; do not commit. |
| **Inspection reports** | NDT training material and published integrity case studies include real thickness survey tables. | Check terms. |
| **Incident reports** | Published accident investigation reports from safety regulators and boards are detailed, well-written, and describe real causal chains — the best possible input for the lessons-learned engine. | Usually free to reuse with attribution. Record the URL. |
| **Regulations** | Factories Act and state Factory Rules, OISD standards, PESO rules, CPCB consent conditions, CEA regulations. See `data/requirements/README.md` for how requirement text provenance is tracked. | Indian gazette text is public domain. ISO/IEC standards are not — do not paste paywalled text into this repo. |

## Required: a manifest entry per document

Every file added here needs a row in `MANIFEST.json` (schema:
`MANIFEST.schema.json`). If a judge or an auditor asks "is this real data?", a
precise answer builds trust and a vague one destroys it.

```json
{
  "filename": "cdu1-sheet3-pid.pdf",
  "doc_class": "pid",
  "origin": "https://example.edu/process-engineering/samples/  (retrieved 2026-09-06)",
  "licence": "CC BY 4.0",
  "contains_personal_data": false,
  "redaction_applied": null,
  "notes": "Sample teaching drawing; tags renamed to match the demo asset spine."
}
```

## Personal data

Real documents contain names, contractor details and signatures. Before ingesting
anything from a plant:

1. redact personal data, or confirm you have a lawful basis to process it;
2. set `contains_personal_data` and `redaction_applied` in the manifest;
3. remember that ingestion copies the file into the blob store under
   `BLOB_ROOT` — deleting the original does not remove the copy.

A PII redaction pass is **not implemented** in this build. Until it is, treat
this tree as unredacted and do not put anything here you would not be willing to
have stored.

## Ingesting

```bash
curl -X POST http://localhost:8000/api/v1/ingest/paths \
  -H 'Content-Type: application/json' \
  -d '{"paths":["data/corpus"],"data_class":"real_source_document","source_system":"corpus","recursive":true}'
```

`data_class` is required by the API and has no default. A document is either a
real source document or clearly-labelled test data, and the caller has to say
which.
