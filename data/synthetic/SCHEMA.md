# Synthetic corpus — schema and provenance

Everything produced by `generate.py` is **SYNTHETIC TEST DATA**. It is not a real
plant, not a real incident, and must never be presented as evidence of anything.
Every generated file carries a `SYNTHETIC TEST DATA` banner in its own content,
every row carries a `data_class` of `synthetic_test_data` in the database, and the
dashboard renders that badge on any value derived from it.

It exists for two reasons: to exercise the pipeline end to end on Day 1, and to
give the evaluation harness a corpus whose ground truth is known exactly.

## Why generated rather than random

Clean synthetic data makes a pipeline look good and then fails publicly on the
first real document. So the generator is built the other way round:

* the **plant model and the event history are scripted**, not sampled. Every
  work order, incident, inspection reading and procedure step below is written
  out explicitly in `generate.py`, so the corpus has a coherent causal story
  rather than noise that happens to typecheck;
* the **surface forms are varied** by a seeded PRNG. That is where the messiness
  lives: which of the six spellings of a tag a given document uses, whether the
  failure code is meaningful or the dropdown default, whether a date is present.

`--seed` defaults to `20260101`. The same seed always produces byte-identical
output, so the golden questions in `eval/golden.jsonl` stay valid and ingestion
stays idempotent across regeneration.

## The plant

A single crude distillation unit, deliberately narrow so that one asset can be
documented in depth.

```
Site    HALDIA (fictional)
└── Plant  CDU-1  Crude Distillation Unit
    └── System  Crude Charge Pumping
        ├── FunctionalLocation  CDU1-PUMP-101   (the position)
        │   ├── Equipment  P-101A   duty pump
        │   └── Equipment  P-101B   standby pump   ← the demo spine
        ├── Equipment  E-104   crude preheat exchanger (P-101B discharge)
        ├── Equipment  V-102   suction drum
        └── Instruments  PIC-101, FT-201, LSHH-305, PSV-204
```

`P-101A` and `P-101B` are a duty/standby pair. They are **siblings, not the same
asset** — the corpus is built so that a resolver which merges them produces
visibly wrong failure counts, which is the point.

## Deliberate messiness

| Defect injected | Where | Why it matters |
|---|---|---|
| Six spellings of the same tag | work orders, incidents, inspections | `P-101B`, `P101B`, `P 101 B`, `10-P-101-B`, `P-101-B`, and `P‑101‑B` with a U+2011 non-breaking hyphen. If entity resolution fails, the graph is six islands. |
| Technician shorthand | work-order long text | "the B pump", "std by pump" — unparseable by grammar, resolvable only from the record's own functional location. |
| Failure code defaulting | `failure_code` column | A documented fraction of corrective work orders carry `OTHER`, the first entry in the CMMS dropdown, while the long text states the real mode. This is what makes "we re-coded them from the technician's own words" a measurable claim rather than a slogan. |
| Missing dates | `closed_on` | Some records never had a close-out date entered. |
| Mixed date formats | across files | `YYYY-MM-DD` and `DD/MM/YYYY` in different exports, as different source systems do. |
| Degradation language | long text | "slight seepage", "topped up", "temporary clamp fitted" — the leading indicators that precede a failure in the record before they appear in any sensor. |

## Files produced

| File | Format | Records | Ingested as |
|---|---|---|---|
| `work_orders_cmms_export.csv` | CSV | 60 | `work_order` |
| `inspection_ut_readings.csv` | CSV | 36 | `inspection_report` |
| `incident_2019_seal_failure.md` | Markdown | 1 | `incident_report` |
| `incident_2022_seal_failure.md` | Markdown | 1 | `incident_report` |
| `sop_4412_crude_charge_pump_startup.md` | Markdown | 1 | `sop` |
| `moc_2023_07_impeller_trim.md` | Markdown | 1 | `moc` |
| `MANIFEST.json` | JSON | — | not ingested; provenance record |

`MANIFEST.json` records the generator version, the seed, and the SHA-256 of every
file, so what was ingested can always be tied back to what was generated.

## Column schema — `work_orders_cmms_export.csv`

| Column | Type | Notes |
|---|---|---|
| `wo_id` | string | `WO-nnnn`, unique |
| `functional_location` | string | `CDU1-PUMP-101` style; present even when the equipment tag is shorthand |
| `equipment_tag` | string | one of the six spellings, varied per row |
| `order_type` | enum | `CORRECTIVE` \| `PREVENTIVE` \| `INSPECTION` |
| `status` | enum | `CLOSED` \| `IN_PROGRESS` |
| `priority` | enum | `1-EMERGENCY` \| `2-HIGH` \| `3-NORMAL` |
| `short_text` | string | one-line description |
| `long_text` | string | the field where the diagnostic information actually lives |
| `as_found` | string | condition on arrival — the only evidence about the failure |
| `as_left` | string | condition on departure |
| `failure_code` | enum | CMMS-coded mode; frequently `OTHER` |
| `opened_on` | date | |
| `closed_on` | date \| empty | |
| `downtime_hrs` | float \| empty | |
| `actual_cost` | int \| empty | INR |

## Column schema — `inspection_ut_readings.csv`

| Column | Type | Notes |
|---|---|---|
| `report_no` | string | |
| `equipment_tag` | string | varied spelling |
| `cml_id` | string | condition monitoring location |
| `ndt_method` | enum | `UT` |
| `inspection_date` | date | multiple surveys per CML across years, so a corrosion rate is computable |
| `thickness_mm` | float | monotonically decreasing per CML by a documented rate |
| `min_thickness` | float | retirement limit |
| `inspector` | string | pseudonymised initials only |
| `observation` | string | |
