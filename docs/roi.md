# ROI model

`web/impact.html`, backed by `web/js/roi.js`. Every output is arithmetic over
inputs the reader can see and change, and **nothing in it is a claim**. There is
no hard-coded payback period, no "typical customer sees…", and no default that
quietly encodes an optimistic answer.

Open it at `/ui/impact.html`.

## The four labels

The organising idea is provenance, shown on every field and repeated on every
result:

| Label | Meaning |
|---|---|
| **USER INPUT** | The reader typed it, or accepted a default they can see |
| **ASSUMPTION** | Plausible, and **not measured by this project** |
| **MEASURED** | This project measured it, with a stated method |
| **CALCULATED** | Derived by the formula shown beside it |

The distinction that carries the weight is ASSUMPTION versus MEASURED. Three
inputs drive most of the answer and **none of them has been measured**:

* minutes to find an answer manually,
* minutes with the system,
* share of downtime events the system could prevent.

The first two would be measured by the time-to-answer study in
[`time-to-answer.md`](time-to-answer.md), which is written up and **has not been
run**. Presenting them as measured would make every downstream number a
fabrication, so the calculator labels them at the field, in the result, and in a
banner at the top of the page.

## What it computes

Three benefit streams, each from inputs to a stated formula:

**Search time.**
`people × searches/person/day × working days × adoption% × (manual − assisted minutes) ÷ 60 × loaded cost/hour`

**Downtime avoided.**
`events/year × avoidable% × cost/event`

**Audit preparation.**
`audits/year × prep hours × reduction% × loaded cost/hour`

Then `annual net = total benefit − annual running cost`, and payback against the
one-off implementation cost.

Payback returns **undefined, not infinity**, when the system does not pay for
itself. Rendering "∞ months" invites the reader to treat it as a large number
rather than as *never*.

## Defaults are conservative on purpose

An ROI model whose defaults flatter it is a sales tool, and the first person to
change a number and watch the answer collapse stops believing any of it. So:

* adoption defaults to 60%, not 100% — tools get used for some questions and not
  others;
* downtime avoidance defaults to 15%, and the field says anything above ~25% is a
  claim needing evidence. Surfacing an open corrective action does not close it;
  someone still has to act;
* assisted search time includes *opening the cited source to check it*. A figure
  that assumes the answer is trusted unread is measuring something nobody should
  want.

Set time saved to zero and the benefit is zero. The model has no floor.

## Sensitivity

A single ROI number invites belief, so the page shows what happens when the
unmeasured inputs are halved and doubled. That range is what the reader is
actually being offered, and it is usually wider than the headline suggests.

If halving the assumptions turns the annual net negative, the model is resting on
guesses rather than on the system — and the reader can see that immediately
rather than discovering it after committing.

## What deliberately is *not* an input

Retrieval quality, entity F1, citation validity and latency are measured by
`eval/run_eval.py` and reported in the README. **None of them feeds this model.**

They answer whether the system works. This model answers what it would be worth
*if* the assumptions hold. Wiring a measured F1 into a financial projection would
dress an assumption in a measurement's clothes, which is precisely the move this
page exists to avoid.
