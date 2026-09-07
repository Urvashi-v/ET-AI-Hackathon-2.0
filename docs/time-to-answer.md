# Time-to-answer study — **NOT YET RUN**

> **No participant data exists.** No engineer has been timed using this system or
> the manual workflow it is meant to replace. Every number in this document is a
> placeholder in a protocol, not a result. Nothing here should be quoted, put on
> a slide, or fed into the ROI model as a measured input.
>
> The ROI calculator marks its time-saving inputs **ASSUMPTION** for exactly this
> reason. When this study runs, those inputs become **MEASURED** and the label
> changes with them.

This document exists because "cuts search time by 80%" is the easiest claim in
this domain to make and the hardest to substantiate, and a protocol nobody has
executed is more honest than a figure nobody has measured.

---

## What would be measured

**Primary outcome.** Wall-clock seconds from a question being posed to the
participant stating an answer *and naming the document it came from*. The second
half is not optional: an unsourced answer in a safety-critical domain is not an
answer, and a protocol that stops at "they said something" measures confidence
rather than retrieval.

**Secondary outcomes.**

| Outcome | Why it is measured |
|---|---|
| Answer correctness | Speed is worthless if the answer is wrong. Graded against a reference by an assessor who does not know which arm produced it. |
| Source correctness | Whether the document named actually contains the answer. |
| Give-up rate | The manual arm's real failure mode is abandonment, not slowness, and a study that only times completed tasks silently drops the worst cases. |
| Self-reported confidence | 1–5, before the answer is checked. Miscalibration is itself a finding. |

## Design

**Within-participant, counterbalanced.** Each participant does both arms.
Between-participant designs need far more people to overcome the variance
between a graduate and someone who has run the unit for twenty years — and that
variance is larger than the effect being measured.

Counterbalancing matters because of an asymmetry: having answered a question
manually teaches you the answer, so a participant who does manual first is faster
on the system arm for reasons that have nothing to do with the system. Half the
participants take question set A manually and set B assisted; the other half take
the reverse. Sets are matched for difficulty by an engineer who is not running
the study.

**Participants.** Target 12–16, drawn from the roles the system serves: field
technicians, reliability engineers, operations shift staff, HSE. Twelve is not a
large sample and the write-up must say so; it is enough to detect a large effect
and not enough to characterise a small one.

**Tasks.** 16 questions drawn from the same categories as the golden set —
lookup, procedural, diagnostic, multi-hop, aggregate, compliance — plus **at
least two with no answer in the corpus**. Those two carry more weight than the
rest: the manual arm's honest outcome is "I could not find it", and if the
assisted arm produces a confident answer instead, that is a finding that
outweighs any time saved.

## Arms

**Manual.** The participant has what they have today: the document share, the
CMMS, the drawing register, search-in-folder, and a colleague they may phone
(timed, and the colleague's time recorded separately — it is a real cost the
manual arm usually hides).

**Assisted.** The participant has this system. They may still consult anything in
the manual arm; forbidding it would measure compliance with the protocol rather
than how the tool is actually used.

## Procedure

1. Consent, and a statement that the tool is under evaluation and not the
   participant. This matters practically: someone who believes they are being
   assessed works differently.
2. Five-minute orientation on the assisted arm. No training on search technique —
   a tool that needs training to beat a folder has not beaten it.
3. Eight questions per arm, one at a time, timer per question.
4. Hard cap of **10 minutes per question**. A capped attempt is recorded as a
   give-up at 600 s and reported separately; excluding it would delete the manual
   arm's characteristic failure.
5. Short debrief. Where did each arm fail, and what did they distrust.

## Analysis

* **Paired comparison** on median time per question, participant by participant.
  Medians rather than means: one participant hunting a drawing for nine minutes
  should not become the headline.
* **Wilcoxon signed-rank** rather than a paired *t*-test. Task times are skewed
  and censored at the cap, which violates the *t*-test's assumptions.
* **Report the effect size and its confidence interval**, not just a *p*-value.
  With twelve participants a significant result may still be compatible with a
  small effect, and the interval says so where a *p*-value does not.
* **Give-ups analysed separately** and reported as a rate, never imputed.
* **Correctness reported alongside time, always.** A speed figure without an
  accuracy figure beside it invites exactly the misreading it should prevent.

## Threats to validity, stated in advance

| Threat | Why it matters here | Mitigation |
|---|---|---|
| **Corpus is synthetic** | The current corpus is generated. A system that answers well on eleven documents it was tuned against says nothing about eleven thousand real ones. | Run on a real corpus, or state the limitation prominently in the result. |
| **Author-as-evaluator** | Whoever built the system is the worst person to run its study. | An independent facilitator; the assessor grading answers must not know which arm produced them. |
| **Learning across arms** | Answering a question teaches you the answer. | Counterbalanced order and matched question sets. |
| **Hawthorne effect** | Participants work harder while watched, in both arms. | Affects both arms similarly; it does not cancel, and the write-up should say so. |
| **Novelty** | A new tool is used more attentively in its first hour than in its thousandth. | A follow-up measurement after several weeks of real use, if the deployment allows. |
| **Selection** | Volunteers for a search-tool study are people comfortable with search tools. | Recruit through the line organisation rather than by volunteering. |

## Reporting

The result belongs in the README's measured-results table with its method named,
alongside every other figure. It is reported whichever way it comes out — a
study that only gets published when it flatters the system is marketing with a
sample size.

Until it runs, the row reads:

| Metric | Result | Method |
|---|---|---|
| Time-to-answer improvement | **Not measured** | Protocol defined; study not yet run — `docs/time-to-answer.md` |
