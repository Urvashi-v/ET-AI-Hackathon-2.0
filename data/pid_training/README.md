# P&ID equipment-symbol detection — what a model would need

**Nothing in this directory is trained, and no accuracy figure is claimed
anywhere in this project.** This document exists because equipment-symbol
detection is the one part of P&ID digitisation that cannot be done with a
classical transform, and the honest response to that is to specify the dataset
rather than to emit boxes with guessed labels.

## What already works without a model

Three of the four detectors in `services/ingest/pid.py` need no training data at
all, which is why they are the ones that are built:

| Detector | Method | Why no training |
|---|---|---|
| Tag localisation | PDF word geometry + the project's tag grammar | The coordinates are already in the file; nothing is inferred |
| Instrument bubbles | `cv2.HoughCircles` | ISA 5.1 draws instruments as circles, and a Hough transform finds circles |
| Process lines | `cv2.HoughLinesP` | Pipe runs are straight and orthogonal |

Between them these recover the tags, the instruments and a lower bound on
connectivity — enough to link a drawing into the knowledge graph, which is the
capability that matters most.

## What needs a model, and why

Telling a centrifugal pump from a vessel from a shell-and-tube exchanger means
recognising a **silhouette**, and there is no transform for that. The symbols
are:

* geometrically varied — a pump is a circle with two tangent lines, a vessel is
  a rounded rectangle, an exchanger is a circle with an internal zigzag;
* drawn at different scales and rotations on the same sheet;
* frequently overlapped by lines, tags and leader arrows;
* not standardised across companies — ISA 5.1, ISO 10628 and most owner-operator
  standards differ in detail.

No pretrained detector ships with ISA symbology. Fine-tuning a general object
detector is the practical route, and that needs labelled sheets.

## The dataset required

```
data/pid_training/
├── README.md            this file
├── dataset.yaml         class list and split paths (Ultralytics format)
├── images/
│   ├── train/           .png renders of P&ID pages at ~300 dpi
│   ├── val/
│   └── test/
└── labels/
    ├── train/           one .txt per image, YOLO format
    ├── val/
    └── test/
```

**Label format** — one line per symbol, normalised to the image:

```
<class_id> <x_centre> <y_centre> <width> <height>
```

**Classes** (`dataset.yaml`), chosen to match the ontology already in the graph
so a detection maps onto an `Equipment` node without translation:

| id | class | ISA/ISO silhouette |
|---|---|---|
| 0 | `centrifugal_pump` | circle with two tangent discharge lines |
| 1 | `positive_displacement_pump` | circle with internal chevron |
| 2 | `vessel` | vertical rounded rectangle |
| 3 | `column` | tall rectangle with internal trays |
| 4 | `heat_exchanger` | circle with internal zigzag, or TEMA rectangle |
| 5 | `air_cooler` | rectangle with fan symbol |
| 6 | `compressor` | trapezoid |
| 7 | `control_valve` | bowtie with actuator |
| 8 | `manual_valve` | plain bowtie |
| 9 | `relief_valve` | angled body with spring |
| 10 | `filter_strainer` | rectangle with diagonal screen |

### Scale needed

These are the honest numbers for this problem class, not a promise about
achievable accuracy:

* **Minimum viable**: ~200 annotated sheets, ~50 instances of each class. Enough
  to fine-tune and to find out whether the approach works on your drawing
  standard at all.
* **Production**: 1,000+ sheets spanning every drawing office and CAD system in
  the estate. Symbol conventions drift between decades and contractors, and a
  model trained on one vendor's sheets degrades sharply on another's.
* **Split**: 70/20/10 train/val/test, **split by sheet and by drawing office**,
  never randomly by image. Random splitting puts near-identical sheets from one
  project on both sides and reports an accuracy that will not survive contact
  with a new contractor's drawings.

### Annotation effort

Roughly 20–40 minutes per sheet for a competent annotator, so 200 sheets is
about two person-weeks. It needs someone who can read a P&ID: the distinction
between a control valve and a manual valve with a hand actuator is not
recoverable by an annotator who has not been taught it, and mislabelled training
data is worse than less training data.

### Suggested approach

Fine-tune a small YOLO variant (`yolov8n`/`yolov8s`) on rendered pages at ~300
dpi. Reasons: symbols are small and numerous, which suits a dense single-stage
detector; sheets render deterministically so augmentation can be limited to
rotation and scale; and the model is small enough to run on the ingest worker's
CPU without a GPU budget.

**Evaluate on held-out sheets from drawing offices absent from training**, and
report mAP@0.5 per class. A single headline number hides exactly the failure that
matters — a model that finds every pump and no exchangers.

## What would be added on top

`services/ingest/pid.py` reports `equipment_symbols` as `not_implemented` and
returns nothing for it. With a trained model, that stage would:

1. Run the detector over the same raster the Hough detectors use.
2. Associate each detected symbol with the nearest tag detection, which already
   has exact coordinates — so a box classified `centrifugal_pump` next to the
   text `P-101B` becomes a typed, tagged, located symbol.
3. Write it into `drawing_detections` with `kind='equipment_symbol'`, the class
   as `text`, and the detector's own score as `confidence` — clearly labelled as
   a model output, distinct from the geometric detectors already there.
4. Let topology extraction connect *symbols* rather than tag rectangles, which
   is what makes real process-route reconstruction possible.

Until that dataset exists, the system links drawings to the graph through tags
and instruments, and says plainly that it does not recognise equipment
silhouettes.
