# V2.1 Evaluation — Context Resolution vs QA Baseline

Generated: `2026-08-28T01:32:12.618929+00:00`

Diagnosis only. No schema/migration/verifier/limit/prompt changes in this sprint.

## Baseline (pre-V2.1)

- Runs: run-ed19de4bcebd4a2f, run-fbb36688ba17480f, run-df4cdd1da0bb45e8, run-5df89dbf77e74152, run-68f2aa1ce8734ae2
- Success: **2/5** (0.4)
- Avg latency: **66725.3 ms** (median 75223.8)
- Avg tool calls: **8.4** (max 10)
- Retrieve-more rate: **0.8**
- Verifier acceptance: **0.4**

Understand consistently left `subject=null` while topic=`fractions`.

## Resolver completeness (live API)

### Default resolve (`grade=CLASS_4`)

- Status: `resolved`
- Grade resolved: `{'id': '0f392a75-cf81-4c62-94e4-6041c23baf0b', 'code': 'CLASS_4', 'name': 'Class 4'}` — **matches** `GradeCurriculum.grade_id` (`b1bcff00-3d07-4e92-b426-97e3bfee12ec`)
- Units: 3 (C4-U04, C4-U05, C4-U06) · Learning outcomes: 10

### Correct grade_id resolve

- Status: `resolved`
- Units (3):
  - `C4-U04` — Number and Numeration FRACTION
  - `C4-U05` — Number and Numeration OPERATION ON FRACTIONS.
  - `C4-U06` — Number and Numeration Operation on Fraction. (Multiplication)
- Learning outcomes (10):
  - `C4U05-LO01` parent=`C4-U05` provenance=True eq=None
  - `C4U06-LO01` parent=`C4-U06` provenance=True eq=None
  - `C4U04-LO01` parent=`C4-U04` provenance=True eq=None
  - `C4U04-LO02` parent=`C4-U04` provenance=True eq=None
  - `C4U06-LO02` parent=`C4-U06` provenance=True eq=None
  - `C4U05-LO02` parent=`C4-U05` provenance=True eq=None
  - `C4U04-LO03` parent=`C4-U04` provenance=True eq=None
  - `C4U05-LO03` parent=`C4-U05` provenance=True eq=None
  - `C4U04-LO04` parent=`C4-U04` provenance=True eq=None
  - `C4U05-LO04` parent=`C4-U05` provenance=True eq=None

### Authoritative GradeCurriculum inventory (false-narrowing check)

- Fraction units in GC tree: **3**
- LO codes: C4U04-LO01, C4U04-LO02, C4U04-LO03, C4U04-LO04, C4U05-LO01, C4U05-LO02, C4U05-LO03, C4U05-LO04, C4U06-LO01, C4U06-LO02
- Resolver with correct grade_id returns the same unit set: **A_complete**

## V2.1 golden question (10 runs)

Question: _What are the learning objectives for fractions in Primary 4?_

- Success: **3/10** (0.3)
- Avg latency: **39098.9 ms** (median 38597.7)
- Avg tool calls: **4.9** (max 8)
- Retrieve-more rate: **0.5**
- Avg resolver calls: **1.4**
- Avg legacy calls: **3.5**
- Sequence classes: `{'acceptable_fallback': 3, 'failure': 7}`
- Verifier acceptance: **0.3**

## Comparison

| Metric | Pre-V2.1 | Broken V2.1 | Fixed V2.1 |
| --- | ---: | ---: | ---: |
| success rate | 0.4 | 0.1 | **0.3** |
| avg latency (ms) | 66725 | 27734 | **39099** |
| avg tool calls | 8.4 | 5.0 | **4.9** |
| retrieve_more rate | 0.8 | 1.0 | **0.5** |
| verifier acceptance | 0.4 | 0.1 | **0.3** |
| resolver `CLASS_4` probe | n/a | not_found | **resolved (3u / 10 LO)** |
| avg resolver calls | n/a | 2.0 | **1.4** |

## Failure classification

- `run-812be1a52e2b4753` — **verification**: no_retrieval_progress
- `run-5f364e80690e450f` — **verification**: no_retrieval_progress
- `run-02f25e00a200468b` — **verification**: no_retrieval_progress
- `run-71d6aa3862e9465b` — **verification**: no_retrieval_progress
- `run-0ee2f0091d024e18` — **verification**: no_retrieval_progress
- `run-77ee2fb278cd47a7` — **verification**: no_retrieval_progress
- `run-41d9a68b158648a4` — **verification**: no_retrieval_progress
- `run-6d5055d0b4f4405b` — **verification**: no_retrieval_progress
- `run-603ddd9d24f84178` — **verification**: no_retrieval_progress
- `run-84d70a71283e4e72` — **verification**: no_retrieval_progress
- `run-b0158f389b04458b` — **verification**: no_retrieval_progress

## Stress matrix

- [stress_direct_lo] status=`insufficient_evidence` tools=['resolve_curriculum_context', 'search_curriculum', 'get_learning_objectives', 'get_learning_objectives', 'resolve_curriculum_context'] class=`failure` resolver=2
- [stress_direct_c4u06] status=`insufficient_evidence` tools=['resolve_curriculum_context', 'search_curriculum', 'get_curriculum_structure', 'get_learning_objectives', 'get_topic', 'get_topic'] class=`failure` resolver=1
- [stress_grade_subject_topics] status=`insufficient_evidence` tools=['get_curriculum_structure', 'resolve_curriculum_context', 'search_curriculum', 'get_learning_objectives'] class=`failure` resolver=1
- [stress_topic_ambiguity_no_grade] status=`insufficient_evidence` tools=['search_curriculum', 'search_curriculum', 'get_curriculum_structure', 'get_learning_objectives', 'resolve_curriculum_context', 'resolve_curriculum_context', 'resolve_curriculum_context', 'get_topic', 'get_topic', 'get_topic'] class=`failure` resolver=3
- [stress_missing_subject_phrasing] status=`completed` tools=['resolve_curriculum_context', 'search_curriculum', 'get_learning_objectives', 'get_learning_objectives'] class=`acceptable_fallback` resolver=1
- [stress_unknown_topic] status=`completed` tools=['resolve_curriculum_context', 'search_curriculum', 'get_curriculum_structure'] class=`acceptable_fallback` resolver=1

## Subject=null

Baseline understand left subject=null. V2.1 tool selection for LO questions can still call resolve with grade+topic only; with the grade-resolution fix, `grade=CLASS_4` + inferred subject now resolves. Observed understand subjects in golden runs: [None, …]. `subject=null` remains a separate follow-up for missing-subject phrasing and needs_context paths.

## Recommendation

**Grade resolution fixed** — resolver probe and golden tool calls now return authoritative Fractions context. End-to-end success (3/10) is still below pre-V2.1 (2/5) due to verifier / legacy-fallback behavior; do not change verifier/limits/prompts in this sprint.

### Next engineering change

- Address `subject=null` from understand for missing-subject phrasing.
- Re-evaluate verifier once structured evidence reliably reaches GENERATE.
- Keep legacy tools as fallback until end-to-end success stabilizes.
