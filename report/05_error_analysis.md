# 5. Error analysis

Subject: the best student before merging, model (c) human + teacher, on Spider dev (it makes 236 errors on
1,034 questions at EX 0.772). 100 failures were sampled at random (seed 0) and categorised by a heuristic that
compares which tables, columns and clauses the predicted and gold SQL use. The labels are approximate. All 100
with gold SQL, prediction and category: `samples/error_samples_100_failures.jsonl`; table:
`data/error_categories.csv`.

| category | count | share |
|---|---|---|
| wrong table/column | 34 | 34% |
| wrong join | 19 | 19% |
| wrong condition | 17 | 17% |
| nesting error (missing/wrong subquery or set operation) | 15 | 15% |
| wrong aggregation/grouping | 12 | 12% |
| other (ordering, limit, select list) | 2 | 2% |
| syntax/execution error | 1 | 1% |

## What it shows
- **The student almost always writes valid SQL.** Only 1% of sampled failures did not execute; the other 99% are
  SQL that runs and answers a slightly different question.
- **The commonest mistake is schema linking:** choosing the wrong column or table (34%), followed by joins
  through the wrong key or table (19%). Example: asked for the three youngest winners' names and ranks, the student
  joined `players` and `rankings` when one column of `matches` was enough.
- **Conditions:** wrong literal or comparison (17%). Example: for "nations which are considered US territory"
  it filtered `GovernmentForm = 'US'` instead of `'US Territory'`.
- **Nesting and set operations** (15%) fail mostly on hard/extra questions: "countries not playing any cartoon
  by Todd Casey" needs `EXCEPT` or `NOT IN`, and the student wrote a `!=` filter, which answers a different
  question.
- **Some failures are benchmark artefacts**, not model mistakes: case-sensitive string matching against a
  lower-cased gold literal (`'Little Lever Urban District'` vs the gold's `"little lever urban district"`), tie
  breaking in `ORDER BY ... LIMIT 1`, and ambiguous questions where either reading is reasonable. Execution
  accuracy is strict, so the real error rate is somewhat lower than 22.8%.

## Student vs teacher, question by question
Final student vs teacher (both zero-shot, 1,034 questions; computed for model (c)):

| | questions |
|---|---|
| both correct | 706 |
| student correct, teacher wrong | 92 |
| teacher correct, student wrong | 103 |
| both wrong | 133 |

The two models fail on largely different questions, which fits the near-equal accuracy. Examples of each case
are in `data/results_generated.md` ("Student vs teacher, side by side"). Typical pattern when the student wins:
the teacher adds an unnecessary join or picks the wrong filter (for "European countries where English is not
official" it wrote `IsOfficial = 'F'` for English instead of excluding countries where English is official); when
the teacher wins, the student invents a join or aggregates the wrong column.

## By hardness
All models degrade with hardness: for the final student 0.92 (easy), 0.81 (medium), 0.71 (hard), 0.51 (extra).
Extra-hard questions (nested queries, multiple set operations, several conditions) remain the main gap
in absolute terms, for the teacher as well (0.49).
