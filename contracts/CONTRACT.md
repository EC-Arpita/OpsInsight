# OpsInsight Analysis JSON: Field Guide (schema v1.0)

One analysis run = **one JSON file**. The API returns it, and the dashboard reads it.
The same structure works for any business (`meta.domain` says which). Only the item names change: tickets vs orders, breach vs late.

Real examples built from your data: `example_incidents.json`, `example_sales.json` (regenerate with `notebooks/08_analysis_json.ipynb`).

**Rules for the team**
- Rates are **ratios from 0 to 1** (0.225 = 22.5%). The dashboard turns them into %.
- Dates are `YYYY-MM-DD`; timestamps are `YYYY-MM-DDTHH:MM`.
- `null` means "not available". The dashboard shows "–", never 0.
- Changing a field name or meaning → bump `schema_version` and tell the whole team.

---

## Sections at a glance

| Section | Dashboard screen | Analyst question it answers |
|---|---|---|
| `meta` | header | What data, which period, what are we predicting? |
| `data_quality` | "Data health" panel | Can I trust it? What was removed? |
| `kpis` | KPI cards (top row) | How are we doing now vs before? |
| `trends` | line charts | How did it change week by week? |
| `breakdowns` | tables / bar charts | Which team, category or region is worst? |
| `root_causes` | "Why is this happening" cards | What *causes* the problems? |
| `anomalies` | alert list | Which weeks were unusual? |
| `forecast` | workload chart | How much work is coming? Will a team be overloaded? |
| `predictions` | risk list + "Why?" panel | Which items are at risk right now, and why? |
| `recommendations` | action cards with Approve / Reject | What should we do and what will it gain? |
| `impact_simulation` | before/after chart | What changes if we act? |
| `model_performance` | "About the model" panel | How reliable are the predictions? |

---

## `meta`
| Field | Meaning |
|---|---|
| `analysis_id` | unique id of this run |
| `generated_at` | when the analysis ran (UTC) |
| `domain` | `support_tickets` or `sales_orders` (later: any) |
| `item_type` | what one row is: `ticket` / `order` |
| `target.name`, `target.definition` | what we predict, in plain words |
| `prediction_moment` | when the prediction is made (only info known then is used) |
| `data_period` | first and last date in the data |
| `current_period` / `previous_period` | the two periods KPIs compare (equal length) |
| `currency`, `timezone` | for money and time display |

## `data_quality`
| Field | Meaning |
|---|---|
| `rows_in_raw_file` | rows before any cleaning |
| `tickets_used_for_analysis` / `orders_used_for_analysis` | rows after cleaning |
| `cleaning_steps[]` | each step: `rows_before`, `rows_after`, `rows_removed`, `note` |
| `warnings[]` | `level` (`info` / `warning` / `critical`) + `message` the analyst must read |
| `modules[]` | which analyses ran; `enabled: false` comes with a `reason` (e.g. no revenue column) |

## `kpis[]`: one object per KPI card
| Field | Meaning |
|---|---|
| `id`, `label` | machine name / display name |
| `value`, `previous_value` | current period vs previous period |
| `change`, `change_pct` | difference (absolute and %) |
| `unit` | `ratio`, `hours`, `days`, `orders`, `BRL`, ... |
| `higher_is_better` | `true` / `false` / `null` (volume metrics are neither good nor bad) |
| `trend` | `improving` / `worsening` / `flat`; for volume: `up` / `down` |
| `status` | card colour: `good` / `warning` / `critical` / `info` |
| `status_rule` | how status was decided: thresholds on the value, or on % change |
| `definition` | exact formula in words; show it as a tooltip |

## `trends`
`granularity` (`week`) and `series[]`, one point per week: `week_ending` + metrics (`tickets`, `breach_rate`, `median_resolution_hours` / `orders`, `late_rate`, `revenue`, `avg_delivery_days`).

## `breakdowns[]`: one table per dimension
`dimension` (column), `label`, `sorted_by`, and `rows[]`:
| Field | Meaning |
|---|---|
| `value` | the team / category / state / seller |
| `tickets` / `orders` | volume |
| `breach_rate` / `late_rate` | problem rate |
| `breaches` / `late_orders` | problem count |
| `share_of_all_breaches` / `share_of_all_late` | its share of all problems, which shows where to focus |
| `vs_overall_…_rate` | difference from the overall rate (+0.17 = 17 points worse) |
| `median_resolution_hours`, `avg_reassignments` / `avg_delivery_days`, `avg_review_score`, `revenue` | context |
| `flag` | `above_average` if the rate is >20% worse than overall |

## `root_causes[]`
| Field | Meaning |
|---|---|
| `id`, `title`, `category` | e.g. `RC1`, "Tickets bounced between teams", `process` |
| `finding` | one plain-English sentence for the card |
| `evidence[]` | the comparison behind it (segment → rate, count): draw as a small bar chart |
| `affected_…_current_period` | how many items it touches now |
| `share_of_…_current_period` | share of current problems linked to it |
| `confidence` | `high` / `medium` / `low` |
| `linked_recommendations` | ids in `recommendations` that fix it |
| `note` | caveats |

## `anomalies[]`
`week_ending`, `metric`, `label`, `actual`, `expected` (normal level), `deviation_score` (how unusual; >2 = unusual, >4 = extreme), `severity`, `direction`.

## `forecast`
`method`, `horizon_days`, and per team (`teams[]`) or platform-wide (`daily[]`): expected volume per day, `expected_capacity`, `overload` (true if volume > capacity +10%), `overloaded_days`.

## `predictions`
`risk_level_rules` explains the high / medium / low cut-offs. `items[]`:
| Field | Meaning |
|---|---|
| `id`, `item_type` | ticket number / order id |
| `context` | what the analyst needs to recognise the item (team, category, state, value...) |
| `prediction.risk` | probability 0–1 of the problem happening |
| `prediction.risk_level` | `high` (top 10%) / `medium` / `low` |
| `prediction.risk_percentile` | riskier than X% of current items |
| `top_reasons[]` | the 3 biggest reasons (SHAP): `label`, `value`, `typical_value`, `direction`, `effect` (strength), `sentence` |
| `explanation_summary` | one sentence for the "Why?" panel |
| `similar_past_cases` | how comparable past items turned out |
| `recommended_action` | what to do now, linked to a recommendation |
| `actual_outcome` | demo only (historical data); **remove in a live system** |

## `recommendations[]`
| Field | Meaning |
|---|---|
| `id`, `title`, `type`, `description` | the action |
| `target` | who or what it applies to |
| `evidence` | why we believe it |
| `expected_impact` | `baseline` → `expected` rate, `change_pct_points`, items avoided, a range (low–high) |
| `estimate_method`, `assumptions` | **how** the impact was estimated; analysts must see this |
| `confidence`, `effort`, `suggested_owner` | for prioritising |
| `linked_root_causes` | which causes it fixes |
| `status` | `pending` → `approved` / `rejected` / `edited` |
| `approval` | `decided_by`, `decided_at`, `decision`, `comment`: filled when the manager clicks |

## `impact_simulation`
`baseline` (current period totals) and `scenarios[]`, one per recommendation. Scenarios overlap, so **don't add them up**.

## `model_performance`
`test_auc` (0.5 = guessing, 1 = perfect), `train_auc`, `base_rate_test`, `precision_top_10pct` (problem rate among the 10% flagged riskiest), `lift_top_10pct` (how many times better than average), rows, features, `validation` method and `test_period`.

---

## Status of each section's numbers
| Section | Computed by | Today |
|---|---|---|
| meta, data_quality, kpis, trends, breakdowns, root_causes | Person 1 | ✅ real |
| anomalies | Person 1/2 | ✅ real (simple median rule) |
| forecast | Person 2 | ⚠ simple seasonal-naive method → replace with a real forecast model |
| predictions + reasons | Person 2 | ✅ real (LightGBM + SHAP) |
| recommendations, impact_simulation | Person 3 | ⚠ simple historical estimates with stated assumptions → replace with the simulator |
| approval | Person 3 + 4 | empty until a manager decides |
