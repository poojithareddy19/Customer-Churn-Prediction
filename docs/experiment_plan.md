# Retention offer experiment plan

This is the plan for a randomised test of whether a retention offer keeps customers the churn model flags. The
model ranks customers by risk; it does not show that an offer changes anyone's behaviour. Only a controlled
experiment can measure that.

Numbers marked "from the data" come from `scripts/experiment_sizing.py`, which reads the deployed model and
the test split and writes `reports/experiment_sizing.md`.

## 1. Hypothesis

Among customers the model flags as likely to churn, offering a retention incentive increases the 90-day
retention rate compared with making no offer.

- H0: retention(offer) = retention(no offer)
- H1: retention(offer) differs from retention(no offer) (two-sided, so a harmful offer is also detected)

## 2. Unit of randomisation

The customer (account). Each eligible customer is assigned once, by a hash of the customer ID with a fixed salt,
so the assignment is reproducible and cannot be changed by the retention team. Customers in the same household
or sharing an account are assigned together if the business can identify them, to limit contamination.

## 3. Eligibility

- Active customers whose model probability is at or above the profit threshold the app uses by default (0.34 in
  the current artifact).
- On the test split that is 523 of 1,409 customers (37.1%), whose observed retention rate is 46.3% (from the data).
- Exclusions, decided before launch: customers already on a retention offer, customers in an open complaint or
  collections process, and staff or test accounts.
- Eligibility is fixed at assignment time. Customers are not re-scored or re-assigned during the test.

## 4. Arms

| Arm | Share | Treatment |
| --- | --- | --- |
| Control | 50% | Normal service, no proactive offer |
| Treatment | 50% | The retention offer, delivered through the usual channel |

The offer, its cost and the delivery channel are fixed for the whole test.

## 5. Metrics

**Primary metric:** 90-day retention rate, the share of customers in each arm still active 90 days after
assignment.

**Secondary metrics:**

- Revenue retained per assigned customer over the 90 days (monthly charges of retained customers, divided by
  customers assigned, so the denominator includes those who left).
- Contract upgrades (month-to-month to one or two year) within 90 days.

**Guardrails** (the test is stopped or the offer rejected if these move the wrong way):

- Offer cost per incremental retained customer: total offer cost divided by (treatment retained minus control
  retained, scaled to the treatment arm size). This must stay below the expected value of a retained customer.
- Complaint rate and downgrade rate (customers moving to a cheaper plan) within 90 days, compared between arms.

All metrics are computed on every assigned customer (intention to treat), including treatment customers who never
saw or accepted the offer.

## 6. Minimum detectable effect, sample size and duration

Two-sided test, alpha = 0.05, power = 0.8, 50/50 split, control retention 46.3% (from the data):

| Minimum detectable effect | Customers per arm | Total |
| --- | --- | --- |
| +2 percentage points | 9,781 | 19,562 |
| +3 percentage points | 4,351 | 8,702 |
| +5 percentage points | 1,568 | 3,136 |
| +10 percentage points | 391 | 782 |

**Planned MDE: +5 percentage points, so 1,568 customers per arm (3,136 in total).** An MDE of +2 or +3 points
would need 19,562 or 8,702 customers, far more than the roughly 2,614 eligible customers estimated below. The
business should confirm that a +5 point lift would justify the offer cost; if only a larger lift would, the MDE
can be raised and the sample reduced.

**Duration.** The snapshot has no timestamps, so the monthly inflow of newly flagged customers is unknown. Applying
the eligible share of 37.1% to all 7,043 customers gives about 2,614 eligible customers today, about 520 short of
the 3,136 needed. Two options, to be decided before launch: enrol the current base at once and top up with newly
flagged customers until 3,136 are assigned, or accept a slightly larger MDE (the current base alone supports about
+5.5 points, 1,296 per arm). Either way, the 90-day window starts from each customer's assignment date, and the
analysis waits about two weeks after the last window closes for data to settle.

## 7. Analysis plan

Fixed before launch and not changed after seeing data:

1. Run the sample-ratio-mismatch check (below). If it fails, stop and fix assignment before any analysis.
2. Compare 90-day retention between arms with a two-proportion z-test (`analyze_ab` in `src/experiment.py`).
3. Report the absolute difference with its 95% confidence interval, not only the p-value.
4. Decision rule: roll out if the lower bound of the 95% CI is above zero and the guardrails pass. If the CI
   includes zero, the offer has not been shown to work at this sample size; do not read that as proof it does
   not work.
5. Secondary metrics and segment cuts (contract type, tenure cohort) are reported as exploratory, with no
   decision attached, because testing many segments inflates false positives.

## 8. Sample-ratio-mismatch check

Before reading any outcome, compare the arm sizes with the planned 50/50 split using a chi-square test
(`srm_check` in `src/experiment.py`). A p-value below 0.001 means assignment or logging is broken (for example,
offers delivered only to customers who could be reached, or one arm dropped from a data pipeline). Results from a
test that fails this check are not used.

## 9. Pitfalls and how the plan handles them

- **Peeking.** Checking the p-value repeatedly and stopping when it dips below 0.05 inflates the false positive
  rate well above 5%. The analysis runs once, after every customer has completed the 90-day window. Guardrails
  can be monitored during the test, but only to stop for harm.
- **Novelty effect.** Customers may react to any contact at first. The 90-day window is long enough to see
  whether retention holds after the first month.
- **Contamination.** Control customers may hear about the offer or ask for it. Agents must not give the offer to
  control customers who call in, and household-level assignment (section 2) reduces spillover. Any control
  customer who receives the offer stays in the control arm for the analysis.
- **Regression to the mean.** Flagged customers include some who would have stayed anyway. Comparing against a
  randomised control, not against last month's churn, removes this bias.
- **Changing the model mid-test.** The model and threshold are frozen for the duration, so eligibility does not
  drift.

## 10. Why a high churn score is not the same as a good offer target

The churn model predicts who is likely to leave. It does not predict who will stay because of an offer. Customers
fall into four groups:

| Group | Without offer | With offer | Value of an offer |
| --- | --- | --- | --- |
| Persuadable | leaves | stays | positive: the offer causes retention |
| Sure thing | stays | stays | wasted cost |
| Lost cause | leaves | leaves | wasted cost |
| Sleeping dog | stays | leaves | negative: the contact prompts them to leave |

A high churn score mixes persuadables with lost causes, and a low score mixes sure things with sleeping dogs. The
segment analysis in `reports/segment_analysis.md` shows, for example, that month-to-month customers paying by
electronic check churn the most, but it cannot say whether they respond to offers. Targeting by risk alone can
spend most of the budget on customers whose behaviour the offer does not change.

**Phase 2: uplift modelling.** Once this experiment has run, its randomised data can train a model of the offer
effect itself:

- **Two-model approach.** Fit one model of retention on treatment customers and another on control customers.
  A customer's predicted uplift is P(retain | offer) minus P(retain | no offer). Simple, but the difference of two
  noisy models can be noisy.
- **Class-transformation approach.** With a 50/50 split, define Z = 1 for treated customers who stayed and for
  control customers who left, and Z = 0 otherwise. A single classifier for Z gives uplift = 2 P(Z = 1) - 1. This
  learns the uplift directly in one model.

Uplift models are evaluated with Qini or uplift curves on held-out experiment data. The offer is then sent to the
customers with the highest predicted uplift (the persuadables) rather than the highest churn risk, and a follow-up
experiment compares uplift targeting against risk targeting.
