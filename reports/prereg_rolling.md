# Pre-registration: rolling-origin check on Yambda (written 2026-10-12, before any run)

**Question.** Is the Yambda result (E2-full beats E1 and E1+) specific to the one test week we used, or
does it hold on other weeks?

**Design.** The same protocol with the whole train | val | test block (21 | 7 | 7 days) shifted back by
K weeks: `yambda_w1` (test days 57–63), `yambda_w2` (50–56), `yambda_w3` (43–49). W0 = the original run
(test days 64–70), already seen, reported separately. For every window the WHOLE pipeline is re-run from
scratch: candidates, E1 (hand, 25 Optuna trials), E1+ (strengthened hand set, 25 trials), E2-full
(grammar enumeration, selection on that window's val, 25 trials). Nothing is carried over between windows.
Earlier windows have less history before the train block (W3: 15 days), so absolute NDCG is expected to
differ; only within-window differences are compared.

**Primary criterion (decided now).** E2-full − E1, test NDCG@10, pooled over the three NEW windows
W1–W3 (per-query differences concatenated; 95% CI by bootstrap clustered by user, 2 000 resamples):
supported if the CI lower bound is > 0 **and** the point estimate is positive in at least 2 of 3 new
windows. Otherwise "not supported" — reported as such.

**Secondary.** The same for E2-full − E1+ and E1+ − E1. Per-window paired bootstrap CIs are reported for
all three comparisons in all four windows.

**Not changed after seeing results:** windows, methods, budgets, criterion.
