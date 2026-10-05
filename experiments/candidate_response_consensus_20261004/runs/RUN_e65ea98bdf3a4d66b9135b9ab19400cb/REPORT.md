# Frozen candidate-response consensus screen

Decision: stop_current_consensus_configuration.
All predictions use the original P, target box, class score and zero mask threshold. Donor pools and consensus weights never use GT.

| Group | Comparison | Macro IoU Δ pp [95% CI] | Candidate Δ pp | Repair/damage | Coverage Δ pp | AUC Δ pp | FPR Δ pp |
|---|---|---:|---:|---:|---:|---:|---:|
| all | SAME_minus_A | -0.0437 [-0.1202,+0.0407] | -0.1475 | 14/15 | +0.0953 | -0.1564 | +0.3749 |
| all | TOP_minus_A | -0.0931 [-0.2152,+0.0256] | -0.1143 | 16/18 | +0.0276 | -0.0610 | +0.3330 |
| all | SCORE_minus_A | -0.0123 [-0.0918,+0.0660] | -0.0859 | 16/17 | +0.0902 | -0.0366 | +0.2439 |
| all | CONS_minus_A | -0.0144 [-0.0946,+0.0637] | -0.0884 | 16/17 | +0.0890 | -0.0369 | +0.2453 |
| all | CONS_MATCH_minus_A | -0.0076 [-0.0888,+0.0732] | -0.0756 | 16/16 | +0.0820 | -0.0376 | +0.2083 |
| all | ONE_MATCH_minus_A | +0.0250 [-0.0045,+0.0533] | +0.0072 | 3/1 | +0.0301 | -0.0167 | +0.0080 |
| all | CONS_minus_TOP | +0.0787 [+0.0031,+0.1684] | +0.0259 | 7/6 | +0.0615 | +0.0241 | -0.0876 |
| all | CONS_minus_SCORE | -0.0021 [-0.0059,+0.0009] | -0.0025 | 0/0 | -0.0011 | -0.0002 | +0.0014 |
| all | CONS_MATCH_minus_ONE_MATCH | -0.0326 [-0.1104,+0.0504] | -0.0829 | 13/15 | +0.0519 | -0.0209 | +0.2003 |
| box_good_mask_bad | SAME_minus_A | +0.1041 [-0.2186,+0.4797] | -0.0089 | 11/0 | +0.3896 | -0.0907 | +0.5275 |
| box_good_mask_bad | TOP_minus_A | -0.0491 [-0.4574,+0.4136] | -0.1018 | 11/0 | +0.1436 | -0.2456 | +0.2524 |
| box_good_mask_bad | SCORE_minus_A | +0.0575 [-0.2455,+0.4249] | -0.0542 | 12/0 | +0.3978 | -0.1371 | +0.5795 |
| box_good_mask_bad | CONS_minus_A | +0.0466 [-0.2562,+0.4151] | -0.0668 | 12/0 | +0.3962 | -0.1383 | +0.6018 |
| box_good_mask_bad | CONS_MATCH_minus_A | +0.0761 [-0.2395,+0.4584] | -0.0142 | 12/0 | +0.3397 | -0.1650 | +0.4173 |
| box_good_mask_bad | ONE_MATCH_minus_A | -0.0102 [-0.1783,+0.1667] | -0.0189 | 3/0 | +0.1150 | -0.1407 | +0.1572 |
| box_good_mask_bad | CONS_minus_TOP | +0.0956 [-0.1617,+0.3588] | +0.0350 | 3/2 | +0.2526 | +0.1073 | +0.3494 |
| box_good_mask_bad | CONS_minus_SCORE | -0.0109 [-0.0268,+0.0029] | -0.0126 | 0/0 | -0.0017 | -0.0013 | +0.0223 |
| box_good_mask_bad | CONS_MATCH_minus_ONE_MATCH | +0.0863 [-0.1895,+0.3921] | +0.0047 | 9/0 | +0.2246 | -0.0243 | +0.2601 |
| original_failure | SAME_minus_A | -0.0496 [-0.3395,+0.2251] | -0.2526 | 14/0 | +0.3932 | -0.5057 | +0.9258 |
| original_failure | TOP_minus_A | +0.0151 [-0.3228,+0.3942] | -0.1010 | 16/0 | +0.2789 | -0.1384 | +0.3628 |
| original_failure | SCORE_minus_A | +0.1196 [-0.1340,+0.4534] | -0.0729 | 16/0 | +0.4183 | -0.1022 | +0.4859 |
| original_failure | CONS_minus_A | +0.1126 [-0.1423,+0.4457] | -0.0778 | 16/0 | +0.4145 | -0.1029 | +0.4927 |
| original_failure | CONS_MATCH_minus_A | +0.1117 [-0.1525,+0.4482] | -0.0588 | 16/0 | +0.3864 | -0.0944 | +0.4213 |
| original_failure | ONE_MATCH_minus_A | +0.0126 [-0.0925,+0.1134] | -0.0291 | 3/0 | +0.0862 | -0.1061 | +0.0657 |
| original_failure | CONS_minus_TOP | +0.0975 [-0.0707,+0.3237] | +0.0232 | 3/3 | +0.1355 | +0.0355 | +0.1299 |
| original_failure | CONS_minus_SCORE | -0.0070 [-0.0185,+0.0017] | -0.0050 | 0/0 | -0.0039 | -0.0007 | +0.0069 |
| original_failure | CONS_MATCH_minus_ONE_MATCH | +0.0992 [-0.1458,+0.4049] | -0.0298 | 13/0 | +0.3002 | +0.0117 | +0.3556 |
| original_success | SAME_minus_A | -0.0215 [-0.0627,+0.0203] | -0.0849 | 0/15 | +0.0223 | -0.0013 | +0.1453 |
| original_success | TOP_minus_A | -0.0988 [-0.2120,-0.0128] | -0.1222 | 0/18 | -0.0407 | -0.0338 | +0.2645 |
| original_success | SCORE_minus_A | -0.0433 [-0.1077,+0.0144] | -0.0937 | 0/17 | -0.0123 | -0.0169 | +0.1482 |
| original_success | CONS_minus_A | -0.0443 [-0.1096,+0.0143] | -0.0947 | 0/17 | -0.0127 | -0.0170 | +0.1484 |
| original_success | CONS_MATCH_minus_A | -0.0361 [-0.0980,+0.0216] | -0.0857 | 0/16 | -0.0122 | -0.0175 | +0.1373 |
| original_success | ONE_MATCH_minus_A | +0.0284 [+0.0127,+0.0470] | +0.0289 | 0/1 | +0.0136 | +0.0091 | -0.0049 |
| original_success | CONS_minus_TOP | +0.0545 [+0.0072,+0.1268] | +0.0275 | 4/3 | +0.0280 | +0.0168 | -0.1161 |
| original_success | CONS_minus_SCORE | -0.0010 [-0.0031,+0.0006] | -0.0010 | 0/0 | -0.0003 | -0.0000 | +0.0002 |
| original_success | CONS_MATCH_minus_ONE_MATCH | -0.0645 [-0.1273,-0.0092] | -0.1145 | 0/15 | -0.0258 | -0.0266 | +0.1422 |

Interpretation limits:

- Mask voting has FCIS precedent; weighted coefficient/logit equivalence is algebra, not novelty.
- CONS−TOP tests whether combining donors adds value; CONS−SCORE tests the extra response-consensus weighting. A gain over SAME alone is insufficient.
- CONS_MATCH and ONE_MATCH use the same per-target donor count; positions and prediction quality still differ, so this is not a proof of a training-assignment cause.
- All donor fallback/empty-support cases remain in the population. Donors are prediction-associated candidates, not GT-authorized positive labels.
- Primary intervals resample whole images. These are reused development images; no blind confirmation, original full inference AP, or trained method claim.
- Stop this fixed configuration if no useful signal. Do not automatically sweep pool size, IoU threshold, weights or start a learned selector.
