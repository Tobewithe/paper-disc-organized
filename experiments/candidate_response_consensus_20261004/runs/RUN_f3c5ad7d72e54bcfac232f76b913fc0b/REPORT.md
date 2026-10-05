# Frozen candidate-response consensus screen

Decision: stop_current_consensus_configuration.
All predictions use the original P, target box, class score and zero mask threshold. Donor pools and consensus weights never use GT.

| Group | Comparison | Macro IoU Δ pp [95% CI] | Candidate Δ pp | Repair/damage | Coverage Δ pp | AUC Δ pp | FPR Δ pp |
|---|---|---:|---:|---:|---:|---:|---:|
| all | SAME_minus_A | -0.6625 [-1.2116,-0.1134] | -0.4272 | 0/0 | +0.1389 | -0.2027 | +1.1593 |
| all | TOP_minus_A | -0.6332 [-1.0726,-0.1938] | -0.4449 | 0/0 | +0.1910 | -0.3025 | +1.1370 |
| all | SCORE_minus_A | -0.5646 [-0.9906,-0.1386] | -0.3820 | 0/0 | +0.1920 | -0.2520 | +1.0706 |
| all | CONS_minus_A | -0.5648 [-0.9906,-0.1391] | -0.3824 | 0/0 | +0.1890 | -0.2498 | +1.0689 |
| all | CONS_MATCH_minus_A | -0.5175 [-0.8970,-0.1379] | -0.3548 | 0/0 | +0.1865 | -0.2305 | +0.9929 |
| all | ONE_MATCH_minus_A | -0.1172 [-0.2041,-0.0303] | -0.0800 | 0/0 | +0.0488 | -0.0167 | +0.2455 |
| all | CONS_minus_TOP | +0.0684 [+0.0547,+0.0820] | +0.0625 | 0/0 | -0.0020 | +0.0527 | -0.0681 |
| all | CONS_minus_SCORE | -0.0002 [-0.0005,+0.0000] | -0.0003 | 0/0 | -0.0030 | +0.0022 | -0.0017 |
| all | CONS_MATCH_minus_ONE_MATCH | -0.4002 [-0.6928,-0.1076] | -0.2748 | 0/0 | +0.1377 | -0.2138 | +0.7474 |
| box_good_mask_bad | SAME_minus_A | -0.5090 [-0.8140,-0.2040] | -0.5090 | 0/0 | -0.3284 | -0.0932 | +0.4857 |
| box_good_mask_bad | TOP_minus_A | -0.6054 [-0.6749,-0.5360] | -0.6054 | 0/0 | -0.0499 | -0.5308 | +0.6703 |
| box_good_mask_bad | SCORE_minus_A | -0.4656 [-0.6280,-0.3033] | -0.4656 | 0/0 | -0.1028 | -0.2843 | +0.5443 |
| box_good_mask_bad | CONS_minus_A | -0.4668 [-0.6280,-0.3056] | -0.4668 | 0/0 | -0.1178 | -0.2738 | +0.5356 |
| box_good_mask_bad | CONS_MATCH_minus_A | -0.3732 [-0.4408,-0.3056] | -0.3732 | 0/0 | -0.1178 | -0.2725 | +0.3975 |
| box_good_mask_bad | ONE_MATCH_minus_A | -0.2012 [-0.4083,+0.0058] | -0.2012 | 0/0 | +0.0649 | -0.0378 | +0.3654 |
| box_good_mask_bad | CONS_minus_TOP | +0.1386 [-0.0920,+0.3692] | +0.1386 | 0/0 | -0.0679 | +0.2570 | -0.1347 |
| box_good_mask_bad | CONS_minus_SCORE | -0.0012 [-0.0023,+0.0000] | -0.0012 | 0/0 | -0.0150 | +0.0106 | -0.0087 |
| box_good_mask_bad | CONS_MATCH_minus_ONE_MATCH | -0.1720 [-0.3115,-0.0326] | -0.1720 | 0/0 | -0.1828 | -0.2348 | +0.0321 |
| original_failure | SAME_minus_A | -0.7078 [-1.2116,-0.2040] | -0.8757 | 0/0 | -0.1404 | -0.2445 | +0.9677 |
| original_failure | TOP_minus_A | -0.8737 [-1.0726,-0.6749] | -0.9400 | 0/0 | +0.1705 | -0.6545 | +1.2904 |
| original_failure | SCORE_minus_A | -0.6469 [-0.9906,-0.3033] | -0.7615 | 0/0 | +0.0852 | -0.4138 | +1.0085 |
| original_failure | CONS_minus_A | -0.6481 [-0.9906,-0.3056] | -0.7622 | 0/0 | +0.0702 | -0.4033 | +0.9998 |
| original_failure | CONS_MATCH_minus_A | -0.6013 [-0.8970,-0.3056] | -0.6999 | 0/0 | +0.0702 | -0.3832 | +0.9308 |
| original_failure | ONE_MATCH_minus_A | -0.0991 [-0.2041,+0.0058] | -0.1341 | 0/0 | +0.0325 | -0.0173 | +0.1813 |
| original_failure | CONS_minus_TOP | +0.2256 [+0.0820,+0.3692] | +0.1778 | 0/0 | -0.1004 | +0.2511 | -0.2906 |
| original_failure | CONS_minus_SCORE | -0.0012 [-0.0023,+0.0000] | -0.0008 | 0/0 | -0.0150 | +0.0105 | -0.0087 |
| original_failure | CONS_MATCH_minus_ONE_MATCH | -0.5022 [-0.6928,-0.3115] | -0.5657 | 0/0 | +0.0377 | -0.3658 | +0.7495 |
| original_success | SAME_minus_A | -0.0908 [-0.0908,-0.0908] | -0.0908 | 0/0 | +0.0415 | -0.0299 | +0.2533 |
| original_success | TOP_minus_A | -0.0735 [-0.0735,-0.0735] | -0.0735 | 0/0 | +0.0813 | -0.0195 | +0.3124 |
| original_success | SCORE_minus_A | -0.0974 [-0.0974,-0.0974] | -0.0974 | 0/0 | +0.0616 | -0.0217 | +0.3230 |
| original_success | CONS_minus_A | -0.0974 [-0.0974,-0.0974] | -0.0974 | 0/0 | +0.0616 | -0.0217 | +0.3230 |
| original_success | CONS_MATCH_minus_A | -0.0960 [-0.0960,-0.0960] | -0.0960 | 0/0 | +0.0553 | -0.0237 | +0.3056 |
| original_success | ONE_MATCH_minus_A | -0.0394 [-0.0394,-0.0394] | -0.0394 | 0/0 | +0.0408 | +0.0010 | +0.1547 |
| original_success | CONS_minus_TOP | -0.0239 [-0.0239,-0.0239] | -0.0239 | 0/0 | -0.0197 | -0.0021 | +0.0106 |
| original_success | CONS_minus_SCORE | +0.0000 [+0.0000,+0.0000] | +0.0000 | 0/0 | +0.0000 | -0.0000 | +0.0000 |
| original_success | CONS_MATCH_minus_ONE_MATCH | -0.0566 [-0.0566,-0.0566] | -0.0566 | 0/0 | +0.0144 | -0.0248 | +0.1509 |

Interpretation limits:

- Mask voting has FCIS precedent; weighted coefficient/logit equivalence is algebra, not novelty.
- CONS−TOP tests whether combining donors adds value; CONS−SCORE tests the extra response-consensus weighting. A gain over SAME alone is insufficient.
- CONS_MATCH and ONE_MATCH use the same per-target donor count; positions and prediction quality still differ, so this is not a proof of a training-assignment cause.
- All donor fallback/empty-support cases remain in the population. Donors are prediction-associated candidates, not GT-authorized positive labels.
- Primary intervals resample whole images. These are reused development images; no blind confirmation, original full inference AP, or trained method claim.
- Stop this fixed configuration if no useful signal. Do not automatically sweep pool size, IoU threshold, weights or start a learned selector.
