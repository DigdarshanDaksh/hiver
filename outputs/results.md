# Results (200 labelled golden examples)

Label distribution:

| intent                  |   count |
|:------------------------|--------:|
| software_update_issue   |      67 |
| app_service_issue       |      26 |
| battery_charging        |      23 |
| how_to_product_question |      19 |
| account_billing         |      19 |
| hardware_damage_safety  |      16 |
| non_english             |      13 |
| feedback_other          |       9 |
| keyboard_i_bug          |       8 |

Gold escalation rate: all 0.33 | uniform slice 0.18

## All examples (stratified: oversamples risky cases)

| metric | trivial | simple |
|---|---|---|
| n | 200 | 200 |
| intent_acc | 0.34 | 0.56 |
| intent_macroF1 | 0.06 [0.05-0.06] | 0.57 [0.50-0.63] |
| esc_recall | 0.00 [0.00-0.00] | 0.46 [0.34-0.59] |
| esc_precision | 0.00 | 0.65 |
| auto_rate | 1.00 | 0.77 |
| missed_escalations | 65 | 35 |
| reply_len_ok | 1.00 | 1.00 |
| invented_url | 0.00 | 0.00 |
| sim_to_real_reply | 0.15 | 0.18 |

## Uniform slice only (natural traffic)

| metric | trivial | simple |
|---|---|---|
| n | 120 | 120 |
| intent_acc | 0.40 | 0.52 |
| intent_macroF1 | 0.06 [0.05-0.07] | 0.50 [0.37-0.57] |
| esc_recall | 0.00 [0.00-0.00] | 0.14 [0.00-0.29] |
| esc_precision | 0.00 | 0.50 |
| auto_rate | 1.00 | 0.95 |
| missed_escalations | 22 | 19 |
| reply_len_ok | 1.00 | 1.00 |
| invented_url | 0.00 | 0.00 |
| sim_to_real_reply | 0.15 | 0.15 |

## LLM judge vs human agreement

_No human ratings yet: run `python label.py replies`._