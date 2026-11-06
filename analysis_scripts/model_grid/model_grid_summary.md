# Model-grid triage (offline, no hardware)

Models: 48 int8 exports (3 other-window exports ignored). Source: `embedded-systme-project origin/main c250591, models/`.

Latency columns are a TRIAGE ESTIMATE from measured MAC/s of the standard-conv DS-CNN with the same width and window; depthwise layers are bracketed between that rate and the overhead-bound depthwise baseline. They are not measurements.

## Operator support
- Unsupported in a runtime: none.
- ESP32 latency sketch resolver (`MicroMutableOpResolver<11>`) lacks: nothing -> register them before measuring (runtime has them).

## Classes
- rp2040_mem: borderline 8, fits 30, no 10
- esp32_mem: borderline 12, fits 27, no (PSRAM arena possible) 9
- rp2040_triage: at_risk 3, meets 25, miss 20
- rp2040_triage_logmel_same_core: at_risk 4, meets 16, miss 28
- esp32_triage: meets 48
- pi1t_triage: meets 48

## Table
| model | val acc | MACs | RP2040 mem (static RAM) | ESP32 mem (static RAM range) | RP2040 ms (opt-pess) | triage | ESP32 ms | Pi 1T ms | plan RP2040 | plan ESP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| DSCNN_64_SC_1_1s | 0.710 | 19,622,144 | fits (140,624) | fits (126,755-215,908) | 2,832-2,832 | miss | 77.2-77.2 | 2.53-2.53 | skip: already measured (same graph as DSCNN_64_SC_1_1s) | skip: already measured (same graph as DSCNN_64_SC_1_1s) |
| DSCNN_64_SC_1_2s | 0.710 | 38,072,576 | fits (224,592) | borderline (202,787-375,652) | 5,482-5,482 | miss | 143.1-143.1 | 5.09-5.09 | skip: already measured (same graph as DSCNN_64_SC_1_2s) | skip: already measured (same graph as DSCNN_64_SC_1_2s) |
| DSCNN_64_SC_25_1s | 0.653 | 1,311,296 | fits (75,088) | fits (67,545-91,507) | 179-179 | meets | 15.3-15.3 | 0.33-0.33 | skip: already measured (same graph as DSCNN_64_SC_25_1s) | skip: already measured (same graph as DSCNN_64_SC_25_1s) |
| DSCNN_64_SC_25_2s | 0.663 | 2,550,848 | fits (96,592) | fits (86,553-131,443) | 349-349 | meets | 30.2-30.2 | 0.65-0.65 | skip: already measured (same graph as DSCNN_64_SC_25_2s) | skip: already measured (same graph as DSCNN_64_SC_25_2s) |
| DSCNN_64_SC_5_1s | 0.681 | 5,018,752 | fits (97,616) | fits (87,281-132,974) | 663-663 | meets | 32.9-32.9 | 0.84-0.84 | skip: already measured (same graph as DSCNN_64_SC_5_1s) | skip: already measured (same graph as DSCNN_64_SC_5_1s) |
| DSCNN_64_SC_5_2s | 0.696 | 9,746,560 | fits (139,600) | fits (125,297-212,846) | 1,282-1,282 | meets | 62.7-62.7 | 1.64-1.64 | skip: already measured (same graph as DSCNN_64_SC_5_2s) | skip: already measured (same graph as DSCNN_64_SC_5_2s) |
| DSCNN_128_SC_1_1s | 0.700 | 39,243,008 | fits (224,592) | borderline (201,773-373,522) | 5,665-5,665 | miss | 154.4-154.4 | 5.06-5.06 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| DSCNN_128_SC_1_2s | 0.695 | 76,143,872 | no (361,808) | no (PSRAM arena possible) (353,837-693,010) | 10,965-10,965 | miss | 286.2-286.2 | 10.18-10.18 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |
| DSCNN_128_SC_25_1s | 0.652 | 2,622,272 | fits (95,568) | fits (86,299-130,911) | 358-358 | meets | 30.6-30.6 | 0.67-0.67 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_128_SC_25_2s | 0.686 | 5,101,376 | fits (138,576) | fits (124,315-210,783) | 697-697 | meets | 60.5-60.5 | 1.31-1.31 | run N=87 warm=5 | run N=100 warm=5 |
| DSCNN_128_SC_5_1s | 0.695 | 10,036,864 | fits (138,576) | fits (124,790-211,781) | 1,326-1,326 | miss | 65.8-65.8 | 1.68-1.68 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| DSCNN_128_SC_5_2s | 0.708 | 19,492,480 | fits (223,568) | borderline (200,822-371,525) | 2,564-2,564 | miss | 125.4-125.4 | 3.29-3.29 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| DSCNN_64_DW_1_1s | 0.686 | 2,790,144 | fits (127,312) | fits (114,716-190,615) | 403-504 | meets | 11.0-53.3 | 0.36-2.38 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_64_DW_1_2s | 0.694 | 5,447,936 | fits (197,968) | fits (178,076-323,735) | 784-983 | meets | 20.5-103.3 | 0.73-4.68 | run N=69 warm=5 | run N=100 warm=5 |
| DSCNN_64_DW_25_1s | 0.630 | 298,176 | fits (75,088) | fits (67,612-91,649) | 41-67 | meets | 3.5-13.7 | 0.08-0.57 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_64_DW_25_2s | 0.635 | 587,840 | fits (96,592) | fits (86,620-131,585) | 80-131 | meets | 7.0-26.8 | 0.15-1.13 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_64_DW_5_1s | 0.667 | 862,592 | fits (90,448) | fits (81,262-120,328) | 114-166 | meets | 5.7-26.5 | 0.14-1.15 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_64_DW_5_2s | 0.690 | 1,691,776 | fits (125,264) | fits (112,942-186,888) | 223-325 | meets | 10.9-51.7 | 0.29-2.26 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_128_DW_1_1s | 0.701 | 5,579,008 | fits (196,944) | fits (177,062-321,605) | 805-1,009 | at_risk | 21.9-106.5 | 0.72-4.76 | run N=67 warm=5 | run N=100 warm=5 |
| DSCNN_128_DW_1_2s | 0.707 | 10,894,592 | no (311,632) | borderline (303,782-587,845) | 1,569-1,967 | meets | 41.0-206.5 | 1.46-9.36 | skip: predicted RAM overflow | run N=100 warm=5 |
| DSCNN_128_DW_25_1s | 0.558 | 596,032 | fits (95,568) | fits (86,240-130,786) | 81-133 | meets | 7.0-27.3 | 0.15-1.15 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_128_DW_25_2s | 0.580 | 1,175,360 | fits (138,576) | fits (124,256-210,658) | 161-262 | meets | 13.9-53.7 | 0.30-2.25 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_128_DW_5_1s | 0.657 | 1,724,544 | fits (125,264) | fits (112,435-185,823) | 228-332 | meets | 11.3-53.1 | 0.29-2.30 | run N=100 warm=5 | run N=100 warm=5 |
| DSCNN_128_DW_5_2s | 0.670 | 3,382,912 | fits (195,920) | fits (175,795-318,943) | 445-649 | meets | 21.8-103.5 | 0.57-4.51 | run N=100 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_64_SC_1_1s | 0.706 | 89,626,080 | borderline (238,928) | borderline (231,742-436,488) | 12,937-12,937 | miss | 352.7-352.7 | 11.55-11.55 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_64_SC_1_2s | 0.717 | 178,164,192 | no (418,128) | no (PSRAM arena possible) (409,150-809,224) | 25,655-25,655 | miss | 669.7-669.7 | 23.83-23.83 | skip: predicted RAM overflow | run N=90 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_64_SC_25_1s | 0.696 | 5,644,056 | fits (104,784) | fits (93,792-146,652) | 771-771 | meets | 65.9-65.9 | 1.43-1.43 | run N=78 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_64_SC_25_2s | 0.692 | 11,220,888 | fits (153,936) | fits (138,144-239,836) | 1,534-1,534 | meets | 133.0-133.0 | 2.88-2.88 | run N=20 warm=2 | run N=100 warm=5 |
| MobileNetV2Tiny_64_SC_5_1s | 0.748 | 22,463,088 | fits (154,960) | fits (139,775-243,264) | 2,968-2,968 | miss | 147.3-147.3 | 3.76-3.76 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_64_SC_5_2s | 0.739 | 44,655,216 | borderline (235,856) | borderline (228,479-429,632) | 5,875-5,875 | miss | 287.2-287.2 | 7.53-7.53 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_128_SC_1_1s | 0.777 | 179,251,680 | no (414,032) | no (PSRAM arena possible) (405,602-801,769) | 25,875-25,875 | miss | 705.3-705.3 | 23.09-23.09 | skip: predicted RAM overflow | run N=86 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_128_SC_1_2s | 0.789 | 356,327,904 | no (772,432) | no (PSRAM arena possible) (760,418-1,547,241) | 51,310-51,310 | miss | 1,339.3-1,339.3 | 47.65-47.65 | skip: predicted RAM overflow | run N=23 warm=2 (PSRAM arena) |
| MobileNetV2Tiny_128_SC_25_1s | 0.670 | 11,287,992 | fits (152,912) | fits (137,256-237,972) | 1,543-1,543 | miss | 131.9-131.9 | 2.87-2.87 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_128_SC_25_2s | 0.674 | 22,441,656 | borderline (232,784) | borderline (225,960-424,340) | 3,068-3,068 | miss | 266.0-266.0 | 5.75-5.75 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_128_SC_5_1s | 0.719 | 44,925,936 | borderline (233,808) | borderline (226,705-425,905) | 5,936-5,936 | miss | 294.6-294.6 | 7.51-7.51 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_128_SC_5_2s | 0.733 | 89,310,192 | no (413,008) | no (PSRAM arena possible) (404,113-798,641) | 11,749-11,749 | miss | 574.4-574.4 | 15.07-15.07 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_64_1_1s | 0.724 | 9,407,136 | borderline (236,880) | borderline (230,031-432,893) | 1,358-2,023 | miss | 37.0-313.7 | 1.21-14.44 | skip: predicted too slow (optimistic > stride) | run N=100 warm=5 |
| MobileNetV2Tiny_64_1_2s | 0.728 | 18,757,920 | no (416,080) | no (PSRAM arena possible) (407,439-805,629) | 2,701-4,030 | miss | 70.5-623.3 | 2.51-28.91 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_64_25_1s | 0.672 | 884,712 | fits (103,760) | fits (93,364-145,754) | 121-290 | meets | 10.3-76.9 | 0.22-3.49 | run N=100 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_64_25_2s | 0.684 | 1,765,704 | fits (152,912) | fits (137,716-238,938) | 241-578 | meets | 20.9-153.7 | 0.45-6.97 | run N=100 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_64_5_1s | 0.702 | 2,747,472 | fits (154,960) | fits (138,920-241,467) | 363-704 | meets | 18.0-154.6 | 0.46-7.05 | run N=100 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_64_5_2s | 0.710 | 5,480,592 | borderline (234,832) | borderline (227,624-427,835) | 721-1,402 | meets | 35.2-308.0 | 0.92-14.08 | run N=30 warm=2 | run N=100 warm=5 |
| MobileNetV2Tiny_128_1_1s | 0.713 | 18,813,792 | no (413,008) | no (PSRAM arena possible) (403,891-798,175) | 2,716-4,046 | miss | 74.0-627.4 | 2.42-28.89 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_128_1_2s | 0.729 | 37,515,360 | no (771,408) | no (PSRAM arena possible) (758,707-1,543,647) | 5,402-8,059 | miss | 141.0-1,246.7 | 5.02-57.82 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |
| MobileNetV2Tiny_128_25_1s | 0.619 | 1,769,304 | fits (151,888) | fits (136,829-237,074) | 242-580 | meets | 20.7-153.8 | 0.45-6.98 | run N=100 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_128_25_2s | 0.618 | 3,531,288 | borderline (232,784) | borderline (225,533-423,442) | 483-1,157 | meets | 41.9-307.3 | 0.91-13.94 | run N=81 warm=5 | run N=100 warm=5 |
| MobileNetV2Tiny_128_5_1s | 0.662 | 5,494,704 | borderline (232,784) | borderline (225,850-424,108) | 726-1,408 | at_risk | 36.0-309.2 | 0.92-14.10 | run N=30 warm=2 | run N=100 warm=5 |
| MobileNetV2Tiny_128_5_2s | 0.667 | 10,960,944 | no (411,984) | no (PSRAM arena possible) (403,258-796,844) | 1,442-2,804 | at_risk | 70.5-616.1 | 1.85-28.16 | skip: predicted RAM overflow | run N=100 warm=5 (PSRAM arena) |

## Measurement plan
- **rp2040**: run 22, skip 26; ~3.8 h (229 min) of bench time.
  - skip: already measured: 6
  - skip: predicted RAM overflow: 10
  - skip: predicted too slow (optimistic > stride): 10
- **esp32s3**: run 42, skip 6; ~2.7 h (160 min) of bench time.
  - skip: already measured: 6
- **rpi4**: run 42, skip 6; ~1.1 h (64 min) of bench time.
  - skip: already measured: 6

## 128 mel bins: front-end table sizes (training parameters, Slaney)
| n_mels | nonzero weights | 1-bin filters | max weight | min weight / max | sparse table bytes |
|---|---|---|---|---|---|
| 64 | 498 | 0 | 0.0211 | 9.74e-05 | 1,380 |
| 128 | 504 | 25 | 0.0424 | 1.11e-03 | 1,776 |
