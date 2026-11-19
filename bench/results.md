# Saddle-search coverage

Stop when the maximum per-atom force norm is under 0.01 eV/Angstrom.
Spring 5.0 eV/Angstrom^2, linear path, 7 and 11 images, 1000 iterations.
Success: code converged, reference index is 1, energy within 0.01 eV,
free-atom RMSD within 0.05 Angstrom. A free cluster is centered and
Kabsch-aligned, and identical species are matched. Medians and IQRs use n=12.
A/A is a Mann-Whitney test of even rounds against odd rounds.
Method p-values are Mann-Whitney, and success p-values are Fisher exact.
ASE ships no Lanczos min-mode search.

## pt_hop

Reference usable: True; index 1; barrier 0.09056 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 12/12 | 0.3075 (0.03911) | 0.07284 (0.01104) | 212 (0) | 0.09057 (0) | 0.0004977 | 1 |
| eon-neb | 7 | 12/12 | 0.1004 (0.01214) | 0.01024 (0.001541) | 57 (0) | 0.09057 (0) | 0.0006239 | 0.3785 |
| ase-neb | 7 | 12/12 | 0.4428 (0.04532) | 0.2478 (0.0297) | 87 (0) | 0.09058 (0) | 0.0005976 | 0.5752 |
| rgsaddle-ci | 7 | 12/12 | 0.3659 (0.0479) | 0.08954 (0.01425) | 250 (0) | 0.09057 (0) | 0.0005525 | 0.6889 |
| eon-ci | 7 | 12/12 | 0.09978 (0.009493) | 0.01003 (0.001866) | 57 (0) | 0.09057 (0) | 0.0005987 | 0.4712 |
| ase-ci | 7 | 12/12 | 0.4476 (0.05489) | 0.2504 (0.03353) | 87 (0) | 0.09058 (0) | 0.0005767 | 0.6889 |
| rgsaddle-neb | 11 | 12/12 | 0.6919 (0.08033) | 0.1654 (0.02438) | 479 (0) | 0.09056 (0) | 0.0002017 | 0.4712 |
| eon-neb | 11 | 12/12 | 0.2115 (0.02132) | 0.01873 (0.002064) | 101 (0) | 0.09058 (0) | 0.0007332 | 0.5752 |
| ase-neb | 11 | 12/12 | 1.162 (0.09262) | 0.8935 (0.07234) | 155 (0) | 0.09058 (0) | 0.0006189 | 0.6889 |
| rgsaddle-ci | 11 | 12/12 | 0.7464 (0.04611) | 0.1793 (0.009864) | 517 (0) | 0.09056 (0) | 0.0006027 | 0.8102 |
| eon-ci | 11 | 12/12 | 0.2174 (0.02283) | 0.01924 (0.001642) | 101 (0) | 0.09058 (0) | 0.0007137 | 0.8102 |
| ase-ci | 11 | 12/12 | 1.191 (0.1029) | 0.9142 (0.1082) | 155 (0) | 0.09058 (0) | 0.0006007 | 0.2298 |
| rgsaddle-dimer |  | 3/12 | 0.3758 (0.1262) | 0.0962 (0.03439) | 258 (89.75) | 0.09053 (3.265e-05) | 0.3379 | 0.3785 |
| eon-dimer |  | 1/12 | 0.2229 (0.07203) | 0.02578 (0.008579) | 142 (60) | 0.09054 (0) | 0.3417 | 0.6889 |
| ase-dimer |  | 1/12 | 0.3961 (0.3989) | 0.1018 (0.09969) | 251.5 (264.5) | 0.09059 (0) | 0.7304 | 0.6889 |
| rgsaddle-lanczos |  | 2/12 | 0.3813 (0.1015) | 0.09724 (0.02552) | 259 (77) | 0.09057 (2.774e-05) | 0.3381 | 0.4712 |
| eon-lanczos |  | 3/12 | 0.2303 (0.03722) | 0.02489 (0.002132) | 144 (16.75) | 0.09059 (1.947e-05) | 0.3405 | 0.298 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 0.008616 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-dimer vs eon-dimer |  | 0.0001558 | 3.658e-05 | 0.0001229 | 1 | 0.5901 |
| rgsaddle-dimer vs ase-dimer |  | 0.3408 | 0.4025 | 0.5833 | 0.3711 | 0.5901 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.0001962 | 3.658e-05 | 0.0001548 | 1 | 1 |

## pt_exchange

Reference usable: True; index 1; barrier 4.569 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 11.4 (1.311) | 1.166 (0.1266) | 2677 (0) |  () | 0.8748 | 0.5752 |
| eon-neb | 7 | 0/12 | 21.5 (1.882) | 1.139 (0.09207) | 5002 (0) |  () | 0.7607 | 0.9362 |
| ase-neb | 7 | 0/12 | 40.8 (4.912) | 19.36 (2.544) | 5007 (0) |  () | 7.601 | 0.4712 |
| rgsaddle-ci | 7 | 0/12 | 11.74 (1.123) | 1.189 (0.1374) | 2677 (0) |  () | 0.8748 | 0.8102 |
| eon-ci | 7 | 0/12 | 1.159 (0.1707) | 0.06307 (0.009553) | 297 (0) |  () | 0.7651 | 0.6889 |
| ase-ci | 7 | 0/12 | 39.54 (5.43) | 18.71 (2.842) | 5007 (0) |  () | 4.048 | 0.2298 |
| rgsaddle-neb | 11 | 0/12 | 4.482 (0.8566) | 0.5416 (0.1083) | 1316 (0) |  () | 0.09887 | 0.0927 |
| eon-neb | 11 | 0/12 | 4.069 (0.6636) | 0.2342 (0.0416) | 1100 (0) |  () | 0.7636 | 0.2298 |
| ase-neb | 11 | 0/12 | 96.57 (13.55) | 59.98 (9.341) | 9011 (0) |  () | 3.999 | 1 |
| rgsaddle-ci | 11 | 12/12 | 3.579 (0.2414) | 0.4709 (0.03809) | 1134 (0) | 4.569 (0) | 0.0002717 | 0.4712 |
| eon-ci | 11 | 0/12 | 2.255 (0.1549) | 0.1384 (0.01728) | 659 (0) |  () | 0.7649 | 0.6889 |
| ase-ci | 11 | 0/12 | 99.39 (11.32) | 61.22 (7.571) | 9011 (0) |  () | 3.902 | 0.6889 |
| rgsaddle-dimer |  | 0/12 | 0.4934 (0.3801) | 0.1242 (0.09208) | 311 (249.2) |  () | 1.018 | 0.6889 |
| eon-dimer |  | 0/12 | 0.271 (0.1282) | 0.03036 (0.02016) | 187 (113.5) |  () | 1.02 | 0.9362 |
| ase-dimer |  | 0/12 | 0.4881 (0.3067) | 0.1237 (0.07499) | 330.5 (234) |  () | 1.164 | 0.8102 |
| rgsaddle-lanczos |  | 0/12 | 0.4316 (0.1889) | 0.1096 (0.04272) | 284.5 (112.5) |  () | 0.975 | 0.06555 |
| eon-lanczos |  | 0/12 | 0.2379 (0.1239) | 0.02783 (0.01608) | 149.5 (105) |  () | 0.8288 | 0.5752 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 0.2366 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs eon-neb | 11 | 0.008616 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 0.006099 | 4.695e-05 | 0.01657 |  | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.7508 | 0.7508 | 0.8625 |  | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.01657 | 3.658e-05 | 0.01937 |  | 1 |

## al_vacancy

Reference usable: True; index 1; barrier 0.2207 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 12/12 | 1.444 (0.1397) | 0.07788 (0.008379) | 157 (0) | 0.2209 (0) | 0.00711 | 0.8102 |
| eon-neb | 7 | 12/12 | 0.6726 (0.05011) | 0.01985 (0.001565) | 72 (0) | 0.2208 (0) | 0.005604 | 0.4712 |
| ase-neb | 7 | 12/12 | 1.937 (0.201) | 0.5378 (0.05107) | 152 (0) | 0.2208 (0) | 0.00616 | 0.9362 |
| rgsaddle-ci | 7 | 12/12 | 1.554 (0.1869) | 0.1087 (0.008509) | 211 (0) | 0.2209 (0) | 0.007109 | 0.6889 |
| eon-ci | 7 | 12/12 | 0.5036 (0.08042) | 0.01465 (0.00171) | 52 (0) | 0.2209 (0) | 0.007145 | 0.5752 |
| ase-ci | 7 | 12/12 | 1.94 (0.2648) | 0.5439 (0.06113) | 152 (0) | 0.2208 (0) | 0.006158 | 0.9362 |
| rgsaddle-neb | 11 | 12/12 | 2.476 (0.2857) | 0.1899 (0.01497) | 398 (0) | 0.2208 (0) | 0.005824 | 0.9362 |
| eon-neb | 11 | 12/12 | 1.408 (0.1271) | 0.04172 (0.002828) | 155 (0) | 0.2208 (0) | 0.005093 | 0.5752 |
| ase-neb | 11 | 12/12 | 5.119 (0.6685) | 3.07 (0.3415) | 362 (0) | 0.2208 (0) | 0.005416 | 0.9362 |
| rgsaddle-ci | 11 | 12/12 | 2.541 (0.2656) | 0.2215 (0.008891) | 452 (0) | 0.2208 (0) | 0.005821 | 0.9362 |
| eon-ci | 11 | 12/12 | 0.8119 (0.07196) | 0.02514 (0.002095) | 92 (0) | 0.221 (0) | 0.007438 | 0.9362 |
| ase-ci | 11 | 12/12 | 5.054 (0.8773) | 3.046 (0.5474) | 362 (0) | 0.2208 (0) | 0.005413 | 0.9362 |
| rgsaddle-dimer |  | 0/12 | 7.347 (2.222) | 1.795 (0.4977) | 3604 (871.2) |  () | 3.374 | 0.298 |
| eon-dimer |  | 2/12 | 0.7572 (0.8511) | 0.09136 (0.1049) | 370.5 (357.2) | 0.2211 (0.0002543) | 0.4307 | 0.9362 |
| ase-dimer |  | 0/12 | 1.134 (4.237) | 0.3209 (1.223) | 582 (2405) |  () | 0.6513 | 0.298 |
| rgsaddle-lanczos |  | 0/12 | 7.17 (5.146) | 1.742 (1.221) | 3414 (2314) |  () | 3.535 | 0.4712 |
| eon-lanczos |  | 0/12 | 30.06 (2.645) | 3.63 (0.2775) | 1.327e+04 (421.2) |  () | 0.4225 | 0.9362 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 7.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-dimer vs eon-dimer |  | 0.001354 | 0.0001962 | 0.001106 |  | 0.4783 |
| rgsaddle-dimer vs ase-dimer |  | 0.008616 | 0.01019 | 0.008616 |  | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 3.658e-05 | 3.658e-05 | 3.658e-05 |  | 1 |

## lj7

Reference usable: True; index 1; barrier 1.061 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 0.9101 (0.06253) | 0.8945 (0.06167) | 3872 (0) |  () | 0.1366 | 1 |
| eon-neb | 7 | 0/12 | 0.09784 (0.006215) | 0.09323 (0.005921) | 1432 (0) |  () | 0.1367 | 0.9362 |
| ase-neb | 7 | 0/12 | 4.804 (0.4459) | 4.776 (0.4415) | 4627 (0) |  () | 0.2826 | 0.9362 |
| rgsaddle-ci | 7 | 12/12 | 0.1417 (0.01168) | 0.1393 (0.01146) | 602 (0) | 1.061 (0) | 0.0002671 | 0.9362 |
| eon-ci | 7 | 12/12 | 0.01861 (0.002313) | 0.01775 (0.002192) | 272 (0) | 1.061 (0) | 0.0006788 | 0.9362 |
| ase-ci | 7 | 0/12 | 5.08 (0.5506) | 4.98 (0.5381) | 5007 (0) |  () | 11.54 | 0.6889 |
| rgsaddle-neb | 11 | 0/12 | 0.3155 (0.03564) | 0.3101 (0.03505) | 1397 (0) |  () | 0.02341 | 0.9362 |
| eon-neb | 11 | 0/12 | 0.4507 (0.04901) | 0.4295 (0.04678) | 6806 (0) |  () | 0.1366 | 0.5752 |
| ase-neb | 11 | 0/12 | 9.181 (1.372) | 9.133 (1.363) | 9011 (0) |  () | 0.1448 | 0.9362 |
| rgsaddle-ci | 11 | 12/12 | 0.2076 (0.01959) | 0.2041 (0.01918) | 917 (0) | 1.061 (0) | 8.82e-06 | 0.5752 |
| eon-ci | 11 | 0/12 | 0.1468 (0.01886) | 0.14 (0.01801) | 2243 (0) |  () | 0.1367 | 0.8102 |
| ase-ci | 11 | 0/12 | 6.441 (0.7633) | 6.408 (0.7584) | 6275 (0) |  () | 0.1367 | 0.6889 |
| rgsaddle-dimer |  | 1/12 | 0.3333 (0.5669) | 0.3276 (0.5579) | 1373 (2318) | 1.061 (0) | 0.3159 | 0.8102 |
| eon-dimer |  | 1/12 | 0.01386 (0.02904) | 0.01318 (0.02766) | 181.5 (442) | 1.061 (0) | 0.3052 | 0.9362 |
| ase-dimer |  | 1/12 | 0.05103 (0.04352) | 0.05006 (0.04272) | 233.5 (200.8) | 1.061 (0) | 0.2478 | 0.1282 |
| rgsaddle-lanczos |  | 0/12 | 1.729 (0.2206) | 1.7 (0.2171) | 6942 (394.8) |  () | 0.4148 | 0.0927 |
| eon-lanczos |  | 3/12 | 0.01555 (0.01121) | 0.01481 (0.01062) | 213 (130) | 1.061 (6.989e-07) | 0.1964 | 0.6889 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 0.0001233 | 9.735e-05 | 0.00355 | 1 | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.002437 | 0.002437 | 0.001652 | 1 | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 3.658e-05 | 3.658e-05 | 3.658e-05 |  | 0.2174 |

## cu4_h

Reference usable: True; index 1; barrier 0.2365 eV.

| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb | 7 | 0/12 | 0.06332 (0.007198) | 0.06059 (0.006763) | 267 (0) |  () | 0.08286 | 0.4712 |
| eon-neb | 7 | 0/12 | 0.009479 (0.001314) | 0.008314 (0.001066) | 127 (0) |  () | 0.1086 | 1 |
| ase-neb | 7 | 0/12 | 0.3606 (0.05208) | 0.3558 (0.05121) | 402 (0) |  () | 0.1349 | 0.298 |
| rgsaddle-ci | 7 | 0/12 | 0.7579 (0.1161) | 0.7251 (0.1102) | 3196 (0) |  () | 0.5242 | 0.3785 |
| eon-ci | 7 | 12/12 | 0.009505 (0.0009125) | 0.008327 (0.0007694) | 127 (0) | 0.2365 (0) | 0.006643 | 0.4712 |
| ase-ci | 7 | 0/12 | 4.662 (0.6059) | 4.599 (0.5962) | 5007 (0) |  () | 11.96 | 0.2298 |
| rgsaddle-neb | 11 | 0/12 | 0.2255 (0.0188) | 0.2163 (0.01785) | 929 (0) |  () | 0.06062 | 0.6889 |
| eon-neb | 11 | 0/12 | 0.0168 (0.002227) | 0.01484 (0.001871) | 227 (0) |  () | 0.06253 | 0.06555 |
| ase-neb | 11 | 0/12 | 3.623 (0.3901) | 3.575 (0.3838) | 4097 (0) |  () | 0.1561 | 0.298 |
| rgsaddle-ci | 11 | 12/12 | 0.2883 (0.02399) | 0.2766 (0.0228) | 1226 (0) | 0.2365 (0) | 1.647e-06 | 0.3785 |
| eon-ci | 11 | 12/12 | 0.02782 (0.001962) | 0.02464 (0.001715) | 389 (0) | 0.2365 (0) | 0.007429 | 0.6889 |
| ase-ci | 11 | 0/12 | 7.393 (0.6845) | 7.3 (0.6752) | 8327 (0) |  () | 12.91 | 0.8102 |
| rgsaddle-dimer |  | 7/12 | 0.07122 (0.1392) | 0.06911 (0.1352) | 284.5 (526.2) | 0.2365 (0.0002102) | 0.01529 | 0.5752 |
| eon-dimer |  | 6/12 | 0.01952 (0.03287) | 0.01786 (0.03015) | 257 (457.5) | 0.2365 (2.091e-05) | 0.2665 | 0.9362 |
| ase-dimer |  | 6/12 | 0.03832 (0.03006) | 0.03705 (0.02908) | 164 (125.8) | 0.2365 (2.793e-05) | 0.2677 | 0.6889 |
| rgsaddle-lanczos |  | 5/12 | 0.09375 (0.06692) | 0.09097 (0.06504) | 343 (239) | 0.2365 (1.89e-05) | 0.5815 | 0.6889 |
| eon-lanczos |  | 4/12 | 0.03607 (0.02615) | 0.03309 (0.02398) | 469 (344) | 0.2365 (3.243e-05) | 0.5232 | 0.298 |

| comparison | images | wall p | outside p | calls p | barrier p | success p |
| --- | --- | --- | --- | --- | --- | --- |
| rgsaddle-neb vs eon-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-ci vs ase-ci | 7 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs eon-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-neb vs ase-neb | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 1 |
| rgsaddle-ci vs eon-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 | 1.912e-06 | 1 |
| rgsaddle-ci vs ase-ci | 11 | 3.658e-05 | 3.658e-05 | 1.912e-06 |  | 7.396e-07 |
| rgsaddle-dimer vs eon-dimer |  | 0.00355 | 0.002437 | 0.5444 | 0.3531 | 1 |
| rgsaddle-dimer vs ase-dimer |  | 0.00355 | 0.00355 | 0.01202 | 0.5203 | 1 |
| rgsaddle-lanczos vs eon-lanczos |  | 0.005108 | 0.004265 | 0.3708 | 0.9025 | 1 |

## Failures

| system | method | images | round | status |
| --- | --- | --- | --- | --- |
| pt_hop | eon-dimer |  | 0 | GOOD |
| pt_hop | eon-lanczos |  | 0 | GOOD |
| pt_exchange | rgsaddle-neb | 7 | 0 | stalled |
| pt_exchange | eon-neb | 7 | 0 | BAD_MAX_ITERATIONS |
| pt_exchange | ase-neb | 7 | 0 | max_iterations |
| pt_exchange | rgsaddle-ci | 7 | 0 | stalled |
| pt_exchange | eon-ci | 7 | 0 | GOOD |
| pt_exchange | ase-ci | 7 | 0 | max_iterations |
| pt_exchange | eon-neb | 11 | 0 | GOOD |
| pt_exchange | ase-neb | 11 | 0 | max_iterations |
| pt_exchange | eon-ci | 11 | 0 | GOOD |
| pt_exchange | ase-ci | 11 | 0 | max_iterations |
| pt_exchange | eon-dimer |  | 0 | GOOD |
| pt_exchange | eon-lanczos |  | 0 | GOOD |
| al_vacancy | rgsaddle-dimer |  | 0 | max_iterations |
| al_vacancy | eon-dimer |  | 0 | GOOD |
| al_vacancy | eon-lanczos |  | 0 | BAD_MAX_ITERATIONS |
| lj7 | eon-neb | 7 | 0 | GOOD |
| lj7 | ase-ci | 7 | 0 | max_iterations |
| lj7 | eon-neb | 11 | 0 | GOOD |
| lj7 | ase-neb | 11 | 0 | max_iterations |
| lj7 | eon-ci | 11 | 0 | GOOD |
| lj7 | eon-dimer |  | 0 | GOOD |
| lj7 | rgsaddle-lanczos |  | 0 | max_iterations |
| cu4_h | eon-neb | 7 | 0 | GOOD |
| cu4_h | ase-ci | 7 | 0 | max_iterations |
| cu4_h | eon-neb | 11 | 0 | GOOD |
| cu4_h | eon-dimer |  | 0 | GOOD |
| cu4_h | eon-lanczos |  | 0 | GOOD |
| cu4_h | eon-neb | 11 | 1 | GOOD |
| cu4_h | ase-ci | 7 | 1 | max_iterations |
| cu4_h | eon-neb | 7 | 1 | GOOD |
| lj7 | eon-lanczos |  | 1 | GOOD |
| lj7 | rgsaddle-lanczos |  | 1 | max_iterations |
| lj7 | eon-dimer |  | 1 | GOOD |
| lj7 | eon-ci | 11 | 1 | GOOD |
| lj7 | ase-neb | 11 | 1 | max_iterations |
| lj7 | eon-neb | 11 | 1 | GOOD |
| lj7 | ase-ci | 7 | 1 | max_iterations |
| lj7 | eon-neb | 7 | 1 | GOOD |

313 further failures are in the JSON.

## RMSD consistency check

Recomputed RMSD values use the stored structures and minimage 0.1.2.
Each recomputed value matches the stored RMSD, so the summary is unchanged.

| system | method | images | round | stored rmsd A | recomputed rmsd A | note |
| --- | --- | --- | --- | --- | --- | --- |
| pt_exchange | ase-neb | 7 | 0 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 1 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 2 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 3 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 4 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 5 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 6 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 7 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 8 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 9 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 10 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 7 | 11 | 7.600849372084 | 7.600849372084 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 0 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 1 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 2 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 3 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 4 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 5 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 6 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 7 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 8 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 9 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 10 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |
| pt_exchange | ase-neb | 11 | 11 | 3.998973588903 | 3.998973588903 | The rejected independent check included fixed slab atoms and used a custom radius-2 lattice-image search; the benchmark uses free atoms only and minimage 0.1.2's Smith/Minkowski-reduced 27-image Euclidean convention. The legacy harness passes raw stored positions directly to minimage::Cell::displacement_euclidean through the C ABI, performs no ASE get_distances call, explicit wrapping, centering, or alignment, and computes sqrt(mean(per-atom squared 3D displacement)) over free atoms. For ase-neb it compares the reference with the highest-energy interior image selected from the final unconverged band. All rounds are identical because ASE BFGS starts from the same linear band without a seed and deterministically reaches the same final selected structure; each run stores that structure independently. |

## Build

| component | revision or version | source |
| --- | --- | --- |
| rgsaddle | 11707696d20994e2839cbc13c0c092799ad58de8 | shas.rgsaddle |
| rgsaddle ABI | 1.43 | versions.rgsaddle_abi |
| rgpot | 97ebba0b062128b7dc4637501c4b70ed4944ae9d, 3.7.0, scalar kernels, fast neighbour list on, Fortran pots on | shas.rgpot + versions.rgpot + harness metadata |
| eOn | 16434cb979c979a152f4e6750bdcfdfb0bf173be, pyeonclient 0.4.2, project version 3.6.0 | shas.eon + versions metadata |
| ASE | 3.29.0 | versions.ase |
| numpy | 2.5.5 | versions.numpy |
| scipy | 1.18.1 | versions.scipy |
| rustc | rustc 1.100.0 (0ef1bc386 2026-11-09) | versions.rustc |
| gcc | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 | versions.gcc |
| g++ | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 | versions.g++ |
| linkcell (rgsaddle) | 0.3.5 | Cargo.lock package |
| minimage (rgsaddle) | 0.1.2 | Cargo.lock package |
| rgmin (rgsaddle) | 0.6.0 | Cargo.lock package |
| readcon-core (rgsaddle) | 0.17.1 | Cargo.lock package |
| linkcell (eOn-side) | unverified (linkcell (eOn-side) not exposed in versions metadata) | versions.eon_linkcell |
| minimage (eOn-side) | unverified (minimage (eOn-side) not exposed in versions metadata) | versions.eon_minimage |
| readcon-core (eOn-side) | unverified (readcon-core (eOn-side) not exposed in versions metadata) | versions.eon_readcon_core |
| native-arch (eOn-side) | unverified (native-arch (eOn-side) not exposed in versions metadata) | versions.eon_native_arch |
| capnp-rpc (eOn-side transport) | unverified (capnp-rpc (eOn-side transport) not exposed in versions metadata) | versions.eon_capnp_transport |
| cpu | Intel(R) Xeon(R) Processor | Linux CPU model metadata |

Revision and ABI are recorded on each run from the next run on.

One core, one thread, 12 rounds.
