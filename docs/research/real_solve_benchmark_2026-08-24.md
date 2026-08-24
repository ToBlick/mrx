# Real-solve benchmark — `apply_inverse_hodge_laplacian` (saddle MINRES)

12x24x12, p=3, tol 1e-10, maxiter 10000. `!` = hit maxiter.

Three preconditioner arms, all on the SAME solve path:

- `jacobi` — `schur.outer='jacobi'`, the library default

- `a0` — `schur.outer='block'`, bc_alpha=`product`, bc_scale=0.10

- `a5` — `schur.outer='block'`, bc_alpha=`penalty`, bc_scale=3.0


## 1. Iterations, before vs after the mass-preconditioner fix

`before` = saddle lower block was a per-DoF jacobi diagonal (the bug).

`after`  = lower block is the production `block_jacobi` mass preconditioner.


| geometry | k | BC | jacobi before | jacobi after | a0 before | a0 after | a5 before | a5 after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 1 | free | 10000! | **1666** | 4316 | **313** | 4313 | **316** |
| toroid | 1 | dbc | 10000! | **1157** | 4277 | **220** | 4282 | **220** |
| toroid | 2 | free | 10000! | **1308** | 9612 | **314** | 9647 | **321** |
| toroid | 2 | dbc | 10000! | **1434** | 9683 | **301** | 9661 | **301** |
| toroid | 3 | free | 7371 | **455** | 3725 | **148** | 3840 | **158** |
| toroid | 3 | dbc | 9158 | **667** | 3303 | **143** | 3303 | **143** |
| w7x | 1 | free | 10000! | **4511** | 10000! | **2164** | 10000! | **2212** |
| w7x | 1 | dbc | 10000! | **2371** | 6451 | **741** | 6464 | **739** |
| w7x | 2 | free | 10000! | **5847** | 10000! | **4659** | 10000! | **4761** |
| w7x | 2 | dbc | 10000! | **2949** | 10000! | **1522** | 10000! | **1525** |
| w7x | 3 | free | 10000! | **1110** | 6229 | **578** | 6171 | **569** |
| w7x | 3 | dbc | 10000! | **1337** | 5608 | **512** | 5617 | **514** |
| quasr9983 | 1 | free | 10000! | **3186** | 6317 | **857** | 6749 | **926** |
| quasr9983 | 1 | dbc | 10000! | **1671** | 4977 | **327** | 4979 | **327** |
| quasr9983 | 2 | free | 10000! | **2111** | 10000! | **778** | 10000! | **873** |
| quasr9983 | 2 | dbc | 10000! | **2872** | 10000! | **574** | 10000! | **574** |
| quasr9983 | 3 | free | 10000! | **741** | 4635 | **225** | 6041 | **346** |
| quasr9983 | 3 | dbc | 10000! | **1383** | 4335 | **288** | 4335 | **288** |

## 2. After the fix: iterations, build time, solve time

| geometry | k | BC | n | jacobi it | a0 it | a0 build | a0 solve | a5 it | a5 build | a5 solve | a5/a0 it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toroid | 1 | free | 8700 | 1666 | 313 | 11.6s | 5.2s | 316 | 1.2s | 4.8s | 1.010 |
| toroid | 1 | dbc | 8124 | 1157 | 220 | 1.6s | 3.4s | 220 | 1.1s | 3.1s | 1.000 |
| toroid | 2 | free | 8664 | 1308 | 314 | 0.7s | 3.3s | 321 | 0.7s | 3.5s | 1.022 |
| toroid | 2 | dbc | 8376 | 1434 | 301 | 0.5s | 6.2s | 301 | 0.5s | 6.0s | 1.000 |
| toroid | 3 | free | 2880 | 455 | 148 | 0.1s | 1.8s | 158 | 0.1s | 1.8s | 1.068 |
| toroid | 3 | dbc | 2880 | 667 | 143 | 0.0s | 3.4s | 143 | 0.0s | 3.4s | 1.000 |
| w7x | 1 | free | 8700 | 4511 | 2164 | 11.7s | 10.4s | 2212 | 1.1s | 10.3s | 1.022 |
| w7x | 1 | dbc | 8124 | 2371 | 741 | 1.6s | 4.3s | 739 | 1.1s | 4.0s | 0.997 |
| w7x | 2 | free | 8664 | 5847 | 4659 | 0.7s | 10.6s | 4761 | 0.7s | 11.0s | 1.022 |
| w7x | 2 | dbc | 8376 | 2949 | 1522 | 0.5s | 10.8s | 1525 | 0.5s | 10.6s | 1.002 |
| w7x | 3 | free | 2880 | 1110 | 578 | 0.1s | 2.1s | 569 | 0.1s | 2.1s | 0.984 |
| w7x | 3 | dbc | 2880 | 1337 | 512 | 0.0s | 4.2s | 514 | 0.0s | 4.2s | 1.004 |
| quasr9983 | 1 | free | 8700 | 3186 | 857 | 11.9s | 6.6s | 926 | 1.1s | 6.5s | 1.081 |
| quasr9983 | 1 | dbc | 8124 | 1671 | 327 | 1.6s | 3.6s | 327 | 1.1s | 3.6s | 1.000 |
| quasr9983 | 2 | free | 8664 | 2111 | 778 | 0.7s | 4.1s | 873 | 0.7s | 4.4s | 1.122 |
| quasr9983 | 2 | dbc | 8376 | 2872 | 574 | 0.5s | 7.2s | 574 | 0.5s | 7.1s | 1.000 |
| quasr9983 | 3 | free | 2880 | 741 | 225 | 0.1s | 1.9s | 346 | 0.1s | 1.9s | 1.538 |
| quasr9983 | 3 | dbc | 2880 | 1383 | 288 | 0.0s | 3.7s | 288 | 0.0s | 3.7s | 1.000 |
| **TOTAL** | | | | **36776** | **14664** | | | **15113** | | | **1.031** |

Block outer vs jacobi outer, total iterations: 36776 vs 14664 = **2.51x**.

Total wall time (build+solve): a0 136.9s, a5 102.8s, a5/a0 = **0.751**.


## 3. Where a5 loses on iterations

| geometry | k | BC | a0 | a5 | a5/a0 |
| --- | --- | --- | --- | --- | --- |
| toroid | 3 | free | 148 | 158 | **1.068** |
| quasr9983 | 1 | free | 857 | 926 | **1.081** |
| quasr9983 | 2 | free | 778 | 873 | **1.122** |
| quasr9983 | 3 | free | 225 | 346 | **1.538** |

Everywhere else the two tie to within 3%. The losses are concentrated on
quasr9983, and the worst is k=3 free (+54%).

