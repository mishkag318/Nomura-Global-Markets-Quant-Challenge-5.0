# Nomura Global Markets Quant Challenge 5.0

This repository contains my solutions for the **Nomura Global Markets Quant Challenge 5.0**, covering:

- **Question 2:** Interest-rate curve construction, swap pricing and analytical risk sensitivities  
- **Question 3:** LP market-making, adversity modelling, externalization and dynamic quoting  

---

## Repository Structure

```text
├── Q2_solution/
│   ├── solution.py
│   ├── Input.csv
│   └── question2_output.csv
│
├── Q3_solution/
│   ├── Task_1/
│   ├── Task_2/
│   ├── Task_3/
│   ├── Task_4/
│   └── Task_5/
│
└── README.md
Question 2: Interest-Rate Curve Construction, Swap Pricing & Risk
Overview

Question 2 focuses on building a fixed-income pricing framework for constructing discount curves, pricing swaps and computing analytical risk sensitivities.

The solution constructs two independent discount-factor curves:

A cash curve from deposit market quotes
A swap curve from par swap rates

It supports two interpolation schemes on log discount factors:

Linear interpolation
Averaged-quadratic interpolation

These curve setups are then used to price a fixed-payer interest-rate swap and compute quote-level analytical Greeks.

Key Features
Built a curve construction engine for cash and swap instruments
Bootstrapped discount factors across multiple maturities
Priced a 25Y fixed-payer interest-rate swap
Computed analytical PV sensitivities to input market quotes
Compared outputs across four curve/interpolation combinations
Cross-checked analytical sensitivities against finite-difference estimates
Methodology
Curve Construction

The cash curve is constructed directly from deposit rates.
The swap curve is bootstrapped sequentially from short to long maturities. For each swap node, the unknown discount factor is solved such that the model swap rate matches the quoted par swap rate.

Interpolation

Interpolation is performed on log discount factors.

Linear interpolation assumes piecewise-constant implied forward rates.
Averaged-quadratic interpolation blends neighbouring quadratic fits to obtain smoother curve behaviour.
Swap Pricing

The swap is valued from the fixed-rate payer’s perspective. The floating leg is computed using discount factors, while the fixed leg is valued using the fixed coupon schedule.

Risk Sensitivities

The solution computes analytical PV sensitivities using the chain rule:

PV → interpolated discount factors → curve nodes → market quotes

The swap curve Jacobian is built using the implicit function theorem, preserving the lower-triangular structure of the bootstrap.

Question 3: Market-Making & Adversity Modelling
Overview

Question 3 focuses on modelling a Liquidity Provider facing informed and uninformed client flow. The solution develops a five-task pipeline covering client adversity, profitability, predictive modelling, externalization and dynamic quoting.

The objective is to improve LP decision-making by identifying adverse flow, selectively externalizing risky trades and adjusting bid/ask spreads under inventory pressure.

Task 1: Adversity Profiling

This task computes the adversity percentage for each client across multiple time horizons.

Adversity is measured as the percentage of trades that move against the LP after execution.

Key Work
Calculated adversity across 6 clients
Evaluated adversity across 6 time horizons
Compared client behaviour from safest to most adverse
Identified clients with consistently higher adverse-flow characteristics
Task 2: Client PnL & Minimum Half-Spread

This task computes expected LP PnL for each client and estimates the minimum half-spread required to make each client non-negative.

Key Work
Calculated client-wise LP profitability
Compared PnL decay across horizons
Estimated minimum half-spread requirements
Linked higher adversity with lower LP profitability
Task 3: Adversity Prediction Model

This task trains machine learning models to predict whether a trade will become adverse for the LP.

Model Used

The solution uses gradient-boosted machine learning models, including:

HistGradientBoostingClassifier
Features Used

The model uses only information available before or at trade time, avoiding look-ahead bias.

Feature groups include:

Trade-level features such as side, volume and spread
Time-based features such as hour, minute and seconds from midnight
Market-state features such as rolling volatility and signed flow
Client-history features based on past adverse behaviour

Future mid-price columns are used only for label construction and not as model inputs.

Task 4: Optimal Externalization Threshold

This task uses predicted adversity probabilities to decide whether each trade should be internalized or externalized.

Decision Rule
Externalize if predicted adversity probability > threshold
Internalize otherwise

The threshold is selected by maximizing validation PnL.

Key Work
Used model-predicted adversity probabilities
Tuned externalization thresholds by client and horizon
Balanced over-externalization against under-externalization
Improved LP decision-making by selectively externalizing risky trades
Task 5: Dynamic Quoting Under Inventory Pressure

This task builds an inventory-aware quoting strategy for the LP.

The quoting function adjusts bid and ask half-spreads based on:

Current inventory
Market volatility
Predicted adversity
Time pressure near close
Strategy Logic

The strategy widens spreads when predicted adversity is high and skews bid/ask quotes when inventory risk increases.

A deadzone is used to avoid unnecessary reaction to small inventory positions, while time pressure amplifies skew near the close.

Final Result

The final inventory-skew strategy improved the risk-adjusted quoting score:

24.83 → 26.77

It also reduced end-of-day inventory risk and drawdown compared to the fixed-spread baseline.

Results Summary
Area	Result
Curve Construction	Built cash and swap discount curves across multiple maturities
Swap Pricing	Priced 25Y fixed-payer swaps across four curve setups
Risk Analytics	Computed analytical quote-level Greeks
Adversity Modelling	Modelled LP adversity across 180K+ trades
Externalization	Tuned thresholds for selective trade externalization
Dynamic Quoting	Improved risk-adjusted score from 24.83 to 26.77
Technologies Used
Python
C++17
NumPy
Pandas
scikit-learn
SciPy
How to Run

Install the required Python dependencies:

pip install numpy pandas scikit-learn scipy

Then run the relevant scripts inside each solution folder.

For Question 2:

cd Q2_solution
python solution.py

For Question 3, run the scripts inside each task folder in order.

Notes

This repository is intended to document my submitted solutions for the Nomura Global Markets Quant Challenge 5.0. The work combines fixed-income analytics, machine learning, market microstructure and dynamic quoting under inventory constraints.
