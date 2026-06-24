# Nomura Global Markets Quant Challenge 5.0

This repository contains my solutions for the Nomura Global Markets Quant Challenge 5.0.

The work mainly covers two problem statements:

- Question 2: Interest-rate curve construction, swap pricing and risk analytics
- Question 3: LP market-making, adversity modelling and dynamic quoting

## Repository Structure

Q2_solution contains the solution files for Question 2.

Q3_solution contains the solution files for Question 3, organized across five tasks.

## Question 2: Interest-Rate Curve Construction and Swap Pricing

Question 2 focuses on building a fixed-income pricing framework.

The solution constructs discount curves from cash and swap market instruments, prices a 25-year fixed-payer interest-rate swap and computes analytical quote-level risk sensitivities.

Key work done:

- Built a C++17 engine for cash and swap curve construction
- Bootstrapped discount factors across multiple maturities
- Priced 25Y fixed-payer swaps across different curve setups
- Computed analytical PV sensitivities to input market quotes
- Compared results across multiple curve and interpolation methods

## Question 3: LP Market-Making and Adversity Modelling

Question 3 focuses on modelling a Liquidity Provider facing different client flow patterns.

The solution develops a full pipeline for measuring client adversity, estimating LP profitability, predicting adverse trades, deciding externalization thresholds and designing dynamic bid/ask quotes.

Key work done:

- Analysed LP trade adversity across clients and time horizons
- Computed client-wise LP PnL and minimum half-spread requirements
- Trained HistGradientBoosting models on 180K+ LP trades to model adversity
- Tuned externalization thresholds to selectively externalize risky trades
- Designed an inventory-aware dynamic quoting strategy
- Improved the risk-adjusted quoting score from 24.83 to 26.77

## Technologies Used

- Python
- C++17
- NumPy
- Pandas
- scikit-learn
- SciPy

## How to Run

For Question 2, open the Q2_solution folder and run the solution script.

For Question 3, open the Q3_solution folder and run the task scripts in order from Task 1 to Task 5.

Install required Python libraries using:

pip install numpy pandas scikit-learn scipy

## Notes

This repository documents my submitted solutions for the Nomura Global Markets Quant Challenge 5.0. The project combines fixed-income analytics, machine learning, market microstructure and inventory-aware quoting.
