# Bank Stock Analysis (2006-2016)

This project analyzes the stock performance of six major banks—Citigroup (C), Bank of America (BAC), Goldman Sachs (GS), JPMorgan Chase (JPM), Morgan Stanley (MS), and Wells Fargo (WFC)—over the period from 2006 to 2016. The analysis focuses on the impact of the 2008 financial crisis and provides insights into future stock price predictions using Monte Carlo simulations.

## Data Source

The data for this analysis was sourced from Yahoo Finance, covering daily stock prices from 2006 to 2016.

## Project Overview

1. **Exploratory Data Analysis (EDA):**
   - Analyzed stock price trends, identifying a significant dip during the 2008 financial crisis.
   - Citigroup (C) experienced the most severe losses, while Goldman Sachs (GS) recovered quickly.

2. **Percentage Change Calculation:**
   - Calculated percentage changes to evaluate potential returns for each bank.

3. **Correlation Analysis:**
   - Discovered near-perfect correlations among the banks' stock prices, with values close to 1.

4. **Monte Carlo Simulations:**
   - Performed Monte Carlo simulations to predict future stock prices, providing a probabilistic outlook based on historical data trends.

## Installation

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- yfinance (for downloading stock data)

## Results

- The analysis confirmed the significant impact of the 2008 financial crisis, with Citigroup (C) being the most affected.
  Goldman Sachs (GS) demonstrated resilience by quickly returning to pre-crisis levels.

- High correlations were found among the bank stocks.

- Monte Carlo simulations provided a range of possible future stock prices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project is for educational purposes only and should not be taken as financial advice or a recommendation to invest in any stocks. Always consult with a qualified financial advisor before making investment decisions.
