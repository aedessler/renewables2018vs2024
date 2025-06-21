# Renewables Save Money: Electricity Market Analysis

This repository analyzes how renewable energy (particularly solar and wind) has transformed electricity markets and reduced wholesale prices across the United States. The analysis focuses on comparing electricity demand, generation, and pricing patterns between 2018 and 2024, with special emphasis on August data when electricity demand is typically highest.

## 📁 Repository Structure

```
renewablesSaveMoney/
├── augustPrice.ipynb              # Main analysis notebook
├── rto_data_reader.py            # Data collection script
├── rto_data/                     # Electricity market data
│   ├── *_hourly_demand_and_price_*.csv  # RTO data files
│   ├── plots/                    # Generated visualizations
│   └── rto_analysis_summary.csv  # Summary statistics
├── *.png                         # Generated charts
└── README.md                     # This file
```

## 📊 Main Analysis: `augustPrice.ipynb`

The primary analysis notebook that examines ERCOT (Texas) electricity market data to demonstrate the impact of renewable energy on wholesale electricity prices.

### Analysis Components

1. **Demand Pattern Analysis**
   - Hourly electricity demand profiles for August 2018 vs 2024
   - Shows increased peak demand but different pricing dynamics

2. **Renewable Generation Growth**
   - Solar + battery generation: 1.2 GW (2018) → 18.8 GW (2024)
   - Wind generation patterns and their impact on net load

3. **Price-Load Relationship Analysis**
   - Median wholesale price vs. total load for both years
   - Net load analysis (total load minus renewables)
   - Demonstrates how renewables "flatten" the price curve

4. **Demand-Weighted Average Price Calculation**
   - Calculates `Σ(load_i × price_i) / Σ(load_i)` for realistic cost estimates
   - Compares actual 2024 prices with projected prices using 2018 relationships

5. **Counterfactual Analysis**
   - **Scenario 1 (Capped)**: What if 2024 demand used 2018 pricing, capped at highest 2018 price
   - **Scenario 2 (Extrapolated)**: Linear extrapolation of 2018 price-demand relationship to 2024 levels

### Generated Visualizations

- `ercot_demand_by_hour_august.png` - Hourly demand patterns
- `ercot_solar_generation_august.png` - Solar generation profiles  
- `ercot_solar_wind_generation_august.png` - Combined renewable generation
- `ercot_price_vs_net_load_august.png` - Price vs. net load relationship
- `ercot_price_vs_total_load_august.png` - Price vs. total load relationship
- `ercot_net_load_vs_total_load_august.png` - Net load fraction analysis
- `ercot_price_load_extrapolation_scenarios.png` - Counterfactual pricing scenarios

## 🔧 Data Collection: `rto_data_reader.py`

A script for downloading electricity market data from all major US Regional Transmission Organizations (RTOs) using the gridstatus.io API.

### Supported RTOs

- **CAISO** (California)
- **ERCOT** (Texas)  
- **ISONE** (New England)
- **MISO** (Midwest)
- **NYISO** (New York)
- **PJM** (Mid-Atlantic)
- **SPP** (Southwest Power Pool)

### Features

- Downloads both demand and price data for specified years
- Handles timezone conversions for each RTO region
- Merges demand and price data into unified datasets
- Generates price-demand relationship plots for each RTO
- Creates summary statistics across all RTOs
- Robust error handling and data validation

### Usage

```bash
# Install required packages
pip install gridstatusio pandas numpy matplotlib

# Run data collection (requires API key)
python rto_data_reader.py --api-key YOUR_API_KEY

# Or run with specific parameters
python rto_data_reader.py --api-key YOUR_API_KEY --years 2018 2024 --rtos ercot caiso
```

### Data Output

The script generates:
- Individual CSV files: `{rto}_hourly_demand_and_price_{year}.csv`
- Summary analysis: `rto_analysis_summary.csv`
- Visualization plots in `rto_data/plots/`

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy jupyter gridstatusio
```

### Running the Analysis

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd renewablesSaveMoney
   ```

2. **Launch Jupyter notebook**
   ```bash
   jupyter notebook augustPrice.ipynb
   ```

3. **Run all cells** to reproduce the analysis and generate visualizations

### Collecting New Data

1. **Get API key** from [gridstatus.io](https://gridstatus.io)

2. **Run data collection script**
   ```bash
   python rto_data_reader.py --api-key YOUR_API_KEY
   ```

## 🔬 Methodology

### Price-Load Relationship Analysis
- Uses median prices within load bins (2.5 GW width) to establish relationships
- Linear interpolation between bins for smooth price curves
- Separate analysis of total load vs. net load (load minus renewables)

### Demand-Weighted Pricing
- Weights each hour's price by its corresponding load
- Provides more realistic cost estimates than simple averages
- Accounts for the fact that high-demand periods typically have higher prices

### Counterfactual Scenarios
- **Capped**: Conservative estimate assuming 2018 pricing structure with price ceiling
- **Extrapolated**: Aggressive estimate using linear extrapolation of 2018 trends

## 📄 License

This project is open source. Please cite this repository if you use the analysis or methodology in your work.
