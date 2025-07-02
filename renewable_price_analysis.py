#!/usr/bin/env python3
"""
Renewable Energy vs Wholesale Price Analysis

This script analyzes the relationship between renewable energy percentage 
(wind + solar as percentage of total load) and load-weighted average wholesale 
electricity prices for different RTOs, comparing 2023 and 2024 data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_rto_data(rto_name, year):
    """
    Load RTO data for a specific year
    
    Parameters:
    -----------
    rto_name : str
        Name of the RTO (e.g., 'miso', 'ercot', etc.)
    year : int
        Year to load data for
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with the loaded data, or None if file doesn't exist
    """
    file_path = Path(f'rto_data/{rto_name}_hourly_demand_and_price_{year}.csv')
    
    if not file_path.exists():
        print(f"  File {file_path} is missing")
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['rto'] = rto_name.upper()
        df['year'] = year
        return df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def calculate_renewable_percentage(df, rto_name):
    """
    Calculate renewable energy percentage for a given RTO dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with RTO data
    rto_name : str
        Name of the RTO to determine column names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with renewable percentage added
    """
    df = df.copy()
    
    # Define renewable energy columns for different RTOs
    renewable_columns = {
        'miso': ['fuel_mix.wind', 'fuel_mix.solar'],
        'ercot': ['fuel_mix.wind', 'fuel_mix.solar'],
        'caiso': ['fuel_mix.wind', 'fuel_mix.solar'],
        'pjm': ['fuel_mix.wind', 'fuel_mix.solar'],
        'nyiso': ['fuel_mix.wind'],  # NYISO doesn't have solar in the data
        'isone': ['fuel_mix.wind', 'fuel_mix.solar'],
        'spp': ['fuel_mix.wind', 'fuel_mix.solar']
    }
    
    if rto_name.lower() not in renewable_columns:
        return None
    
    # Get renewable columns for this RTO
    renewable_cols = renewable_columns[rto_name.lower()]
    
    # Check if columns exist in the dataframe
    available_cols = [col for col in renewable_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    # Calculate total renewable generation
    df['renewable_generation'] = df[available_cols].sum(axis=1)
    
    # Calculate renewable percentage (avoid division by zero)
    df['renewable_percentage'] = np.where(
        df['load'] > 0,
        (df['renewable_generation'] / df['load']) * 100,
        0
    )
    
    # Remove rows with invalid data
    df = df.dropna(subset=['renewable_percentage', 'price', 'load'])
    df = df[df['load'] > 0]
    
    return df

def calculate_weighted_average_price(df):
    """
    Calculate load-weighted average price for the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with load and price data
    
    Returns:
    --------
    float
        Load-weighted average price
    """
    if df.empty or df['load'].sum() == 0:
        return np.nan
    
    # Calculate weighted average price
    weighted_price = (df['price'] * df['load']).sum() / df['load'].sum()
    return weighted_price

def analyze_rtos_by_year():
    """
    Analyze RTO data separately for 2023 and 2024
    
    Returns:
    --------
    tuple
        (results_2023_df, results_2024_df) - DataFrames with summary statistics for each year
    """
    # List of RTOs to analyze
    rtos = ['miso', 'ercot', 'caiso', 'pjm', 'spp', 'isone', 'nyiso']
    
    results_2023 = []
    results_2024 = []
    
    print("Analyzing 2023 data...")
    print("-" * 40)
    
    for rto_name in rtos:
        # Analyze 2023
        df_2023 = load_rto_data(rto_name, 2023)
        if df_2023 is not None:
            df_2023 = calculate_renewable_percentage(df_2023, rto_name)
            if df_2023 is not None and not df_2023.empty:
                # Calculate statistics
                total_renewable_generation = df_2023['renewable_generation'].sum()
                total_load = df_2023['load'].sum()
                renewable_pct = (total_renewable_generation / total_load) * 100
                weighted_price = calculate_weighted_average_price(df_2023)
                
                results_2023.append({
                    'RTO': rto_name.upper(),
                    'Renewable_Percentage': renewable_pct,
                    'Weighted_Average_Price': weighted_price,
                    'Total_Hours': len(df_2023)
                })
                
                print(f"{rto_name.upper()}: {renewable_pct:.1f}% renewable, ${weighted_price:.2f}/MWh ({len(df_2023):,} hours)")
    
    print(f"\nAnalyzing 2024 data...")
    print("-" * 40)
    
    for rto_name in rtos:
        # Analyze 2024
        df_2024 = load_rto_data(rto_name, 2024)
        if df_2024 is not None:
            df_2024 = calculate_renewable_percentage(df_2024, rto_name)
            if df_2024 is not None and not df_2024.empty:
                # Calculate statistics
                total_renewable_generation = df_2024['renewable_generation'].sum()
                total_load = df_2024['load'].sum()
                renewable_pct = (total_renewable_generation / total_load) * 100
                weighted_price = calculate_weighted_average_price(df_2024)
                
                results_2024.append({
                    'RTO': rto_name.upper(),
                    'Renewable_Percentage': renewable_pct,
                    'Weighted_Average_Price': weighted_price,
                    'Total_Hours': len(df_2024)
                })
                
                print(f"{rto_name.upper()}: {renewable_pct:.1f}% renewable, ${weighted_price:.2f}/MWh ({len(df_2024):,} hours)")
    
    return pd.DataFrame(results_2023), pd.DataFrame(results_2024)

def create_two_panel_plot(results_2023_df, results_2024_df):
    """
    Create a two-panel plot comparing 2023 and 2024 data
    
    Parameters:
    -----------
    results_2023_df : pd.DataFrame
        DataFrame with 2023 RTO summary statistics
    results_2024_df : pd.DataFrame
        DataFrame with 2024 RTO summary statistics
    """
    # Create the two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Renewable Energy vs Wholesale Electricity Prices by RTO', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Define colors for consistency across panels - assign each RTO a specific color
    rto_color_map = {
        'MISO': '#1f77b4',    # blue
        'ERCOT': '#ff7f0e',   # orange
        'CAISO': '#2ca02c',   # green
        'PJM': '#d62728',     # red
        'SPP': '#9467bd',     # purple
        'ISONE': '#8c564b',   # brown
        'NYISO': '#e377c2'    # pink
    }
    
    # Plot 2023 data (left panel)
    if not results_2023_df.empty:
        colors_2023 = [rto_color_map[rto] for rto in results_2023_df['RTO']]
        scatter1 = ax1.scatter(results_2023_df['Renewable_Percentage'], 
                              results_2023_df['Weighted_Average_Price'],
                              s=200, alpha=0.7, c=colors_2023, 
                              edgecolors='black', linewidth=1)
        
        # Add RTO labels
        for i, row in results_2023_df.iterrows():
            ax1.annotate(row['RTO'], 
                        (row['Renewable_Percentage'], row['Weighted_Average_Price']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add trend line for 2023
        if len(results_2023_df) > 1:
            z = np.polyfit(results_2023_df['Renewable_Percentage'], results_2023_df['Weighted_Average_Price'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(results_2023_df['Renewable_Percentage'].min(), 
                                 results_2023_df['Renewable_Percentage'].max(), 100)
            ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            
            # Calculate and display correlation
            corr = results_2023_df['Renewable_Percentage'].corr(results_2023_df['Weighted_Average_Price'])
            ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax1.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
            
            ax1.legend()
    
    # Plot 2024 data (right panel)
    if not results_2024_df.empty:
        colors_2024 = [rto_color_map[rto] for rto in results_2024_df['RTO']]
        scatter2 = ax2.scatter(results_2024_df['Renewable_Percentage'], 
                              results_2024_df['Weighted_Average_Price'],
                              s=200, alpha=0.7, c=colors_2024, 
                              edgecolors='black', linewidth=1)
        
        # Add RTO labels
        for i, row in results_2024_df.iterrows():
            ax2.annotate(row['RTO'], 
                        (row['Renewable_Percentage'], row['Weighted_Average_Price']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add trend line for 2024
        if len(results_2024_df) > 1:
            z = np.polyfit(results_2024_df['Renewable_Percentage'], results_2024_df['Weighted_Average_Price'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(results_2024_df['Renewable_Percentage'].min(), 
                                 results_2024_df['Renewable_Percentage'].max(), 100)
            ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            
            # Calculate and display correlation
            corr = results_2024_df['Renewable_Percentage'].corr(results_2024_df['Weighted_Average_Price'])
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax2.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
            
            ax2.legend()
    
    # Customize both panels
    for ax, year in zip([ax1, ax2], ['2023', '2024']):
        ax.set_xlabel('Renewable Energy Percentage (Wind + Solar) (%)', fontsize=12)
        ax.set_ylabel('Load-Weighted Average Price ($/MWh)', fontsize=12)
        ax.set_title(f'{year} Data', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Make sure both panels have the same scale for easy comparison
    if not results_2023_df.empty and not results_2024_df.empty:
        # Get combined ranges
        all_renewable = pd.concat([results_2023_df['Renewable_Percentage'], 
                                  results_2024_df['Renewable_Percentage']])
        all_prices = pd.concat([results_2023_df['Weighted_Average_Price'], 
                               results_2024_df['Weighted_Average_Price']])
        
        # Set same limits for both panels
        x_margin = (all_renewable.max() - all_renewable.min()) * 0.1
        y_margin = (all_prices.max() - all_prices.min()) * 0.1
        
        ax1.set_xlim(all_renewable.min() - x_margin, all_renewable.max() + x_margin)
        ax1.set_ylim(all_prices.min() - y_margin, all_prices.max() + y_margin)
        ax2.set_xlim(all_renewable.min() - x_margin, all_renewable.max() + x_margin)
        ax2.set_ylim(all_prices.min() - y_margin, all_prices.max() + y_margin)
    
    # Adjust layout and save
    plt.tight_layout()
    # plt.savefig('renewable_vs_price_2023_2024_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_tables(results_2023_df, results_2024_df):
    """
    Create and display summary tables for both years
    
    Parameters:
    -----------
    results_2023_df : pd.DataFrame
        DataFrame with 2023 RTO summary statistics
    results_2024_df : pd.DataFrame
        DataFrame with 2024 RTO summary statistics
    """
    print("\n" + "="*80)
    print("SUMMARY COMPARISON: 2023 vs 2024")
    print("="*80)
    
    # 2023 Summary
    if not results_2023_df.empty:
        print("\n2023 DATA:")
        print("-" * 50)
        results_2023_sorted = results_2023_df.sort_values('Renewable_Percentage', ascending=False)
        print(f"{'RTO':<8} {'Renewable %':<12} {'Avg Price':<12} {'Total Hours':<12}")
        print("-" * 50)
        for _, row in results_2023_sorted.iterrows():
            print(f"{row['RTO']:<8} {row['Renewable_Percentage']:<12.1f} "
                  f"${row['Weighted_Average_Price']:<11.2f} {row['Total_Hours']:<12,}")
        
        if len(results_2023_df) > 1:
            corr_2023 = results_2023_df['Renewable_Percentage'].corr(results_2023_df['Weighted_Average_Price'])
            print(f"\n2023 Correlation: {corr_2023:.3f}")
    
    # 2024 Summary
    if not results_2024_df.empty:
        print("\n2024 DATA:")
        print("-" * 50)
        results_2024_sorted = results_2024_df.sort_values('Renewable_Percentage', ascending=False)
        print(f"{'RTO':<8} {'Renewable %':<12} {'Avg Price':<12} {'Total Hours':<12}")
        print("-" * 50)
        for _, row in results_2024_sorted.iterrows():
            print(f"{row['RTO']:<8} {row['Renewable_Percentage']:<12.1f} "
                  f"${row['Weighted_Average_Price']:<11.2f} {row['Total_Hours']:<12,}")
        
        if len(results_2024_df) > 1:
            corr_2024 = results_2024_df['Renewable_Percentage'].corr(results_2024_df['Weighted_Average_Price'])
            print(f"\n2024 Correlation: {corr_2024:.3f}")

def main():
    """
    Main function to run the analysis
    """
    print("Starting Renewable Energy vs Wholesale Price Analysis...")
    print("Comparing 2023 and 2024 data side by side")
    print("="*80)
    
    # Analyze RTOs for both years
    results_2023_df, results_2024_df = analyze_rtos_by_year()
    
    if results_2023_df.empty and results_2024_df.empty:
        print("No data found for analysis!")
        return
    
    # Display summary tables
    create_summary_tables(results_2023_df, results_2024_df)
    
    # Create the two-panel plot
    print("\nCreating two-panel comparison plot...")
    create_two_panel_plot(results_2023_df, results_2024_df)
    
    print("\nAnalysis complete!")
    print("Generated file: renewable_vs_price_2023_2024_comparison.png")

if __name__ == "__main__":
    main() 