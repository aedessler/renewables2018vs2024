#!/usr/bin/env python3
"""
RTO Data Reader - Download and process electricity demand and price data for all RTOs

This script downloads electricity demand and price data for all major RTOs using the gridstatus.io API
for both 2018 and 2024, following the same data structure as the existing analysis.

Usage:
    python rto_data_reader.py --api-key YOUR_API_KEY

Requirements:
    - gridstatusio library (pip install gridstatusio)
    - Valid gridstatus.io API key
    - pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import sys
from gridstatusio import GridStatusClient

# RTO configuration mapping for demand data
RTO_DEMAND_CONFIG = {
    'caiso': {
        'dataset': 'caiso_standardized_hourly',
        'load_column': 'load.load',
        'name': 'CAISO',
        'timezone': 'US/Pacific'
    },
    'ercot': {
        'dataset': 'ercot_standardized_hourly', 
        'load_column': 'load.load',
        'name': 'ERCOT',
        'timezone': 'US/Central'
    },
    'isone': {
        'dataset': 'isone_standardized_hourly',
        'load_column': 'load.load', 
        'name': 'ISONE',
        'timezone': 'US/Eastern'
    },
    'miso': {
        'dataset': 'miso_standardized_hourly',
        'load_column': 'load.load',
        'name': 'MISO',
        'timezone': 'US/Central'
    },
    'nyiso': {
        'dataset': 'nyiso_standardized_hourly',
        'load_column': 'load.load',
        'name': 'NYISO',
        'timezone': 'US/Eastern'
    },
    'pjm': {
        'dataset': 'pjm_standardized_hourly',
        'load_column': 'load.load',
        'name': 'PJM',
        'timezone': 'US/Eastern'
    },
    'spp': {
        'dataset': 'spp_standardized_hourly',
        'load_column': 'load.load',
        'name': 'SPP',
        'timezone': 'US/Central'
    }
}

# RTO configuration mapping for price data
RTO_PRICE_CONFIG = {
    'caiso': {
        'dataset': 'caiso_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND'],
        'name': 'CAISO',
        'timezone': 'US/Pacific'
    },
    'ercot': {
        'dataset': 'ercot_spp_day_ahead_hourly',
        'price_column': 'spp',
        'location_filter': 'HB_HUBAVG',
        'name': 'ERCOT',
        'timezone': 'US/Central'
    },
    'isone': {
        'dataset': 'isone_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['.H.INTERNAL_HUB', '.Z.CONNECTICUT', '.Z.MAINE', '.Z.NEWHAMPSHIRE', 
                           '.Z.RHODEISLAND', '.Z.VERMONT', '.Z.SEMASS', '.Z.WCMASS', '.Z.NEMASSBOST'],
        'name': 'ISONE',
        'timezone': 'US/Eastern'
    },
    'miso': {
        'dataset': 'miso_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['ARKANSAS.HUB', 'ILLINOIS.HUB', 'INDIANA.HUB', 'LOUISIANA.HUB', 
                           'MICHIGAN.HUB', 'MINN.HUB', 'MS.HUB', 'TEXAS.HUB'],
        'name': 'MISO',
        'timezone': 'US/Central'
    },
    'nyiso': {
        'dataset': 'nyiso_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['WEST', 'GENESE', 'CENTRL', 'NORTH', 'MHK VL', 'CAPITL', 
                           'HUD VL', 'MILLWD', 'DUNWOD', 'N.Y.C.', 'LONGIL'],
        'name': 'NYISO',
        'timezone': 'US/Eastern'
    },
    'pjm': {
        'dataset': 'pjm_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['AEP GEN HUB', 'ATSI GEN HUB', 'CHICAGO GEN HUB', 'DOMINION HUB', 
                           'EASTERN HUB', 'NEW JERSEY HUB', 'OHIO HUB'],
        'name': 'PJM',
        'timezone': 'US/Eastern'
    },
    'spp': {
        'dataset': 'spp_lmp_day_ahead_hourly',
        'price_column': 'lmp',
        'location_filter': ['SPPNORTH_HUB', 'SPPSOUTH_HUB'],
        'name': 'SPP',
        'timezone': 'US/Central'
    }
}

def download_rto_demand_data(client, rto_key, year, output_dir='rto_data'):
    """
    Download demand data for a specific RTO and year
    
    Parameters:
    -----------
    client : GridStatusClient
        Authenticated gridstatus client
    rto_key : str
        RTO key from RTO_DEMAND_CONFIG
    year : int
        Year to download data for
    output_dir : str
        Directory to save CSV files
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with demand data or None if failed
    """
    
    config = RTO_DEMAND_CONFIG[rto_key]
    
    print(f"\nDownloading {config['name']} demand data for {year}...")
    
    # Set date range for full year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # Calculate approximate limit (365 days * 24 hours + buffer)
    limit_rows = 365 * 24 + 100
    
    try:
        # Download data from gridstatus
        df = client.get_dataset(
            config['dataset'],
            start=start_date,
            end=end_date,
            limit=limit_rows
        )
        
        print(f"Downloaded {len(df)} rows for {config['name']} demand")
        
        if len(df) == 0:
            print(f"No demand data found for {config['name']} in {year}")
            return None
        
        # Process timestamp - convert to local time for the RTO
        if 'interval_start_utc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['interval_start_utc']).dt.tz_convert(config['timezone'])
        
        # Extract load data and other relevant columns
        columns_to_keep = ['timestamp', config['load_column']]
        
        # Add fuel mix columns if available (similar to ERCOT example)
        fuel_mix_columns = [col for col in df.columns if col.startswith('fuel_mix.')]
        if fuel_mix_columns:
            columns_to_keep.extend(fuel_mix_columns)
        
        # Add net load if available
        if 'net_load' in df.columns:
            columns_to_keep.append('net_load')
        
        # Filter to available columns
        available_columns = [col for col in columns_to_keep if col in df.columns]
        demand_data = df[available_columns].copy()
        
        # Rename load column for consistency
        demand_data = demand_data.rename(columns={config['load_column']: 'load'})
        
        # If load column is missing or all NaN, try to calculate from fuel mix
        if 'load' not in demand_data.columns or demand_data['load'].isnull().all():
            fuel_mix_cols = [col for col in demand_data.columns if col.startswith('fuel_mix.')]
            if fuel_mix_cols:
                print(f"  Load column missing/empty, calculating from fuel mix ({len(fuel_mix_cols)} sources)")
                demand_data['load'] = demand_data[fuel_mix_cols].sum(axis=1)
                print(f"  Calculated load from fuel mix: {demand_data['load'].count()} valid values")
            else:
                print(f"  Warning: No load data or fuel mix data available for {config['name']} {year}")
        
        return demand_data
        
    except Exception as e:
        print(f"Error downloading {config['name']} demand data: {str(e)}")
        return None

def download_rto_price_data(client, rto_key, year, output_dir='rto_data'):
    """
    Download price data for a specific RTO and year
    
    Parameters:
    -----------
    client : GridStatusClient
        Authenticated gridstatus client
    rto_key : str
        RTO key from RTO_PRICE_CONFIG
    year : int
        Year to download data for
    output_dir : str
        Directory to save CSV files
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with price data or None if failed
    """
    
    config = RTO_PRICE_CONFIG[rto_key]
    
    print(f"Downloading {config['name']} price data for {year}...")
    
    # Set date range for full year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    try:
        if isinstance(config['location_filter'], list):
            # For RTOs with multiple specific locations/hubs, download and average them
            print(f"Fetching data for multiple locations: {len(config['location_filter'])} hubs")
            limit_rows = 365 * 24 * len(config['location_filter']) + 100
            
            location_data = []
            for location in config['location_filter']:
                print(f"  Downloading {location}...")
                df_location = client.get_dataset(
                    config['dataset'],
                    start=start_date,
                    end=end_date,
                    filter_column="location",
                    filter_value=location,
                    limit=limit_rows
                )
                if len(df_location) > 0:
                    location_data.append(df_location)
                    print(f"    Downloaded {len(df_location)} rows for {location}")
                else:
                    print(f"    No data found for {location}")
            
            if not location_data:
                print(f"No price data found for any locations in {config['name']}")
                return None
            
            # Combine location data and calculate average
            df_all_locations = pd.concat(location_data, ignore_index=True)
            print(f"Calculating average across {len(config['location_filter'])} locations...")
            
            price_col = config['price_column']
            df_avg = df_all_locations.groupby('interval_start_utc')[price_col].mean().reset_index()
            
            # Add back other columns
            df_avg['location'] = f"{config['name']}_AVERAGE"
            
            df = df_avg
            print(f"Calculated location average for {len(df)} time periods")
            
        else:
            # For single location (like ERCOT hub average)
            print(f"Fetching data with location filter: {config['location_filter']}")
            limit_rows = 365 * 24 + 100
            df = client.get_dataset(
                config['dataset'],
                start=start_date,
                end=end_date,
                filter_column="location",
                filter_value=config['location_filter'],
                limit=limit_rows
            )
            print(f"Downloaded {len(df)} rows for {config['name']} price")
            
            if len(df) == 0:
                print(f"No price data found for {config['name']} in {year}")
                return None
        
        # Process timestamp - convert to local time for the RTO
        if 'interval_start_utc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['interval_start_utc']).dt.tz_convert(config['timezone'])
        
        # Extract price data
        price_col = config['price_column']
        if price_col in df.columns:
            price_data = df[['timestamp', price_col]].copy()
            # Rename price column for consistency
            price_data = price_data.rename(columns={price_col: 'price'})
        else:
            print(f"Warning: Price column '{price_col}' not found in {config['name']} data")
            return None
        
        return price_data
        
    except Exception as e:
        print(f"Error downloading {config['name']} price data: {str(e)}")
        return None

def merge_demand_and_price_data(demand_df, price_df, rto_name):
    """
    Merge demand and price data on timestamp
    
    Parameters:
    -----------
    demand_df : pd.DataFrame
        DataFrame with demand data
    price_df : pd.DataFrame
        DataFrame with price data
    rto_name : str
        Name of the RTO for logging
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    
    if demand_df is None or price_df is None:
        print(f"Cannot merge data for {rto_name} - missing demand or price data")
        return None
    
    print(f"Merging demand and price data for {rto_name}...")
    print(f"  Demand data: {len(demand_df)} rows")
    print(f"  Price data: {len(price_df)} rows")
    
    # Merge on timestamp
    merged_df = pd.merge(
        demand_df,
        price_df,
        on="timestamp",
        how="inner"
    )
    
    print(f"  Merged data: {len(merged_df)} rows")
    
    return merged_df

def analyze_price_demand_relationship(df, rto_name, year, output_dir='rto_data'):
    """
    Analyze the relationship between price and demand, similar to readGridStatus.ipynb
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame with demand and price data
    rto_name : str
        Name of the RTO
    year : int
        Year of the data
    output_dir : str
        Directory to save analysis results
    """
    
    if df is None or len(df) == 0:
        print(f"No data to analyze for {rto_name} {year}")
        return
    
    print(f"\nAnalyzing price-demand relationship for {rto_name} {year}...")
    
    # Create daily averages (similar to the notebook)
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.set_index('timestamp')
    df_daily = df_copy.resample('D')[['load', 'price']].mean().reset_index()
    
    print(f"Daily averages calculated: {len(df_daily)} days")
    
    # Bin daily average load and calculate median price by load bin
    bin_size = 2500  # 2.5 GW in MW
    
    # Handle NaN values by dropping them before binning
    df_daily_clean = df_daily.dropna(subset=['load', 'price'])
    
    if len(df_daily_clean) == 0:
        print(f"Warning: No valid data after removing NaN values for {rto_name} {year}")
        return None
    
    # Create bins with cleaned data
    load_min = int(df_daily_clean['load'].min())
    load_max = int(df_daily_clean['load'].max())
    
    df_daily_clean['load_bin'] = pd.cut(
        df_daily_clean['load'], 
        bins=range(load_min, load_max + bin_size, bin_size)
    )
    
    median_price_by_daily_load_bin = df_daily_clean.groupby('load_bin', observed=True)['price'].median().reset_index()
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of daily data
    plt.scatter(df_daily_clean['load'], df_daily_clean['price'], alpha=0.5, label='Daily Average Data')
    
    # Calculate the center of each bin and plot median prices
    median_price_by_daily_load_bin['bin_center'] = median_price_by_daily_load_bin['load_bin'].apply(lambda x: x.mid)
    plt.plot(median_price_by_daily_load_bin['bin_center'], median_price_by_daily_load_bin['price'], 
             color='red', marker='o', linestyle='-', label='Median Price by Load Bin')
    
    plt.xlabel('Load (MW)')
    plt.ylabel('Price ($/MWh)')
    plt.title(f'{rto_name} {year}: Daily Average Price vs. Load with Median Price by Load Bin')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    plt.savefig(f"{output_dir}/plots/{rto_name.lower()}_price_demand_{year}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis results
    analysis_results = {
        'rto': rto_name,
        'year': year,
        'total_hours': len(df),
        'total_days': len(df_daily_clean),
        'avg_daily_load_mw': df_daily_clean['load'].mean(),
        'avg_daily_price_mwh': df_daily_clean['price'].mean(),
        'max_daily_load_mw': df_daily_clean['load'].max(),
        'min_daily_load_mw': df_daily_clean['load'].min(),
        'max_daily_price_mwh': df_daily_clean['price'].max(),
        'min_daily_price_mwh': df_daily_clean['price'].min(),
        'load_price_correlation': df_daily_clean['load'].corr(df_daily_clean['price'])
    }
    
    return analysis_results

def process_rto_data(client, rto_key, years, output_dir='rto_data'):
    """
    Process data for a single RTO across multiple years
    
    Parameters:
    -----------
    client : GridStatusClient
        Authenticated gridstatus client
    rto_key : str
        RTO key
    years : list
        List of years to process
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    
    rto_name = RTO_DEMAND_CONFIG[rto_key]['name']
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Processing {rto_name} data")
    print(f"{'='*60}")
    
    for year in years:
        print(f"\n--- Processing {rto_name} {year} ---")
        
        # Download demand data
        demand_df = download_rto_demand_data(client, rto_key, year, output_dir)
        
        # Download price data
        price_df = download_rto_price_data(client, rto_key, year, output_dir)
        
        # Merge data
        merged_df = merge_demand_and_price_data(demand_df, price_df, rto_name)
        
        if merged_df is not None and len(merged_df) > 0:
            # Save merged data
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/{rto_key}_hourly_demand_and_price_{year}.csv"
            merged_df.to_csv(filename, index=False)
            print(f"Saved merged data to {filename}")
            
            # Analyze price-demand relationship
            analysis_result = analyze_price_demand_relationship(merged_df, rto_name, year, output_dir)
            if analysis_result:
                results[f"{rto_name}_{year}"] = analysis_result
        
        else:
            print(f"Failed to process {rto_name} {year} - no merged data available")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Download and analyze RTO demand and price data')
    parser.add_argument('--api-key', type=str, help='GridStatus.io API key (if not provided, will use GRIDSTATUS_API_KEY environment variable)')
    parser.add_argument('--output-dir', type=str, default='rto_data', help='Output directory for data and analysis')
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2024], help='Years to download (default: 2018 2024)')
    parser.add_argument('--rtos', nargs='+', choices=list(RTO_DEMAND_CONFIG.keys()), 
                       default=list(RTO_DEMAND_CONFIG.keys()), help='RTOs to download (default: all)')
    
    args = parser.parse_args()
    
    # Get API key from command line argument or environment variable
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get('GRIDSTATUS_API_KEY')
        if not api_key:
            print("Error: No API key provided. Please either:")
            print("  1. Use --api-key argument: python rto_data_reader.py --api-key YOUR_KEY")
            print("  2. Set environment variable: export GRIDSTATUS_API_KEY=YOUR_KEY")
            sys.exit(1)
        else:
            print("Using API key from GRIDSTATUS_API_KEY environment variable")
    else:
        print("Using API key from command line argument")
    
    print(f"RTO Data Reader")
    print(f"Years: {args.years}")
    print(f"RTOs: {args.rtos}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize GridStatus client
    try:
        client = GridStatusClient(api_key=api_key)
        print("GridStatus client initialized successfully")
    except Exception as e:
        print(f"Error initializing GridStatus client: {str(e)}")
        sys.exit(1)
    
    # Process each RTO
    all_results = {}
    
    for rto_key in args.rtos:
        try:
            rto_results = process_rto_data(client, rto_key, args.years, args.output_dir)
            all_results.update(rto_results)
        except Exception as e:
            print(f"Error processing {rto_key}: {str(e)}")
            continue
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_filename = f"{args.output_dir}/rto_analysis_summary.csv"
        results_df.to_csv(results_filename)
        print(f"\nSaved analysis summary to {results_filename}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string())
    
    print(f"\nRTO data processing complete!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()