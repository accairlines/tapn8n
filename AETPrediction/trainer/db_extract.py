#!/usr/bin/env python3
"""
Database extraction module for AET prediction training
Executes SQL queries from DB_Extract.sql per month and saves results to CSV files
"""

import os
import re
import logging
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import pymysql
from calendar import monthrange

# Load environment variables
load_dotenv()

# Database connection settings from environment
DB_HOST = os.environ.get('SQL_HOST')
DB_USER = os.environ.get('SQL_USER')
DB_PASSWORD = os.environ.get('SQL_PASSWORD')
DB_NAME = os.environ.get('SQL_DATABASE', 'taphubtm')
DB_PORT = int(os.environ.get('SQL_PORT', 3306))
DB_CA = os.environ.get('SQL_CA')

# Data path from environment
DATA_PATH = os.environ.get('AET_DATA_PATH')
LOG_PATH = os.environ.get('AET_LOG_PATH', './logs')

# Ensure log directory exists
os.makedirs(LOG_PATH, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, 'db_extract.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def get_db_connection():
    """Create and return a database connection"""
    try:
        if DB_CA:
            # SSL connection
            ssl_config = {
                'ca': DB_CA
            }
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT,
                ssl=ssl_config,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        else:
            # Non-SSL connection
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise


def parse_sql_file(sql_file_path):
    """Parse SQL file and extract individual queries"""
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by semicolon and filter out empty queries
    queries = []
    for query in content.split(';'):
        query = query.strip()
        if query and not query.startswith('--'):
            queries.append(query)
    
    return queries


def get_month_date_range(year, month):
    """Get start and end date for a given month"""
    start_date = datetime(year, month, 1)
    # Get last day of month
    last_day = monthrange(year, month)[1]
    end_date = datetime(year, month, last_day)
    # End date should be exclusive (first day of next month)
    end_date = end_date + timedelta(days=1)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def replace_date_in_query(query, start_date, end_date, year):
    """Replace date ranges and year placeholders in SQL query"""
    # First, replace %year% placeholder with actual year
    query = query.replace('%year%', str(year))
    
    # Replace %s placeholders with dates
    # For ACARS query with INTERVAL, replace in order: first %s with start_date, second %s with end_date
    # MySQL will handle the INTERVAL arithmetic
    # For other queries, same logic applies
    query = query.replace("'%s'", f"'{start_date}'", 1)  # Replace first occurrence
    query = query.replace("'%s'", f"'{end_date}'", 1)     # Replace second occurrence
    
    return query


def get_output_filename(query, year, month):
    """Determine output filename based on query content"""
    query_upper = query.upper()
    query_lower = query.lower()
    month_str = f"{year}-{month:02d}"
    
    if 'OSUSR_UUK_FLT_INFO' in query_upper:
        return f"flight_{month_str}.csv"
    elif 'OSUSR_FAM_FP_ARINC633' in query_upper and 'wp' in query_lower:
        return f"fp_arinc633_wp_{month_str}.csv"
    elif 'OSUSR_FAM_FP_ARINC633' in query_upper and 'mel' in query_lower:
        return f"fp_arinc633_mel_{month_str}.csv"
    elif 'OSUSR_FAM_FP_ARINC633' in query_upper:
        return f"fp_arinc633_fp_{month_str}.csv"
    elif 'OSUSR_FAM_TAP_ACARS' in query_upper:
        return f"acars_{month_str}.csv"
    else:
        return f"unknown_{month_str}.csv"


def execute_query_and_save(connection, query, output_path, month_str):
    """Execute a SQL query and save results to CSV"""
    try:
        logger.info(f"Executing query for {month_str}...")
        logger.debug(f"Query: {query[:200]}...")  # Log first 200 chars of query
        
        # Use cursor directly instead of pd.read_sql to avoid issues with DictCursor
        # pd.read_sql can have problems with pymysql DictCursor connections
        with connection.cursor() as cursor:
            cursor.execute(query)
            
            # Get column names from cursor description
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Create DataFrame from results
                if rows:
                    # DictCursor returns dict rows, convert to DataFrame
                    if isinstance(rows[0], dict):
                        df = pd.DataFrame(rows)
                    else:
                        # Regular cursor returns tuples
                        df = pd.DataFrame(rows, columns=columns)
                    
                    # Verify first row is not just column names (sanity check)
                    if len(df) > 0:
                        first_row_values = list(df.iloc[0].values)
                        if first_row_values == columns:
                            logger.error(f"Query returned column names as data for {month_str}! Query may be malformed.")
                            logger.error(f"Full query: {query}")
                            return False
                else:
                    # No rows returned, create empty DataFrame with column names
                    df = pd.DataFrame(columns=columns)
                    logger.warning(f"No data returned for {month_str}, creating empty file with columns")
            else:
                # No description means query didn't return a result set (e.g., DDL statement)
                logger.error(f"Query did not return a result set for {month_str}")
                return False
        
        if df.empty:
            logger.warning(f"No data returned for {month_str}, creating empty file")
        else:
            logger.info(f"Retrieved {len(df)} rows for {month_str}")
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='latin1')
        logger.info(f"Saved data to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error executing query for {month_str}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"Query that failed: {query}")
        return False


def extract_data_per_month(start_year, start_month, end_year=None, end_month=None):
    """
    Extract data from database per month based on SQL queries in DB_Extract.sql
    
    Args:
        start_year: Starting year
        start_month: Starting month (1-12)
        end_year: Ending year (defaults to current year)
        end_month: Ending month (defaults to current month)
    """
    # Default to current month if not specified
    if end_year is None or end_month is None:
        now = datetime.now()
        end_year = now.year
        end_month = now.month
    
    # Get current month for comparison
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Read SQL queries from file
    sql_file_path = os.path.join(os.path.dirname(__file__), 'DB_Extract.sql')
    if not os.path.exists(sql_file_path):
        logger.error(f"SQL file not found: {sql_file_path}")
        return False
    
    queries = parse_sql_file(sql_file_path)
    logger.info(f"Parsed {len(queries)} queries from {sql_file_path}")
    
    # Connect to database
    try:
        connection = get_db_connection()
        logger.info("Connected to database successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False
    
    success_count = 0
    skip_count = 0
    
    try:
        # Iterate through each month
        current_iter = datetime(start_year, start_month, 1)
        end_iter = datetime(end_year, end_month, 1)
        
        while current_iter <= end_iter:
            year = current_iter.year
            month = current_iter.month
            month_str = f"{year}-{month:02d}"
            
            # Check if this is the current month
            is_current_month = (year == current_year and month == current_month)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing month: {month_str} (Current month: {is_current_month})")
            logger.info(f"{'='*60}")
            
            # Create month directory
            month_dir = os.path.join(DATA_PATH, month_str)
            os.makedirs(month_dir, exist_ok=True)
            
            # Get date range for this month
            start_date, end_date = get_month_date_range(year, month)
            logger.info(f"Date range: {start_date} to {end_date}")
            
            # Process each query
            for i, query in enumerate(queries, 1):
                # Replace dates and year in query
                modified_query = replace_date_in_query(query, start_date, end_date, year)
                logger.debug(f"Query {i} for {month_str}:\n{modified_query}")
                
                # Determine output filename
                output_filename = get_output_filename(query, year, month)
                output_path = os.path.join(month_dir, output_filename)
                
                # Check if file already exists (either .csv or .done)
                # Also check for .done file (processed file)
                done_filename = output_filename.replace('.csv', '.done')
                done_path = os.path.join(month_dir, done_filename)
                
                # Skip if either .csv or .done file exists (unless it's current month)
                if (os.path.exists(output_path) or os.path.exists(done_path)) and not is_current_month:
                    logger.info(f"Skipping query {i} for {month_str} - file already exists: {output_filename} or {done_filename}")
                    skip_count += 1
                    continue
                
                # For current month, also check if .done file exists (already processed)
                if is_current_month and os.path.exists(done_path):
                    logger.info(f"Skipping query {i} for {month_str} - file already processed (.done exists): {done_filename}")
                    skip_count += 1
                    continue
                
                # Execute query and save
                if execute_query_and_save(connection, modified_query, output_path, f"{month_str} (query {i})"):
                    success_count += 1
                else:
                    logger.warning(f"Failed to execute query {i} for {month_str}")
            
            # Move to next month
            current_iter += relativedelta(months=1)
    
    finally:
        connection.close()
        logger.info("Database connection closed")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction completed:")
    logger.info(f"  - Successful: {success_count}")
    logger.info(f"  - Skipped: {skip_count}")
    logger.info(f"{'='*60}")
    
    return True


def main():
    """Main function to extract data"""
    logger.info("=== Starting database extraction ===")
    
    # You can customize the date range here
    # For example, extract data from January 2024 to current month
    start_year = 2024
    start_month = 1
    
    try:
        success = extract_data_per_month(start_year, start_month)
        if success:
            logger.info("=== Database extraction completed successfully ===")
            return 0
        else:
            logger.error("=== Database extraction failed ===")
            return 1
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

