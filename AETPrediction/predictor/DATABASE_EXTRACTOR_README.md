# Database Extractor for AET Prediction

This module provides functionality to extract flight data from the Django database and create pandas DataFrames with the same column structure expected by the XGBoost model.

## Features

- Connects to Django database using the default database configuration
- Extracts data from all required tables (flights, flight plans, waypoints, MEL, ACARS, equipment, aircraft, stations)
- Joins data from multiple sources
- Creates features in the exact format expected by the XGBoost model
- Handles missing values and data preprocessing

## Usage

### Basic Usage

```python
from aet_api.db_extractor import get_flight_data_for_prediction

# Extract data for the last 30 days
df = get_flight_data_for_prediction()

# Extract data for specific date range
df = get_flight_data_for_prediction(
    start_date='2025-01-01', 
    end_date='2025-01-31'
)

# Extract data for last 7 days
df = get_flight_data_for_prediction(days_back=7)

# Extract data for a specific flight by ID
df = get_flight_data_for_prediction(flight_id=12345)
```

### Advanced Usage

```python
from aet_api.db_extractor import DatabaseExtractor

extractor = DatabaseExtractor()

# Extract data for date range
df = extractor.extract_flight_data(
    start_date='2025-01-01',
    end_date='2025-01-31'
)

# Extract data for specific flight
df = extractor.extract_flight_data(flight_id=12345)
```

### API Endpoint

You can also use the HTTP API endpoint:

```
# Extract data for date range
GET /api/extract-data/?start_date=2025-01-01&end_date=2025-01-31&days_back=30

# Extract data for specific flight
GET /api/extract-data/?flight_id=12345
```

## Data Sources

The extractor pulls data from the following tables:

1. **osusr_uuk_flt_info2025** - Main flight information
2. **osusr_fam_fp_arinc6332025** - Flight plans
3. **osusr_fam_fp_arinc633_wp2025** - Waypoints data
4. **osusr_fam_fp_arinc633_mel2025** - MEL (Minimum Equipment List)
5. **osusr_fam_tap_acars_plane202** - ACARS messages
6. **equipments** - Equipment information
7. **aircrafts** - Aircraft information
8. **stations_utc** - Station timezone information

## Output Format

The output DataFrame contains the following columns:

### Base Flight Columns (33 columns)
- `OFFBLOCK`, `MVT`, `ATA`, `ONBLOCK`, `ETA`
- `FROM_IATA`, `STD`, `CALL_SIGN`, `AC_REGISTRATION`
- `OPERATOR`, `FLT_NR`, `TO_IATA`, `ETD`, `ATD`, `STA`, `FROM_STAND`, `TO_STAND`
- `AC_READY`, `TSAT`, `TOBT`, `CTOT`, `SERV_TYP_COD`
- `FROM_TERMINAL`, `TO_TERMINAL`, `FROM_GATE`, `CAPTAIN`, `AIRCRAFT_ICAO_TYPE`
- `AIRLINE_SPEC`, `PERFORMANCE_FACTOR`, `ROUTE_NAME`, `ROUTE_OPTIMIZATION`, `CRUISE_CI`, `CLIMB_PROC`
- `CRUISE_PROC`, `DESCENT_PRO`, `GREAT_CIRC`, `ZERO_FUEL_WEIGHT`
- `BODYTYPE`, `EQUIPTYPE`, `EQUIPTYPE2`

### Waypoint Features (150 columns)
- `wp1_SEG_WIND_DIRECTION` to `wp50_SEG_WIND_DIRECTION`
- `wp1_SEG_WIND_SPEED` to `wp50_SEG_WIND_SPEED`
- `wp1_SEG_TEMPERATURE` to `wp50_SEG_TEMPERATURE`

### ACARS Features (40 columns)
- `ac1_WINDDIRECTION` to `ac20_WINDDIRECTION`
- `ac1_WINDSPEED` to `ac20_WINDSPEED`

**Total: 223 columns** (same as expected by the XGBoost model)

## Testing

Run the test script to verify the database extraction works:

```bash
cd predictor
python test_db_extractor.py
```

This will:
- Test database connectivity
- Extract sample data
- Verify column structure
- Check for missing values
- Display sample output

## Requirements

- Django database connection configured in settings
- Required environment variables set:
  - `SQL_USER`
  - `SQL_PASSWORD` 
  - `SQL_HOST`
  - `SQL_CA`
- pandas, numpy, django packages installed

## Error Handling

The extractor includes comprehensive error handling:
- Database connection errors
- Missing table/data errors
- Data type conversion errors
- Missing value handling (filled with -1)

All errors are logged and exceptions are raised with descriptive messages.
