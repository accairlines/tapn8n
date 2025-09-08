def get_data_cols():
    # Define required columns for each file type
    flights_cols = [
        'OFFBLOCK', 'MVT', 'ATA', 'ONBLOCK', 'ETA',
        'FROM_IATA', 'STD', 'CALL_SIGN', 'AC_REGISTRATION',
        'OPERATOR', 'FLT_NR', 'TO_IATA', 'ETD', 'ATD', 'STA', 'FROM_STAND', 'TO_STAND',
        'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'SERV_TYP_COD',
        'FROM_TERMINAL', 'TO_TERMINAL', 'FROM_GATE'
    ]
    flight_plan_cols = [
        'CALLSIGN', 'DEPARTURE_AIRP', 'STD', 'TS', 'FLP_FILE_NAME',
        'CAPTAIN', 'AIRCRAFT_ICAO_TYPE', 'AIRLINE_SPEC', 'PERFORMANCE_FACTOR',
        'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC',
        'CRUISE_PROC', 'DESCENT_PRO', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT',
        'TRIP_DURATION', 'TAXI_OUT_TIME', 'FLIGHT_TIME', 'TAXI_IN_TIME', 
        'ARRIVAL_AIRP'
    ]
    waypoints_cols = [
        'ALTITUDE', 'SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE',
        'FLP_FILE_NAME', 'CUMULATIVE_FLIGHT_TIME'
    ]
    mel_cols = ['FLP_FILE_NAME']
    acars_cols = ['FLIGHT', 'REPORTTIME', 'WINDDIRECTION', 'WINDSPEED']
    equipments_cols = ['ID', 'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2']
    aircrafts_cols = ['ACREGISTRATION', 'EQUIPTYPEID']
    stations_cols = ['STATION', 'TIMEDIFF_MINUTES', 'DAY_NUM']
    
    