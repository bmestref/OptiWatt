import cvxpy as cp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry

def EnergyDispatchOptimizer(latitude, longitude, panel_azimuth, panel_tilt, panel_size,
                            efficiency, country_id, end_date, S_max, S_init, both_ways,
                            S_newday = None, B_max = None, V_total_max = None, V_max = None,
                            export = False, target_dir = None, chart = False):
     
     
    """
    Energy Dispatch Optimizer for a Photovoltaic + Battery System.

    This function calculates the optimal strategy to either store or sell energy produced
    by a photovoltaic system, using a battery to maximize revenue based on dynamic electricity prices.
    It uses mixed-integer linear programming to enforce energy storage and dispatch constraints.

    External data from the Spanish electricity market (ESIOS API) and solar irradiance forecasts 
    from Open-Meteo are fetched to build hourly energy production and price profiles.

    Parameters
    ----------
    latitude : float
        Latitude of the photovoltaic installation.
    longitude : float
        Longitude of the photovoltaic installation.
    panel_azimuth : float
        Azimuth angle of the solar panel (0° = North, 180° = South).
    panel_tilt : float
        Tilt angle of the solar panel.
    panel_size : float
        Total panel area or size [m²].
    efficiency : float
        Efficiency factor of the solar system (typically between 0 and 1).
    S_max : float
        Maximum energy storage capacity of the battery [Wh].
    S_init : float
        Initial battery state of charge at start of simulation [Wh].
    country_id : int
        Numeric code representing the country for electricity prices:
            - 1: Portugal
            - 2: France
            - 3: Spain
            - 8824: United Kingdom
            - 8826: Germany
            - 8827: Belgium
            - 8828: Netherlands
    end_date : str
        End of the simulation period in 'YYYY-MM-DD' format.
    both_ways : boolean
        If True, charge and discharge operations of the battery are enabled simultaneously.
    export : bool, optional
        If True, exports the optimization results to a CSV file. Default is False.
    target_dir : str or None, optional
        Directory to save the exported CSV. If None and export is True, saves to current working directory.
    chart : bool, optional
        If True, plots graphs of energy dispatch, battery charge/discharge and sales. Default is False.
    B_max : float, optional
        Maximum charge/discharge rate of the battery per hour [W]. Default is None.
    V_max : float, optional
        Maximum allowable sale of energy per hour [W] coming directly from the PV panel. Default is None.
    V_total_max : float, optional
        Maximum allowable sale of energy per hour [W] in total. Default is None.
    S_newday : float, optional
        Battery state at the start of the next day. Default is None.

    Returns
    -------
    If chart and export are False:
        tuple
            - total_revenue : float
                Total optimized revenue from energy sales [€].
            - energy_sold : np.ndarray
                Energy sold to the market at each hour.
            - battery_state : np.ndarray
                Battery charge level at each hour.
            - battery_input : np.ndarray
                Energy charged into the battery at each hour.
            - battery_output : np.ndarray
                Energy discharged from the battery at each hour.

    If chart is True:
        Displays plots for battery behavior and energy dispatch.

    If export is True:
        Saves CSV to target_dir or working directory, with energy flows and timestamps.

    Notes
    -----
    - Uses convex optimization with binary variables (`carga_flag`) to prevent simultaneous battery charge/discharge.
    - Weather data retrieved from Open-Meteo (global tilted irradiance).
    - Electricity prices pulled from ESIOS (REE Spain) API.
    - Timezones automatically adjusted based on selected country.
    - Problem solved using the GUROBI solver. Ensure GUROBI is installed and licensed.
    """
    
    # ---- Type Checks ----
    if not isinstance(latitude, (int, float)):
        raise TypeError("Latitude must be a number.")
    if not isinstance(longitude, (int, float)):
        raise TypeError("Longitude must be a number.")
    if not isinstance(panel_azimuth, (int, float)):
        raise TypeError("Panel azimuth must be a number.")
    if not isinstance(panel_tilt, (int, float)):
        raise TypeError("Panel tilt must be a number.")
    if not isinstance(panel_size, (int, float)):
        raise TypeError("Panel size must be a number.")
    if not isinstance(efficiency, (int, float)):
        raise TypeError("Efficiency must be a number.")
    if country_id not in [1, 2, 3, 8824, 8826, 8827, 8828]:
        raise TypeError("Country ID not valid, see valid IDs: 1 : Portugal," + \
                         "2 : France, 3 : Spain, 8824 : England, 8826 : Germany," + \
                              "8827 : Belgique, 8828 : Netherlands.")
    if not isinstance(S_max, (int, float)):
        raise TypeError("S_max must be a number.")
    if not isinstance(S_init, (int, float)):
        raise TypeError("S_init must be a number.")
    if not isinstance(both_ways, bool):
        raise TypeError("both_ways must be a True or False.")
    
    # Optional inputs
    if B_max is not None and not isinstance(B_max, (int, float)):
        raise TypeError("B_max must be a number.")
    if V_max is not None and not isinstance(V_max, (int, float)):
        raise TypeError("V_max must be a number.")
    if V_total_max is not None and not isinstance(V_total_max, (int, float)):
        raise TypeError("V_total_max must be a number.")
    if S_newday is not None and not isinstance(S_newday, (int, float)):
        raise TypeError("S_newday must be a number.")
    if not isinstance(export, bool):
        raise TypeError("export must be a boolean.")
    if not isinstance(chart, bool):
        raise TypeError("chart must be a boolean.")
    if target_dir is not None and not isinstance(target_dir, str):
        raise TypeError("target_dir must be a string or None.")
    
    try:
        datetime.strptime(end_date, "%Y-%m-%dT%H")
    except ValueError:
        raise ValueError("end_date must be in 'YYYY-MM-DDTHH' format, e.g., '2024-03-21T00'.")

    # ---- Get Country SPOT Price ----
    api_token = '196f0f3df07fbb9d99ecbaf511cf4a553ee9df59bd33f635e69c2e95e2bb9295'
    url_base = 'https://api.esios.ree.es/'
    endpoint = 'indicators/'
    indicator = '600'
    url = url_base + endpoint + indicator

    headers = {
    'Host': 'api.esios.ree.es',
    'x-api-key': api_token
    }

    params = {
    'start_date': f'{datetime.now().strftime("%Y-%m-%dT%H")}',
    'end_date': f'{end_date}'
    }

    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    df = pd.DataFrame(data['indicator']['values'])
    
    df = df[df['geo_id'] == country_id]
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%dT%H")
    # ---- Get Solar Irradiance variables from OpenMeteoAPI ----
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "global_tilted_irradiance",
        "forecast_days": 3,
        "tilt": panel_tilt,
	    "azimuth": panel_azimuth
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    hourly = response.Hourly()
    hourly_global_tilted_irradiance = hourly.Variables(0).ValuesAsNumpy()

    country_timezones = {
    1: "Europe/Lisbon",
    2: "Europe/Paris",
    3: "Europe/Madrid",
    8824: "Europe/London",
    8826: "Europe/Berlin",
    8827: "Europe/Brussels",
    8828: "Europe/Amsterdam"
    }

    hourly_data = {"datetime": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
    )}

    hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    dt_utc = hourly_dataframe['datetime'].apply(lambda x: x.replace(tzinfo=ZoneInfo("UTC")))

    dt_country = dt_utc.apply(lambda x: x.astimezone(ZoneInfo(country_timezones[country_id])))

    country_str = dt_country.apply(lambda x: x.strftime("%Y-%m-%dT%H"))

    hourly_dataframe['datetime'] = country_str

    df = pd.merge(df, hourly_dataframe, how = 'inner', on = 'datetime')

    prod = np.array(df['global_tilted_irradiance']\
        .apply(lambda x : panel_size * x * efficiency))
    
    price = np.array(df['value'].apply(lambda x: x * 10**(-6))) # Convert the MWh price to Wh price

    # ---- Optimize the Dispatch Schedule ----
    n = price.shape[0]         # Number of hours
    v = cp.Variable(n)         # Energy directly sold to the electricity market
    v_total = cp.Variable(n)   # Total energy sold to the electricity market
    b_in = cp.Variable(n)      # Energy input of the battery per hour
    b_out = cp.Variable(n)     # Energy output of the battery per hour
    s = cp.Variable(n)         # Battery charge state (including initial state)
    carga_flag = cp.Variable(n, boolean = True) # Force the battery to not be discharged and charged simultaneously
    
    constraints = []

    constraints.append(s[0] == S_init)
    
    if S_newday != None:
        new_day = df[df['datetime'].str.endswith("T00")].index.tolist()

        if isinstance(S_newday, (int, float)):
            for index in new_day:
                constraints.append(s[index] == S_newday)
        else:
            raise TypeError('No Battery State at the Ending of the Day inputed, or format not supported.')

    for t in range(n):
         
        constraints.append(0 <= s[t])
        constraints.append(s[t] <= S_max)

        constraints.append(s[t] == s[t-1] + b_in[t] - b_out[t])
        constraints.append(0 <= b_in[t])
        
        if not both_ways:
            if B_max is not None:
                constraints.append(b_in[t] <= B_max * carga_flag[t])
                constraints.append(b_out[t] <= B_max * (1 - carga_flag[t]))
            else:
                constraints.append(b_in[t] <= 1e15 * carga_flag[t])
                constraints.append(b_out[t] <= 1e15 * (1 - carga_flag[t]))

        constraints.append(0 <= b_out[t])


        constraints.append(0 <= v[t])
        constraints.append(0 <= v_total[t])

        if V_max != None:
            constraints.append(v[t] <= V_max)
        
        if V_total_max != None:
            constraints.append(v_total[t] <= V_total_max)

        constraints.append(v[t] == prod[t] - b_in[t]) 

        constraints.append(v[t] <= prod[t]) 
        constraints.append(v_total[t] == b_out[t] + v[t]) 

        if prod[t] == 0:
            constraints.append(b_in[t] == 0)
        
    objective = cp.Maximize(cp.sum(cp.multiply(price, v_total)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.GUROBI, reoptimize=True)
    
    dir_sold_cost = [i*j for i, j in zip(v.value, price)]
    bat_sold_cost = [i*j for i, j in zip(b_out.value, price)]
    total_sold = [i+j for i,j in zip(dir_sold_cost, bat_sold_cost)]

    if chart == True:
        
        fig, ax = plt.subplots(figsize = (15,5))
        ax.bar(x = df['datetime'], height = s.value, color = 'g', label = 'Battery State')
        ax.plot(df['datetime'], b_in.value, c = 'b', label = 'Battery Charge')
        ax.plot(df['datetime'] , b_out.value, c = 'r', label = 'Battery Discharge')
        ax.set(title = 'Energy Dispatch Schedule', xlabel = 'Datetime')
        ax.set_ylabel('Energy [W]', color='k')
        ax.tick_params(axis='y', labelcolor='k')
        ax.legend()

        ax2 = ax.twinx()
        ax2.plot(df['datetime'], price, c = 'k', linestyle = '--', label = 'SPOT Price W/h')
        ax2.set_ylabel('€/W', color='k')
        ax2.tick_params(axis='y', labelcolor='k')

        ax.set_xticklabels(df['datetime'], rotation=90)
        ax2.set_xticklabels(df['datetime'], rotation=90)
        ax2.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

        fig, ax = plt.subplots(figsize = (15,5))
        bars1 = ax.bar(df.index - 0.2, bat_sold_cost, 0.2, label='Revenue from Discharging Battery', color = 'b')
        bars2 = ax.bar(df.index, dir_sold_cost, 0.2, label='Revenue from Direct Sold', color = 'y')
        bars3 = ax.bar(df.index + 0.2, total_sold , 0.2, label='Total Revenue', color = 'g')
        ax.fill_between(x = df.index, y1 = np.full(df.shape[0], 0), y2 = np.cumsum(total_sold), color = 'g', alpha = 0.3)
        ax.set_xticks(df.index)
        ax.set_xticklabels(df['datetime'], rotation=90)        
        ax.set(title = 'Energy Sold in the Dispatch Schedule', xlabel = 'Datetime', ylabel = '€')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


    if export == True:
        export_data = pd.DataFrame({'datetime': df['datetime'], '€_sold':total_sold, 'energy_direct_sold':v.value,
                                     'energy_total_sold':v_total.value, '€_battery_sold':bat_sold_cost, '€_direct_sold':dir_sold_cost,
                                       'battery_state':s.value, 'energy_bat_input':b_in.value, 'energy_bat_output':b_out.value}) 
        
        file_name = f'EnergyDispatchSchedule_{df['datetime'].iloc[0]}_{df['datetime'].iloc[-1]}.csv'
        if target_dir != None:
            path = os.path.join(target_dir, file_name)
            export_data.to_csv(path)

        else:
            path = os.path.join(os.getcwd(), file_name)
            export_data.to_csv(path)
        
    else:
        return(total_sold, bat_sold_cost, dir_sold_cost, v.value, v_total.value, s.value, b_in.value, b_out.value)
    

  


