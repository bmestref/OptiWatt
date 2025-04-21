# OptiWatt
This repository provides an advanced optimization tool for managing energy flows in photovoltaic (PV) systems with battery storage. The EnergyDispatchOptimizer function uses real-time solar irradiance forecasts and electricity market prices to determine the most profitable dispatch strategy, i.e., when to store energy in the battery or sell it to the grid.

🚀 Features <br>
- Mixed-integer optimization using cvxpy with the GUROBI solver.
- Forecasts solar irradiance from Open-Meteo.
- Retrieves spot electricity prices from the REE ESIOS API.
- Models battery constraints such as capacity, charge/discharge rate, and prevents simultaneous charge/discharge.
- Optionally exports results to CSV and visualizes dispatch plans with matplotlib.
- Handles timezone conversion automatically for multiple European countries.

📦 Requirements <br>
Install the necessary libraries using:

`pip install -r requirements.txt`

Additionally, you must have a working installation and license of GUROBI for solving the optimization problem.

📈 What It Solves <br>
This function calculates the optimal energy dispatch from a solar PV system with an optional battery, deciding:

- When to charge the battery (if solar production is higher than current demand or price is low),
- When to discharge the battery (if the price is high),
- When to sell energy directly to the grid, and
- How to maximize revenue from selling energy at varying hourly electricity prices.

🌍 Supported Countries <br>
Electricity price data is supported for:

- 🇵🇹 Portugal (1)
- 🇫🇷 France (2)
- 🇪🇸 Spain (3)
- 🇬🇧 United Kingdom (8824)
- 🇩🇪 Germany (8826)
- 🇧🇪 Belgium (8827)
- 🇳🇱 Netherlands (8828)

📤 Example Use
`
EnergyDispatchOptimizer(
    latitude = 40.4168,
    longitude = -3.7038,
    panel_azimuth = 180,
    panel_tilt = 30,
    panel_size = 20,
    efficiency = 0.18,
    country_id = 3,
    end_date = "2025-04-23T00",
    S_max = 5000,
    S_init = 1000,
    both_ways = False,
    chart = True,
    export = True
)`

📁 Outputs <br>
Charts showing:
- Battery state, charge/discharge flows, and spot prices
- Revenue breakdown from direct sale vs battery discharge
- Optional CSV export with:
    - Hourly energy flows
    - Battery state
    - Revenue data

🧠 Applications <br>
- Smart energy management for residential PV installations
- Feasibility studies for solar and battery economics
- Real-time microgrid optimization
- Integration with home energy management systems (HEMS)

📝 License <br>
This project is open-source under the MIT License. See LICENSE for details.
