# OptiWatt
This repository provides an advanced optimization tool for managing energy flows in photovoltaic (PV) systems with battery storage. The EnergyDispatchOptimizer function uses real-time solar irradiance forecasts and electricity market prices to determine the most profitable dispatch strategy, i.e., when to store energy in the battery or sell it to the grid.

🚀 Features
- Mixed-integer optimization using cvxpy with the GUROBI solver.
- Forecasts solar irradiance from Open-Meteo.
- Retrieves spot electricity prices from the REE ESIOS API.
- Models battery constraints such as capacity, charge/discharge rate, and prevents simultaneous charge/discharge.
- Optionally exports results to CSV and visualizes dispatch plans with matplotlib.
- Handles timezone conversion automatically for multiple European countries.
