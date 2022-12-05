import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
import warnings
from scipy.optimize import curve_fit
# Read df
df_water_times = pd.read_excel('Data.xlsx', sheet_name='water_times', skiprows=[1])
df_water_flow = pd.read_excel('Data.xlsx', sheet_name='water_flow', skiprows=[1])
df_radiation = pd.read_excel('Data.xlsx', sheet_name='radiation', skiprows=[1])
df_lai = pd.read_excel('Data.xlsx', sheet_name='leaf_area_biomass', skiprows=[1])
df_dm = pd.read_excel('Data.xlsx', sheet_name='fresh_biomass_DM', skiprows=[1])
df_biomass_curves = pd.read_excel('Data.xlsx', sheet_name='biomass', skiprows=[1])

print(df_water_times)
print(df_water_flow)
print(df_radiation)
print(df_lai)
print(df_dm)
class AeroponicModel:
    def __init__(self):
        self.water_times_losses = None
        self.water_flow_losses = None
        self.radiation_losses = None
        self.biomass_to_leaf = None
        self.fresh_biomass_to_dry_biomass = None
        self.growing_curves = None

    def calibrate(self, plot = False):

        # Calibrate biomass to leaf area
        self.biomass_to_leaf = np.polynomial.Chebyshev.fit(df_lai['fresh_biomass'], df_lai['leaf_area'], 5)
        if plot:
            plt.title('Biomass to leaf area')
            plt.plot(df_lai['fresh_biomass'], df_lai['leaf_area'], 'o', label='data')
            plt.plot(np.linspace(0, 200, 100), self.biomass_to_leaf(np.linspace(0, 200, 100)), label='chebyshev')
            plt.legend(loc='best')
            plt.xlabel('Fresh biomass (g/plant)')
            plt.ylabel('Leaf area (cm2/plant)')
            plt.show()

        # Calibrate fresh biomass to dry biomass
        self.fresh_biomass_to_dry_biomass = np.polynomial.Chebyshev.fit(df_dm['fresh_biomass'], df_dm['dry_biomass'], 4)
        if plot:
            plt.title('Fresh biomass to dry biomass')
            plt.plot(df_dm['fresh_biomass'], df_dm['dry_biomass'], 'o', label='data')
            plt.plot(np.linspace(0, 200, 100), self.fresh_biomass_to_dry_biomass(np.linspace(0, 200, 100)), label='chebyshev')
            plt.legend(loc='best')
            plt.xlabel('Fresh biomass (g/plant)')
            plt.ylabel('Dry biomass (g/plant)')
            plt.show()

        # Calibrate growing curves
        #sort by radiation and group by radiation
        df_biomass_grouped = df_biomass_curves.sort_values(by=['radiation']).groupby('radiation')
        # Fit splines to each group
        self.growing_curves = {}
        sigmoid = lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) + d
        for radiation, group in df_biomass_grouped:
            # Fit spline
            df = group.sort_values(by=['day'])
            params = curve_fit(sigmoid, df['day'], df['biomass_dry'], p0=[0, 1, 20, 40], maxfev=10000)
            # Store spline
            self.growing_curves[radiation] = params[0]
        # Plot
        if plot:
            plt.title('Growing curves')
            for i in self.growing_curves:
                plt.plot(np.linspace(11, 50, 100), sigmoid(np.linspace(11, 50, 100), *self.growing_curves[i]), label='Scenario = ' + str(i))
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
            plt.show()
            # Plot derivatives
            plt.title('Growing curves derivatives')
            for i in self.growing_curves:
                derivative = lambda x: sigmoid(x, *self.growing_curves[i]) * self.growing_curves[i][1] * self.growing_curves[i][0] * np.exp(-self.growing_curves[i][1] * (x - self.growing_curves[i][2])) / (1 + np.exp(-self.growing_curves[i][1] * (x - self.growing_curves[i][2]))) ** 2
                plt.plot(np.linspace(11, 50, 100), derivative(np.linspace(11, 50, 100)), label=f"Scenario {i}")
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass derivative (g/plant/day)')
            plt.show()

        # Get growing curves for each water times
        bet_scenario_params = self.growing_curves[22]
        time_curves = {}
        days = df_water_times['days'].mean()
        # Group water times by scenario
        for group, df in df_water_times.groupby('off'):
            # Get growing curve
            params = bet_scenario_params
            params[0] = df['biomass_dry']
            print(f"Scenario {group} params: {params}")
            # Store growing params
            time_curves[group] = np.array(params)
        # Plot
        if plot:
            plt.title('Growing curves for each water times')
            for group in time_curves:
                plt.plot(np.linspace(11, days, 100), sigmoid(np.linspace(11, days, 100), *time_curves[group]), label='Water times = ' + str(group))
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
            plt.show()

        # Get growing curves for each water flow
        bet_scenario_params = self.growing_curves[22]
        flow_curves = {}
        # Group water flow by scenario
        days = df_water_flow['days'].mean()
        for group, df in df_water_flow.groupby('rate'):
            # Get growing curve
            params = bet_scenario_params
            params[0] = self.fresh_biomass_to_dry_biomass(df['biomass_fresh'])
            print(f"Scenario {group} params: {params}")
            # Store growing params
            flow_curves[group] = np.array(params)
        # Plot
        if plot:
            plt.title('Growing curves for each water flow')
            for group in flow_curves:
                plt.plot(np.linspace(11, days, 100), sigmoid(np.linspace(11, days, 100), *flow_curves[group]), label='Water flow = ' + str(group))
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
            plt.show()


