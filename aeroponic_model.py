import numpy as np
from matplotlib import cm
import pandas as pd
from scipy.interpolate import CubicSpline
import warnings
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import itertools
import matplotlib.pyplot as plt


# Read df
df_water_times = pd.read_excel('Data.xlsx', sheet_name='water_times', skiprows=[1])
df_water_flow = pd.read_excel('Data.xlsx', sheet_name='water_flow', skiprows=[1])
df_radiation = pd.read_excel('Data.xlsx', sheet_name='radiation', skiprows=[1])
df_lai = pd.read_excel('Data.xlsx', sheet_name='leaf_area_biomass', skiprows=[1])
df_dm = pd.read_excel('Data.xlsx', sheet_name='fresh_biomass_DM', skiprows=[1])
df_biomass_curves = pd.read_excel('Data.xlsx', sheet_name='biomass', skiprows=[1])
df_wue = pd.read_excel('Data.xlsx', sheet_name='accumulated_water_light', skiprows=[1])
# # print(df_water_times)
# # print(df_water_flow)
# # print(df_radiation)
# # print(df_lai)
# # print(df_dm)


sigmoid = lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) + d
sigmoid_x = lambda y, a, b, c, d: -1 * np.log((a / (y-d)) - 1) / (b) + c
sigmoid_derivative = lambda biomass,x,a, b, c, d: a * b * np.exp(-b * (x - c)) / (1 + np.exp(-b * (x - c))) ** 2
plateau = lambda x, a, b, l: np.maximum((a*x+b), l)


class AeroponicModel:
    def __init__(self):
        self.radiation_losses = None
        self.biomass_to_leaf = None
        self.fresh_biomass_to_dry_biomass = None
        self.growing_curves_rad = None
        self.light_loss = None
        self.water_flow_loss = None
        self.water_times_loss = None
        self.best_params = None


    def calibrate(self, plot = False):

        # Calibrate biomass to leaf area
        self.biomass_to_leaf = np.polynomial.Chebyshev.fit(df_lai['fresh_biomass'], df_lai['leaf_area'], 2)
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.set_title('Biomass to leaf area')
            ax1.plot(df_lai['fresh_biomass'], df_lai['leaf_area'], 'o', label='data')
            ax1.plot(np.linspace(0, 300, 100), self.biomass_to_leaf(np.linspace(0, 300, 100)), label='chebyshev')
            ax1.legend(loc='best')
            ax1.set_xlabel('Fresh biomass (g/plant)')
            ax1.set_ylabel('Leaf area (cm2/plant)')

        # Calibrate fresh biomass to dry biomass
        self.fresh_biomass_to_dry_biomass = np.polynomial.Chebyshev.fit(df_dm['fresh_biomass'], df_dm['dry_biomass'], 2)
        self.dry_biomass_to_fresh_biomass = np.polynomial.Chebyshev.fit(df_dm['dry_biomass'], df_dm['fresh_biomass'], 2)
        if plot:
            ax2.set_title('Fresh biomass to dry biomass')
            ax2.plot(df_dm['fresh_biomass'], df_dm['dry_biomass'], 'o', label='data')
            ax2.plot(np.linspace(0, 300, 100), self.fresh_biomass_to_dry_biomass(np.linspace(0, 300, 100)), label='chebyshev')
            ax2.legend(loc='best')
            ax2.set_xlabel('Fresh biomass (g/plant)')
            ax2.set_ylabel('Dry biomass (g/plant)')
            plt.savefig("calibration_LAI_DM.png")
            

        # Calibrate growing curves
        #sort by radiation and group by radiation
        df_biomass_grouped = df_biomass_curves.sort_values(by=['radiation']).groupby('radiation')
        # Fit splines to each group
        self.growing_curves_rad = {}
        for radiation, group in df_biomass_grouped:
            # Fit spline
            df = group.sort_values(by=['day'])
            params = curve_fit(sigmoid, df['day'], df['biomass_dry'], p0=[0, 1, 20, 40], maxfev=10000)
            print(f"Radiation : {radiation} | Params : {params[0]}")
            self.growing_curves_rad[radiation] = np.array(params[0])
        # Plot
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.scatter(df_biomass_curves['day'], df_biomass_curves['biomass_dry'], label='data')
            ax1.set_title('Growing curves')
            for i in self.growing_curves_rad:
                ax1.plot(np.linspace(11, 50, 100), sigmoid(np.linspace(11, 50, 100), *self.growing_curves_rad[i]), label='Light = ' + str(i))
            ax1.legend(loc='best')
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('Biomass (g/plant)')
            # Plot derivatives
            ax2.set_title('Growing curves derivatives')
            for i in self.growing_curves_rad:
                ax2.plot(np.linspace(11, 50, 100), sigmoid_derivative(0, np.linspace(11, 50, 100), *self.growing_curves_rad[i]), label='Light = ' + str(i))
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Biomass derivative (g/plant/day)')
            plt.savefig("growing_curves_light.png")
            

        # Get growing curves for each water times
        self.best_params = np.array(self.growing_curves_rad[22])
        time_curves = {}
        days = df_water_times['days'].mean()
        # Group water times by scenario
        for group, df in df_water_times.groupby('off'):
            # Get growing curve
            params = np.array(self.best_params)
            params[0] = df['biomass_dry']
            # # print(f"Scenario {group} params: {params}")
            # Store growing params
            time_curves[group] = np.array(params)
        # Store growing curves
        self.growing_curves_time = time_curves
        # Plot
        if plot:
            plt.figure()
            plt.title('Growing curves for each water times')
            for group in time_curves:
                plt.plot(np.linspace(11, days, 100), sigmoid(np.linspace(11, days, 100), *time_curves[group]), label='Water times = ' + str(group))
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
            plt.savefig("growing_curves_water_times.png")

        # Get growing curves for each water flow
        bet_scenario_params = np.array(self.best_params)
        flow_curves = {}
        # Group water flow by scenario
        days = df_water_flow['days'].mean()
        for group, df in df_water_flow.groupby('rate'):
            # Get growing curve
            params = bet_scenario_params
            params[0] = self.fresh_biomass_to_dry_biomass(df['biomass_fresh'])
            # # print(f"Scenario {group} params: {params}")
            # Store growing params
            flow_curves[group] = np.array(params)
        # Store growing curves
        self.growing_curves_flow = flow_curves
        # Plot
        if plot:
            plt.figure()
            plt.title('Growing curves for each water flow')
            for group in flow_curves:
                plt.plot(np.linspace(11, days, 100), sigmoid(np.linspace(11, days, 100), *flow_curves[group]), label='Water flow = ' + str(group))
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
            plt.savefig("growing_curves_water_flow.png")
            

        # Fit derivatives estimations
        # Get first param for each radiation
        params = [i[0] for i in self.growing_curves_rad.values()]
        loss = [(i/np.max(params)) for i in params]
        light = list(self.growing_curves_rad.keys())
        # Fit polynomial
        self.light_loss = np.polynomial.Chebyshev.fit(light, loss, 2)
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

            ax1.set_title('Derivative to light')
            ax1.plot(light, loss, 'o', label='data')
            ax1.plot(np.linspace(0, 25, 100), self.light_loss(np.linspace(0, 25, 100)), label='chebyshev')
            ax1.legend(loc='best')
            ax1.set_xlabel('Light mol/m2/day')
            ax1.set_ylabel('a-parameter reduction')

        # Get first param for each water times
        params = [i[0] for i in self.growing_curves_time.values()]
        loss = [(i/np.max(params)) for i in params]
        water_times = list(self.growing_curves_time.keys())
        # Fit polynomial
        self.water_times_loss = curve_fit(plateau, water_times, loss, maxfev=10000)[0]
        if plot:
            ax2.set_title('Derivative to water times')
            ax2.plot(water_times, loss, 'o', label='data')
            ax2.plot(np.linspace(30, 60, 100), plateau(np.linspace(30, 60, 100), *self.water_times_loss), label='plateau')
            ax2.legend(loc='best')
            ax2.set_xlabel('Water times')
            ax2.set_ylabel('a-parameter reduction')

        #Get first param for each water flow
        params = [i[0] for i in self.growing_curves_flow.values()]
        loss = [(i/np.max(params)) for i in params]
        water_flow = list(self.growing_curves_flow.keys())
        # Fit linear
        self.water_flow_loss = np.polyfit(water_flow, loss, 1)
        if plot:
            ax3.set_title('Derivative to water flow')
            ax3.plot(water_flow, loss, 'o', label='data')
            ax3.plot(np.linspace(0.25, 2, 100), self.water_flow_loss[0]*np.linspace(0.25, 2, 100) + self.water_flow_loss[1], label='linear')
            ax3.legend(loc='best')
            ax3.set_xlabel('Water flow')
            ax3.set_ylabel('a-parameter reduction')
            plt.tight_layout(pad=4)
            plt.savefig("calibration_parameters_loss_influence.png")
            plt.show()

        # Get water use efficiency
        wues = []
        lights = []
        a_params = []
        for group, df in df_wue.groupby('light'):
            wue = (df['biomass_dry'] / df['accumulated_water']).mean()
            wues.append(wue)
            lights.append(group)
            light_factor = self.light_loss(group)
            a = light_factor
            a_params.append(a)
        # Fit chebyshev
        self.wue_from_light = np.polynomial.Chebyshev.fit(lights, wues, 2)
        self.wue_from_rel_rate = np.polynomial.Chebyshev.fit(a_params, wues, 2)

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plt.tight_layout(pad=5.0)
            ax[0].set_title('WUE from light')
            ax[0].plot(lights, wues, 'o', label='data')
            ax[0].plot(np.linspace(0, 25, 100), self.wue_from_light(np.linspace(0, 25, 100)), label='Fitted quadratic')
            ax[0].legend(loc='best')
            ax[0].set_xlabel('Light mol/m2/day')
            ax[0].set_ylabel('WUE (g biomass/ml water)')
            ax[1].set_title('WUE from a-parameter')
            ax[1].plot(a_params, wues, 'o', label='data')
            ax[1].plot(np.linspace(0, 1, 100), self.wue_from_rel_rate(np.linspace(0,1, 100)), label='Fitted quadratic')
            ax[1].legend(loc='best')
            ax[1].set_xlabel('a-parameter')
            ax[1].set_ylabel('WUE (g biomass/ml water)')
            plt.savefig("calibration_parameters_wue.png")
            plt.show()




    # Simulate growing season base on light, water times and water flow using odeint and plot results
    # Assuming constant scenario
    def simulate_growing_season(self, light, water_times, water_flow, plot=True, season_length=50):
        # Get a-param reduction
        light_factor = self.light_loss(light)
        water_times_factor = plateau(water_times, *self.water_times_loss)
        water_flow_factor = self.water_flow_loss[0]*water_flow + self.water_flow_loss[1]
        a = float(self.best_params[0])*np.min([light_factor, water_times_factor, water_flow_factor])
        # Get growing curve
        params = np.array(self.best_params)
        params[0] = a
        print(f"Biomass sigmoid params: {params}")
        # Simulate
        days = np.linspace(11, season_length, season_length-11)
        rates = sigmoid_derivative(0,days, *params)
        biomass = odeint(sigmoid_derivative, 0, days, args=(*params,))
        # Plot
        if plot:
            plt.figure()
            plt.title(f"Simulation with light = {light}, water times = {water_times}s and water flow = {water_flow}L/h")
            plt.plot(days, biomass, label='Biomass')
            plt.plot(days, rates, label='Rate')
            plt.legend(loc='best')
            plt.xlabel('Time (days)')
            plt.ylabel('Biomass (g/plant)')
        plt.show()
        return days, rates, biomass


    # Estimate growing rate on certain day based on light, water times and water flow and final biomass
    def estimate_growing_rate(self, day, biomass, light, water_times, water_flow, final_biomass=None):
        if final_biomass is None:
            final_biomass = self.best_params[0]
        # Get a-param reduction
        light_factor = self.light_loss(light)
        water_times_factor = plateau(water_times, *self.water_times_loss)
        water_flow_factor = self.water_flow_loss[0]*water_flow + self.water_flow_loss[1]
        a = final_biomass*np.min([light_factor, water_times_factor, water_flow_factor])
        # Get growing curve
        params = np.array(self.best_params)
        params[0] = a
        # Get growing rate
        rate_day = sigmoid_derivative(biomass,day, *params)
        day_biomass = sigmoid_x(biomass, *params)
        rate_biomass = sigmoid_derivative(biomass,day_biomass, *params)
        if rate_day > rate_biomass:
            return rate_biomass
        else:
            return rate_day


    # Plot the different growing scenaios in a 3D plot
    def plot_growing_scenarios(self, season_length=50):
        # Get all combinations
        light = np.linspace(0, 25, 2)
        water_times = np.linspace(30, 60, 2)
        water_flow = np.linspace(0.5, 2, 2)
        combinations = list(itertools.product(light, water_times, water_flow))
        ax = plt.axes(projection='3d')
        lights_, water_times_, water_flows_ = [], [], []
        b = []
        for i, combination in enumerate(combinations):
            print(f"Simulating scenario {i+1}/{len(combinations)}")
            days, rates, biomass = self.simulate_growing_season(*combination, plot=False, season_length=season_length)
            lights_.append(combination[0])
            water_times_.append(combination[1])
            water_flows_.append(combination[2])
            b.append(biomass[-1][0])
            print(combination[0], combination[1], combination[2], biomass[-1][0])
        ax.scatter3D(lights_, water_times_, water_flows_, c=b, cmap='viridis')
        ax.set_xlabel('Light')
        ax.set_ylabel('Water times')
        ax.set_zlabel('Water flow')
        plt.show()


