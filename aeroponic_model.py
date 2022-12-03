import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
import warnings
# Read df
df_water_times = pd.read_excel('Data.xlsx', sheet_name='water_times', skiprows=[1])
df_water_flow = pd.read_excel('Data.xlsx', sheet_name='water_flow', skiprows=[1])
df_radiation = pd.read_excel('Data.xlsx', sheet_name='radiation', skiprows=[1])

print(df_water_times)
print(df_water_flow)
class AeroponicModel:
    def __init__(self):
        self.water_flow_losses = None
        self.water_times_losses = None
        self.radiation_losses = None

    def calibrate(self, plot = False):
        global Water_flow_losses, Water_times_losses
        # Calibrate biomass loss due to water flow
        df_water_flow['loss'] = df_water_flow['biomass_fresh']/df_water_flow['biomass_fresh'].max() - 1
        # Fit spline to loss and rate
        loss = df_water_flow['loss']
        rate = df_water_flow['rate']
        # Fit spline
        cs_flow = CubicSpline(rate, loss)
        # Plot
        if plot:
            plt.title('Biomass loss due to water flow')
            plt.plot(rate, loss, 'o', label='data')
            plt.plot(np.linspace(0, 2, 100), cs_flow(np.linspace(0, 2, 100)), label='spline')
            plt.xlabel('Water flow rate (L/h)')
            plt.ylabel('Biomass loss (fraction)')
            plt.legend(loc='best')
            plt.show()
        self.water_flow_losses = cs_flow

        # Calibrate biomass loss due to water times
        df_water_times['loss'] = df_water_times['biomass_dry']/df_water_times['biomass_dry'].max() - 1
        # Fit spline to loss and rate
        loss = df_water_times['loss']
        off_time = df_water_times['off']
        cs_times = CubicSpline(off_time, loss)
        # Plot
        if plot:
            plt.title('Water timing losses')
            plt.plot(off_time, loss, 'o', label='data')
            plt.plot(np.linspace(0, 60, 100), cs_times(np.linspace(0, 60, 100)), label='spline')
            plt.legend(loc='best')
            plt.xlabel('Watering interval (min)')
            plt.ylabel('Biomass loss (fraction)')
            plt.show()
        self.water_times_losses = cs_times

        # Calibrate biomass loss due to radiation
        df_radiation['loss'] = df_radiation['biomass_fresh']/df_radiation['biomass_fresh'].max() - 1
        # Fit spline to loss and rate
        loss = df_radiation['loss']
        radiation = df_radiation['radiation']
        cs_radiation = CubicSpline(radiation, loss)
        # Plot
        if plot:
            plt.title('Radiation losses')
            plt.plot(radiation, loss, 'o', label='data')
            plt.plot(np.linspace(0, 400, 100), cs_radiation(np.linspace(0, 400, 100)), label='spline')
            plt.legend(loc='best')
            plt.xlabel('Light flux (umol/m2/s)')
            plt.ylabel('Biomass loss (fraction)')
            plt.show()
        self.radiation_losses = cs_radiation

    def get_loss(self, water_flow, water_times, radiation):
        if water_flow > df_water_flow['rate'].max():
            warnings.warn(f'Water flow rate is outside the scope of the calibration data. Max value is {df_water_flow["rate"].max()}')
        if water_times > df_water_times['off'].max():
            warnings.warn(f'Watering interval is outside the scope of the calibration data. Max value is {df_water_times["off"].max()}')
        if radiation > df_radiation['radiation'].max():
            warnings.warn(f'Radiation is outside the scope of the calibration data. Max value is {df_radiation["radiation"].max()}')
        if water_flow < df_water_flow['rate'].min():
            warnings.warn(f'Water flow rate is outside the scope of the calibration data. Min value is {df_water_flow["rate"].min()}')
        if water_times < df_water_times['off'].min():
            warnings.warn(f'Watering interval is outside the scope of the calibration data. Min value is {df_water_times["off"].min()}')
        if radiation < df_radiation['radiation'].min():
            warnings.warn(f'Radiation is outside the scope of the calibration data. Min value is {df_radiation["radiation"].min()}')


        # Calculate biomass loss due to water flow
        loss_flow = self.water_flow_losses(water_flow)
        # Calculate biomass loss due to
        loss_times = self.water_times_losses(water_times)
        # Calculate biomass loss due to
        loss_radiation = self.radiation_losses(radiation)
        # Calculate total biomass loss
        loss_total = loss_flow + loss_times + loss_radiation
        return loss_total





