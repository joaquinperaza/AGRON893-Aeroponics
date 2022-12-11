import numpy as np

from aeroponic_model import AeroponicModel
import pandas as pd
import matplotlib.pyplot as plt


def validate(model):
    rates = []
    print(" ######## Validation 2 ######## ")
    print("Model validation used data reported by ")


    df = pd.read_excel('Data.xlsx', sheet_name='radiation', skiprows=[1])
    print(df)
    RMSE = 0
    for rad, df in df.groupby('radiation'):
        biomass = 0.1
        r = list(np.zeros(11))
        for day in range(30-11):
            umol_m2_s = df['radiation'].mean()
            mol_m2_day = umol_m2_s * 86400 / 1000000
            water_times = 30
            water_flow = 0.5
            rate = model.estimate_growing_rate(day=day+11, biomass=biomass, light=mol_m2_day,
                                                   water_times=water_times, water_flow=water_flow)
            r.append(rate)
            biomass += rate
        print(f"Experiment Radiation {mol_m2_day} mol/m2/day")
        print(f"Model biomass {model.dry_biomass_to_fresh_biomass(biomass)} g/plant")
        print(f"Experiment biomass {df['biomass_fresh'].mean()} g/plant")
        print(f"Error {model.dry_biomass_to_fresh_biomass(biomass) - df['biomass_fresh'].mean()} g/plant")
        print()
        rates.append(r)
        RMSE += (model.dry_biomass_to_fresh_biomass(biomass) - df['biomass_fresh'].mean())**2
    RMSE = np.sqrt(RMSE/len(rates))
    print(f"RMSE {RMSE} g/plant")

    return rates


if __name__ == '__main__':
    model = AeroponicModel()
    model.calibrate(plot=True)
    r1=validate(model)





