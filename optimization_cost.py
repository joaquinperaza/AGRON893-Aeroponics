from scipy.optimize import Bounds
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

class Optimization:

    def __init__(self, aeroponic_model):
        self.aeroponic_model = aeroponic_model
        self.aeroponic_model.calibrate(plot=False)
        self.light_bounds = [(5, 22)]
        self.water_flow_bounds = [(0.5, 2.6)]
        self.water_times_bounds = [(30, 60)]
        self.light_cost_per_day = 0.5
        self.water_cost_per_litre = 0.1
        self.motor_kwh = 0.1
        self.pwr_cost_per_kwh = 0.1
        self.price_per_kg = 1.0


    def set_motor_kwh(self, motor_kwh):
        self.motor_kwh = motor_kwh

    def set_light_cost_per_day(self, light_cost_per_day):
        self.light_cost_per_day = light_cost_per_day

    def set_water_cost_per_litre(self, water_cost_per_litre):
        self.water_cost_per_litre = water_cost_per_litre

    def set_pwr_cost_per_kwh(self, pwr_cost_per_kwh):
        self.pwr_cost_per_kwh = pwr_cost_per_kwh

    def set_price_per_kg(self, price_per_kg):
        self.price_per_kg = price_per_kg

    def set_n_plants(self, n_plants):
        self.n_plants=n_plants

    def simulate(self, light, water_times, water_flow):
        biomass = 0
        sz = light.size
        for i in range(sz):
            biomass += self.aeroponic_model.estimate_growing_rate(day=i, biomass=biomass, light=light[i], water_times=water_times[i], water_flow=water_flow[i])
        return biomass

    def calculate_light_cost(self, light):
        return np.sum(light * self.light_cost_per_day)

    def calculate_water_cost(self, water_flow):
        return np.sum(water_flow * 24 * self.water_cost_per_litre)

    def calculate_motor_cost(self, water_times):
        motor_working_time = np.sum(5/(water_times+5)) * 24
        return np.sum(motor_working_time * self.motor_kwh * self.pwr_cost_per_kwh)

    def cost(self, x):
        #split one d array into 3 arrays of equal size
        sz = x.size//3
        light = x[:sz]
        water_times = x[sz:2*sz]
        water_flow = x[2*sz:]
        yield_kg = self.simulate(light, water_times, water_flow)
        cost = self.calculate_light_cost(light) +\
               self.calculate_water_cost(water_flow) +\
               self.calculate_motor_cost(water_times) - self.aeroponic_model.dry_biomass_to_fresh_biomass(yield_kg)/1000 * self.price_per_kg * self.n_plants
        return cost

    def optimize(self, days, light_efficiency=None, water_cost=None, motor_kwh=None, price_per_kg=None, pwr_cost_per_kwh=None, n_plants=10):
        if pwr_cost_per_kwh is not None and light_efficiency is not None:
            lamp_consumption = 200 #watts
            radius = 2.5 #meters (lamp_to_plants radius)
            light_area = 4 * np.pi * radius**2 #meters^2
            watts_m2_day = lamp_consumption * light_efficiency * 24 / light_area
            #Convert watts to umol/m2/s
            umols_s = watts_m2_day * 4.6 #umol/m2/s
            #Convert umol/m2/s to mol/m2/day
            mol_day = umols_s * 24 * 60 * 60 / 1000000
            power_consumption = lamp_consumption * 24 * 0.001 #kWh
            power_cost = power_consumption * pwr_cost_per_kwh
            power_cost_per_mol = power_cost / mol_day
            self.set_light_cost_per_day(power_cost_per_mol)
        if water_cost is not None:
            self.set_water_cost_per_litre(water_cost)
        if motor_kwh is not None:
            self.set_motor_kwh(motor_kwh)
        if price_per_kg is not None:
            self.set_price_per_kg(price_per_kg)
        if pwr_cost_per_kwh is not None:
            self.set_pwr_cost_per_kwh(pwr_cost_per_kwh)
        if n_plants is not None:
            self.set_n_plants(n_plants)

        x0 = np.array([22]*days + [30]*days + [1]*days)
        res = minimize(self.cost, x0, method='SLSQP', bounds=[*self.light_bounds*days, *self.water_times_bounds*days, *self.water_flow_bounds*days], options={'maxiter': 5000})
        return res


def output_results(res, opt, scenario):
    print("Optimal solution found")
    print(f"We made {-res.fun} dollars")
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"Using {no_plants} plants, light cost {opt.light_cost_per_day:.2f} usd/mol/day \n"
                 f", water cost {opt.water_cost_per_litre:.4f} usd/L, motor {opt.motor_kwh:.2f} kwh, \n"
                 f"power cost {opt.pwr_cost_per_kwh:.2f} usd/kwh, price per kg {opt.price_per_kg} usd/kg")
    sz = res.x.size // 3
    light = res.x[:sz]
    water_times = res.x[sz:2 * sz]
    water_flow = res.x[2 * sz:]
    plt.figtext(0.5, 0.02, f"Optimized profit {-res.fun:.2f} dollars", wrap=True, horizontalalignment='center', fontsize=12)
    ax[0].plot(light, c='tomato')
    ax[0].set_title("Light")
    ax[0].set_ylabel("Light intensity (mol/m2/day)")
    ax[0].set_ylim([0, 23])
    ax[1].plot(water_times)
    ax[1].set_title("Water times")
    ax[1].set_ylabel("Irrrigation interval (min)")
    ax[1].set_ylim([0, 65])
    ax[2].plot(water_flow)
    ax[2].set_title("Water flow")
    ax[2].set_ylabel("Water flow (L/h)")
    ax[2].set_ylim([0, 1.5])
    ax[3].set_title("Biomass")
    ax[3].set_ylabel("Biomass (g/plant)")
    biomass = []
    max_biomass = []
    for i in range(sz):
        b = 0
        m_b = 0
        if len(biomass) > 0:
            b = biomass[-1]
            m_b = max_biomass[-1]
        biomass.append(
            b + opt.aeroponic_model.estimate_growing_rate(day=i, biomass=b, light=light[i], water_times=water_times[i],
                                                          water_flow=water_flow[i]))
        max_biomass.append(m_b + opt.aeroponic_model.estimate_growing_rate(day=i, biomass=m_b, light=24, water_times=30,
                                                                           water_flow=.5))
    fresh_biomass = []
    for b in biomass:
        fresh_biomass.append(opt.aeroponic_model.dry_biomass_to_fresh_biomass(b))
    max_fresh_biomass = []
    for b in max_biomass:
        max_fresh_biomass.append(opt.aeroponic_model.dry_biomass_to_fresh_biomass(b))
    ax[3].plot(fresh_biomass, label="Simulated", color="green")
    ax[3].plot(max_fresh_biomass, label="Max", color="blue", linestyle="--")
    ax[3].legend()
    ax[3].legend()
    print(biomass)
    plt.savefig(f"results_{scenario}.png")
    plt.show()

    print(f"Light cost: {opt.calculate_light_cost(light)}")
    print(f"Water cost: {opt.calculate_water_cost(water_flow)}")
    print(f"Motor cost: {opt.calculate_motor_cost(water_times)}")
    print(f"Yield: {opt.simulate(light, water_times, water_flow)}")
    print(f"Light : {light}")
    print(f"Water times : {water_times}")
    print(f"Water flow : {water_flow}")
    print("Total light : ", np.sum(light))
    print(f"Light use efficiency: {b / np.sum(light):.2f}")
    wue = opt.aeroponic_model.wue_from_light(light).mean()
    print(f"Water use efficiency: {wue:.2f}")

if __name__ == "__main__":
    from aeroponic_model import AeroponicModel

    no_plants = 80

    #Scenario 1:
    #Kansas prices of light, water, electricity, and product lettuce

    # Scenario 2:
    # New York prices of light, water, electricity, and product lettuce

    # Scenario 3:
    # Kansas prices of light, water, electricity, and product lettuce
    # Incandescent light bulbs efficiency of (30%)

    scenarios = [
    # (light_efficiency, water_l, water_kwh, lettuce_price)
        ("Kansas LED", 0.5, 0.001, 0.11, 3),
        ("Kansas Incandescent", 0.3, 0.001, 0.11, 3),
        ("California LED", 0.5, 0.001, 0.28, 12),
    ]

    for scenario, light_eff, water_cost, cost_kwh, price in scenarios:
        model = AeroponicModel()
        opt = Optimization(model)
        res = opt.optimize(40, light_efficiency=light_eff, water_cost=water_cost, motor_kwh=.3, price_per_kg=price,
                           pwr_cost_per_kwh=cost_kwh, n_plants=no_plants)
        if (res.success):
            output_results(res, opt, scenario)
        else:
            warnings.warn("The optimizer failed to find a solution after 5000 iterations")





