# AGRON893-Aeroponics
 
Project for AGRON893 - Crop modeling and simulation

## Project Description

First, the AeroponicsModel.py file is run to fit literature data to calibrate the model.
The model can predict growing rate based on the following parameters:

- Light intensity
- Water flow rate
- Irrigation frequency

The model only consider the effect of the most limiting factor on the growth rate. Looking to equally optimize all three factors.
The predicted growth rate can be limited by the following parameters:
- Age
- Accumulated biomass

If the model optimization reduces costs by reducing factors more than the expected growth dynamics, the accumulated biomass will start limiting the potential growing rate.

Being possible to optimize the parameters to maximize the biomass production by minimizing the cost of the system in a daily basis.
The optimization is done by the OptimizeAeroponics.py file.

SLSQP seems to be the best optimizer for this problem. The other optimizers did not converge or had a lot of iterations to find the best solution.

## How to run

To calibrate the model, just type in the terminal:

```
python3 main.py
```

To run the optimization, just type in the terminal:

```
python3 optimization_costs.py
```
