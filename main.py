from aeroponic_model import AeroponicModel


def test(model):
    # Test best scenario
    model.simulate_growing_season(light=22, water_times=30, water_flow=0.5)
    # Test worst scenario
    model.simulate_growing_season(light=8, water_times=60, water_flow=1.5)
    # Tetst average scenario
    model.simulate_growing_season(light=15, water_times=45, water_flow=1)


if __name__ == '__main__':
    model = AeroponicModel()
    model.calibrate(plot=False)
    model.plot_growing_scenarios()
    #model.test_rate()
    #test(model)
