# presto-measure
Collection of measurement and analysis scripts for the Presto platform.

The target use case is the characterization of
a circuit-QED device with two superconducting transmon qubits coupled by a tunable coupler. Each qubit has a readout
resonator and a dedicated control line. The resonators are coupled to a single readout line in a notch configuration.
The coupler accepts a DC bias and an AC drive. The readout line is preamplified by a Josephson parametric amplifier
which is biased and pumped. See [Bengtsson 2020](https://doi.org/10.1103/PhysRevApplied.14.034010) for a very similar
setup.

## Usage

In general, the use is like the following:
```python
# import the experiment class
from some_module import SomeExperiment

# initialize the experiment
experiment = SomeExperiment(
    # some parameters required
    # ...
)

# run the experiment: it might take time, and it will save at the end
presto_address = "192.168.42.50"  # or whatever else
save_filename = experiment.run(presto_address)

# analyze the data and get nice plots
experiment.analyze()

# you can also load older data
old_experiment = SomeExperiment.load("/path/to/saved/data.h5")
old_experiment.analyze()
```

## Summary of modules (scripts)

### `sweep`
Simple single-frequency sweep on the resonator using **lockin** mode. If
[resonator_tools](https://github.com/sebastianprobst/resonator_tools) is available, perform fit to extract resonance
frequency and internal and external quality factors.

### `sweep_power`
2D sweep of drive amplitude and frequency on the resonator using **lockin** mode. If
[resonator_tools](https://github.com/sebastianprobst/resonator_tools) is available, perform fit to extract resonance
frequency and internal and external quality factors for a slice of the data at a given drive amplitude.

### `two_tone_power`
Two tone spectroscopy using **lockin** mode. 2D sweep of qubit drive amplitude and frequency, with fixed resonator
drive frequency and amplitude.

## Older stuff
Scripts in `bak` folder may or may not work, with higher chances on the may-not case.
