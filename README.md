# presto-measure

Collection of measurement and analysis scripts for the Presto platform.

The target use case is the characterization of a circuit-QED device with two superconducting transmon qubits coupled by
a tunable coupler. Each qubit has a readout resonator and a dedicated control line. The resonators are coupled to a
single readout line in a notch configuration. The coupler accepts a DC bias and an AC drive. The readout line is
preamplified by a Josephson parametric amplifier (JPA) which is biased and pumped. See [Bengtsson
2020](https://doi.org/10.1103/PhysRevApplied.14.034010) for a very similar setup.



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

### `two_tone_pulsed`
Two tone spectroscopy using **pulsed** mode. 1D sweep of qubit drive frequency, with fixed amplitude and fixed
resonator drive frequency and amplitude.

### `rabi_amp`
Rabi oscillations by driving the qubit with a pulse of variable amplitude and constant duration. Multiple pulses can be
applied to increase accuracy of fitted results.

### `ramsey_single`
Measure Ramsey oscillations by changing the delay between two π/2 pulses. Fit detuning of control drive frequency from
qubit, and T2*. The control pulse has a sin^2 envelope, while the readout pulse is square.

### `ramsey_chevron`
Measure a Ramsey chevron pattern by changing the delay between two π/2 pulses and their frequency.

### `excited_sweep`
Pulsed frequency sweep on the resonator with and without a π/2 control pulse: plot (and fit) resonance curves for
ground and excited states.

### `t1`
Measure the energy-relaxation time T1.

### `ramsey_echo`
Measure the decoherence time T2 with a Ramsey echo experiment.

### `ac_stark_shift`
Measure AC-Stark shift and measurement-induced dephasing by performing a Ramsey measurement while driving the resonator
with variable power.



## JPA calibration

### `jpa_sweep_bias`
2D sweep of DC bias (JPA pump line) and frequency of probe (qubits readout line) to find the modulation curve of the
JPA.

### `jpa_sweep_power_bias`
3D sweep of AC pump power and DC bias (JPA pump line) and frequency of probe (qubits readout line) to find operating
point of JPA, i.e. where there is gain. It can be run with a single value of pump power, in which case it's a 2D sweep.



## Older stuff
Scripts in `bak` folder may or may not work, with higher chances on the may-not case.
