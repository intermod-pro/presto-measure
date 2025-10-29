# -*- coding: utf-8 -*-
from datetime import datetime
import os
import shutil
import sys

import numpy as np

from qubit_settings import Settings

_NR_AVG = 1_000

_AEC_RED = "\x1b[1;31m"
_AEC_GREEN = "\x1b[1;32m"
_AEC_YELLOW = "\x1b[1;33m"
_AEC_RESET = "\x1b[0m"


def main(settings: Settings):
    settings._drag_use = False
    sweep(settings)
    two_tone_pulsed(settings)
    rabi_amp(settings, num_pulses=1)
    ramsey_single(settings, coarse=True)
    ramsey_single(settings)
    rabi_amp(settings, num_pulses=10)
    two_tone_ef(settings)
    settings._drag_use = True
    rabi_amp(settings, num_pulses=10)
    ramsey_single(settings)
    rabi_amp(settings, num_pulses=10)
    excited_sweep(settings)


def rprint(body: str):
    print(f"{_AEC_GREEN}RECAL:{_AEC_RESET} {body:s}", file=sys.stderr)


def sweep(settings: Settings):
    from sweep import Sweep

    exp = Sweep(
        freq_center=settings.readout_freq,
        freq_span=10 * 1e6,
        df=5e3,
        num_averages=max(_NR_AVG // 10, 1),
        amp=settings.readout_amp,
        output_port=settings.readout_port,
        input_port=settings.sample_port,
        dither=True,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    f_min = round(exp.analyze(batch=True), -3)
    rprint(f"Readout frequency: {settings.readout_freq} --> {f_min}")
    settings.readout_freq = f_min


def two_tone_pulsed(settings: Settings):
    from two_tone_pulsed import TwoTonePulsed

    exp = TwoTonePulsed(
        readout_freq=settings.readout_freq,
        control_freq_center=settings.control_freq,
        control_freq_span=100 * 1e6,
        control_freq_nr=256,
        readout_amp=settings.readout_amp,
        control_amp=settings.control_amp_180 / 10,
        readout_duration=settings.readout_duration,
        control_duration=settings.control_duration * 10,
        sample_duration=settings.sample_duration,
        readout_port=settings.readout_port,
        control_port=settings.control_port,
        sample_port=settings.sample_port,
        wait_delay=settings.wait_delay,
        readout_sample_delay=settings.readout_sample_delay,
        num_averages=_NR_AVG,
        jpa_params=settings.jpa,
        drag=settings.drag,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    f_q = round(exp.analyze(batch=True), -3)
    rprint(f"Control frequency: {settings.control_freq} --> {f_q}")
    settings.control_freq = f_q


def rabi_amp(settings: Settings, num_pulses: int):
    from rabi_amp import RabiAmp

    exp = RabiAmp(
        readout_freq=settings.readout_freq,
        control_freq=settings.control_freq,
        readout_amp=settings.readout_amp,
        control_amp_arr=np.linspace(0.0, 0.707, 256),
        readout_duration=settings.readout_duration,
        control_duration=settings.control_duration,
        sample_duration=settings.sample_duration,
        readout_port=settings.readout_port,
        control_port=settings.control_port,
        sample_port=settings.sample_port,
        wait_delay=settings.wait_delay,
        readout_sample_delay=settings.readout_sample_delay,
        num_averages=_NR_AVG,
        num_pulses=num_pulses,
        jpa_params=settings.jpa,
        drag=settings.drag,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    control_amp_180, control_amp_90 = exp.analyze(batch=True)
    control_amp_180 = round(control_amp_180, 5)
    control_amp_90 = round(control_amp_90, 5)
    rprint(f"Control amplitude 180: {settings.control_amp_180} --> {control_amp_180}")
    rprint(f"Control amplitude 90: {settings.control_amp_90} --> {control_amp_90}")
    settings.control_amp_180 = control_amp_180
    settings.control_amp_90 = control_amp_90


def ramsey_single(settings: Settings, coarse: bool = False):
    from ramsey_single import RamseySingle

    if coarse:
        detuning = 11e6  ## Hz
        step = 2e-9
    else:
        detuning = 1.1e6  ## Hz
        step = 20e-9

    exp = RamseySingle(
        readout_freq=settings.readout_freq,
        control_freq=settings.control_freq + detuning,
        readout_amp=settings.readout_amp,
        control_amp=settings.control_amp_90,
        readout_duration=settings.readout_duration,
        control_duration=settings.control_duration,
        sample_duration=settings.sample_duration,
        delay_arr=step * np.arange(256),
        readout_port=settings.readout_port,
        control_port=settings.control_port,
        sample_port=settings.sample_port,
        wait_delay=settings.wait_delay,
        readout_sample_delay=settings.readout_sample_delay,
        num_averages=_NR_AVG,
        jpa_params=settings.jpa,
        drag=settings.drag,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    measured = exp.analyze(batch=True)
    control_freq = round(settings.control_freq + detuning - measured, -3)
    rprint(f"Control frequency: {settings.control_freq} --> {control_freq}")
    settings.control_freq = control_freq


def two_tone_ef(settings: Settings):
    from two_tone_ef import TwoToneEF

    exp = TwoToneEF(
        readout_freq=settings.readout_freq,
        ge_freq=settings.control_freq,
        alpha_center=-210e6,
        alpha_span=100e6,
        alpha_nr=256,
        readout_amp=settings.readout_amp,
        ge_amp=settings.control_amp_180,
        ef_amp=settings.control_amp_180 / 5,
        readout_duration=settings.readout_duration,
        ge_duration=settings.control_duration,
        ef_duration=settings.control_duration * 5,
        sample_duration=settings.sample_duration,
        readout_port=settings.readout_port,
        control_port=settings.control_port,
        sample_port=settings.sample_port,
        wait_delay=settings.wait_delay,
        readout_sample_delay=settings.readout_sample_delay,
        num_averages=_NR_AVG,
        jpa_params=settings.jpa,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    anharmonicity = round(exp.analyze(batch=True), -5)
    rprint(f"Anharmonicity: {settings.anharmonicity} --> {anharmonicity}")
    settings.anharmonicity = anharmonicity


def excited_sweep(settings: Settings):
    from excited_sweep import ExcitedSweep

    exp = ExcitedSweep(
        readout_freq_center=settings.readout_freq,
        readout_freq_span=5 * 1e6,
        readout_freq_nr=128,
        control_freq=settings.control_freq,
        readout_amp=settings.readout_amp,
        control_amp=settings.control_amp_180,
        readout_duration=settings.readout_duration,
        control_duration=settings.control_duration,
        sample_duration=settings.sample_duration,
        readout_port=settings.readout_port,
        control_port=settings.control_port,
        sample_port=settings.sample_port,
        wait_delay=settings.wait_delay,
        readout_sample_delay=settings.readout_sample_delay,
        num_averages=_NR_AVG,
        drag=settings.drag,
    )
    exp.DAC_CURRENT = settings.dac_current
    exp.run(settings.address, settings.port)

    f_opt = round(exp.analyze(batch=True), -3)
    rprint(f"Readout frequency: {settings.readout_freq} --> {f_opt}")
    settings.readout_freq = f_opt


def compare(old: Settings, new: Settings):
    quantities = [
        ("readout_freq", "Readout frequency"),
        ("control_freq", "Control frequency"),
        ("control_amp_180", "Control amplitude 180"),
        ("control_amp_90", "Control amplitude 90"),
        ("anharmonicity", "Anharmonicity"),
    ]

    col = min(80, shutil.get_terminal_size().columns)

    def centered(msg: str):
        print(f"{_AEC_GREEN}{msg:=^{col}}{_AEC_RESET}", file=sys.stderr)

    centered("  Recalibration summary  ")

    for quantity, name in quantities:
        q_old = getattr(old, quantity)
        q_new = getattr(new, quantity)
        eps = q_new / q_old - 1.0
        if abs(eps) < 0.01:
            color = _AEC_RESET
        elif abs(eps) < 0.1:
            color = _AEC_YELLOW
        else:
            color = _AEC_RED
        print(f"  {name}: {q_old} --> {q_new}: {color}{eps:+.2%}{_AEC_RESET}", file=sys.stderr)

    centered("")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        _which_qubit = int(sys.argv[1])
    elif len(sys.argv) == 3:
        _which_qubit = int(sys.argv[1])
        _NR_AVG = int(sys.argv[2])
    else:
        raise RuntimeError("Wrong number of arguments! Usage: `python autocal.py WHICH_QUBIT`")

    _settings = Settings.from_latest(_which_qubit)
    _old_filename = _settings._path
    rprint(f"Loaded qubit parameters from {_old_filename}")

    main(_settings)

    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _new_filename = os.path.join("data", f"qubit{_which_qubit}_{_timestamp}.toml")
    _settings.save(_new_filename)
    rprint(f"Saved qubit parameters to {_new_filename}")

    compare(Settings.load(_old_filename), Settings.load(_new_filename))
