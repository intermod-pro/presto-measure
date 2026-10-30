# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import tomlkit
from tomlkit.exceptions import NonExistentKey

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode


@dataclass
class Settings:
    address: str
    port: int
    dc_params: dict

    readout_port: int
    readout_duration: float
    readout_amp: float
    readout_freq: float
    readout_current: int

    sample_port: int
    sample_duration: float
    sample_attenuation: float
    wait_delay: float
    readout_sample_delay: float

    control_port: int
    control_duration: float
    control_freq: float
    control_amp_180: float
    control_amp_90: float
    control_current: int

    anharmonicity: float
    _drag_use: bool
    _drag_lambda: float

    _jpa: dict | None

    _path: str
    _doc: tomlkit.TOMLDocument

    @property
    def drag(self) -> float:
        if self._drag_use:
            alpha = math.tau * self.anharmonicity
            return self._drag_lambda / alpha
        else:
            return 0.0

    @property
    def jpa(self) -> dict | None:
        if self._jpa is not None:
            pump_freq = 2 * (self.readout_freq + self._jpa["_pump_offset"])
            jpa = self._jpa.copy()
            del jpa["_pump_offset"]
            jpa["pump_freq"] = pump_freq
            return jpa

    @classmethod
    def from_qubit(cls, which_qubit: int):
        if which_qubit == 1:
            settings_path = "qubit1.toml"
        elif which_qubit == 2:
            settings_path = "qubit2.toml"
        else:
            raise ValueError(f"unknown qubit number: {which_qubit}")

        return cls.load(settings_path)

    @classmethod
    def from_latest(cls, which_qubit: int) -> Settings:
        filename = max(
            filter(
                lambda x: x.startswith(f"qubit{which_qubit}") and x.endswith(".toml"),
                os.listdir("data/"),
            )
        )
        return cls.load(os.path.join("data", filename))

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as f:
            doc = tomlkit.load(f)

        if "dc-params" in doc["hardware"]:
            params = doc["hardware"]["dc-params"]
            adc_mode = params.get("adc-mode", AdcMode.Mixed.value)
            dac_mode = params.get("dac-mode", DacMode.Mixed.value)
            adc_fsample = params.get("adc-fsample", None)
            dac_fsample = params.get("dac-fsample", None)
            if isinstance(adc_mode, int):
                adc_mode = AdcMode(adc_mode)
            elif isinstance(adc_mode, list):
                assert len(adc_mode) == 4
                assert [isinstance(m, int) for m in adc_mode]
                adc_mode = [AdcMode(m) for m in adc_mode]
            else:
                raise TypeError(f"unknown type for adc-mode: {type(adc_mode) = }, {adc_mode = }")
            if isinstance(dac_mode, int):
                dac_mode = DacMode(dac_mode)
            elif isinstance(dac_mode, list):
                assert len(dac_mode) == 4
                assert [isinstance(m, int) for m in dac_mode]
                dac_mode = [DacMode(m) for m in dac_mode]
            else:
                raise TypeError(f"unknown type for dac-mode: {type(dac_mode) = }, {dac_mode = }")
            if adc_fsample is not None:
                if isinstance(adc_fsample, int):
                    adc_fsample = AdcFSample(adc_fsample)
                elif isinstance(adc_fsample, list):
                    assert len(adc_fsample) == 4
                    assert [isinstance(m, int) for m in adc_fsample]
                    adc_fsample = [AdcFSample(m) for m in adc_fsample]
                else:
                    raise TypeError(
                        f"unknown type for adc-fsample: {type(adc_fsample) = }, {adc_fsample = }"
                    )
            if dac_fsample is not None:
                if isinstance(dac_fsample, int):
                    dac_fsample = DacFSample(dac_fsample)
                elif isinstance(dac_fsample, list):
                    assert len(dac_fsample) == 4
                    assert [isinstance(m, int) for m in dac_fsample]
                    dac_fsample = [DacFSample(m) for m in dac_fsample]
                else:
                    raise TypeError(
                        f"unknown type for dac-fsample: {type(dac_fsample) = }, {dac_fsample = }"
                    )
            dc_params = {
                "adc_mode": adc_mode,
                "adc_fsample": adc_fsample,
                "dac_mode": dac_mode,
                "dac_fsample": dac_fsample,
            }
        else:
            dc_params = {
                "adc_mode": AdcMode.Mixed,
                "adc_fsample": None,
                "dac_mode": DacMode.Mixed,
                "dac_fsample": None,
            }

        if doc["jpa"]["use"]:
            jpa = {
                "pump_port": int(doc["jpa"]["pump-port"]),
                "bias_port": int(doc["jpa"]["bias-port"]),
                "pump_pwr": int(doc["jpa"]["pump-power"]),
                "_pump_offset": float(doc["jpa"]["pump-frequency-offset"]),
                "bias": float(doc["jpa"]["dc-bias"]),
            }
        else:
            jpa = None

        try:
            readout_current = int(doc["readout"]["current"])
        except NonExistentKey:
            try:
                readout_current = int(doc["hardware"]["dac-current"])
                print("WARN: readout.current not found, fall back to legacy hardware.dac-current")
            except NonExistentKey:
                readout_current = 32_000
                print("WARN: readout.current not found, fall back to default 32000")
        try:
            control_current = int(doc["control"]["current"])
        except NonExistentKey:
            try:
                control_current = int(doc["hardware"]["dac-current"])
                print("WARN: control.current not found, fall back to legacy hardware.dac-current")
            except NonExistentKey:
                control_current = 32_000
                print("WARN: control.current not found, fall back to default 32000")
        try:
            sample_attenuation = float(doc["sample"]["attenuation"])
        except NonExistentKey:
            sample_attenuation = 0.0
            print("WARN: sample.attenuation not found, fall back to default 0.0")

        settings = cls(
            _path=filename,
            _doc=doc,
            address=str(doc["hardware"]["address"]),
            port=int(doc["hardware"]["port"]),
            dc_params=dc_params,
            readout_port=int(doc["readout"]["port"]),
            readout_duration=float(doc["readout"]["duration"]),
            readout_amp=float(doc["readout"]["amplitude"]),
            readout_freq=float(doc["readout"]["frequency"]),
            readout_current=readout_current,
            sample_port=int(doc["sample"]["port"]),
            sample_duration=float(doc["sample"]["duration"]),
            sample_attenuation=sample_attenuation,
            wait_delay=float(doc["measurement"]["delay"]),
            readout_sample_delay=float(doc["sample"]["delay"]),
            control_port=int(doc["control"]["port"]),
            control_duration=float(doc["control"]["duration"]),
            control_freq=float(doc["control"]["frequency"]),
            control_amp_180=float(doc["control"]["amplitude_180"]),
            control_amp_90=float(doc["control"]["amplitude_90"]),
            control_current=control_current,
            anharmonicity=float(doc["control"]["drag"]["anharmonicity"]),
            _drag_use=bool(doc["control"]["drag"]["use"]),
            _drag_lambda=float(doc["control"]["drag"]["lambda"]),
            _jpa=jpa,
        )

        return settings

    def save(self, filename: str):
        self._update_toml()
        self._dump_toml(filename)

    def _update_toml(self):
        self._doc["hardware"]["address"] = self.address
        self._doc["hardware"]["port"] = self.port
        self._doc["readout"]["port"] = self.readout_port
        self._doc["readout"]["duration"] = self.readout_duration
        self._doc["readout"]["amplitude"] = self.readout_amp
        self._doc["readout"]["frequency"] = self.readout_freq
        self._doc["readout"]["current"] = self.readout_current
        self._doc["sample"]["port"] = self.sample_port
        self._doc["sample"]["duration"] = self.sample_duration
        self._doc["sample"]["attenuation"] = self.sample_attenuation
        self._doc["measurement"]["delay"] = self.wait_delay
        self._doc["sample"]["delay"] = self.readout_sample_delay
        self._doc["control"]["port"] = self.control_port
        self._doc["control"]["duration"] = self.control_duration
        self._doc["control"]["frequency"] = self.control_freq
        self._doc["control"]["amplitude_180"] = self.control_amp_180
        self._doc["control"]["amplitude_90"] = self.control_amp_90
        self._doc["control"]["current"] = self.control_current

        self._doc["control"]["drag"]["use"] = self._drag_use
        self._doc["control"]["drag"]["lambda"] = self._drag_lambda
        self._doc["control"]["drag"]["anharmonicity"] = self.anharmonicity

        self._doc["hardware"]["dc-params"]["adc-mode"] = [
            m.value for m in self.dc_params["adc_mode"]
        ]
        self._doc["hardware"]["dc-params"]["dac-mode"] = [
            m.value for m in self.dc_params["dac_mode"]
        ]
        if self.dc_params["adc_fsample"] is not None:
            self._doc["hardware"]["dc-params"]["adc-fsample"] = [
                m.value for m in self.dc_params["adc_fsample"]
            ]
        if self.dc_params["dac_fsample"] is not None:
            self._doc["hardware"]["dc-params"]["dac-fsample"] = [
                m.value for m in self.dc_params["dac_fsample"]
            ]

        if self._jpa is None:
            self._doc["jpa"]["use"] = False
        else:
            self._doc["jpa"]["use"] = True
            self._doc["jpa"]["pump-port"] = self._jpa["pump_port"]
            self._doc["jpa"]["bias-port"] = self._jpa["bias_port"]
            self._doc["jpa"]["pump-power"] = self._jpa["pump_pwr"]
            self._doc["jpa"]["pump-frequency-offset"] = self._jpa["_pump_offset"]
            self._doc["jpa"]["dc-bias"] = self._jpa["bias"]

    def _dump_toml(self, filename: str):
        with open(filename, "w") as f:
            tomlkit.dump(self._doc, f)
