# -*- coding: utf-8 -*-
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false

from dataclasses import dataclass
import math
import os

import tomlkit


@dataclass
class Settings:
    address: str
    port: int

    dac_current: int

    readout_port: int
    readout_duration: float
    readout_amp: float
    readout_freq: float

    sample_port: int
    sample_duration: float
    wait_delay: float
    readout_sample_delay: float

    control_port: int
    control_duration: float
    control_freq: float
    control_amp_180: float
    control_amp_90: float

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
    def from_latest(cls, which_qubit: int):
        filename = sorted(
            filter(
                lambda x: x.startswith(f"qubit{which_qubit}") and x.endswith(".toml"),
                os.listdir("data/"),
            )
        )[-1]
        return cls.load(os.path.join("data", filename))

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as f:
            doc = tomlkit.load(f)

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

        settings = cls(
            _path=filename,
            _doc=doc,
            address=str(doc["hardware"]["address"]),
            port=int(doc["hardware"]["port"]),
            dac_current=int(doc["hardware"]["dac-current"]),
            readout_port=int(doc["readout"]["port"]),
            readout_duration=float(doc["readout"]["duration"]),
            readout_amp=float(doc["readout"]["amplitude"]),
            readout_freq=float(doc["readout"]["frequency"]),
            sample_port=int(doc["sample"]["port"]),
            sample_duration=float(doc["sample"]["duration"]),
            wait_delay=float(doc["measurement"]["delay"]),
            readout_sample_delay=float(doc["sample"]["delay"]),
            control_port=int(doc["control"]["port"]),
            control_duration=float(doc["control"]["duration"]),
            control_freq=float(doc["control"]["frequency"]),
            control_amp_180=float(doc["control"]["amplitude_180"]),
            control_amp_90=float(doc["control"]["amplitude_90"]),
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
        self._doc["hardware"]["dac-current"] = self.dac_current
        self._doc["readout"]["port"] = self.readout_port
        self._doc["readout"]["duration"] = self.readout_duration
        self._doc["readout"]["amplitude"] = self.readout_amp
        self._doc["readout"]["frequency"] = self.readout_freq
        self._doc["sample"]["port"] = self.sample_port
        self._doc["sample"]["duration"] = self.sample_duration
        self._doc["measurement"]["delay"] = self.wait_delay
        self._doc["sample"]["delay"] = self.readout_sample_delay
        self._doc["control"]["port"] = self.control_port
        self._doc["control"]["duration"] = self.control_duration
        self._doc["control"]["frequency"] = self.control_freq
        self._doc["control"]["amplitude_180"] = self.control_amp_180
        self._doc["control"]["amplitude_90"] = self.control_amp_90

        self._doc["control"]["drag"]["use"] = self._drag_use
        self._doc["control"]["drag"]["lambda"] = self._drag_lambda
        self._doc["control"]["drag"]["anharmonicity"] = self.anharmonicity

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
