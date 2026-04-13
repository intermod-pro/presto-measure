from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, OptimizeResult

from presto.utils import asarray

FreqTy: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
ScatTy: TypeAlias = np.ndarray[tuple[int], np.dtype[np.complex128]]
DbTy: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
JacTy: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


@dataclass
class Notch:
    _ar: float
    _ai: float
    _tau: float
    _fr: float
    _ql: float
    _kr: float
    _ki: float
    _fit_result: OptimizeResult | None = None

    def f_res(self) -> float:
        """Resonance frequency"""
        return self._fr

    def q_loaded(self) -> float:
        """Loaded quality factor"""
        return self._ql

    def q_coupling(self) -> float:
        """Coupling quality factor"""
        return 1.0 / self._kr

    def q_internal(self) -> float:
        """Internal quality factor"""
        return 1.0 / (1.0 / self._ql - self._kr)

    def _to_list(self) -> list[float]:
        return [self._ar, self._ai, self._tau, self._fr, self._ql, self._kr, self._ki]

    def _to_list_baseline(self) -> list[float]:
        return [self._ar, self._ai, self._tau]

    def _to_list_norm(self) -> list[float]:
        return [self._fr, self._ql, self._kr, self._ki]

    @classmethod
    def _from_list(cls, x: list[float]) -> Notch:
        a_real, a_imag, tau, fr, q, k_real, k_imag = x
        return cls(a_real, a_imag, tau, fr, q, k_real, k_imag)

    def copy(self) -> Notch:
        """Returns a copy"""
        return Notch._from_list(self._to_list())

    def _s_baseline(self, f: FreqTy) -> ScatTy:
        a = np.complex128(self._ar + 1j * self._ai)
        s: npt.NDArray[np.complex128] = a * np.exp(-2j * np.pi * f * self._tau)
        return s  # pyright: ignore[reportReturnType]

    def _s_norm(self, f: FreqTy) -> ScatTy:
        # M. S. Khalil, M. J. A. Stoutimore, F. C. Wellstood, and K. D. Osborn.
        # An analysis method for asymmetric resonator transmission applied to superconducting devices.
        # Journal of Applied Physics 111, 054510 (2012).
        # https://doi.org/10.1063/1.3692073
        # Eq. (11), with k = 1 / Qe
        k = np.complex128(self._kr + 1j * self._ki)
        s: npt.NDArray[np.complex128] = 1.0 - self._ql * k / (
            1.0 + 2j * self._ql * (f / self._fr - 1.0)
        )
        return s  # pyright: ignore[reportReturnType]

    def s(self, f: npt.NDArray[np.floating]) -> ScatTy:
        """Evaluate the the response function

        Args:
            f: frequency array in hertz

        Returns:
            linear-response function evaluated at ``f``
        """
        f = asarray(f, np.float64)
        s: npt.NDArray[np.complex128] = self._s_baseline(f) * self._s_norm(f)
        return s  # pyright: ignore[reportReturnType]

    def s_db(self, f: npt.NDArray[np.floating]) -> DbTy:
        """Evaluate the the magnitude of the response function, in dB

        Args:
            f: frequency array in hertz

        Returns:
            magnitude of the linear-response function evaluated at ``f``, in dB
        """
        s_db: npt.NDArray[np.float64] = 20.0 * np.log10(np.abs(self.s(f)))
        return s_db  # pyright: ignore[reportReturnType]

    def _jac_baseline(self, f: FreqTy) -> JacTy:
        # a_real, a_imag, tau = x
        jac = np.zeros((2 * len(f), 3))
        # ds/da_real
        jac[:, 0] = _c2r(np.exp(-2j * np.pi * f * self._tau) * self._s_norm(f))
        # ds/da_imag
        jac[:, 1] = _c2r(1j * np.exp(-2j * np.pi * f * self._tau) * self._s_norm(f))
        # ds/dtau
        jac[:, 2] = _c2r(-2j * np.pi * f * self._s_baseline(f) * self._s_norm(f))
        return jac

    def _jac_norm(self, f: FreqTy) -> JacTy:
        # fr, q, k_real, k_imag = x
        k = self._kr + 1j * self._ki
        jac = np.zeros((2 * len(f), 4))
        # ds/dfr
        jac[:, 0] = _c2r(
            self._s_baseline(f)
            * (2j * f * k * self._ql**2)
            / (-2 * f * self._ql + (2 * self._ql + 1j) * self._fr) ** 2
        )
        # ds/q
        jac[:, 1] = _c2r(
            self._s_baseline(f)
            * (k * self._fr**2)
            / (-2 * f * self._ql + (2 * self._ql + 1j) * self._fr) ** 2
        )
        # ds/k_real
        jac[:, 2] = _c2r(-self._s_baseline(f) * self._ql / (1 + 2j * self._ql * (f / self._fr - 1)))

        # ds/k_imag
        jac[:, 3] = _c2r(
            -1j * self._s_baseline(f) * self._ql / (1 + 2j * self._ql * (f / self._fr - 1))
        )
        return jac

    def _jac_all(self, f: FreqTy) -> JacTy:
        # a_real, a_imag, tau, fr, q, k_real, k_imag = x
        jac = np.zeros((2 * len(f), 7))
        jac[:, :3] = self._jac_baseline(f)
        jac[:, 3:] = self._jac_norm(f)
        return jac

    def _refit_norm(self, f_data: FreqTy, s_data: ScatTy) -> tuple[Notch, OptimizeResult]:
        def erf(x):
            p = self.copy()
            p._fr, p._ql, p._kr, p._ki = x
            err = p.s(f_data) - s_data
            return _c2r(err)

        def jac(x):
            p = self.copy()
            p._fr, p._ql, p._kr, p._ki = x
            return p._jac_norm(f_data)

        res = _fit(erf, self._to_list_norm(), jac)

        new_p = self.copy()
        new_p._fr, new_p._ql, new_p._kr, new_p._ki = res.x
        return new_p, res

    def _refit_baseline(self, f_data: FreqTy, s_data: ScatTy) -> tuple[Notch, OptimizeResult]:
        def erf(x):
            p = self.copy()
            p._ar, p._ai, p._tau = x
            err = p.s(f_data) - s_data
            return _c2r(err)

        def jac(x):
            p = self.copy()
            p._ar, p._ai, p._tau = x
            return p._jac_baseline(f_data)

        res = _fit(erf, self._to_list_baseline(), jac)

        new_p = self.copy()
        new_p._ar, new_p._ai, new_p._tau = res.x
        return new_p, res

    def _refit_all(self, f_data: FreqTy, s_data: ScatTy) -> tuple[Notch, OptimizeResult]:
        def erf(x):
            p = Notch._from_list(x)
            err = p.s(f_data) - s_data
            return _c2r(err)

        def jac(x):
            p = Notch._from_list(x)
            return p._jac_all(f_data)

        res = _fit(erf, self._to_list(), jac)

        p = Notch._from_list(res.x)
        return p, res

    @classmethod
    def _guess_from_data(cls, f_data: FreqTy, s_data: ScatTy, x0: dict[str, float]) -> Notch:
        fr = x0.get("fr")
        if fr is None:
            # fr is at the minimum
            idx_min = np.argmin(np.abs(s_data))
            fr = f_data[idx_min]

        tau = x0.get("tau")
        if tau is None:
            # tau is slope in phase
            pu = np.unwrap(np.angle(s_data))
            dp = pu[-1] - pu[0]
            df = f_data[-1] - f_data[0]
            tau = -dp / (2 * np.pi * df)

        # a is s at zero, accounting for tau
        a = s_data[0] / np.exp(1j * (-2 * np.pi * f_data[0] * tau))

        ql = x0.get("ql")
        if ql is None:
            # Ql and Qc, seems reasonable
            ql = 10_000.0

        k_real = 0.9 / ql
        k_imag = 0.0

        return cls(
            _ar=np.real(a),
            _ai=np.imag(a),
            _tau=tau,
            _fr=fr,
            _ql=ql,
            _kr=k_real,
            _ki=k_imag,
        )

    @classmethod
    def autofit(
        cls,
        f_data: npt.NDArray[np.floating],
        s_data: npt.NDArray[np.complexfloating],
        x0: dict[str, float] | None = None,
    ) -> Notch:
        """Perform a least-squares fit of the resonance model to the data.

        Args:
            f_data: the frequency array, in hertz
            s_data: the measured scattering data, complex valued
            x0: optional dictionary with initial guess for the fit, if ``None`` (default), extract
                initial guess from the data

        Returns:
            a ``Notch`` object with parameters from the result of the fit
        """
        f_data = asarray(f_data, np.float64)
        s_data = asarray(s_data, np.complex128)
        if x0 is None:
            x0 = dict()

        p = cls._guess_from_data(f_data, s_data, x0)
        p, res = p._refit_norm(f_data, s_data)
        p, res = p._refit_baseline(f_data, s_data)
        p, res = p._refit_all(f_data, s_data)

        p._fit_result = res

        return p

    def fit_result(self) -> OptimizeResult:
        """Return the result object of the least-squares fit

        Raises:
            RuntimeError: if no fit result is available
        """
        if self._fit_result is None:
            raise RuntimeError("fit result not available, no fit was performed")
        return self._fit_result


def _c2r(data: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    return np.frombuffer(data, np.float64)


def _fit(erf: Callable, x0: list[float], jac: Callable) -> OptimizeResult:
    return least_squares(
        fun=erf,
        x0=x0,
        jac=jac,  # pyright: ignore[reportArgumentType]
        method="lm",
        x_scale="jac",  # pyright: ignore[reportArgumentType]
    )
