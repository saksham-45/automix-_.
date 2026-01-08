"""
Curve Parameterization Module

Fits volume/EQ curves to mathematical functions for smoother ML learning.
Instead of storing raw points, we store function parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class CurveParameterizer:
    """Parameterizes curves to functions for ML training"""
    
    @staticmethod
    def parameterize_volume_curve(
        times: np.ndarray,
        gain_db: np.ndarray,
        curve_type: str = "auto"
    ) -> Dict:
        """
        Fit volume curve to a function.
        
        Returns:
            {
                "type": "exponential" | "linear" | "logarithmic",
                "params": {...},
                "r2_score": float,
                "reconstruction": List[float]  # Reconstructed curve for validation
            }
        """
        if len(times) < 3:
            return {
                "type": "linear",
                "params": {"a": 0, "b": gain_db[0] if len(gain_db) > 0 else 0},
                "r2_score": 0.0,
                "reconstruction": gain_db.tolist()
            }
        
        # Normalize times to [0, 1]
        times_norm = (times - times[0]) / (times[-1] - times[0] + 1e-8)
        
        # Try different curve types
        best_fit = None
        best_r2 = -np.inf
        best_type = "linear"
        
        # 1. Exponential decay/growth: y = a * exp(b*x) + c
        try:
            if np.all(gain_db <= 0):  # Fade out
                # Exponential decay: y = a * exp(-b*x) + c
                def exp_decay(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                popt, _ = curve_fit(
                    exp_decay,
                    times_norm,
                    gain_db,
                    p0=[gain_db[0], 2.0, gain_db[-1]],
                    maxfev=1000
                )
                y_pred = exp_decay(times_norm, *popt)
                r2 = CurveParameterizer._compute_r2(gain_db, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_type = "exponential_decay"
                    best_fit = {
                        "type": "exponential_decay",
                        "params": {
                            "a": float(popt[0]),
                            "b": float(popt[1]),
                            "c": float(popt[2])
                        },
                        "r2_score": float(r2),
                        "reconstruction": y_pred.tolist()
                    }
            else:  # Fade in
                # Exponential growth: y = a * (1 - exp(-b*x)) + c
                def exp_growth(x, a, b, c):
                    return a * (1 - np.exp(-b * x)) + c
                
                popt, _ = curve_fit(
                    exp_growth,
                    times_norm,
                    gain_db,
                    p0=[gain_db[-1] - gain_db[0], 2.0, gain_db[0]],
                    maxfev=1000
                )
                y_pred = exp_growth(times_norm, *popt)
                r2 = CurveParameterizer._compute_r2(gain_db, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_type = "exponential_growth"
                    best_fit = {
                        "type": "exponential_growth",
                        "params": {
                            "a": float(popt[0]),
                            "b": float(popt[1]),
                            "c": float(popt[2])
                        },
                        "r2_score": float(r2),
                        "reconstruction": y_pred.tolist()
                    }
        except:
            pass
        
        # 2. Linear: y = a*x + b
        try:
            popt = np.polyfit(times_norm, gain_db, 1)
            y_pred = np.polyval(popt, times_norm)
            r2 = CurveParameterizer._compute_r2(gain_db, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_type = "linear"
                best_fit = {
                    "type": "linear",
                    "params": {
                        "a": float(popt[0]),
                        "b": float(popt[1])
                    },
                    "r2_score": float(r2),
                    "reconstruction": y_pred.tolist()
                }
        except:
            pass
        
        # 3. Logarithmic: y = a * log(b*x + 1) + c
        try:
            def log_curve(x, a, b, c):
                return a * np.log(b * x + 1) + c
            
            popt, _ = curve_fit(
                log_curve,
                times_norm,
                gain_db,
                p0=[gain_db[-1] - gain_db[0], 1.0, gain_db[0]],
                maxfev=1000
            )
            y_pred = log_curve(times_norm, *popt)
            r2 = CurveParameterizer._compute_r2(gain_db, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_type = "logarithmic"
                best_fit = {
                    "type": "logarithmic",
                    "params": {
                        "a": float(popt[0]),
                        "b": float(popt[1]),
                        "c": float(popt[2])
                    },
                    "r2_score": float(r2),
                    "reconstruction": y_pred.tolist()
                }
        except:
            pass
        
        # Fallback to linear if nothing worked
        if best_fit is None:
            popt = np.polyfit(times_norm, gain_db, 1)
            y_pred = np.polyval(popt, times_norm)
            best_fit = {
                "type": "linear",
                "params": {
                    "a": float(popt[0]),
                    "b": float(popt[1])
                },
                "r2_score": float(CurveParameterizer._compute_r2(gain_db, y_pred)),
                "reconstruction": y_pred.tolist()
            }
        
        return best_fit
    
    @staticmethod
    def parameterize_eq_curve(
        times: np.ndarray,
        freq_hz: np.ndarray,
        curve_type: str = "filter_sweep"
    ) -> Dict:
        """
        Parameterize EQ/filter curve.
        
        For filter sweeps, we typically want: freq = a * (1 - exp(-b*t)) + c
        """
        if len(times) < 3:
            return {
                "type": "linear",
                "params": {"a": freq_hz[0] if len(freq_hz) > 0 else 1000},
                "r2_score": 0.0
            }
        
        times_norm = (times - times[0]) / (times[-1] - times[0] + 1e-8)
        
        # Filter sweep: exponential growth
        try:
            def filter_sweep(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
            
            popt, _ = curve_fit(
                filter_sweep,
                times_norm,
                freq_hz,
                p0=[freq_hz[-1] - freq_hz[0], 2.0, freq_hz[0]],
                maxfev=1000
            )
            y_pred = filter_sweep(times_norm, *popt)
            r2 = CurveParameterizer._compute_r2(freq_hz, y_pred)
            
            return {
                "type": "filter_sweep",
                "params": {
                    "start_freq_hz": float(freq_hz[0]),
                    "end_freq_hz": float(freq_hz[-1]),
                    "a": float(popt[0]),
                    "b": float(popt[1]),
                    "c": float(popt[2])
                },
                "r2_score": float(r2),
                "reconstruction": y_pred.tolist()
            }
        except:
            # Fallback to linear
            popt = np.polyfit(times_norm, freq_hz, 1)
            return {
                "type": "linear",
                "params": {
                    "a": float(popt[0]),
                    "b": float(popt[1])
                },
                "r2_score": float(CurveParameterizer._compute_r2(freq_hz, np.polyval(popt, times_norm)))
            }
    
    @staticmethod
    def reconstruct_curve(params: Dict, times: np.ndarray) -> np.ndarray:
        """Reconstruct curve from parameters"""
        curve_type = params["type"]
        p = params["params"]
        times_norm = (times - times[0]) / (times[-1] - times[0] + 1e-8)
        
        if curve_type == "linear":
            return np.polyval([p["a"], p["b"]], times_norm)
        elif curve_type == "exponential_decay":
            return p["a"] * np.exp(-p["b"] * times_norm) + p["c"]
        elif curve_type == "exponential_growth":
            return p["a"] * (1 - np.exp(-p["b"] * times_norm)) + p["c"]
        elif curve_type == "logarithmic":
            return p["a"] * np.log(p["b"] * times_norm + 1) + p["c"]
        elif curve_type == "filter_sweep":
            return p["a"] * (1 - np.exp(-p["b"] * times_norm)) + p["c"]
        else:
            # Default to linear
            return np.polyval([p.get("a", 0), p.get("b", 0)], times_norm)
    
    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

