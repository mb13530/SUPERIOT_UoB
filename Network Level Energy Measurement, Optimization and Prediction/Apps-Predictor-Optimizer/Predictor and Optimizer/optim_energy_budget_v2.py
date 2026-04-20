import sys
import os
import glob
import pickle
import traceback
import joblib
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
    QScrollArea, QGridLayout, QFrame, QMessageBox,
    QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap

from scipy.optimize import differential_evolution


# -----------------------------
# Configuration / conventions
# -----------------------------
FEATURE_COLUMNS = [
    "State",
    "VLC_Payload",
    "BLE_Payload",
    "TxPower",
    "AdvInterval",
    "ConnInterval",
    "Comm_Mode",
]

COMM_MODE_DISCONNECTED = 1
COMM_MODE_CONNECTED = 2

STATE_LABELS = {
    1: "State 1 (Initialization)",
    2: "State 2 (Advertising)",
    3: "State 3 (Operation)",
    4: "State 4 (Sleep)",
}

OBJECTIVE_MIN_ENERGY = "Minimize Energy"
OBJECTIVE_MAX_WORK_UNDER_BUDGET = "Maximize Useful Work under Energy Budget"
OBJECTIVE_BALANCED = "Balanced Score (Recommended)"


def safe_load_model_file(model_file):
    joblib_error = None
    pickle_error = None

    try:
        obj = joblib.load(model_file)
        return obj, "joblib"
    except Exception as e:
        joblib_error = e
        print(f"joblib failed for {model_file}: {e}")

    try:
        with open(model_file, "rb") as f:
            obj = pickle.load(f)
        return obj, "pickle"
    except Exception as e:
        pickle_error = e

    raise RuntimeError(
        f"Could not load {model_file} with either joblib or pickle.\n"
        f"joblib error: {joblib_error}\n"
        f"pickle error: {pickle_error}"
    )


def normalize_loaded_model_payload(obj, source_name="unknown"):
    if isinstance(obj, dict):
        if "model" in obj:
            return obj

        for alt_key in ("estimator", "regressor", "clf"):
            if alt_key in obj:
                return {
                    "model": obj[alt_key],
                    "scaler": obj.get("scaler", None),
                    "r2_score": obj.get("r2_score", obj.get("score", "N/A")),
                }

        raise ValueError(f"{source_name}: dict found, but no 'model' key. Keys: {list(obj.keys())}")

    if hasattr(obj, "predict"):
        return {
            "model": obj,
            "scaler": None,
            "r2_score": "N/A",
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size > 0 and all(isinstance(x, str) for x in obj.flat):
            raise ValueError(f"{source_name}: file contains feature names only, not a trained model")

        if obj.shape == () or obj.size == 1:
            try:
                inner = obj.item()
                return normalize_loaded_model_payload(inner, source_name)
            except Exception:
                pass

        items = list(obj.flat)
        if len(items) >= 1 and hasattr(items[0], "predict"):
            return {
                "model": items[0],
                "scaler": items[1] if len(items) > 1 else None,
                "r2_score": items[2] if len(items) > 2 else "N/A",
            }

        raise ValueError(
            f"{source_name}: ndarray is not a recognized model payload. "
            f"shape={obj.shape}, dtype={obj.dtype}"
        )

    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if len(items) >= 1 and hasattr(items[0], "predict"):
            return {
                "model": items[0],
                "scaler": items[1] if len(items) > 1 else None,
                "r2_score": items[2] if len(items) > 2 else "N/A",
            }

    raise ValueError(f"{source_name}: unsupported payload type {type(obj)}")


def get_expected_feature_order(model_data):
    scaler = model_data.get("scaler", None)
    model = model_data.get("model", None)

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return FEATURE_COLUMNS


def build_input_dataframe(state, config):
    return pd.DataFrame({
        "State": [state],
        "VLC_Payload": [config.get("vlc_payload", 0)],
        "BLE_Payload": [config.get("ble_payload", 0)],
        "TxPower": [config["txpower"]],
        "AdvInterval": [config["adv_interval"]],
        "ConnInterval": [config["conn_interval"]],
        "Comm_Mode": [config.get("comm_mode", COMM_MODE_DISCONNECTED)],
    })


def validate_loaded_model(model_data):
    if not isinstance(model_data, dict):
        raise ValueError(f"Loaded object is not a dict. Got: {type(model_data)}")

    if "model" not in model_data:
        raise ValueError(f"Loaded dict missing required key 'model'. Keys: {list(model_data.keys())}")

    model = model_data["model"]
    scaler = model_data.get("scaler", None)

    test_config = {
        "txpower": 0,
        "adv_interval": 100,
        "conn_interval": 100,
        "vlc_payload": 0,
        "ble_payload": 0,
        "comm_mode": COMM_MODE_DISCONNECTED,
    }

    df = build_input_dataframe(state=1, config=test_config)
    expected_cols = get_expected_feature_order(model_data)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    df = df[expected_cols]

    if scaler is not None:
        x = scaler.transform(df)
    else:
        x = df

    pred = model.predict(x)
    if len(pred) == 0:
        raise ValueError("Model returned empty prediction during validation.")

    return float(pred[0])


class BudgetSettings:
    def __init__(
        self,
        enabled=False,
        capacitance_mF=38.4,
        v_start=3.3,
        v_min=1.6,
        opv_area_cm2=1.0,
        opv_power_density_uW_cm2=0.0,
        harvest_efficiency=1.0,
    ):
        self.enabled = bool(enabled)
        self.capacitance_mF = float(capacitance_mF)
        self.v_start = float(v_start)
        self.v_min = float(v_min)
        self.opv_area_cm2 = float(opv_area_cm2)
        self.opv_power_density_uW_cm2 = float(opv_power_density_uW_cm2)
        self.harvest_efficiency = float(harvest_efficiency)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "capacitance_mF": self.capacitance_mF,
            "v_start": self.v_start,
            "v_min": self.v_min,
            "opv_area_cm2": self.opv_area_cm2,
            "opv_power_density_uW_cm2": self.opv_power_density_uW_cm2,
            "harvest_efficiency": self.harvest_efficiency,
        }


class ObjectiveSettings:
    def __init__(
        self,
        objective_mode=OBJECTIVE_BALANCED,
        ble_weight=1.0,
        vlc_weight=1.0,
        energy_weight=0.35,
        connected_bonus=0.05,
    ):
        self.objective_mode = str(objective_mode)
        self.ble_weight = float(ble_weight)
        self.vlc_weight = float(vlc_weight)
        self.energy_weight = float(energy_weight)
        self.connected_bonus = float(connected_bonus)

    def to_dict(self):
        return {
            "objective_mode": self.objective_mode,
            "ble_weight": self.ble_weight,
            "vlc_weight": self.vlc_weight,
            "energy_weight": self.energy_weight,
            "connected_bonus": self.connected_bonus,
        }


class EnergyCalculator:
    """
    Shared calculator used by both the worker and the UI for:
    - validity checks
    - total energy calculation
    - per-state breakdown generation
    - energy budget calculation
    - useful-work score calculation
    """

    def __init__(
        self,
        model_data,
        state_durations,
        ble_connected_duration,
        budget_settings=None,
        objective_settings=None,
        vlc_ms_per_byte=39.48,
        ble_ms_per_byte=0.02722,
    ):
        self.model_data = model_data
        self.state_durations = state_durations
        self.ble_connected_duration = ble_connected_duration
        self.vlc_ms_per_byte = vlc_ms_per_byte
        self.ble_ms_per_byte = ble_ms_per_byte
        self.budget_settings = budget_settings or BudgetSettings()
        self.objective_settings = objective_settings or ObjectiveSettings()

        self.sleep_txpower = 0.0
        self.sleep_adv_interval = 100.0
        self.sleep_conn_interval = 100.0

        self.init_txpower = 0.0
        self.init_adv_interval = 100.0
        self.init_conn_interval = 100.0

        self.adv_default_conn_interval = 100.0
        self._no_budget_energy_ref_cache = {}

    def _normalize_config(self, config):
        comm_mode = int(config.get("comm_mode", COMM_MODE_DISCONNECTED))
        comm_mode = COMM_MODE_CONNECTED if comm_mode == COMM_MODE_CONNECTED else COMM_MODE_DISCONNECTED

        return {
            "txpower": float(config["txpower"]),
            "adv_interval": float(config["adv_interval"]),
            "conn_interval": float(config["conn_interval"]),
            "vlc_payload": float(config.get("vlc_payload", 0)),
            "ble_payload": float(config.get("ble_payload", 0)),
            "comm_mode": comm_mode,
        }

    def _get_state_specific_config(self, state, config):
        cfg = self._normalize_config(config).copy()

        if state == 1:
            cfg["ble_payload"] = 0.0
            cfg["vlc_payload"] = 0.0
            cfg["comm_mode"] = COMM_MODE_DISCONNECTED
            cfg["txpower"] = self.init_txpower
            cfg["adv_interval"] = self.init_adv_interval
            cfg["conn_interval"] = self.init_conn_interval

        elif state == 2:
            cfg["ble_payload"] = 0.0
            cfg["vlc_payload"] = 0.0
            cfg["comm_mode"] = COMM_MODE_DISCONNECTED
            cfg["conn_interval"] = self.adv_default_conn_interval

        elif state == 4:
            cfg["ble_payload"] = 0.0
            cfg["vlc_payload"] = 0.0
            cfg["comm_mode"] = COMM_MODE_DISCONNECTED
            cfg["txpower"] = self.sleep_txpower
            cfg["adv_interval"] = self.sleep_adv_interval
            cfg["conn_interval"] = self.sleep_conn_interval

        return cfg

    def _effective_connected_duration(self, total_duration, config):
        if config["comm_mode"] == COMM_MODE_DISCONNECTED:
            return 0.0
        return min(max(0.0, self.ble_connected_duration), max(0.0, total_duration))

    def _get_raw_tx_durations_seconds(self, config):
        ble_tx_duration = max(0.0, config["ble_payload"]) * self.ble_ms_per_byte / 1000.0
        vlc_tx_duration = max(0.0, config["vlc_payload"]) * self.vlc_ms_per_byte / 1000.0
        return ble_tx_duration, vlc_tx_duration

    def _predict_current(self, input_data):
        model = self.model_data["model"]
        scaler = self.model_data.get("scaler", None)

        expected_cols = get_expected_feature_order(self.model_data)
        missing = [c for c in expected_cols if c not in input_data.columns]
        if missing:
            raise ValueError(f"Input data missing expected columns: {missing}")

        input_data = input_data[expected_cols]

        if scaler is not None:
            x = scaler.transform(input_data)
        else:
            x = input_data

        current_uA = model.predict(x)[0]
        return float(current_uA)

    def total_duration_seconds(self):
        return sum(float(v) for v in self.state_durations.values())

    def get_energy_budget_mJ(self):
        bs = self.budget_settings
        if not bs.enabled:
            return None

        C = max(0.0, bs.capacitance_mF) / 1000.0
        v_hi = max(bs.v_start, bs.v_min)
        v_lo = min(bs.v_start, bs.v_min)

        cap_energy_J = 0.5 * C * max(0.0, v_hi * v_hi - v_lo * v_lo)

        total_duration_s = self.total_duration_seconds()
        opv_power_W = max(0.0, bs.opv_power_density_uW_cm2) * max(0.0, bs.opv_area_cm2) * 1e-6
        harvest_energy_J = opv_power_W * total_duration_s * np.clip(bs.harvest_efficiency, 0.0, 1.0)

        return (cap_energy_J + harvest_energy_J) * 1000.0

    def get_budget_breakdown(self):
        bs = self.budget_settings
        if not bs.enabled:
            return {
                "enabled": False,
                "budget_mJ": None,
                "cap_energy_mJ": None,
                "harvest_energy_mJ": None,
            }

        C = max(0.0, bs.capacitance_mF) / 1000.0
        v_hi = max(bs.v_start, bs.v_min)
        v_lo = min(bs.v_start, bs.v_min)

        cap_energy_J = 0.5 * C * max(0.0, v_hi * v_hi - v_lo * v_lo)

        total_duration_s = self.total_duration_seconds()
        opv_power_W = max(0.0, bs.opv_power_density_uW_cm2) * max(0.0, bs.opv_area_cm2) * 1e-6
        harvest_energy_J = opv_power_W * total_duration_s * np.clip(bs.harvest_efficiency, 0.0, 1.0)

        return {
            "enabled": True,
            "budget_mJ": (cap_energy_J + harvest_energy_J) * 1000.0,
            "cap_energy_mJ": cap_energy_J * 1000.0,
            "harvest_energy_mJ": harvest_energy_J * 1000.0,
        }

    def get_useful_work_score(self, config, constraints):
        config = self._normalize_config(config)
        os_ = self.objective_settings

        ble_max = max(float(constraints["ble_payload"][1]), 1e-9)
        vlc_max = max(float(constraints["vlc_payload"][1]), 1e-9)

        ble_norm = max(0.0, config["ble_payload"]) / ble_max
        vlc_norm = max(0.0, config["vlc_payload"]) / vlc_max
        connected_bonus = os_.connected_bonus if config["comm_mode"] == COMM_MODE_CONNECTED else 0.0

        return os_.ble_weight * ble_norm + os_.vlc_weight * vlc_norm + connected_bonus

    def _constraints_cache_key(self, constraints):
        keys = ["txpower", "adv_interval", "conn_interval", "vlc_payload", "ble_payload", "comm_mode"]
        return tuple((k, float(constraints[k][0]), float(constraints[k][1])) for k in keys)

    def _get_no_budget_energy_reference_mJ(self, constraints):
        """
        When budget is disabled, keep a meaningful energy penalty in the
        balanced objective by using a reference energy from the search bounds.
        """
        cache_key = self._constraints_cache_key(constraints)
        if cache_key in self._no_budget_energy_ref_cache:
            return self._no_budget_energy_ref_cache[cache_key]

        def cmin(name):
            return float(constraints[name][0])

        def cmax(name):
            return float(constraints[name][1])

        candidates = [
            {
                "txpower": cmin("txpower"),
                "adv_interval": cmax("adv_interval"),
                "conn_interval": cmax("conn_interval"),
                "vlc_payload": 0.0,
                "ble_payload": 0.0,
                "comm_mode": COMM_MODE_DISCONNECTED,
            },
            {
                "txpower": cmax("txpower"),
                "adv_interval": cmin("adv_interval"),
                "conn_interval": cmin("conn_interval"),
                "vlc_payload": 0.0,
                "ble_payload": 0.0,
                "comm_mode": COMM_MODE_DISCONNECTED,
            },
            {
                "txpower": cmax("txpower"),
                "adv_interval": cmin("adv_interval"),
                "conn_interval": cmin("conn_interval"),
                "vlc_payload": cmax("vlc_payload"),
                "ble_payload": cmax("ble_payload"),
                "comm_mode": COMM_MODE_CONNECTED,
            },
            {
                "txpower": cmin("txpower"),
                "adv_interval": cmax("adv_interval"),
                "conn_interval": cmax("conn_interval"),
                "vlc_payload": cmax("vlc_payload"),
                "ble_payload": cmax("ble_payload"),
                "comm_mode": COMM_MODE_CONNECTED,
            },
        ]

        energies = []
        for cfg in candidates:
            cfg = self._normalize_config(cfg)
            valid, _ = self.is_valid_config(cfg)
            if valid:
                e = self.calculate_total_energy(cfg)
                if np.isfinite(e):
                    energies.append(float(e))

        ref_mJ = max(energies) if energies else 1.0
        ref_mJ = max(ref_mJ, 1.0)
        self._no_budget_energy_ref_cache[cache_key] = ref_mJ
        return ref_mJ

    def is_valid_config(self, config):
        config = self._normalize_config(config)
        state3_total = float(self.state_durations.get(3, 0.0))
        connected_duration = self._effective_connected_duration(state3_total, config)

        if config["comm_mode"] == COMM_MODE_DISCONNECTED:
            if config["ble_payload"] > 0 or config["vlc_payload"] > 0:
                return False, "Disconnected mode cannot have BLE/VLC payload"
            return True, ""

        if connected_duration <= 0:
            return False, "Connected mode requires positive State 3 BLE connected duration"

        raw_ble_tx_duration, raw_vlc_tx_duration = self._get_raw_tx_durations_seconds(config)
        if raw_ble_tx_duration + raw_vlc_tx_duration > connected_duration + 1e-12:
            return False, "BLE/VLC TX duration exceeds available connected duration"

        return True, ""

    def get_breakdown(self, config):
        config = self._normalize_config(config)
        valid, reason = self.is_valid_config(config)

        budget_mJ = self.get_energy_budget_mJ()
        budget_enabled = budget_mJ is not None

        breakdown = {
            "valid": valid,
            "reason": reason,
            "config": config,
            "total_energy_mJ": float("inf") if not valid else 0.0,
            "states": {},
            "budget": self.get_budget_breakdown(),
            "budget_enabled": budget_enabled,
            "within_budget": None,
            "budget_margin_mJ": None,
        }

        if not valid:
            return breakdown

        voltage = 3.3
        total_energy = 0.0

        # States 1, 2, 4
        for state in [1, 2, 4]:
            duration = float(self.state_durations.get(state, 0.0))
            if duration <= 0:
                breakdown["states"][state] = {
                    "label": STATE_LABELS[state],
                    "duration_s": 0.0,
                    "current_uA": 0.0,
                    "energy_mJ": 0.0,
                }
                continue

            state_config = self._get_state_specific_config(state, config)
            input_data = build_input_dataframe(state=state, config=state_config)
            current_uA = self._predict_current(input_data)
            current_mA = current_uA / 1000.0
            energy_mJ = current_mA * voltage * duration

            breakdown["states"][state] = {
                "label": STATE_LABELS[state],
                "duration_s": duration,
                "current_uA": current_uA,
                "energy_mJ": energy_mJ,
            }
            total_energy += energy_mJ

        # State 3 composite
        state3_total = float(self.state_durations.get(3, 0.0))
        state3_info = {
            "label": STATE_LABELS[3],
            "duration_s": state3_total,
            "avg_current_uA": 0.0,
            "energy_mJ": 0.0,
            "substates": [],
        }

        if state3_total > 0:
            if config["comm_mode"] == COMM_MODE_DISCONNECTED:
                config_idle = config.copy()
                config_idle.update({
                    "vlc_payload": 0.0,
                    "ble_payload": 0.0,
                    "comm_mode": COMM_MODE_DISCONNECTED,
                })
                input_data = build_input_dataframe(state=3, config=config_idle)
                current_uA = self._predict_current(input_data)
                current_mA = current_uA / 1000.0
                energy_mJ = current_mA * voltage * state3_total

                state3_info["substates"].append({
                    "name": "BLE disconnected idle",
                    "duration_s": state3_total,
                    "current_uA": current_uA,
                    "energy_mJ": energy_mJ,
                })
                state3_info["energy_mJ"] = energy_mJ
                total_energy += energy_mJ

            else:
                connected_duration = self._effective_connected_duration(state3_total, config)
                disconnected_duration = max(0.0, state3_total - connected_duration)
                ble_tx_duration, vlc_tx_duration = self._get_raw_tx_durations_seconds(config)

                if ble_tx_duration + vlc_tx_duration > connected_duration + 1e-12:
                    breakdown["valid"] = False
                    breakdown["reason"] = "BLE/VLC TX duration exceeds available connected duration"
                    breakdown["total_energy_mJ"] = float("inf")
                    return breakdown

                connected_idle_duration = max(0.0, connected_duration - ble_tx_duration - vlc_tx_duration)
                state3_energy = 0.0

                if disconnected_duration > 0:
                    config_1 = config.copy()
                    config_1.update({
                        "vlc_payload": 0.0,
                        "ble_payload": 0.0,
                        "comm_mode": COMM_MODE_DISCONNECTED,
                    })
                    input_data = build_input_dataframe(state=3, config=config_1)
                    current_uA = self._predict_current(input_data)
                    current_mA = current_uA / 1000.0
                    energy_mJ = current_mA * voltage * disconnected_duration
                    state3_info["substates"].append({
                        "name": "BLE disconnected idle",
                        "duration_s": disconnected_duration,
                        "current_uA": current_uA,
                        "energy_mJ": energy_mJ,
                    })
                    state3_energy += energy_mJ

                if connected_idle_duration > 0:
                    config_2 = config.copy()
                    config_2.update({
                        "vlc_payload": 0.0,
                        "ble_payload": 0.0,
                        "comm_mode": COMM_MODE_CONNECTED,
                    })
                    input_data = build_input_dataframe(state=3, config=config_2)
                    current_uA = self._predict_current(input_data)
                    current_mA = current_uA / 1000.0
                    energy_mJ = current_mA * voltage * connected_idle_duration
                    state3_info["substates"].append({
                        "name": "BLE connected idle",
                        "duration_s": connected_idle_duration,
                        "current_uA": current_uA,
                        "energy_mJ": energy_mJ,
                    })
                    state3_energy += energy_mJ

                if ble_tx_duration > 0 and config["ble_payload"] > 0:
                    config_3 = config.copy()
                    config_3.update({
                        "vlc_payload": 0.0,
                        "ble_payload": config["ble_payload"],
                        "comm_mode": COMM_MODE_CONNECTED,
                    })
                    input_data = build_input_dataframe(state=3, config=config_3)
                    current_uA = self._predict_current(input_data)
                    current_mA = current_uA / 1000.0
                    energy_mJ = current_mA * voltage * ble_tx_duration
                    state3_info["substates"].append({
                        "name": "BLE TX",
                        "duration_s": ble_tx_duration,
                        "current_uA": current_uA,
                        "energy_mJ": energy_mJ,
                    })
                    state3_energy += energy_mJ

                if vlc_tx_duration > 0 and config["vlc_payload"] > 0:
                    config_4 = config.copy()
                    config_4.update({
                        "vlc_payload": config["vlc_payload"],
                        "ble_payload": 0.0,
                        "comm_mode": COMM_MODE_CONNECTED,
                    })
                    input_data = build_input_dataframe(state=3, config=config_4)
                    current_uA = self._predict_current(input_data)
                    current_mA = current_uA / 1000.0
                    energy_mJ = current_mA * voltage * vlc_tx_duration
                    state3_info["substates"].append({
                        "name": "VLC TX",
                        "duration_s": vlc_tx_duration,
                        "current_uA": current_uA,
                        "energy_mJ": energy_mJ,
                    })
                    state3_energy += energy_mJ

                state3_info["energy_mJ"] = state3_energy
                total_energy += state3_energy

            if state3_total > 0:
                state3_info["avg_current_uA"] = (state3_info["energy_mJ"] / (voltage * state3_total)) * 1000.0

        breakdown["states"][3] = state3_info
        breakdown["total_energy_mJ"] = total_energy

        if budget_enabled:
            breakdown["within_budget"] = (total_energy <= budget_mJ + 1e-9)
            breakdown["budget_margin_mJ"] = budget_mJ - total_energy
        else:
            breakdown["within_budget"] = None
            breakdown["budget_margin_mJ"] = None

        return breakdown

    def calculate_total_energy(self, config):
        return float(self.get_breakdown(config)["total_energy_mJ"])

    def evaluate_config(self, config, constraints):
        breakdown = self.get_breakdown(config)
        budget_mJ = self.get_energy_budget_mJ()
        budget_enabled = budget_mJ is not None

        if not breakdown["valid"]:
            return {
                "valid": False,
                "score": -1e18,
                "energy": float("inf"),
                "work_score": -1e18,
                "within_budget": None if not budget_enabled else False,
                "budget_enabled": budget_enabled,
                "breakdown": breakdown,
            }

        energy_mJ = float(breakdown["total_energy_mJ"])
        within_budget = None if not budget_enabled else (energy_mJ <= budget_mJ + 1e-9)
        work_score = self.get_useful_work_score(config, constraints)

        mode = self.objective_settings.objective_mode
        score = -energy_mJ

        if mode == OBJECTIVE_MIN_ENERGY:
            score = -energy_mJ
            if budget_enabled and not within_budget:
                score -= 1e9 + 1000.0 * max(0.0, energy_mJ - budget_mJ)

        elif mode == OBJECTIVE_MAX_WORK_UNDER_BUDGET:
            if budget_enabled:
                if within_budget:
                    score = 1e6 * work_score - energy_mJ
                else:
                    score = -1e9 - 1000.0 * max(0.0, energy_mJ - budget_mJ)
            else:
                score = 1e6 * work_score - energy_mJ

        elif mode == OBJECTIVE_BALANCED:
            max_possible_work = (
                self.objective_settings.ble_weight
                + self.objective_settings.vlc_weight
                + max(0.0, self.objective_settings.connected_bonus)
            )
            work_norm = work_score / max(max_possible_work, 1e-9)

            if budget_enabled:
                energy_norm = energy_mJ / max(budget_mJ, 1e-9)
                if within_budget:
                    score = 1000.0 * work_norm - 100.0 * self.objective_settings.energy_weight * energy_norm
                else:
                    score = -1e9 - 1000.0 * max(0.0, energy_mJ - budget_mJ)
            else:
                energy_ref_mJ = self._get_no_budget_energy_reference_mJ(constraints)
                energy_norm = energy_mJ / max(energy_ref_mJ, 1e-9)
                score = 1000.0 * work_norm - 100.0 * self.objective_settings.energy_weight * energy_norm

        return {
            "valid": True,
            "score": float(score),
            "energy": energy_mJ,
            "work_score": float(work_score),
            "within_budget": within_budget,
            "budget_enabled": budget_enabled,
            "breakdown": breakdown,
        }


class OptimizationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(
        self,
        model_data,
        constraints,
        algorithm,
        state_durations,
        ble_connected_duration=0,
        budget_settings=None,
        objective_settings=None,
    ):
        super().__init__()
        self.model_data = model_data
        self.constraints = constraints
        self.algorithm = algorithm
        self.state_durations = state_durations
        self.ble_connected_duration = ble_connected_duration
        self.is_running = True

        self.calculator = EnergyCalculator(
            model_data=self.model_data,
            state_durations=self.state_durations,
            ble_connected_duration=self.ble_connected_duration,
            budget_settings=budget_settings,
            objective_settings=objective_settings,
        )

    def run(self):
        try:
            if self.algorithm == "Grid Search":
                results = self.grid_search_optimization()
            elif self.algorithm == "Random Search":
                results = self.random_search_optimization()
            elif self.algorithm == "Differential Evolution":
                results = self.differential_evolution_optimization()
            else:
                results = self.grid_search_optimization()

            self.finished.emit(results)

        except Exception as e:
            print(f"Optimization error: {e}")
            traceback.print_exc()
            self.finished.emit([])

    def stop(self):
        self.is_running = False

    def _get_allowed_comm_modes(self):
        mode_min, mode_max = self.constraints["comm_mode"]
        modes = []

        if mode_min <= COMM_MODE_DISCONNECTED <= mode_max:
            modes.append(COMM_MODE_DISCONNECTED)
        if mode_min <= COMM_MODE_CONNECTED <= mode_max:
            modes.append(COMM_MODE_CONNECTED)

        if not modes:
            midpoint = (mode_min + mode_max) / 2.0
            modes = [COMM_MODE_DISCONNECTED if midpoint < 1.5 else COMM_MODE_CONNECTED]

        return modes

    def _normalize_config(self, config):
        return self.calculator._normalize_config(config)

    def _candidate_result(self, config):
        out = self.calculator.evaluate_config(config, self.constraints)
        if not out["valid"]:
            return None
        return {
            "config": config,
            "energy": float(out["energy"]),
            "score": float(out["score"]),
            "work_score": float(out["work_score"]),
            "within_budget": out["within_budget"],
            "budget_enabled": bool(out.get("budget_enabled", False)),
        }

    def _sort_results(self, results):
        return sorted(
            results,
            key=lambda x: (
                -float(x.get("score", -1e18)),
                float(x.get("energy", float("inf"))),
            )
        )

    def grid_search_optimization(self):
        self.status_update.emit("Starting Grid Search...")

        txpower_range = np.linspace(
            self.constraints["txpower"][0],
            self.constraints["txpower"][1],
            self.constraints["txpower_steps"],
        )
        adv_range = np.linspace(
            self.constraints["adv_interval"][0],
            self.constraints["adv_interval"][1],
            self.constraints["adv_interval_steps"],
        )
        conn_range = np.linspace(
            self.constraints["conn_interval"][0],
            self.constraints["conn_interval"][1],
            self.constraints["conn_interval_steps"],
        )
        vlc_range = np.linspace(
            self.constraints["vlc_payload"][0],
            self.constraints["vlc_payload"][1],
            self.constraints["vlc_payload_steps"],
        )
        ble_range = np.linspace(
            self.constraints["ble_payload"][0],
            self.constraints["ble_payload"][1],
            self.constraints["ble_payload_steps"],
        )
        comm_modes = self._get_allowed_comm_modes()

        total = (
            len(txpower_range) * len(adv_range) * len(conn_range) *
            len(vlc_range) * len(ble_range) * len(comm_modes)
        )

        self.status_update.emit(f"Evaluating {total} raw configurations...")

        results = []
        count = 0

        for txp in txpower_range:
            for adv in adv_range:
                for conn in conn_range:
                    for vlc in vlc_range:
                        for ble in ble_range:
                            for comm in comm_modes:
                                if not self.is_running:
                                    return self._sort_results(results)

                                config = self._normalize_config({
                                    "txpower": txp,
                                    "adv_interval": adv,
                                    "conn_interval": conn,
                                    "vlc_payload": vlc,
                                    "ble_payload": ble,
                                    "comm_mode": comm,
                                })

                                candidate = self._candidate_result(config)
                                if candidate is not None:
                                    results.append(candidate)

                                count += 1
                                if count % max(1, total // 100) == 0:
                                    self.progress.emit(int((count / total) * 100))

        return self._sort_results(results)

    def random_search_optimization(self):
        self.status_update.emit("Starting Random Search...")

        n_samples = 1000
        max_attempts = max(2000, 5 * n_samples)
        allowed_modes = self._get_allowed_comm_modes()

        results = []
        attempts = 0

        while len(results) < n_samples and attempts < max_attempts:
            if not self.is_running:
                return self._sort_results(results)

            config = self._normalize_config({
                "txpower": np.random.uniform(*self.constraints["txpower"]),
                "adv_interval": np.random.uniform(*self.constraints["adv_interval"]),
                "conn_interval": np.random.uniform(*self.constraints["conn_interval"]),
                "vlc_payload": np.random.uniform(*self.constraints["vlc_payload"]),
                "ble_payload": np.random.uniform(*self.constraints["ble_payload"]),
                "comm_mode": np.random.choice(allowed_modes),
            })

            candidate = self._candidate_result(config)
            if candidate is not None:
                results.append(candidate)

            attempts += 1
            if attempts % 10 == 0:
                progress = int((len(results) / n_samples) * 100)
                self.progress.emit(min(progress, 100))
                self.status_update.emit(
                    f"Generated {len(results)}/{n_samples} valid configurations after {attempts} attempts..."
                )

        return self._sort_results(results)

    def differential_evolution_optimization(self):
        self.status_update.emit("Starting Differential Evolution...")

        allowed_modes = self._get_allowed_comm_modes()
        mode_min = min(allowed_modes)
        mode_max = max(allowed_modes)

        bounds = [
            self.constraints["txpower"],
            self.constraints["adv_interval"],
            self.constraints["conn_interval"],
            self.constraints["vlc_payload"],
            self.constraints["ble_payload"],
            (mode_min, mode_max),
        ]

        def objective(x):
            if not self.is_running:
                return 1e12

            if len(allowed_modes) == 1:
                mode = allowed_modes[0]
            else:
                mode = COMM_MODE_DISCONNECTED if x[5] < 1.5 else COMM_MODE_CONNECTED

            config = self._normalize_config({
                "txpower": x[0],
                "adv_interval": x[1],
                "conn_interval": x[2],
                "vlc_payload": x[3],
                "ble_payload": x[4],
                "comm_mode": mode,
            })

            out = self.calculator.evaluate_config(config, self.constraints)
            return -float(out["score"])

        def callback(xk, convergence):
            try:
                progress = int(max(0, min(100, (1 - convergence) * 100)))
                self.progress.emit(progress)
            except Exception:
                pass
            return not self.is_running

        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            callback=callback,
            polish=True,
        )

        if len(allowed_modes) == 1:
            best_mode = allowed_modes[0]
        else:
            best_mode = COMM_MODE_DISCONNECTED if result.x[5] < 1.5 else COMM_MODE_CONNECTED

        best_config = self._normalize_config({
            "txpower": result.x[0],
            "adv_interval": result.x[1],
            "conn_interval": result.x[2],
            "vlc_payload": result.x[3],
            "ble_payload": result.x[4],
            "comm_mode": best_mode,
        })

        results = []
        candidate = self._candidate_result(best_config)
        if candidate is not None:
            results.append(candidate)

        for _ in range(50):
            varied_config = best_config.copy()
            varied_config["txpower"] += np.random.normal(0, 0.5)
            varied_config["adv_interval"] += np.random.normal(0, 5)
            varied_config["conn_interval"] += np.random.normal(0, 10)

            varied_config["txpower"] = float(np.clip(varied_config["txpower"], *bounds[0]))
            varied_config["adv_interval"] = float(np.clip(varied_config["adv_interval"], *bounds[1]))
            varied_config["conn_interval"] = float(np.clip(varied_config["conn_interval"], *bounds[2]))

            candidate = self._candidate_result(varied_config)
            if candidate is not None:
                results.append(candidate)

        return self._sort_results(results)


class MedalWidget(QWidget):
    def __init__(self, rank, config, energy, score, work_score,
                 baseline_energy=None, within_budget=None, budget_enabled=False):
        super().__init__()
        self.rank = rank
        self.config = config
        self.energy = energy
        self.score = score
        self.work_score = work_score
        self.baseline_energy = baseline_energy
        self.within_budget = within_budget
        self.budget_enabled = budget_enabled
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        medal_icon = medals.get(self.rank, "🏅")

        ble_status = "BLE Connected" if self.config["comm_mode"] == COMM_MODE_CONNECTED else "BLE Disconnected"

        config_str = (
            f"TxPower={self.config['txpower']:.1f}dBm, "
            f"AdvInt={self.config['adv_interval']:.0f}ms, "
            f"ConnInt={self.config['conn_interval']:.0f}ms, "
            f"VLC={self.config['vlc_payload']:.0f}B, "
            f"BLE={self.config['ble_payload']:.0f}B, "
            f"{ble_status}"
        )

        saves_str = ""
        if self.baseline_energy and self.baseline_energy > 0:
            savings = ((self.baseline_energy - self.energy) / self.baseline_energy) * 100
            saves_str = f", Saves {savings:.1f}%"

        status_parts = [
            f"Work Score {self.work_score:.4f}",
            f"Objective Score {self.score:.4f}",
        ]
        if self.budget_enabled:
            status_parts.append("Within budget" if self.within_budget else "Over budget")
        else:
            status_parts.append("Budget check: disabled")

        result_text = (
            f"Top {self.rank}: {medal_icon}\n"
            f"{config_str}\n"
            f"Energy {self.energy:.2f}mJ{saves_str}\n"
            f"{' | '.join(status_parts)}"
        )

        label = QLabel(result_text)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
                padding: 2px;
            }
        """)

        layout.addWidget(label)
        self.setLayout(layout)


class BreakdownWidget(QWidget):
    def __init__(self, breakdown):
        super().__init__()
        self.breakdown = breakdown
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)

        title = QLabel("🔍 Best Configuration Breakdown")
        title.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #006600;
            padding: 6px 0px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        text = QLabel(self._format_breakdown_text())
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
                padding: 6px;
                line-height: 1.35;
            }
        """)
        layout.addWidget(text)

        self.setLayout(layout)

    def _format_breakdown_text(self):
        b = self.breakdown
        if not b.get("valid", False):
            return f"Invalid breakdown: {b.get('reason', 'Unknown reason')}"

        states = b["states"]
        lines = []
        lines.append(f"Total Energy = {b['total_energy_mJ']:.2f} mJ")

        budget = b.get("budget", {})
        if budget.get("enabled", False):
            lines.append(
                f"Energy Budget = {budget['budget_mJ']:.2f} mJ "
                f"(Capacitor {budget['cap_energy_mJ']:.2f} mJ + Harvested {budget['harvest_energy_mJ']:.2f} mJ)"
            )
            lines.append(
                f"Budget Margin = {b.get('budget_margin_mJ', 0.0):.2f} mJ | "
                f"Within Budget = {b.get('within_budget', False)}"
            )
        else:
            lines.append("Budget check: disabled")

        lines.append("")

        for state in [1, 2]:
            s = states[state]
            lines.append(
                f"{s['label']}: Duration={s['duration_s']:.4f}s, "
                f"Current={s['current_uA']:.2f}µA, "
                f"Energy={s['energy_mJ']:.4f}mJ"
            )

        s3 = states[3]
        lines.append(
            f"{s3['label']}: Duration={s3['duration_s']:.4f}s, "
            f"Avg Current={s3['avg_current_uA']:.2f}µA, "
            f"Energy={s3['energy_mJ']:.4f}mJ"
        )

        for sub in s3["substates"]:
            lines.append(
                f"  • {sub['name']}: Duration={sub['duration_s']:.4f}s, "
                f"Current={sub['current_uA']:.2f}µA, "
                f"Energy={sub['energy_mJ']:.4f}mJ"
            )

        s4 = states[4]
        lines.append(
            f"{s4['label']}: Duration={s4['duration_s']:.4f}s, "
            f"Current={s4['current_uA']:.2f}µA, "
            f"Energy={s4['energy_mJ']:.4f}mJ"
        )

        return "\n".join(lines)


class EnergyOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ Energy Consumption Optimizer")
        self.setGeometry(100, 100, 1280, 820)

        self.models = {}
        self.current_model = None
        self.optimization_worker = None
        self.is_optimizing = False
        self.optimization_results = []
        self.baseline_energy = None
        self.best_breakdown = None

        self.load_available_models()
        self.init_ui()

    def load_available_models(self):
        model_patterns = [
            "*_enhanced_model.joblib",
            "current_prediction_enhanced_model.joblib",
            "*_model.joblib",
            "current_prediction_model.joblib",
            "*.pkl",
            "*.pickle",
        ]

        print("Loading trained models...")
        models_loaded = 0
        errors = []
        already_seen = set()

        for pattern in model_patterns:
            model_files = glob.glob(pattern)
            for model_file in model_files:
                if model_file in already_seen:
                    continue
                already_seen.add(model_file)

                try:
                    raw_obj, loader_used = safe_load_model_file(model_file)
                    model_data = normalize_loaded_model_payload(raw_obj, model_file)
                    pred = validate_loaded_model(model_data)

                    base_name = os.path.splitext(model_file)[0]
                    if "current_prediction" in base_name:
                        if "enhanced" in base_name:
                            model_name = "Best Model (7-Features)"
                        else:
                            model_name = "Best Model (3-Features)"
                    else:
                        model_name = base_name.replace("_model", "").replace("_", " ").title()

                    original_name = model_name
                    suffix = 2
                    while model_name in self.models:
                        model_name = f"{original_name} [{suffix}]"
                        suffix += 1

                    self.models[model_name] = model_data
                    models_loaded += 1

                    r2_score = model_data.get("r2_score", "N/A")
                    print(f"✓ Loaded: {model_name} via {loader_used} (R² = {r2_score}) | test pred = {pred:.3f}")

                    if self.current_model is None:
                        self.current_model = model_data
                        print("  -> Set as default model")

                except Exception as e:
                    msg = f"{model_file}: {e}"
                    errors.append(msg)
                    print(f"✗ Failed to load {msg}")

        if models_loaded == 0:
            error_text = (
                "No trained models found or no compatible models could be loaded.\n\n"
                "Common reasons:\n"
                "- incompatible sklearn/pickle/joblib environment\n"
                "- file is not in expected model format\n"
                "- file loads but predict() validation fails\n\n"
                "First errors:\n"
                + "\n".join(errors[:8])
            )
            QMessageBox.critical(self, "Error", error_text)
            sys.exit(1)

        print(f"Successfully loaded {models_loaded} model(s)!")

    def create_custom_title_bar(self, layout):
        title_bar = QWidget()
        title_bar.setFixedHeight(50)
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-bottom: 1px solid #d0d0d0;
            }
        """)

        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 8, 15, 8)

        logo_label = QLabel()
        try:
            pixmap = QPixmap("icon/icon.png")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setText("⚡")
                logo_label.setStyleSheet("font-size: 24px; color: black;")
        except Exception:
            logo_label.setText("⚡")
            logo_label.setStyleSheet("font-size: 24px; color: black;")

        logo_label.setFixedSize(34, 34)
        title_layout.addWidget(logo_label)
        title_layout.addStretch(1)

        title_label = QLabel("Energy Consumption Optimizer")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: black;
            font-size: 16px;
            font-weight: bold;
            margin: 0px;
            padding: 0px;
        """)
        title_layout.addWidget(title_label)

        title_layout.addStretch(1)
        spacer_label = QLabel()
        spacer_label.setFixedSize(34, 34)
        title_layout.addWidget(spacer_label)
        layout.addWidget(title_bar)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.create_custom_title_bar(main_layout)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(content_widget)

        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 62)

        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 38)

        self.setStyleSheet("""
            QMainWindow { background-color: #F5F5F5; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(self.create_model_selection_group())
        scroll_layout.addWidget(self.create_state_durations_group())
        scroll_layout.addWidget(self.create_constraints_group())
        scroll_layout.addWidget(self.create_objective_group())
        scroll_layout.addWidget(self.create_energy_budget_group())
        scroll_layout.addWidget(self.create_display_options_group())
        scroll_layout.addWidget(self.create_algorithm_group())
        scroll_layout.addWidget(self.create_control_group())
        scroll_layout.addStretch()

        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        panel.setLayout(layout)
        return panel

    def create_model_selection_group(self):
        group = QGroupBox("🤖 Model Selection")
        layout = QVBoxLayout()

        # self.model_combo = QComboBox()
        # for model_name, model_data in self.models.items():
        #     r2_score = model_data.get("r2_score", 0)
        #     try:
        #         text = f"{model_name} (R²={float(r2_score):.4f})"
        #     except Exception:
        #         text = f"{model_name} (R²={r2_score})"
        #     self.model_combo.addItem(text)
            
        self.model_combo = QComboBox()
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)
######################################################################################
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo)
        group.setLayout(layout)
        return group

    def create_state_durations_group(self):
        group = QGroupBox("⏱️ State Durations (seconds)")
        layout = QGridLayout()

        self.state_duration_spins = {}
        states = [
            (1, "State 1 (Initialization)", 1.0),
            (2, "State 2 (Advertising)", 1.0),
            (3, "State 3 (Operation)", 1.0),
            (4, "State 4 (Sleep)", 1.0),
        ]

        row = 0
        for state_num, state_name, default_val in states:
            label = QLabel(state_name)
            spin = QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setDecimals(4)
            spin.setValue(default_val)
            spin.setSuffix(" s")

            self.state_duration_spins[state_num] = spin
            layout.addWidget(label, row, 0)
            layout.addWidget(spin, row, 1)
            row += 1

            if state_num == 3:
                ble_connected_label = QLabel("  └─ State 3 BLE Connected Duration")
                ble_connected_label.setStyleSheet("color: #666; font-size: 11px;")

                self.ble_connected_duration_spin = QDoubleSpinBox()
                self.ble_connected_duration_spin.setRange(0, 100000)
                self.ble_connected_duration_spin.setDecimals(4)
                self.ble_connected_duration_spin.setValue(0.5)
                self.ble_connected_duration_spin.setSuffix(" s")
                self.ble_connected_duration_spin.setToolTip(
                    "Duration of BLE connected time within State 3 (must be ≤ State 3 duration)"
                )

                layout.addWidget(ble_connected_label, row, 0)
                layout.addWidget(self.ble_connected_duration_spin, row, 1)
                row += 1

        group.setLayout(layout)
        return group

    def create_constraints_group(self):
        group = QGroupBox("📊 Parameter Constraints")
        main_layout = QVBoxLayout()

        steps_explanation = QLabel(
            "💡 <b>Steps</b>: Number of values to test between Min and Max. "
            "Example: Min=0, Max=100, Steps=5 → tests [0, 25, 50, 75, 100]"
        )
        steps_explanation.setStyleSheet("color: #0066CC; font-size: 10px; padding: 5px;")
        steps_explanation.setWordWrap(True)
        main_layout.addWidget(steps_explanation)

        layout = QGridLayout()
        self.constraint_inputs = {}

        parameters = [
            ("txpower", "Tx Power", -4, 4, 3, " dBm"),
            ("adv_interval", "Advertising Interval", 40, 150, 3, " ms"),
            ("conn_interval", "Connection Interval", 50, 500, 3, " ms"),
            ("vlc_payload", "VLC Payload", 0, 100, 2, " bytes"),
            ("ble_payload", "BLE Payload", 0, 100, 2, " bytes"),
            ("comm_mode", "BLE Disconnected/Connected", 1, 2, 2, ""),
        ]

        row = 0
        for param_key, param_label, min_default, max_default, step_default, suffix in parameters:
            label = QLabel(f"{param_label}:")
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row, 0)

            min_label = QLabel("Min:")
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-100, 100000)
            min_spin.setValue(min_default)
            min_spin.setSuffix(suffix)
            min_spin.setMaximumWidth(100)
            layout.addWidget(min_label, row, 1)
            layout.addWidget(min_spin, row, 2)

            max_label = QLabel("Max:")
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-100, 100000)
            max_spin.setValue(max_default)
            max_spin.setSuffix(suffix)
            max_spin.setMaximumWidth(100)
            layout.addWidget(max_label, row, 3)
            layout.addWidget(max_spin, row, 4)

            steps_label = QLabel("Steps:")
            steps_spin = QSpinBox()
            steps_spin.setRange(1, 100)
            steps_spin.setValue(step_default)
            steps_spin.setMaximumWidth(60)
            layout.addWidget(steps_label, row, 5)
            layout.addWidget(steps_spin, row, 6)

            self.constraint_inputs[param_key] = {
                "min": min_spin,
                "max": max_spin,
                "steps": steps_spin,
            }
            row += 1

        main_layout.addLayout(layout)
        group.setLayout(main_layout)
        return group

    def create_objective_group(self):
        group = QGroupBox("🎯 Optimization Formulation")
        layout = QGridLayout()

        layout.addWidget(QLabel("Goal:"), 0, 0)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            OBJECTIVE_BALANCED,
            OBJECTIVE_MAX_WORK_UNDER_BUDGET,
            OBJECTIVE_MIN_ENERGY,
        ])
        layout.addWidget(self.objective_combo, 0, 1, 1, 3)

        help_label = QLabel(
            "Recommended formulation: maximize useful work (BLE/VLC payload utility) while respecting the energy budget, "
            "and use energy only as a secondary penalty."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 10px; padding: 4px;")
        layout.addWidget(help_label, 1, 0, 1, 4)

        layout.addWidget(QLabel("BLE weight:"), 2, 0)
        self.ble_weight_spin = QDoubleSpinBox()
        self.ble_weight_spin.setRange(0, 100)
        self.ble_weight_spin.setDecimals(3)
        self.ble_weight_spin.setValue(1.0)
        layout.addWidget(self.ble_weight_spin, 2, 1)

        layout.addWidget(QLabel("VLC weight:"), 2, 2)
        self.vlc_weight_spin = QDoubleSpinBox()
        self.vlc_weight_spin.setRange(0, 100)
        self.vlc_weight_spin.setDecimals(3)
        self.vlc_weight_spin.setValue(1.0)
        layout.addWidget(self.vlc_weight_spin, 2, 3)

        layout.addWidget(QLabel("Energy penalty weight:"), 3, 0)
        self.energy_weight_spin = QDoubleSpinBox()
        self.energy_weight_spin.setRange(0, 100)
        self.energy_weight_spin.setDecimals(3)
        self.energy_weight_spin.setValue(0.35)
        layout.addWidget(self.energy_weight_spin, 3, 1)

        layout.addWidget(QLabel("Connected bonus:"), 3, 2)
        self.connected_bonus_spin = QDoubleSpinBox()
        self.connected_bonus_spin.setRange(0, 100)
        self.connected_bonus_spin.setDecimals(3)
        self.connected_bonus_spin.setValue(0.05)
        layout.addWidget(self.connected_bonus_spin, 3, 3)

        group.setLayout(layout)
        return group

    def create_energy_budget_group(self):
        group = QGroupBox("🔋 Energy Budget")
        layout = QGridLayout()

        self.enable_budget_checkbox = QCheckBox("Enable energy budget from supercapacitor + OPV harvesting")
        self.enable_budget_checkbox.setChecked(True)
        self.enable_budget_checkbox.stateChanged.connect(self._update_budget_ui_state)
        layout.addWidget(self.enable_budget_checkbox, 0, 0, 1, 4)

        layout.addWidget(QLabel("Capacitance:"), 1, 0)
        self.capacitance_spin = QDoubleSpinBox()
        self.capacitance_spin.setRange(0, 100000)
        self.capacitance_spin.setDecimals(4)
        self.capacitance_spin.setValue(38.4)
        self.capacitance_spin.setSuffix(" mF")
        layout.addWidget(self.capacitance_spin, 1, 1)

        layout.addWidget(QLabel("Start Voltage:"), 1, 2)
        self.v_start_spin = QDoubleSpinBox()
        self.v_start_spin.setRange(0, 10)
        self.v_start_spin.setDecimals(4)
        self.v_start_spin.setValue(3.3)
        self.v_start_spin.setSuffix(" V")
        layout.addWidget(self.v_start_spin, 1, 3)

        layout.addWidget(QLabel("Minimum Voltage:"), 2, 0)
        self.v_min_spin = QDoubleSpinBox()
        self.v_min_spin.setRange(0, 10)
        self.v_min_spin.setDecimals(4)
        self.v_min_spin.setValue(1.6)
        self.v_min_spin.setSuffix(" V")
        layout.addWidget(self.v_min_spin, 2, 1)

        layout.addWidget(QLabel("OPV Area:"), 2, 2)
        self.opv_area_spin = QDoubleSpinBox()
        self.opv_area_spin.setRange(0, 100000)
        self.opv_area_spin.setDecimals(4)
        self.opv_area_spin.setValue(1.0)
        self.opv_area_spin.setSuffix(" cm²")
        layout.addWidget(self.opv_area_spin, 2, 3)

        layout.addWidget(QLabel("OPV Power Density:"), 3, 0)
        self.opv_power_density_spin = QDoubleSpinBox()
        self.opv_power_density_spin.setRange(0, 10000000)
        self.opv_power_density_spin.setDecimals(4)
        self.opv_power_density_spin.setValue(0.0)
        self.opv_power_density_spin.setSuffix(" µW/cm²")
        layout.addWidget(self.opv_power_density_spin, 3, 1)

        layout.addWidget(QLabel("Harvest Efficiency:"), 3, 2)
        self.harvest_efficiency_spin = QDoubleSpinBox()
        self.harvest_efficiency_spin.setRange(0, 1)
        self.harvest_efficiency_spin.setDecimals(4)
        self.harvest_efficiency_spin.setSingleStep(0.05)
        self.harvest_efficiency_spin.setValue(1.0)
        layout.addWidget(self.harvest_efficiency_spin, 3, 3)

        note = QLabel(
            "Budget = 0.5·C·(Vstart² − Vmin²) + OPV_power_density·OPV_area·mission_time·efficiency"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-size: 10px; padding: 4px;")
        layout.addWidget(note, 4, 0, 1, 4)

        group.setLayout(layout)
        self._update_budget_ui_state()
        return group

    def create_display_options_group(self):
        group = QGroupBox("🖥️ Display Options")
        layout = QVBoxLayout()
        self.show_breakdown_checkbox = QCheckBox("Show best configuration energy breakdown")
        self.show_breakdown_checkbox.setChecked(True)
        self.show_breakdown_checkbox.stateChanged.connect(self._refresh_results_if_available)
        layout.addWidget(self.show_breakdown_checkbox)
        group.setLayout(layout)
        return group

    def create_algorithm_group(self):
        group = QGroupBox("🔬 Optimization Algorithm")
        layout = QVBoxLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Grid Search (Exhaustive)",
            "Random Search (Fast)",
            "Differential Evolution (Smart)",
        ])
        layout.addWidget(self.algorithm_combo)

        self.algo_desc = QLabel()
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setStyleSheet("color: #666; font-style: italic;")
        self.update_algorithm_description()
        self.algorithm_combo.currentIndexChanged.connect(self.update_algorithm_description)
        layout.addWidget(self.algo_desc)

        group.setLayout(layout)
        return group

    def update_algorithm_description(self):
        descriptions = {
            0: "Tries all possible combinations within constraints. Most thorough but slowest.",
            1: "Randomly samples valid configurations. Faster, good balance of speed and quality.",
            2: "Uses evolutionary search to find good valid solutions efficiently.",
        }
        self.algo_desc.setText(descriptions.get(self.algorithm_combo.currentIndex(), ""))

    def create_control_group(self):
        group = QGroupBox("🚀 Optimization Control")
        layout = QVBoxLayout()

        button_container = QHBoxLayout()
        button_container.addStretch()

        self.optimize_button = QPushButton("▶️ Start Optimization")
        self.optimize_button.setFixedWidth(190)
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12px;
                padding: 8px 15px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.optimize_button.clicked.connect(self.toggle_optimization)
        button_container.addWidget(self.optimize_button)
        button_container.addStretch()
        layout.addLayout(button_container)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #CCC;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk { background-color: #4CAF50; }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to optimize")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        group.setLayout(layout)
        return group

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                background-color: #FAFAFA;
            }
        """)

        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setSpacing(15)
        self.results_layout.setContentsMargins(10, 10, 10, 10)

        initial_msg = QLabel("🎯 Start optimization to see results")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setStyleSheet("""
            font-size: 12px;
            color: #999;
            padding: 10px;
        """)
        initial_msg.setWordWrap(True)
        self.results_layout.addWidget(initial_msg)
        self.results_layout.addStretch()

        self.results_widget.setLayout(self.results_layout)
        scroll.setWidget(self.results_widget)
        layout.addWidget(scroll)
        panel.setLayout(layout)
        return panel

    def _update_budget_ui_state(self):
        enabled = self.enable_budget_checkbox.isChecked()
        for w in [
            self.capacitance_spin,
            self.v_start_spin,
            self.v_min_spin,
            self.opv_area_spin,
            self.opv_power_density_spin,
            self.harvest_efficiency_spin,
        ]:
            w.setEnabled(enabled)

    def _refresh_results_if_available(self):
        if self.optimization_results:
            self.display_top_results()

    # def on_model_changed(self):
    #     model_text = self.model_combo.currentText()
    #     model_name = model_text.split(" (R²=")[0]
    #     self.current_model = self.models.get(model_name)
    #     print(f"Selected model: {model_name}")

    def on_model_changed(self):
        model_text = self.model_combo.currentText()
        model_name = model_text.split(" (R²=")[0]
        self.current_model = self.models.get(model_name)
        print(f"Selected model: {model_name}")


    def _read_budget_settings(self):
        return BudgetSettings(
            enabled=self.enable_budget_checkbox.isChecked(),
            capacitance_mF=self.capacitance_spin.value(),
            v_start=self.v_start_spin.value(),
            v_min=self.v_min_spin.value(),
            opv_area_cm2=self.opv_area_spin.value(),
            opv_power_density_uW_cm2=self.opv_power_density_spin.value(),
            harvest_efficiency=self.harvest_efficiency_spin.value(),
        )

    def _read_objective_settings(self):
        return ObjectiveSettings(
            objective_mode=self.objective_combo.currentText(),
            ble_weight=self.ble_weight_spin.value(),
            vlc_weight=self.vlc_weight_spin.value(),
            energy_weight=self.energy_weight_spin.value(),
            connected_bonus=self.connected_bonus_spin.value(),
        )

    def toggle_optimization(self):
        if self.is_optimizing:
            self.stop_optimization()
        else:
            self.start_optimization()

    def start_optimization(self):
        constraints = {}
        for param_key, inputs in self.constraint_inputs.items():
            min_val = inputs["min"].value()
            max_val = inputs["max"].value()
            if min_val > max_val:
                QMessageBox.warning(self, "Warning", f"Invalid range for {param_key}: Min > Max")
                return
            constraints[param_key] = (min_val, max_val)
            constraints[f"{param_key}_steps"] = inputs["steps"].value()

        state_durations = {state_num: spin.value() for state_num, spin in self.state_duration_spins.items()}
        algorithm = self.algorithm_combo.currentText().split(" (")[0]

        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model selected!")
            return

        if sum(state_durations.values()) == 0:
            QMessageBox.warning(self, "Warning", "Total duration is zero!")
            return

        if self.ble_connected_duration_spin.value() > state_durations[3]:
            QMessageBox.warning(self, "Warning", "State 3 BLE Connected Duration cannot exceed State 3 duration.")
            return

        budget_settings = self._read_budget_settings()
        objective_settings = self._read_objective_settings()

        if budget_settings.enabled and budget_settings.v_start < budget_settings.v_min:
            QMessageBox.warning(self, "Warning", "Start voltage should be greater than or equal to minimum voltage.")
            return

        self.is_optimizing = True
        self.optimize_button.setText("⏹️ Stop Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        self.progress_bar.setValue(0)
        self.status_label.setText("Optimizing...")

        self.optimization_worker = OptimizationWorker(
            self.current_model,
            constraints,
            algorithm,
            state_durations,
            self.ble_connected_duration_spin.value(),
            budget_settings=budget_settings,
            objective_settings=objective_settings,
        )

        self.optimization_worker.progress.connect(self.update_progress)
        self.optimization_worker.status_update.connect(self.update_status)
        self.optimization_worker.finished.connect(self.on_optimization_finished)
        self.optimization_worker.start()

        print(f"Starting {algorithm} optimization...")
        print(f"Constraints: {constraints}")
        print(f"State durations: {state_durations}")
        print(f"Budget settings: {budget_settings.to_dict()}")
        print(f"Objective settings: {objective_settings.to_dict()}")

    def stop_optimization(self):
        if self.optimization_worker:
            self.optimization_worker.stop()
            self.optimization_worker.wait()

        self.is_optimizing = False
        self.optimize_button.setText("▶️ Start Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.status_label.setText("Optimization stopped")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def _create_calculator_for_current_inputs(self):
        state_durations = {state_num: spin.value() for state_num, spin in self.state_duration_spins.items()}
        return EnergyCalculator(
            model_data=self.current_model,
            state_durations=state_durations,
            ble_connected_duration=self.ble_connected_duration_spin.value(),
            budget_settings=self._read_budget_settings(),
            objective_settings=self._read_objective_settings(),
        )

    def on_optimization_finished(self, results):
        self.is_optimizing = False
        self.optimize_button.setText("▶️ Start Optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

        if not results:
            self.status_label.setText("Optimization failed, cancelled, or no valid configurations were found")
            QMessageBox.warning(self, "Warning", "Optimization returned no valid results!")
            return

        self.optimization_results = results
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✅ Optimization complete! Found {len(results)} valid configurations")

        if len(results) > 0:
            self.baseline_energy = results[-1]["energy"]
            calculator = self._create_calculator_for_current_inputs()
            self.best_breakdown = calculator.get_breakdown(results[0]["config"])

        self.display_top_results()

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Score: {result['score']:.6f}")
            print(f"   Energy: {result['energy']:.2f} mJ")
            print(f"   Work score: {result['work_score']:.4f}")
            if result.get("budget_enabled", False):
                print(f"   Within budget: {result['within_budget']}")
            else:
                print("   Budget check: disabled")
            print(f"   Config: {result['config']}")

        if len(results) > 0:
            baseline_energy = results[-1]["energy"]
            baseline_config = results[-1]["config"]
            savings = ((baseline_energy - results[0]["energy"]) / baseline_energy) * 100 if baseline_energy > 0 else 0.0

            print("\n" + "=" * 80)
            print(f"BASELINE (Worst Valid Configuration - Rank #{len(results)}):")
            print("=" * 80)
            print(f"Energy: {baseline_energy:.2f} mJ")
            print(f"Score: {results[-1]['score']:.6f}")
            print(f"Work score: {results[-1]['work_score']:.4f}")
            if results[-1].get("budget_enabled", False):
                print(f"Within budget: {results[-1]['within_budget']}")
            else:
                print("Budget check: disabled")
            print(f"Config: {baseline_config}")
            print(f"\n✅ Improvement in energy vs worst valid config: {savings:.2f}%")

        if self.best_breakdown and self.best_breakdown.get("valid", False):
            print("\n" + "=" * 80)
            print("BEST CONFIGURATION BREAKDOWN")
            print("=" * 80)
            budget = self.best_breakdown.get("budget", {})
            if budget.get("enabled", False):
                print(
                    f"Budget = {budget['budget_mJ']:.2f} mJ "
                    f"(Capacitor {budget['cap_energy_mJ']:.2f} mJ + Harvested {budget['harvest_energy_mJ']:.2f} mJ)"
                )
                print(
                    f"Within budget = {self.best_breakdown.get('within_budget', False)}, "
                    f"Margin = {self.best_breakdown.get('budget_margin_mJ', 0.0):.2f} mJ"
                )
            else:
                print("Budget check: disabled")

            for state in [1, 2]:
                s = self.best_breakdown["states"][state]
                print(f"{s['label']}: duration={s['duration_s']:.4f}s, current={s['current_uA']:.2f}uA, energy={s['energy_mJ']:.4f}mJ")

            s3 = self.best_breakdown["states"][3]
            print(f"{s3['label']}: duration={s3['duration_s']:.4f}s, avg_current={s3['avg_current_uA']:.2f}uA, energy={s3['energy_mJ']:.4f}mJ")
            for sub in s3["substates"]:
                print(f"  - {sub['name']}: duration={sub['duration_s']:.4f}s, current={sub['current_uA']:.2f}uA, energy={sub['energy_mJ']:.4f}mJ")

            s4 = self.best_breakdown["states"][4]
            print(f"{s4['label']}: duration={s4['duration_s']:.4f}s, current={s4['current_uA']:.2f}uA, energy={s4['energy_mJ']:.4f}mJ")

    def create_baseline_widget(self, result, rank):
        config = result["config"]
        energy = result["energy"]
        score = result.get("score", 0.0)
        work_score = result.get("work_score", 0.0)
        within_budget = result.get("within_budget", None)
        budget_enabled = result.get("budget_enabled", False)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)

        ble_status = "BLE Connected" if config["comm_mode"] == COMM_MODE_CONNECTED else "BLE Disconnected"

        config_str = (
            f"TxPower={config['txpower']:.1f}dBm, "
            f"AdvInt={config['adv_interval']:.0f}ms, "
            f"ConnInt={config['conn_interval']:.0f}ms, "
            f"VLC={config['vlc_payload']:.0f}B, "
            f"BLE={config['ble_payload']:.0f}B, "
            f"{ble_status}"
        )

        status_parts = [
            f"Work Score {work_score:.4f}",
            f"Objective Score {score:.4f}",
        ]
        if budget_enabled:
            status_parts.append("Within budget" if within_budget else "Over budget")
        else:
            status_parts.append("Budget check: disabled")

        result_text = (
            f"Baseline: 😫\n"
            f"{config_str}\n"
            f"Energy {energy:.2f}mJ\n"
            f"{' | '.join(status_parts)}"
        )

        label = QLabel(result_text)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #333;
                padding: 8px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    def display_top_results(self):
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        top3_caption = QLabel("🏆 Top 3 Optimal Configurations")
        top3_caption.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #0066CC;
            padding: 8px;
            margin-bottom: 5px;
        """)
        top3_caption.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(top3_caption)

        for i, result in enumerate(self.optimization_results[:3], 1):
            medal_widget = MedalWidget(
                rank=i,
                config=result["config"],
                energy=result["energy"],
                score=result.get("score", 0.0),
                work_score=result.get("work_score", 0.0),
                baseline_energy=self.baseline_energy,
                within_budget=result.get("within_budget", None),
                budget_enabled=result.get("budget_enabled", False),
            )
            self.results_layout.addWidget(medal_widget)

            if i < min(3, len(self.optimization_results)):
                separator_mini = QFrame()
                separator_mini.setFrameShape(QFrame.HLine)
                separator_mini.setFrameShadow(QFrame.Sunken)
                separator_mini.setStyleSheet("background-color: #DDDDDD; max-height: 1px; margin: 5px 0px;")
                self.results_layout.addWidget(separator_mini)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #CCCCCC; max-height: 2px; margin: 10px 0px;")
        self.results_layout.addWidget(separator)

        baseline_caption = QLabel("😫 Baseline (Worst Valid Configuration)")
        baseline_caption.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #CC6600;
            padding: 8px;
            margin-top: 5px;
        """)
        baseline_caption.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(baseline_caption)

        if len(self.optimization_results) > 0:
            worst_result = self.optimization_results[-1]
            baseline_widget = self.create_baseline_widget(worst_result, len(self.optimization_results))
            self.results_layout.addWidget(baseline_widget)

        if self.show_breakdown_checkbox.isChecked() and self.best_breakdown is not None:
            separator_breakdown = QFrame()
            separator_breakdown.setFrameShape(QFrame.HLine)
            separator_breakdown.setFrameShadow(QFrame.Sunken)
            separator_breakdown.setStyleSheet("background-color: #BBBBBB; max-height: 2px; margin: 10px 0px;")
            self.results_layout.addWidget(separator_breakdown)

            breakdown_widget = BreakdownWidget(self.best_breakdown)
            self.results_layout.addWidget(breakdown_widget)

        self.results_layout.addStretch()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EnergyOptimizerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()