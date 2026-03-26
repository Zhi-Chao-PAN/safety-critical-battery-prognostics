"""
War Zone 3: FMEA LLM Diagnostic Agent

Converts raw physics tensors (accumulated_stress, max_grad_anode) and anomalous
macro signals (dC/dt > 0) into structured ISO 26262 FMEA reports via LLM API calls.

Interface:
    When PhysicsFeatureExtractor outputs abnormal stress or the macro detector
    spots thermodynamically impossible capacity recovery, this agent intercepts
    and generates a standardized Failure Mode report.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

@dataclass
class FMEARecord:
    """ISO 26262 compliant FMEA record."""
    failure_mode: str
    failure_effect: str
    severity: int          # 1-10 scale
    occurrence: int        # 1-10 scale
    detection: int         # 1-10 scale
    rpn: int               # Risk Priority Number = S * O * D
    intervention: str
    raw_tensor_evidence: dict

@dataclass
class AnomalyTrigger:
    """Encapsulates the anomaly detection results from physics and macro layers."""
    cycle_index: int
    accumulated_stress: float
    max_concentration_gradient: float
    capacity_derivative: float  # dC/dt, positive = thermodynamic violation
    predicted_capacity: float
    true_capacity: float

class FMEAAgent:
    """
    LLM-powered FMEA diagnostic agent.
    
    Monitors physics feature tensors and macro capacity signals.
    When anomalies breach configurable thresholds, constructs a structured
    JSON payload and queries an LLM API for ISO 26262 failure analysis.
    """

    def __init__(self,
                 stress_threshold: float = 1e6,
                 gradient_threshold: float = 5000.0,
                 api_key: str | None = None,
                 api_base: str | None = None,
                 model_name: str = "deepseek-chat"):
        self.stress_threshold = stress_threshold
        self.gradient_threshold = gradient_threshold
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.model_name = model_name
        self.alert_history: list[FMEARecord] = []

        self.system_prompt = """You are an ISO 26262 certified battery safety engineer.
You will receive JSON telemetry data from a Battery Management System (BMS) 
physics engine monitoring lithium-ion cell degradation.

Your task is to analyze the anomaly and return a STRICT JSON response with exactly these fields:
{
  "failure_mode": "<specific electrochemical failure mechanism, e.g. 'Lithium plating at anode surface'>",
  "failure_effect": "<system-level consequence, e.g. 'Capacity fade acceleration leading to premature EOL'>",
  "severity": <integer 1-10>,
  "occurrence": <integer 1-10>,
  "detection": <integer 1-10>,
  "intervention": "<specific corrective action for BMS firmware>"
}

Severity scale: 1=negligible, 5=degraded performance, 8=safety hazard, 10=catastrophic thermal runaway.
Be extremely precise. Use electrochemistry terminology. Do NOT add any text outside the JSON."""

    def detect_anomalies(self,
                         cycle_idx: int,
                         accumulated_stress: float,
                         max_grad: float,
                         capacity_current: float,
                         capacity_previous: float,
                         predicted_capacity: float) -> AnomalyTrigger | None:
        """
        Checks if physics features or macro signals breach safety thresholds.
        
        Returns AnomalyTrigger if anomaly detected, None otherwise.
        """
        dc_dt = capacity_current - capacity_previous

        is_anomalous = False
        reasons = []

        # Check 1: Accumulated mechanical stress exceeds particle cracking threshold
        if accumulated_stress > self.stress_threshold:
            is_anomalous = True
            reasons.append(f"Stress={accumulated_stress:.2f} > threshold={self.stress_threshold}")

        # Check 2: Concentration gradient indicates lithium plating risk
        if max_grad > self.gradient_threshold:
            is_anomalous = True
            reasons.append(f"Gradient={max_grad:.2f} > threshold={self.gradient_threshold}")

        # Check 3: Thermodynamic violation (capacity INCREASING = impossible in normal aging)
        if dc_dt > 0.01:  # 1% tolerance for measurement noise
            is_anomalous = True
            reasons.append(f"dC/dt={dc_dt:.4f} > 0 (thermodynamic violation)")

        if is_anomalous:
            logger.warning(f"🚨 ANOMALY DETECTED at Cycle {cycle_idx}: {'; '.join(reasons)}")
            return AnomalyTrigger(
                cycle_index=cycle_idx,
                accumulated_stress=accumulated_stress,
                max_concentration_gradient=max_grad,
                capacity_derivative=dc_dt,
                predicted_capacity=predicted_capacity,
                true_capacity=capacity_current
            )
        return None

    def _build_llm_payload(self, trigger: AnomalyTrigger) -> dict:
        """Construct the JSON payload for the LLM API."""
        return {
            "telemetry": {
                "cycle_index": trigger.cycle_index,
                "accumulated_mechanical_stress_Pa": trigger.accumulated_stress,
                "max_radial_concentration_gradient_mol_m3": trigger.max_concentration_gradient,
                "capacity_time_derivative_Ah_per_cycle": trigger.capacity_derivative,
                "predicted_remaining_capacity_Ah": trigger.predicted_capacity,
                "measured_capacity_Ah": trigger.true_capacity,
            },
            "context": {
                "cell_chemistry": "LiFePO4 / Graphite",
                "nominal_capacity_Ah": 1.1,
                "operating_temperature_C": 25.0,
                "physics_model": "Single Particle Model (FDM, N=5 shells)"
            }
        }

    def generate_fmea_report(self, trigger: AnomalyTrigger) -> FMEARecord:
        """
        Calls the LLM API with anomaly data and parses the structured FMEA response.
        Falls back to rule-based analysis if API is unavailable.
        """
        payload = self._build_llm_payload(trigger)

        # Attempt LLM API call
        if self.api_key:
            try:
                import requests
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": json.dumps(payload, indent=2)}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    # Strip markdown code fences if present
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                    llm_result = json.loads(content)

                    record = FMEARecord(
                        failure_mode=llm_result["failure_mode"],
                        failure_effect=llm_result["failure_effect"],
                        severity=llm_result["severity"],
                        occurrence=llm_result["occurrence"],
                        detection=llm_result["detection"],
                        rpn=llm_result["severity"] * llm_result["occurrence"] * llm_result["detection"],
                        intervention=llm_result["intervention"],
                        raw_tensor_evidence=payload["telemetry"]
                    )
                    self.alert_history.append(record)
                    logger.info(f"LLM FMEA Report: {record.failure_mode} | RPN={record.rpn}")
                    return record

            except Exception as e:
                logger.warning(f"LLM API call failed ({e}), falling back to rule-based analysis.")

        # Fallback: Rule-based FMEA when no API key or API failure
        return self._rule_based_fmea(trigger, payload)

    def _rule_based_fmea(self, trigger: AnomalyTrigger, payload: dict) -> FMEARecord:
        """Deterministic rule-based FMEA when LLM is unavailable."""
        if trigger.capacity_derivative > 0:
            record = FMEARecord(
                failure_mode="Anomalous capacity recovery (possible lithium plating dissolution)",
                failure_effect="Temporary capacity increase masking accelerated degradation; "
                               "risk of internal short circuit and thermal runaway",
                severity=8,
                occurrence=3,
                detection=6,
                rpn=144,
                intervention="Reduce charge rate to C/3, increase rest period between cycles, "
                             "activate thermal monitoring at 5s intervals",
                raw_tensor_evidence=payload["telemetry"]
            )
        elif trigger.accumulated_stress > self.stress_threshold:
            record = FMEARecord(
                failure_mode="Excessive mechanical stress on active material particles",
                failure_effect="Particle cracking leading to loss of electrical contact, "
                               "accelerated capacity fade and impedance rise",
                severity=6,
                occurrence=5,
                detection=4,
                rpn=120,
                intervention="Limit discharge depth to 80% SOC, reduce C-rate below 0.5C, "
                             "schedule impedance spectroscopy diagnostic",
                raw_tensor_evidence=payload["telemetry"]
            )
        else:
            record = FMEARecord(
                failure_mode="Elevated concentration gradient at particle surface",
                failure_effect="SEI layer thickening and lithium inventory loss",
                severity=5,
                occurrence=4,
                detection=5,
                rpn=100,
                intervention="Apply CC-CV charging protocol with reduced CV hold voltage, "
                             "monitor coulombic efficiency trend",
                raw_tensor_evidence=payload["telemetry"]
            )

        self.alert_history.append(record)
        logger.info(f"Rule-Based FMEA: {record.failure_mode} | RPN={record.rpn}")
        return record

    def export_report(self, output_path: str) -> None:
        """Export all accumulated FMEA records as a JSON report."""
        records = [asdict(r) for r in self.alert_history]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(records)} FMEA records to {output_path}")
