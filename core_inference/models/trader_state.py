from dataclasses import dataclass

from core_inference.models.brokerage_state import BrokerageState


@dataclass
class TraderState:
    allocation: dict[str: float]
    shares_hold: dict[str: float]
    brokerage_states: dict[str: BrokerageState]