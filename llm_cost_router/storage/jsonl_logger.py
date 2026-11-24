"""Structured JSONL logging."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class JSONLLogger:
    """Logger that writes structured JSON lines."""

    def __init__(self, log_path: Path):
        """Initialize JSONL logger."""
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]):
        """Log an event as a JSON line."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **data,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
