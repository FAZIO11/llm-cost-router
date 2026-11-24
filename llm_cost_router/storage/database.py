"""SQLite database for audit trail and analytics."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class RequestLog:
    """Request log entry."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    prompt: str = ""
    complexity_tier: str = ""
    complexity_score: float = 0.0
    selected_model: str = ""
    provider: str = ""
    tier: str = ""
    response_content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    frontier_baseline_cost_usd: float = 0.0
    savings_usd: float = 0.0
    savings_percent: float = 0.0


@dataclass
class EvaluationLog:
    """Evaluation log entry."""

    id: Optional[int] = None
    request_id: int = 0
    timestamp: Optional[str] = None
    quality_score: float = 0.0
    passed: bool = False
    heuristic_score: float = 0.0
    judge_score: Optional[float] = None
    judge_rationale: Optional[str] = None
    failure_reasons: Optional[str] = None


@dataclass
class ThresholdAdjustment:
    """Threshold adjustment event."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    tier: str = ""
    old_threshold: float = 0.0
    new_threshold: float = 0.0
    reason: str = ""
    rolling_quality_score: float = 0.0


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self):
        """Create tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                complexity_tier TEXT NOT NULL,
                complexity_score REAL NOT NULL,
                selected_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                tier TEXT NOT NULL,
                response_content TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                cost_usd REAL NOT NULL,
                frontier_baseline_cost_usd REAL NOT NULL,
                savings_usd REAL NOT NULL,
                savings_percent REAL NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                quality_score REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                heuristic_score REAL NOT NULL,
                judge_score REAL,
                judge_rationale TEXT,
                failure_reasons TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threshold_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tier TEXT NOT NULL,
                old_threshold REAL NOT NULL,
                new_threshold REAL NOT NULL,
                reason TEXT NOT NULL,
                rolling_quality_score REAL NOT NULL
            )
        """)

        self.conn.commit()

    def log_request(self, log: RequestLog) -> int:
        """Log a request and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt, complexity_tier, complexity_score,
                selected_model, provider, tier, response_content,
                input_tokens, output_tokens, latency_ms, cost_usd,
                frontier_baseline_cost_usd, savings_usd, savings_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.timestamp or datetime.utcnow().isoformat(),
                log.prompt,
                log.complexity_tier,
                log.complexity_score,
                log.selected_model,
                log.provider,
                log.tier,
                log.response_content,
                log.input_tokens,
                log.output_tokens,
                log.latency_ms,
                log.cost_usd,
                log.frontier_baseline_cost_usd,
                log.savings_usd,
                log.savings_percent,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_evaluation(self, log: EvaluationLog) -> int:
        """Log an evaluation and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO evaluations (
                request_id, timestamp, quality_score, passed,
                heuristic_score, judge_score, judge_rationale, failure_reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.request_id,
                log.timestamp or datetime.utcnow().isoformat(),
                log.quality_score,
                log.passed,
                log.heuristic_score,
                log.judge_score,
                log.judge_rationale,
                log.failure_reasons,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_threshold_adjustment(self, log: ThresholdAdjustment) -> int:
        """Log a threshold adjustment and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO threshold_adjustments (
                timestamp, tier, old_threshold, new_threshold, reason, rolling_quality_score
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                log.timestamp or datetime.utcnow().isoformat(),
                log.tier,
                log.old_threshold,
                log.new_threshold,
                log.reason,
                log.rolling_quality_score,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_requests(
        self, limit: int = 100, offset: int = 0, tier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get request logs with optional filtering."""
        query = "SELECT * FROM requests"
        params = []

        if tier:
            query += " WHERE tier = ?"
            params.append(tier)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_evaluations(
        self, limit: int = 100, passed: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation logs with optional filtering."""
        query = "SELECT * FROM evaluations"
        params = []

        if passed is not None:
            query += " WHERE passed = ?"
            params.append(passed)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics."""
        # Total requests
        total_requests = self.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

        # Cost metrics
        cost_data = self.conn.execute(
            """
            SELECT 
                SUM(cost_usd) as total_cost,
                SUM(frontier_baseline_cost_usd) as baseline_cost,
                SUM(savings_usd) as total_savings,
                AVG(latency_ms) as avg_latency
            FROM requests
            """
        ).fetchone()

        # Quality metrics
        quality_data = self.conn.execute(
            """
            SELECT 
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pass_rate
            FROM evaluations
            """
        ).fetchone()

        # Tier distribution
        tier_dist = self.conn.execute(
            """
            SELECT tier, COUNT(*) as count
            FROM requests
            GROUP BY tier
            """
        ).fetchall()

        # Escalations (requests that went to higher tier than predicted)
        escalations = self.conn.execute(
            """
            SELECT COUNT(*) FROM requests
            WHERE (complexity_tier = 'simple' AND tier != 'local')
               OR (complexity_tier = 'moderate' AND tier NOT IN ('local', 'budget'))
            """
        ).fetchone()[0]

        return {
            "total_requests": total_requests,
            "total_cost_usd": cost_data[0] or 0.0,
            "baseline_cost_usd": cost_data[1] or 0.0,
            "total_savings_usd": cost_data[2] or 0.0,
            "savings_percent": (
                (cost_data[2] / cost_data[1] * 100) if cost_data[1] and cost_data[1] > 0 else 0.0
            ),
            "avg_latency_ms": cost_data[3] or 0.0,
            "avg_quality_score": quality_data[0] or 0.0,
            "quality_pass_rate": quality_data[1] or 0.0,
            "tier_distribution": {row[0]: row[1] for row in tier_dist},
            "escalation_count": escalations,
        }

    def get_rolling_quality_by_tier(self, tier: str, window: int = 20) -> float:
        """Get rolling average quality score for a tier."""
        result = self.conn.execute(
            """
            SELECT AVG(e.quality_score) as avg_quality
            FROM evaluations e
            JOIN requests r ON e.request_id = r.id
            WHERE r.tier = ?
            ORDER BY e.timestamp DESC
            LIMIT ?
            """,
            (tier, window),
        ).fetchone()

        return result[0] if result[0] is not None else 1.0

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
