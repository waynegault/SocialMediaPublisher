"""Analytics data warehouse for Social Media Publisher.

This module provides time-series storage and aggregation for analytics
data, enabling historical trend analysis and reporting.

Features:
- Time-series storage for metrics
- Data aggregation (hourly, daily, weekly, monthly)
- Historical trend analysis
- Data export for external tools
- Separate analytics DB from operational data

Example:
    warehouse = AnalyticsWarehouse()
    warehouse.record_metric("impressions", 1500, {"post_id": "123"})

    trends = warehouse.get_trends("impressions", days=30)
    report = warehouse.generate_report(start_date, end_date)
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# =============================================================================
# Enums and Constants
# =============================================================================


class AggregationPeriod(Enum):
    """Time periods for data aggregation."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MetricType(Enum):
    """Types of metrics tracked."""

    COUNTER = "counter"  # Cumulative count (e.g., total posts)
    GAUGE = "gauge"  # Point-in-time value (e.g., current followers)
    HISTOGRAM = "histogram"  # Distribution (e.g., engagement rates)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metric_type": self.metric_type.value,
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time period."""

    name: str
    period: AggregationPeriod
    period_start: datetime
    period_end: datetime
    count: int
    sum: float
    min: float
    max: float
    avg: float
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def stddev(self) -> float:
        """Standard deviation (placeholder - would need variance tracking)."""
        return 0.0


@dataclass
class TrendData:
    """Trend analysis results."""

    metric_name: str
    start_date: datetime
    end_date: datetime
    data_points: list[tuple[datetime, float]]
    trend_direction: str  # "up", "down", "stable"
    percent_change: float
    avg_value: float
    min_value: float
    max_value: float


@dataclass
class AnalyticsReport:
    """Complete analytics report."""

    generated_at: datetime
    start_date: datetime
    end_date: datetime
    metrics: dict[str, AggregatedMetric]
    trends: dict[str, TrendData]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "summary": self.summary,
            "metrics": {
                k: {
                    "count": v.count,
                    "sum": v.sum,
                    "avg": v.avg,
                    "min": v.min,
                    "max": v.max,
                }
                for k, v in self.metrics.items()
            },
            "trends": {
                k: {
                    "direction": v.trend_direction,
                    "percent_change": v.percent_change,
                    "avg": v.avg_value,
                }
                for k, v in self.trends.items()
            },
        }


# =============================================================================
# Analytics Warehouse
# =============================================================================


class AnalyticsWarehouse:
    """Time-series analytics data warehouse."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize analytics warehouse.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory DB.
        """
        self.db_path = db_path or ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        For in-memory databases, we keep a persistent connection
        since a new connection would create a new empty database.
        """
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close connection if it's not the persistent in-memory connection."""
        if self.db_path != ":memory:":
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Raw metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT,
                metric_type TEXT DEFAULT 'gauge',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Aggregated metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                period TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                count INTEGER NOT NULL,
                sum REAL NOT NULL,
                min REAL NOT NULL,
                max REAL NOT NULL,
                avg REAL NOT NULL,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, period, period_start, tags)
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agg_name_period ON aggregated_metrics(name, period, period_start)"
        )

        conn.commit()
        self._close_connection(conn)

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
        metric_type: MetricType = MetricType.GAUGE,
    ) -> int:
        """Record a metric data point.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for filtering
            timestamp: Optional timestamp (defaults to now)
            metric_type: Type of metric

        Returns:
            ID of inserted record
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO metrics (name, value, timestamp, tags, metric_type)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                name,
                value,
                timestamp.isoformat(),
                json.dumps(tags or {}),
                metric_type.value,
            ),
        )

        metric_id = cursor.lastrowid
        conn.commit()
        self._close_connection(conn)

        return metric_id if metric_id else 0

    def record_batch(self, metrics: list[MetricPoint]) -> int:
        """Record multiple metrics at once.

        Args:
            metrics: List of MetricPoint objects

        Returns:
            Number of records inserted
        """
        if not metrics:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        data = [
            (
                m.name,
                m.value,
                m.timestamp.isoformat(),
                json.dumps(m.tags),
                m.metric_type.value,
            )
            for m in metrics
        ]

        cursor.executemany(
            """
            INSERT INTO metrics (name, value, timestamp, tags, metric_type)
            VALUES (?, ?, ?, ?, ?)
        """,
            data,
        )

        count = cursor.rowcount
        conn.commit()
        self._close_connection(conn)

        return count

    def get_metrics(
        self,
        name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        tags: dict[str, str] | None = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Get raw metric data points.

        Args:
            name: Metric name
            start_date: Optional start date filter
            end_date: Optional end date filter
            tags: Optional tag filter
            limit: Maximum records to return

        Returns:
            List of MetricPoint objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM metrics WHERE name = ?"
        params: list[Any] = [name]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._close_connection(conn)

        results = []
        for row in rows:
            tags_data = json.loads(row["tags"]) if row["tags"] else {}

            # Filter by tags if specified
            if tags:
                if not all(tags_data.get(k) == v for k, v in tags.items()):
                    continue

            results.append(
                MetricPoint(
                    name=row["name"],
                    value=row["value"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=tags_data,
                    metric_type=MetricType(row["metric_type"]),
                )
            )

        return results

    def aggregate(
        self,
        name: str,
        period: AggregationPeriod,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AggregatedMetric]:
        """Aggregate metrics over time periods.

        Args:
            name: Metric name
            period: Aggregation period
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of aggregated metrics
        """
        metrics = self.get_metrics(name, start_date, end_date, limit=100000)

        if not metrics:
            return []

        # Group by period
        buckets: dict[str, list[float]] = {}

        for m in metrics:
            bucket_key = self._get_period_key(m.timestamp, period)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(m.value)

        # Calculate aggregates
        results = []
        for bucket_key, values in sorted(buckets.items()):
            period_start = self._parse_period_key(bucket_key, period)
            period_end = self._get_period_end(period_start, period)

            results.append(
                AggregatedMetric(
                    name=name,
                    period=period,
                    period_start=period_start,
                    period_end=period_end,
                    count=len(values),
                    sum=sum(values),
                    min=min(values),
                    max=max(values),
                    avg=sum(values) / len(values),
                )
            )

        return results

    def _get_period_key(self, dt: datetime, period: AggregationPeriod) -> str:
        """Get bucket key for a datetime based on period."""
        if period == AggregationPeriod.HOURLY:
            return dt.strftime("%Y-%m-%d-%H")
        elif period == AggregationPeriod.DAILY:
            return dt.strftime("%Y-%m-%d")
        elif period == AggregationPeriod.WEEKLY:
            # ISO week
            return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        else:  # MONTHLY
            return dt.strftime("%Y-%m")

    def _parse_period_key(self, key: str, period: AggregationPeriod) -> datetime:
        """Parse a period key back to datetime."""
        if period == AggregationPeriod.HOURLY:
            return datetime.strptime(key, "%Y-%m-%d-%H").replace(tzinfo=timezone.utc)
        elif period == AggregationPeriod.DAILY:
            return datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        elif period == AggregationPeriod.WEEKLY:
            year, week = key.split("-W")
            return datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w").replace(
                tzinfo=timezone.utc
            )
        else:  # MONTHLY
            return datetime.strptime(key + "-01", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )

    def _get_period_end(self, start: datetime, period: AggregationPeriod) -> datetime:
        """Get the end of a period given its start."""
        if period == AggregationPeriod.HOURLY:
            return start + timedelta(hours=1)
        elif period == AggregationPeriod.DAILY:
            return start + timedelta(days=1)
        elif period == AggregationPeriod.WEEKLY:
            return start + timedelta(weeks=1)
        else:  # MONTHLY
            # Approximate - add 30 days
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1)
            return start.replace(month=start.month + 1)

    def get_trends(
        self,
        name: str,
        days: int = 30,
        period: AggregationPeriod = AggregationPeriod.DAILY,
    ) -> TrendData:
        """Analyze trends for a metric.

        Args:
            name: Metric name
            days: Number of days to analyze
            period: Aggregation period for trend data

        Returns:
            TrendData with analysis results
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        aggregates = self.aggregate(name, period, start_date, end_date)

        if not aggregates:
            return TrendData(
                metric_name=name,
                start_date=start_date,
                end_date=end_date,
                data_points=[],
                trend_direction="stable",
                percent_change=0.0,
                avg_value=0.0,
                min_value=0.0,
                max_value=0.0,
            )

        data_points = [(a.period_start, a.avg) for a in aggregates]
        values = [a.avg for a in aggregates]

        # Calculate trend
        if len(values) >= 2:
            first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
            second_half = sum(values[len(values) // 2 :]) / (
                len(values) - len(values) // 2
            )

            if first_half > 0:
                percent_change = ((second_half - first_half) / first_half) * 100
            else:
                percent_change = 0.0

            if percent_change > 5:
                trend_direction = "up"
            elif percent_change < -5:
                trend_direction = "down"
            else:
                trend_direction = "stable"
        else:
            percent_change = 0.0
            trend_direction = "stable"

        return TrendData(
            metric_name=name,
            start_date=start_date,
            end_date=end_date,
            data_points=data_points,
            trend_direction=trend_direction,
            percent_change=percent_change,
            avg_value=sum(values) / len(values),
            min_value=min(values),
            max_value=max(values),
        )

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_names: list[str] | None = None,
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report.

        Args:
            start_date: Report start date
            end_date: Report end date
            metric_names: Optional list of metrics to include

        Returns:
            AnalyticsReport object
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all metric names if not specified
        if not metric_names:
            cursor.execute("SELECT DISTINCT name FROM metrics")
            metric_names = [row["name"] for row in cursor.fetchall()]

        self._close_connection(conn)

        metrics = {}
        trends = {}

        for name in metric_names:
            # Get aggregated data
            agg_list = self.aggregate(
                name, AggregationPeriod.DAILY, start_date, end_date
            )
            if agg_list:
                # Combine all aggregates
                total_count = sum(a.count for a in agg_list)
                total_sum = sum(a.sum for a in agg_list)
                all_mins = [a.min for a in agg_list]
                all_maxs = [a.max for a in agg_list]

                metrics[name] = AggregatedMetric(
                    name=name,
                    period=AggregationPeriod.DAILY,
                    period_start=start_date,
                    period_end=end_date,
                    count=total_count,
                    sum=total_sum,
                    min=min(all_mins) if all_mins else 0,
                    max=max(all_maxs) if all_maxs else 0,
                    avg=total_sum / total_count if total_count > 0 else 0,
                )

            # Get trends
            days = (end_date - start_date).days
            trends[name] = self.get_trends(name, days=max(1, days))

        # Generate summary
        summary = {
            "total_metrics": len(metric_names),
            "date_range_days": (end_date - start_date).days,
            "metrics_with_data": len(metrics),
            "trending_up": sum(1 for t in trends.values() if t.trend_direction == "up"),
            "trending_down": sum(
                1 for t in trends.values() if t.trend_direction == "down"
            ),
        }

        return AnalyticsReport(
            generated_at=datetime.now(timezone.utc),
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            trends=trends,
            summary=summary,
        )

    def export_csv(
        self,
        name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Export metric data to CSV format.

        Args:
            name: Metric name
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            CSV string
        """
        metrics = self.get_metrics(name, start_date, end_date)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["timestamp", "value", "metric_type", "tags"])

        # Data
        for m in metrics:
            writer.writerow(
                [
                    m.timestamp.isoformat(),
                    m.value,
                    m.metric_type.value,
                    json.dumps(m.tags),
                ]
            )

        return output.getvalue()

    def export_json(
        self,
        name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Export metric data to JSON format.

        Args:
            name: Metric name
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            JSON string
        """
        metrics = self.get_metrics(name, start_date, end_date)
        return json.dumps([m.to_dict() for m in metrics], indent=2)

    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Remove data older than specified days.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff.isoformat(),))
        deleted = cursor.rowcount

        conn.commit()
        self._close_connection(conn)

        return deleted

    def get_metric_names(self) -> list[str]:
        """Get all unique metric names.

        Returns:
            List of metric names
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT name FROM metrics ORDER BY name")
        names = [row["name"] for row in cursor.fetchall()]

        self._close_connection(conn)
        return names

    def close(self) -> None:
        """Close the warehouse (for cleanup in tests)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_analytics_warehouse(db_path: str | None = None) -> AnalyticsWarehouse:
    """Create an analytics warehouse instance.

    Args:
        db_path: Optional database path

    Returns:
        AnalyticsWarehouse instance
    """
    return AnalyticsWarehouse(db_path)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for this module."""
    from test_framework import TestSuite

    suite = TestSuite("Analytics Data Warehouse", "analytics_db.py")
    suite.start_suite()

    def test_metric_point():
        mp = MetricPoint(
            name="test",
            value=100.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert mp.name == "test"
        assert mp.value == 100.0
        data = mp.to_dict()
        assert "name" in data
        assert "value" in data

    def test_aggregated_metric():
        am = AggregatedMetric(
            name="test",
            period=AggregationPeriod.DAILY,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            count=10,
            sum=1000.0,
            min=50.0,
            max=150.0,
            avg=100.0,
        )
        assert am.count == 10
        assert am.avg == 100.0

    def test_warehouse_init():
        warehouse = AnalyticsWarehouse()
        assert warehouse.db_path == ":memory:"

    def test_record_metric():
        warehouse = AnalyticsWarehouse()
        metric_id = warehouse.record_metric("impressions", 1500)
        assert metric_id > 0

    def test_record_metric_with_tags():
        warehouse = AnalyticsWarehouse()
        metric_id = warehouse.record_metric(
            "clicks", 50, tags={"post_id": "123", "campaign": "test"}
        )
        assert metric_id > 0

    def test_record_batch():
        warehouse = AnalyticsWarehouse()
        now = datetime.now(timezone.utc)
        metrics = [
            MetricPoint("test", 100, now),
            MetricPoint("test", 200, now),
            MetricPoint("test", 300, now),
        ]
        count = warehouse.record_batch(metrics)
        assert count == 3

    def test_get_metrics():
        warehouse = AnalyticsWarehouse()
        warehouse.record_metric("views", 100)
        warehouse.record_metric("views", 200)

        results = warehouse.get_metrics("views")
        assert len(results) == 2

    def test_get_metrics_date_filter():
        warehouse = AnalyticsWarehouse()
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        warehouse.record_metric("test", 100, timestamp=yesterday)
        warehouse.record_metric("test", 200, timestamp=now)

        results = warehouse.get_metrics("test", start_date=now - timedelta(hours=1))
        assert len(results) == 1

    def test_aggregate_daily():
        warehouse = AnalyticsWarehouse()
        now = datetime.now(timezone.utc)

        # Add metrics for today
        for i in range(5):
            warehouse.record_metric("test", i * 10, timestamp=now)

        aggregates = warehouse.aggregate("test", AggregationPeriod.DAILY)
        assert len(aggregates) >= 1
        assert aggregates[0].count == 5

    def test_get_trends():
        warehouse = AnalyticsWarehouse()
        now = datetime.now(timezone.utc)

        # Add increasing metrics
        for i in range(10):
            ts = now - timedelta(days=9 - i)
            warehouse.record_metric("growth", i * 100, timestamp=ts)

        trends = warehouse.get_trends("growth", days=10)
        assert trends.metric_name == "growth"
        assert trends.trend_direction == "up"

    def test_generate_report():
        warehouse = AnalyticsWarehouse()
        now = datetime.now(timezone.utc)

        warehouse.record_metric("metric1", 100)
        warehouse.record_metric("metric2", 200)

        report = warehouse.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
        )
        assert report.summary["total_metrics"] >= 2

    def test_export_csv():
        warehouse = AnalyticsWarehouse()
        warehouse.record_metric("test", 100)
        warehouse.record_metric("test", 200)

        csv_data = warehouse.export_csv("test")
        assert "timestamp" in csv_data
        assert "100" in csv_data

    def test_export_json():
        warehouse = AnalyticsWarehouse()
        warehouse.record_metric("test", 100)

        json_data = warehouse.export_json("test")
        parsed = json.loads(json_data)
        assert len(parsed) == 1
        assert parsed[0]["value"] == 100

    def test_cleanup_old_data():
        warehouse = AnalyticsWarehouse()
        old = datetime.now(timezone.utc) - timedelta(days=100)

        warehouse.record_metric("old", 100, timestamp=old)
        warehouse.record_metric("new", 200)

        deleted = warehouse.cleanup_old_data(days_to_keep=30)
        assert deleted == 1

        remaining = warehouse.get_metrics("old")
        assert len(remaining) == 0

    suite.run_test(
        test_name="MetricPoint creation",
        test_func=test_metric_point,
        test_summary="Tests MetricPoint creation functionality",
        method_description="Calls MetricPoint and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="AggregatedMetric creation",
        test_func=test_aggregated_metric,
        test_summary="Tests AggregatedMetric creation functionality",
        method_description="Calls AggregatedMetric and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Warehouse init",
        test_func=test_warehouse_init,
        test_summary="Tests Warehouse init functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Record metric",
        test_func=test_record_metric,
        test_summary="Tests Record metric functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Record metric with tags",
        test_func=test_record_metric_with_tags,
        test_summary="Tests Record metric with tags functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Record batch",
        test_func=test_record_batch,
        test_summary="Tests Record batch functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function correctly processes multiple items",
    )
    suite.run_test(
        test_name="Get metrics",
        test_func=test_get_metrics,
        test_summary="Tests Get metrics functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get metrics date filter",
        test_func=test_get_metrics_date_filter,
        test_summary="Tests Get metrics date filter functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Aggregate daily",
        test_func=test_aggregate_daily,
        test_summary="Tests Aggregate daily functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get trends",
        test_func=test_get_trends,
        test_summary="Tests Get trends functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Generate report",
        test_func=test_generate_report,
        test_summary="Tests Generate report functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Export CSV",
        test_func=test_export_csv,
        test_summary="Tests Export CSV functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Export JSON",
        test_func=test_export_json,
        test_summary="Tests Export JSON functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Cleanup old data",
        test_func=test_cleanup_old_data,
        test_summary="Tests Cleanup old data functionality",
        method_description="Calls AnalyticsWarehouse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()
if __name__ == "__main__":
    # Demo usage
    print("Analytics Data Warehouse Demo")
    print("=" * 50)

    warehouse = AnalyticsWarehouse()

    # Record some metrics
    now = datetime.now(timezone.utc)
    for i in range(30):
        ts = now - timedelta(days=29 - i)
        warehouse.record_metric("impressions", 1000 + i * 50, timestamp=ts)
        warehouse.record_metric("clicks", 50 + i * 2, timestamp=ts)
        warehouse.record_metric("engagement", 5.0 + i * 0.1, timestamp=ts)

    # Get trends
    for metric in ["impressions", "clicks", "engagement"]:
        trends = warehouse.get_trends(metric, days=30)
        print(f"\n{metric}:")
        print(f"  Direction: {trends.trend_direction}")
        print(f"  Change: {trends.percent_change:.1f}%")
        print(f"  Avg: {trends.avg_value:.1f}")

    # Generate report
    report = warehouse.generate_report(
        start_date=now - timedelta(days=30),
        end_date=now,
    )
    print(f"\nReport Summary: {report.summary}")


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
