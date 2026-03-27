"""
database.py
SQLite database layer for the Traffic Congestion Prediction System.
Provides full CRUD operations for:
  - roads             : road master data
  - traffic_records   : raw traffic readings
  - predictions       : saved prediction results
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "traffic.db")


# ─────────────────────────────────────────────
# Connection Helper
# ─────────────────────────────────────────────

@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Schema Creation
# ─────────────────────────────────────────────

def init_db():
    """Create all tables if they don't already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS roads (
                road_id     TEXT PRIMARY KEY,
                road_name   TEXT NOT NULL,
                road_type   TEXT DEFAULT 'Urban',
                capacity    INTEGER DEFAULT 700,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS traffic_records (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                road_id           TEXT NOT NULL,
                timestamp         TEXT NOT NULL,
                vehicle_count     INTEGER NOT NULL,
                average_speed     REAL NOT NULL,
                congestion_level  TEXT NOT NULL,
                weather_condition TEXT DEFAULT 'Clear',
                holiday_flag      INTEGER DEFAULT 0,
                special_events    TEXT DEFAULT 'None',
                created_at        TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (road_id) REFERENCES roads(road_id)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                road_id          TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                vehicle_count    INTEGER,
                average_speed    REAL,
                predicted_level  TEXT NOT NULL,
                prob_low         REAL,
                prob_medium      REAL,
                prob_high        REAL,
                suggested_action TEXT,
                created_at       TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (road_id) REFERENCES roads(road_id)
            );

            CREATE INDEX IF NOT EXISTS idx_traffic_road ON traffic_records(road_id);
            CREATE INDEX IF NOT EXISTS idx_traffic_ts   ON traffic_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pred_road    ON predictions(road_id);
        """)
    print(f"[DB] Initialised → {DB_PATH}")


def seed_roads():
    """Insert default roads (skips duplicates)."""
    defaults = [
        ("R101", "City Center Road",  "Urban",   700),
        ("R102", "Highway 21",        "Highway", 1000),
        ("R103", "Outer Ring Road",   "Ring",    800),
        ("R104", "Express Highway",   "Highway", 950),
        ("R105", "Northern Link",     "Urban",   650),
    ]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO roads (road_id, road_name, road_type, capacity) VALUES (?,?,?,?)",
            defaults,
        )
    print("[DB] Default roads seeded.")


# ═════════════════════════════════════════════
# ROADS — CRUD
# ═════════════════════════════════════════════

def create_road(road_id: str, road_name: str,
                road_type: str = "Urban", capacity: int = 700) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO roads (road_id, road_name, road_type, capacity) VALUES (?,?,?,?)",
                (road_id, road_name, road_type, capacity),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def read_all_roads() -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM roads ORDER BY road_id").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def read_road(road_id: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM roads WHERE road_id = ?", (road_id,)).fetchone()
    return dict(row) if row else None


def update_road(road_id: str, road_name: str = None,
                road_type: str = None, capacity: int = None) -> bool:
    fields, values = [], []
    if road_name is not None: fields.append("road_name = ?"); values.append(road_name)
    if road_type is not None: fields.append("road_type = ?"); values.append(road_type)
    if capacity  is not None: fields.append("capacity = ?");  values.append(capacity)
    if not fields:
        return False
    values.append(road_id)
    with get_connection() as conn:
        cur = conn.execute(f"UPDATE roads SET {', '.join(fields)} WHERE road_id = ?", values)
    return cur.rowcount > 0


def delete_road(road_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM roads WHERE road_id = ?", (road_id,))
    return cur.rowcount > 0


# ═════════════════════════════════════════════
# TRAFFIC RECORDS — CRUD
# ═════════════════════════════════════════════

def create_traffic_record(road_id: str, timestamp: str,
                           vehicle_count: int, average_speed: float,
                           congestion_level: str,
                           weather_condition: str = "Clear",
                           holiday_flag: int = 0,
                           special_events: str = "None") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO traffic_records
               (road_id, timestamp, vehicle_count, average_speed,
                congestion_level, weather_condition, holiday_flag, special_events)
               VALUES (?,?,?,?,?,?,?,?)""",
            (road_id, timestamp, vehicle_count, average_speed,
             congestion_level, weather_condition, holiday_flag, special_events),
        )
    return cur.lastrowid


def read_traffic_records(road_id: str = None, start_date: str = None,
                          end_date: str = None, congestion_level: str = None,
                          limit: int = 500) -> pd.DataFrame:
    query, params = "SELECT * FROM traffic_records WHERE 1=1", []
    if road_id:          query += " AND road_id = ?";          params.append(road_id)
    if start_date:       query += " AND timestamp >= ?";       params.append(start_date)
    if end_date:         query += " AND timestamp <= ?";       params.append(end_date)
    if congestion_level: query += " AND congestion_level = ?"; params.append(congestion_level)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def read_traffic_record_by_id(record_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM traffic_records WHERE id = ?", (record_id,)).fetchone()
    return dict(row) if row else None


def update_traffic_record(record_id: int, **kwargs) -> bool:
    allowed = {"vehicle_count", "average_speed", "congestion_level",
               "weather_condition", "holiday_flag", "special_events"}
    fields = [f"{k} = ?" for k in kwargs if k in allowed]
    values = [v for k, v in kwargs.items() if k in allowed]
    if not fields:
        return False
    values.append(record_id)
    with get_connection() as conn:
        cur = conn.execute(
            f"UPDATE traffic_records SET {', '.join(fields)} WHERE id = ?", values
        )
    return cur.rowcount > 0


def delete_traffic_record(record_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM traffic_records WHERE id = ?", (record_id,))
    return cur.rowcount > 0


def bulk_insert_traffic(df: pd.DataFrame) -> int:
    cols = ["road_id","timestamp","vehicle_count","average_speed",
            "congestion_level","weather_condition","holiday_flag","special_events"]
    rows = df[cols].values.tolist()
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO traffic_records
               (road_id, timestamp, vehicle_count, average_speed,
                congestion_level, weather_condition, holiday_flag, special_events)
               VALUES (?,?,?,?,?,?,?,?)""", rows,
        )
    return len(rows)


# ═════════════════════════════════════════════
# PREDICTIONS — CRUD
# ═════════════════════════════════════════════

def save_prediction(road_id: str, timestamp: str, vehicle_count: int,
                    average_speed: float, predicted_level: str,
                    prob_low: float = None, prob_medium: float = None,
                    prob_high: float = None, suggested_action: str = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (road_id, timestamp, vehicle_count, average_speed,
                predicted_level, prob_low, prob_medium, prob_high, suggested_action)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (road_id, timestamp, vehicle_count, average_speed,
             predicted_level, prob_low, prob_medium, prob_high, suggested_action),
        )
    return cur.lastrowid


def read_predictions(road_id: str = None, limit: int = 200) -> pd.DataFrame:
    query, params = "SELECT * FROM predictions WHERE 1=1", []
    if road_id:
        query += " AND road_id = ?"; params.append(road_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def delete_prediction(pred_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    return cur.rowcount > 0


def clear_old_predictions(days: int = 30) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM predictions WHERE created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
    return cur.rowcount


# ═════════════════════════════════════════════
# Analytics Queries
# ═════════════════════════════════════════════

def get_congestion_summary() -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT road_id, congestion_level, COUNT(*) as count
            FROM traffic_records
            GROUP BY road_id, congestion_level
            ORDER BY road_id, congestion_level
        """).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_hourly_avg(road_id: str = None) -> pd.DataFrame:
    query = """
        SELECT road_id,
               CAST(strftime('%H', timestamp) AS INTEGER) as hour,
               ROUND(AVG(vehicle_count), 1) as avg_vehicles,
               ROUND(AVG(average_speed), 1)  as avg_speed
        FROM traffic_records
    """
    params = []
    if road_id:
        query += " WHERE road_id = ?"; params.append(road_id)
    query += " GROUP BY road_id, hour ORDER BY road_id, hour"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_db_stats() -> dict:
    with get_connection() as conn:
        stats = {}
        for t in ("roads", "traffic_records", "predictions"):
            stats[t] = conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()["n"]
    return stats


# ─────────────────────────────────────────────
# CLI quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    seed_roads()

    rid = create_traffic_record("R101", datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                540, 22.5, "High")
    print(f"Inserted id={rid}")
    print(read_traffic_records(road_id="R101", limit=3))
    update_traffic_record(rid, vehicle_count=560)
    print("After update:", read_traffic_record_by_id(rid))
    delete_traffic_record(rid)
    print("DB stats:", get_db_stats())