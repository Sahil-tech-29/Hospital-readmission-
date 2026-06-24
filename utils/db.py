import sqlite3
import json
from datetime import datetime

DB_PATH = 'predictions.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            risk_score  REAL,
            risk_label  TEXT,
            features    TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(risk_score, risk_label, features_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, risk_score, risk_label, features)
        VALUES (?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        round(float(risk_score), 4),
        risk_label,
        json.dumps(features_dict)
    ))
    conn.commit()
    conn.close()


def get_recent_predictions(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, risk_score, risk_label
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*), AVG(risk_score) FROM predictions')
    total, avg_score = c.fetchone()
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_label = 'High Risk'")
    high_risk = c.fetchone()[0]
    conn.close()
    return {
        'total': total or 0,
        'avg_score': round((avg_score or 0) * 100, 1),
        'high_risk': high_risk or 0,
        'high_risk_pct': round((high_risk / total * 100) if total else 0, 1)
    }