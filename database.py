"""
Database module for storing video and music recommendations
"""
import sqlite3
import os
import random

DB_PATH = 'recommendations.db'


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database with tables and default data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create video_recommendation table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_recommendation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            posture_type TEXT NOT NULL,
            video_url TEXT NOT NULL,
            title TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create music_recommendation table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS music_recommendation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            music_url TEXT NOT NULL,
            title TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) FROM video_recommendation')
    video_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM music_recommendation')
    music_count = cursor.fetchone()[0]
    
    # Insert video recommendations if table is empty
    if video_count == 0:
        video_recommendations = [
            ('Squats', 'https://youtu.be/xqvCmoLULNY?si=14dwCOmshs1v6zXY', 'Squats Tutorial', 'Learn proper squats form'),
            ('Arm_Raise', 'https://youtu.be/JGeRYIZdojU?si=GWVwNWdrCjNUuCV7', 'Arm Raise Tutorial', 'Learn proper arm raise form'),
            ('Knee_Extension', 'https://youtu.be/o90ocSBDJis?si=vnPKQ2Lt8UVZGFvr', 'Knee Extension Tutorial', 'Learn proper knee extension form')
        ]
        cursor.executemany('''
            INSERT INTO video_recommendation (posture_type, video_url, title, description)
            VALUES (?, ?, ?, ?)
        ''', video_recommendations)
    
    # Insert music recommendations if table is empty
    if music_count == 0:
        music_recommendations = [
            ('https://youtu.be/D2CKOIiabfg?si=Nv6oc4fZo_kC49X9', 'Zumba Music 1', 'Energetic Zumba track'),
            ('https://youtu.be/hw-xiFZiMb8?si=mSUceUiZDwdCs-BT', 'Zumba Music 2', 'Upbeat Zumba track'),
            ('https://youtu.be/a1CwygQ13VI?si=liu8aFt2cdPGwcEE', 'Zumba Music 3', 'Motivational Zumba track')
        ]
        cursor.executemany('''
            INSERT INTO music_recommendation (music_url, title, description)
            VALUES (?, ?, ?)
        ''', music_recommendations)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def get_video_recommendations(posture_types):
    """Get video recommendations for given posture types"""
    if not posture_types:
        return []
    
    conn = get_db_connection()
    placeholders = ','.join(['?'] * len(posture_types))
    cursor = conn.execute(f'''
        SELECT * FROM video_recommendation 
        WHERE posture_type IN ({placeholders})
    ''', posture_types)
    
    recommendations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return recommendations


def get_random_music_recommendation():
    """Get a random music recommendation"""
    conn = get_db_connection()
    cursor = conn.execute('SELECT * FROM music_recommendation ORDER BY RANDOM() LIMIT 1')
    recommendation = cursor.fetchone()
    conn.close()
    
    if recommendation:
        return dict(recommendation)
    return None


def get_all_music_recommendations():
    """Get all music recommendations"""
    conn = get_db_connection()
    cursor = conn.execute('SELECT * FROM music_recommendation')
    recommendations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return recommendations

