"""
Database schema definition and CSV import functionality for TableTalk SQL Agent
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "database/students.db"
DB_URL = f"sqlite:///{DB_PATH}"

def create_database_schema(db_path: str = DB_PATH) -> None:
    """
    Create the database schema for student data
    """
    # Ensure database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop table if it exists (for development)
    cursor.execute("DROP TABLE IF EXISTS students")
    
    # Create students table with proper schema
    create_table_sql = """
    CREATE TABLE students (
        -- Primary key
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        -- Demographics
        student_id TEXT,
        student_name TEXT NOT NULL,
        student_age INTEGER,
        student_gender TEXT CHECK (student_gender IN ('male', 'female')),
        school_name TEXT,
        city TEXT,
        grade_level INTEGER,
        
        -- Categories
        student_category TEXT,
        
        -- Program information
        program_name TEXT,
        track_chosen TEXT CHECK (track_chosen IN ('Explorers', 'Wizards', 'Pathfinders')),
        courses_selected TEXT,
        
        -- Course details
        course_1 TEXT,
        instructor_1 TEXT,
        course_2 TEXT,
        instructor_2 TEXT,
        course_3 TEXT,
        instructor_3 TEXT,
        teaching_assistant TEXT,
        
        -- Administrative
        weeks_attending TEXT,
        payment_status TEXT,
        rc_name TEXT,
        
        -- Parent information
        parent_name TEXT,
        parent_phone_primary TEXT,
        parent_phone_secondary TEXT,
        parent_email TEXT,
        
        -- Timestamps
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    cursor.execute(create_table_sql)
    
    # Create indexes for common query patterns
    indexes = [
        "CREATE INDEX idx_student_category ON students(student_category)",
        "CREATE INDEX idx_city ON students(city)", 
        "CREATE INDEX idx_school_name ON students(school_name)",
        "CREATE INDEX idx_track_chosen ON students(track_chosen)",
        "CREATE INDEX idx_grade_level ON students(grade_level)",
        "CREATE INDEX idx_program_name ON students(program_name)",
        "CREATE INDEX idx_rc_name ON students(rc_name)",
        "CREATE INDEX idx_instructor_1 ON students(instructor_1)",
        "CREATE INDEX idx_instructor_2 ON students(instructor_2)",
        "CREATE INDEX idx_instructor_3 ON students(instructor_3)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database schema created successfully at {db_path}")

def clean_data_value(value: Any) -> Any:
    """
    Clean and standardize data values
    """
    if pd.isna(value) or value == '' or value == 'NA':
        return None
    
    # Convert to string and strip whitespace
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['na', 'n/a', '', 'nan', 'null']:
            return None
    
    return value

def import_csv_to_database(csv_path: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Import CSV data into the SQLite database
    
    Returns:
        Dict with import statistics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Log initial data info
    logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Define column mapping - only include columns up to parent_email as requested
    column_mapping = {
        'student_id': 'student_id',
        'student_name': 'student_name', 
        'student_age': 'student_age',
        'student_gender': 'student_gender',
        'school_name': 'school_name',
        'city': 'city',
        'grade_level': 'grade_level',
        'student_category': 'student_category',
        'program_name': 'program_name',
        'track_chosen': 'track_chosen',
        'courses_selected': 'courses_selected',
        'course_1': 'course_1',
        'instructor_1': 'instructor_1',
        'course_2': 'course_2', 
        'instructor_2': 'instructor_2',
        'course_3': 'course_3',
        'instructor_3': 'instructor_3',
        'teaching_assistant': 'teaching_assistant',
        'weeks_attending': 'weeks_attending',
        'payment_status': 'payment_status',
        'rc_name': 'rc_name',
        'parent_name': 'parent_name',
        'parent_phone_primary': 'parent_phone_primary',
        'parent_phone_secondary': 'parent_phone_secondary',
        'parent_email': 'parent_email'
    }
    
    # Select and rename columns
    available_columns = {col: col for col in df.columns if col in column_mapping}
    df_filtered = df[list(available_columns.keys())].copy()
    
    # Clean the data
    for col in df_filtered.columns:
        df_filtered[col] = df_filtered[col].apply(clean_data_value)
    
    # Handle specific data type conversions
    if 'student_age' in df_filtered.columns:
        df_filtered['student_age'] = pd.to_numeric(df_filtered['student_age'], errors='coerce')
    
    if 'grade_level' in df_filtered.columns:
        df_filtered['grade_level'] = pd.to_numeric(df_filtered['grade_level'], errors='coerce')
    
    # Filter out rows where student_name is empty (invalid records)
    initial_count = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['student_name'])
    final_count = len(df_filtered)
    
    logger.info(f"Filtered data: {final_count} valid records (removed {initial_count - final_count} invalid records)")
    
    # Connect to database and import
    conn = sqlite3.connect(db_path)
    
    try:
        # Import data using pandas to_sql
        df_filtered.to_sql('students', conn, if_exists='replace', index=False)
        
        # Get statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT student_category) FROM students WHERE student_category IS NOT NULL")
        unique_categories = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT city) FROM students WHERE city IS NOT NULL")
        unique_cities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT school_name) FROM students WHERE school_name IS NOT NULL")
        unique_schools = cursor.fetchone()[0]
        
        conn.commit()
        
        stats = {
            'total_records': total_records,
            'unique_categories': unique_categories,
            'unique_cities': unique_cities,
            'unique_schools': unique_schools,
            'csv_path': csv_path,
            'db_path': db_path
        }
        
        logger.info(f"Data imported successfully: {total_records} records")
        return stats
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error importing data: {e}")
        raise
    finally:
        conn.close()

def get_database_engine() -> Engine:
    """
    Get SQLAlchemy engine for database operations
    """
    return create_engine(DB_URL, echo=False)

def get_database_schema_info() -> Dict[str, Any]:
    """
    Get information about the database schema
    """
    if not os.path.exists(DB_PATH):
        return {"error": "Database not found"}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get table info
        cursor.execute("PRAGMA table_info(students)")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM students")
        row_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute("SELECT * FROM students LIMIT 3")
        sample_data = cursor.fetchall()
        
        # Format column information
        column_info = []
        for col in columns:
            column_info.append({
                'name': col[1],
                'type': col[2],
                'not_null': bool(col[3]),
                'primary_key': bool(col[5])
            })
        
        return {
            'table_name': 'students',
            'columns': column_info,
            'row_count': row_count,
            'sample_data': sample_data[:3],  # Limit to 3 rows
            'db_path': DB_PATH
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def validate_database() -> Dict[str, Any]:
    """
    Validate the database and return health information
    """
    if not os.path.exists(DB_PATH):
        return {"valid": False, "error": "Database file not found"}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
        if not cursor.fetchone():
            return {"valid": False, "error": "Students table not found"}
        
        # Check data integrity
        cursor.execute("SELECT COUNT(*) FROM students")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM students WHERE student_name IS NOT NULL")
        valid_names = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "valid": True,
            "total_records": total_records,
            "valid_names": valid_names,
            "data_quality": valid_names / total_records if total_records > 0 else 0
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

# Main function for testing
if __name__ == "__main__":
    # Create database schema
    create_database_schema()
    
    # Import CSV data
    csv_path = "uploads/GSP Standardized Sheet - May_2025_standard.csv"
    if os.path.exists(csv_path):
        stats = import_csv_to_database(csv_path)
        print("Import completed:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print(f"CSV file not found: {csv_path}")
    
    # Validate database
    validation = validate_database()
    print("Database validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")