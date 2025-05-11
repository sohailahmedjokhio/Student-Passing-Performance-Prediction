# SQL Connector for Students Performance Prediction
# Contributor: Sohail Ahmed

import mysql.connector
import pandas as pd
from mysql.connector import Error
from sqlalchemy import create_engine
import os

def setup_database_and_table(csv_path='student-data.csv'):
    """Create database and students table, and populate with data from CSV if they don't exist."""
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='1234'
        )
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS student_performance")
        print("Database 'student_performance' created or already exists")
        cursor.execute("USE student_performance")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS students (
            school VARCHAR(2),
            sex VARCHAR(1),
            age INT,
            address VARCHAR(1),
            famsize VARCHAR(3),
            Pstatus VARCHAR(1),
            Medu INT,
            Fedu INT,
            Mjob VARCHAR(20),
            Fjob VARCHAR(20),
            reason VARCHAR(20),
            guardian VARCHAR(20),
            traveltime INT,
            studytime INT,
            failures INT,
            schoolsup VARCHAR(3),
            famsup VARCHAR(3),
            paid VARCHAR(3),
            activities VARCHAR(3),
            nursery VARCHAR(3),
            higher VARCHAR(3),
            internet VARCHAR(3),
            romantic VARCHAR(3),
            famrel INT,
            freetime INT,
            goout INT,
            Dalc INT,
            Walc INT,
            health INT,
            absences INT,
            passed VARCHAR(3)
        )
        """
        cursor.execute(create_table_query)
        print("Table 'students' created or already exists")
        cursor.execute("SELECT COUNT(*) FROM students")
        if cursor.fetchone()[0] == 0:
            if not os.path.exists(csv_path):
                print(f"Error: {csv_path} not found in project directory")
                return False
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO students (school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
                                         reason, guardian, traveltime, studytime, failures, schoolsup, famsup,
                                         paid, activities, nursery, higher, internet, romantic, famrel, freetime,
                                         goout, Dalc, Walc, health, absences, passed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(row))
            connection.commit()
            print(f"Inserted {len(df)} rows into 'students' table")
        else:
            print("Table 'students' already contains data")
        return True
    except Error as e:
        print(f"Error setting up database: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL setup connection closed")

def connect_to_mysql():
    """Connect to MySQL database and return SQLAlchemy engine."""
    try:
        engine = create_engine("mysql+mysqlconnector://root:1234@127.0.0.1/student_performance")
        print("Successfully connected to MySQL database")
        return engine
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_student_data():
    """Fetch student data from MySQL table and return as Pandas DataFrame."""
    engine = connect_to_mysql()
    if engine is None:
        return None
    try:
        query = """
        SELECT school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
               reason, guardian, traveltime, studytime, failures, schoolsup, famsup,
               paid, activities, nursery, higher, internet, romantic, famrel, freetime,
               goout, Dalc, Walc, health, absences, passed
        FROM students
        """
        df = pd.read_sql(query, engine)
        print("Data fetched successfully from MySQL")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        engine.dispose()
        print("MySQL connection closed")

if __name__ == "__main__":
    if setup_database_and_table():
        df = fetch_student_data()
        if df is not None:
            print("Dataset Shape:", df.shape)
            print("Columns:", df.columns)