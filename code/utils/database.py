import sqlite3


def create_connection(database):
    conn = None
    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        print(e)
    return conn


def create_table(conn):
    sql_create_results_table = """CREATE TABLE IF NOT EXISTS expdisc (
                                    id INTEGER PRIMARY KEY,
                                    model_name TEXT NOT NULL UNIQUE,
                                    mean_accuracy REAL NOT NULL,
                                    min_accuracy REAL NOT NULL,
                                    max_accuracy REAL NOT NULL,
                                    median_accuracy REAL NOT NULL
                                );"""

    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_results_table)
    except sqlite3.Error as e:
        print(e)


def insert_result(conn, result):
    sql = """INSERT INTO expdisc (model_name, mean_accuracy, min_accuracy, max_accuracy, median_accuracy) 
    VALUES(?, ?, ?, ?, ?);"""
    cursor = conn.cursor()
    cursor.execute(sql, result)
    conn.commit()


def check_model_exists(conn, model_name):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM expdisc WHERE model_name=?", (model_name,))
    return cursor.fetchone() is not None

# Renvoie "True" si existe déjà
