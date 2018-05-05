import sqlite3

DATABASE_FILE = "database.db"


def initialise(database_file):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""CREATE TABLE batches
                 (id integer primary key not null, submitted_on date not null, location text,
                images_processed integer, crops_generated integer, elapsed_time date, size integer, status text)""")

    conn.commit()
    conn.close()


initialise(DATABASE_FILE)