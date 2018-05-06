import sqlite3

DATABASE_FILE = "database.db"


def initialise(database_file):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""CREATE TABLE batches
                 (id integer primary key not null, submitted_on timestamp not null, location text,
                images_processed integer, crops_generated integer, elapsed_time time, size integer, status text,
                                                                                            batch_type integer)""")

    conn.commit()
    conn.close()


initialise(DATABASE_FILE)