import sqlite3


STATUS_PENDING = "pending"


def create_batch(db_cursor, submission_date, images_processed, generated_crops):
    """
    Inserts a batch instanse in the database
    :param db_cursor: sqlite3 cursor
    :param submission_date: datetime object
    :param images_processed: integer, number of processed images
    :param generated_crops: integer, number of generated crops
    :return: integer, the id of the inserted row
    """
    query = """insert into batches (submitted_on, images_processed, crops_generated, status) values (?, ?, ?, ?)"""
    db_cursor.execute(query, (submission_date, images_processed, generated_crops, STATUS_PENDING))
    return db_cursor.lastrowid