import sqlite3


STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"

def create_batch(db_cursor, submission_date):
    """
    Inserts a batch instanse in the database
    :param db_cursor: sqlite3 cursor
    :param submission_date: datetime object
    :return: integer, the id of the inserted row
    """
    query = """INSERT INTO batches (submitted_on, status) VALUES (?, ?)"""
    db_cursor.execute(query, (submission_date, STATUS_PENDING))
    return db_cursor.lastrowid


def update_batch(db_cursor, batch_id, images_processed, generated_crops):
    """
    Add image counts to batch
    :param db_cursor: sqlite3 cursor
    :param batch_id: integer, id of the batch to update
    :param images_processed: integer, number of processed images
    :param generated_crops: integer, number of generated crops
    :return: None
    """
    query = """UPDATE batches SET(images_processed = ?, crops_generated = ?) WHERE id = ? """
    db_cursor.execute(query, (images_processed, generated_crops, batch_id))


def finish_batch(db_cursor, batch_id, location, elapsed_time, size):
    """
    Completes a batch
    :param db_cursor: sqlite3 cursor
    :param batch_id: integer, id of the batch to update
    :param location: string, filename of the archive containing the batch data
    :param elapsed_time: datetime.time object
    :param size: integer, size of the archive
    :return: None
    """
    query = """UPDATE batches SET(location = ?, elapsed_time = ?, size = ?, status = ?) WHERE id = ? """
    db_cursor.execute(query, (location, elapsed_time, size, STATUS_COMPLETED, batch_id))