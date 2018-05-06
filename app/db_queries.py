import sqlite3
import datetime
import time

import Batch


def create_batch(db_cursor, submission_date):
    """
    Inserts a batch instanse in the database
    :param db_cursor: sqlite3 cursor
    :param submission_date: datetime object
    :return: integer, the id of the inserted row
    """
    query = """INSERT INTO batches (submitted_on, status, batch_type) VALUES (?, ?, ?)"""
    db_cursor.execute(query, (time.mktime(submission_date.timetuple()), Batch.BatchStatus.PENDING.value,
                                                Batch.BatchType.EXTRACTION_AND_CLASSIFICATION.value))
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
    query = """UPDATE batches SET images_processed = ?, crops_generated = ? WHERE id = ? """
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
    print("location: {}, elapsed time: {}".format(location, elapsed_time))
    query = """UPDATE batches SET location = ?, elapsed_time = ?, size = ?, status = ? WHERE id = ? """
    db_cursor.execute(query, (location, elapsed_time.total_seconds(), size, Batch.BatchStatus.COMPLETED.value, batch_id))


def get_batches(db_cursor, limit=1000):
    query = """SELECT * FROM batches ORDER BY submitted_on DESC LIMIT ?"""
    db_cursor.execute(query, (limit,))

    batches = []
    while True:
        row_data = db_cursor.fetchone()
        if row_data is None: break

        id = row_data[0]
        submitted_on = datetime.datetime.fromtimestamp(row_data[1])
        location = row_data[2]
        images_processed = row_data[3]
        crops_generated = row_data[4]
        elapsed_time = None if row_data[5] is None else datetime.timedelta(seconds=row_data[5])
        size = row_data[6]
        status = row_data[7]
        batch_type = row_data[8]
        batch = Batch.Batch(id, submitted_on, location, images_processed, crops_generated, elapsed_time, size, status,
                                                                                                            batch_type)
        batches.append(batch)

    return batches
