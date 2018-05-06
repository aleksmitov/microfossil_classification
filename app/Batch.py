from enum import Enum


class Batch:
    def __init__(self, id, submitted_on, location, images_processed, crops_generated, elapsed_time, size, status,
                                                                                                        batch_type):
        self.id = id
        self.submitted_on = submitted_on
        self.location = location
        self.images_processed = images_processed
        self.crops_generated = crops_generated
        self.elapsed_time = elapsed_time
        self.size = size
        self.status = status
        self.batch_type = batch_type


class BatchType(Enum):
    EXTRACTION_AND_CLASSIFICATION = 0
    EXTRACTION = 1
    FILTERING = 2
    CLASSIFICATION = 3


class BatchStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"

