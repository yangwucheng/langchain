from peewee import *


class TaskModel(Model):
    id = AutoField()

    source = CharField(max_length=255, null=False)
    source_type = CharField(max_length=255, null=False)
    destination = CharField(max_length=255, null=False)
    qa_count_per_chunk = IntegerField(default=5)

    chunk_size = IntegerField(default=500)
    chunk_overlap = IntegerField(default=50)
    total_chunk = IntegerField(null=True)
    success_chunk = IntegerField(default=0)
    failed_chunk = IntegerField(default=0)
    # Submitted, Running, Finished(completely success), Finished(partial failed), Failed
    status = CharField(max_length=50, default="Submitted")

    class Meta:
        database = None

    def to_dict(self):
        return {
            'id': self.id,
            'source': self.source,
            'source_type': self.source_type,
            'destination': self.destination,
            'qa_count_per_chunk': self.qa_count_per_chunk,
            'chunk_size': self.chunk_size,

            'total_chunk': self.total_chunk,

            'success_chunk': self.success_chunk,
            'failed_chunk': self.failed_chunk,
            'status': self.status,
        }
