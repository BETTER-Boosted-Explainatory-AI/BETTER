import uuid

def is_valid_uuid(val):
    try:
        uuid_obj = uuid.UUID(str(val))
        return True
    except (ValueError, TypeError):
        return False
