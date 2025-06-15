import uuid

def is_valid_uuid(val):
    try:
        _ = uuid.UUID(str(val))
        return True
    except (ValueError, TypeError):
        return False
