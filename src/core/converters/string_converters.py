import json


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def json_to_dictionary(myjson):
    try:
        return json.loads(myjson)
    except ValueError as e:
        return e


def byte_string_to_string(value):
    return value.decode('utf-8')
