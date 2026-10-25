"""One byte label per original header byte; high nibble describes username→ID reuse."""
import importlib.util
import json
from pathlib import Path
import sys

NAMES = ['format', 'page_delta', 'old_revision', 'new_revision', 'old_timestamp',
         'new_timestamp', 'username_first', 'username_repeat', 'user_id_first',
         'user_id_repeat', 'ip']
ASSOCIATIONS = ['outside_user_id', 'new_username', 'seen_username_same_id', 'seen_username_changed_id']


def make_labels(data, codec):
    labels = bytearray(len(data))
    users, ids = {}, set()
    cursor = records = 0
    fields = [0] * len(NAMES)
    associations = [0] * len(ASSOCIATIONS)
    def assign(a, b, kind, association=0):
        assert a < b and not any(labels[a:b]), 'empty/overlapping field'
        labels[a:b] = bytes([kind | association << 4]) * (b-a)
        fields[kind] += 1
    while cursor < len(data):
        m = codec.RECORD.match(data, cursor)
        if m is None:
            raise ValueError('incomplete header at byte %d' % cursor)
        old = codec.decimal(m[2]) < codec.THRESHOLD
        assign(*m.span(1), 1)
        assign(*m.span(2), 2 if old else 3)
        assign(*m.span(3), 4 if old else 5)
        if m[5] is not None:
            a = m.start(4) + len(b'\xdf\x8fN\n\xdf\x87N')
            b = data.index(b'\n', a)
            username, user_id = data[a:b], codec.decimal(m[5])
            seen = username in users
            association = 1 if not seen else 2 if users[username] == user_id else 3
            assign(a, b, 7 if seen else 6)
            assign(*m.span(5), 9 if user_id in ids else 8, association)
            associations[association] += 1
            users[username] = user_id
            ids.add(user_id)
        else:
            a = m.start(4) + len(b'\xdf\x8fN\nipN')
            assign(a, data.index(b'\n', a), 10)
        cursor = m.end()
        records += 1
    if not records:
        raise ValueError('empty header')
    byte_counts = [sum((v & 15) == i for v in labels) for i in range(len(NAMES))]
    assert cursor == len(data) == sum(byte_counts)
    return labels, {'records': records, 'input_bytes': len(data), 'names': NAMES,
                    'field_occurrences': fields, 'field_bytes': byte_counts,
                    'association_names': ASSOCIATIONS, 'association_records': associations,
                    'unique_usernames': len(users), 'unique_user_ids': len(ids),
                    'labels_cover_every_input_byte': True,
                    'repeat_definition': 'same raw username / same integer user-ID seen earlier in this cold slice'}


if __name__ == '__main__':
    codec_path, source, target = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location('metadata_codec', codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    labels, report = make_labels(source.read_bytes(), codec)
    target.write_bytes(labels)
    target.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
