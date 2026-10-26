"""Basic checksum behavior for the distributable archive."""
import tempfile
from pathlib import Path
import unittest

from paper_suite.checksum import checksum_bytes, checksum_file, checksum_json


class ChecksumsTests(unittest.TestCase):
    def test_file_size_and_crc_detect_accidental_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.txt'
            path.write_bytes(b'alpha')
            first = checksum_file(path)
            self.assertEqual(first, checksum_bytes(b'alpha'))
            self.assertEqual(first['size_bytes'], 5)
            path.write_bytes(b'omega')
            self.assertEqual(first['size_bytes'], checksum_file(path)['size_bytes'])
            self.assertNotEqual(first['crc32'], checksum_file(path)['crc32'])

    def test_canonical_json_ignores_dictionary_key_order(self):
        self.assertEqual(checksum_json({'a': 1, 'b': 2}),
                         checksum_json({'b': 2, 'a': 1}))
        self.assertNotEqual(checksum_json({'a': 1}), checksum_json({'a': 2}))


if __name__ == '__main__':
    unittest.main()
