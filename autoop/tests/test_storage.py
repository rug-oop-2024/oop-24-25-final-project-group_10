import os
# we changed "/" to os.path.sep because we use windows
import unittest
from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile


class TestStorage(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """
        Test storage initialization.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """
        Test storing and loading data.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.path.sep}path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = f"test{os.path.sep}otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self):
        """
        Test deleting a key.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.path.sep}path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self):
        """
        Test listing keys in a directory.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test{os.path.sep}{random.randint(0, 100)}"
                       for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = [os.path.sep.join(key.split(os.path.sep)[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
