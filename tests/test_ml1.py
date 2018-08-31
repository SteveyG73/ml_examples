import unittest
import ml1


class TestML1Methods(unittest.TestCase):

    def test_get_dataset(self):
        actual = ml1.get_dataset()
        self.assertGreaterEqual(actual.shape[0], 1)


if __name__ == '__main__':
    unittest.main()