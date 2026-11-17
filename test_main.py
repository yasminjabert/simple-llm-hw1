import unittest
from main import TestLLM, data

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = TestLLM()
        self.model.train(data)

    def test_most_frequent_after_kot(self):
        self.assertEqual(self.model.predict_next_word("кот"), "спит")

    def test_next_after_spit(self):
        self.assertEqual(self.model.predict_next_word("спит"), "на")

    def test_unknown_word(self):
        self.assertEqual(self.model.predict_next_word("собака"),
                         "Слово не найдено в обучающей выборке.")

    def test_last_word_in_sentence(self):
        self.assertEqual(self.model.predict_next_word("диване"),
                         "Слово не найдено в обучающей выборке.")

if __name__ == "__main__":
    unittest.main()