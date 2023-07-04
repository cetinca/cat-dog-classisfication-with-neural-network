import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy

from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pickle

def test_labels():
    return numpy.array([0] * 25 + [1] * 25)


class UnfreezeTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=10000000)]

    def check(self, reply: str, attach):

        if 'stage_five_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_five_history` is not in SavedHistory directory")

        with open('../SavedHistory/stage_five_history', 'rb') as stage_five:
            answer = pickle.load(stage_five)

        if not isinstance(answer, numpy.ndarray):
            return CheckResult.wrong("`stage_five_history` should be a numpy array")

        labels = test_labels()
        accuracy = labels == answer
        test_accuracy = accuracy.mean()

        if test_accuracy < 0.95:
            return CheckResult.wrong(f"Your model's accuracy is {test_accuracy * 100}%\n"
                                     "The goal is to score at least 95%")

        print(f"Test accuracy: {round(test_accuracy, 3)}")
        return CheckResult.correct()


if __name__ == '__main__':
    UnfreezeTest().run_tests()
