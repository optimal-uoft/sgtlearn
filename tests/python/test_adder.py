from sgtlearn.adder import Adder


def test_adder_delegates_to_cpp():
    adder = Adder()
    assert adder.add(2.0, 3.0) == 5.0
