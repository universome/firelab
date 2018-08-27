from firelab.utils.training_utils import determine_turn


def test_determine_turn():
    assert determine_turn(0, [5,2]) == 0
    assert determine_turn(4, [5,2]) == 0
    assert determine_turn(5, [5,2]) == 1
    assert determine_turn(6, [5,2]) == 1
    assert determine_turn(7, [5,2]) == 0
    assert determine_turn(13, [3,3,3]) == 1
    assert determine_turn(8, [3,3,3]) == 2
    assert determine_turn(9, [3,3,3]) == 0
