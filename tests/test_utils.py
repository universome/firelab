from firelab.utils.training_utils import determine_turn
from firelab.utils.training_utils import PiecewiseLinearScheme as PLS


def test_determine_turn():
    assert determine_turn(0, [5,2]) == 0
    assert determine_turn(4, [5,2]) == 0
    assert determine_turn(5, [5,2]) == 1
    assert determine_turn(6, [5,2]) == 1
    assert determine_turn(7, [5,2]) == 0
    assert determine_turn(13, [3,3,3]) == 1
    assert determine_turn(8, [3,3,3]) == 2
    assert determine_turn(9, [3,3,3]) == 0


def test_piecewise_linear_scheme():
    # TODO: add proper tests?
    assert PLS([(0., 0., 0.)]).evaluate(3) == 0
    assert PLS([(0., 0., 5.), (0., 2., 10.)]).scheme_idx_for_iteration(3) == 0
    assert PLS([(0., 3., 5.), (3., 0., 10.)]).scheme_idx_for_iteration(5) == 0
    assert PLS([(0., 3., 5.), (3., 0., 10.)]).scheme_idx_for_iteration(6) == 1
    assert PLS([(0., 3., 5.), (3., 0., 10.)]).scheme_idx_for_iteration(10) == 1
    assert PLS([(0., 3., 5.), (3., 0., 10.)]).scheme_idx_for_iteration(20) == 1
