from indexes.python import splitmix


def test_reference_values():
    assert splitmix.next_u64(splitmix.new(42)) == 13679457532755275413
    assert splitmix.next_u64(splitmix.new(0)) == 16294208416658607535


def test_ranges():
    rng = splitmix.new(7)
    for _ in range(1000):
        assert 0.0 <= splitmix.next_f64(rng) < 1.0
        assert 0 <= splitmix.next_below(rng, 10) < 10
