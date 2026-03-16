import inspect

import photonic_jordan as pj


def test_exported_symbols_have_docstrings():
    missing = []
    for name in pj.__all__:
        obj = getattr(pj, name)
        if not inspect.getdoc(obj):
            missing.append(name)
    assert not missing, f"Missing docstrings for exported symbols: {missing}"


def test_key_methods_use_structured_sections():
    checks = [
        (pj.PhotonicSystem.ensure_scope_filtration, ["Parameters", "Returns", "Notes"]),
        (pj.StateBuilder.from_modes_and_gram, ["Parameters", "Returns"]),
        (pj.StateInvariantView.I_exact, ["Parameters"]),
        (pj.PhotonicState.project_jordan, ["Parameters"]),
        (pj.ObservableFactory.from_matrix, ["Parameters"]),
        (pj.SingleParticleObservable, ["Parameters", "Notes"]),
        (pj.SchurWeylDecomposition, ["References"]),
    ]

    for obj, tokens in checks:
        doc = inspect.getdoc(obj)
        assert doc is not None
        for token in tokens:
            assert token in doc, f"Expected section '{token}' in docstring of {obj}"
