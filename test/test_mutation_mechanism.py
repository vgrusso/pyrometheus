import os
import pytest
import numpy as np


mutation_available = True
try:
    import mutationpp as mpp
    from pyrometheus.bandit.impl.mutation import MutationMechanism
except ImportError:
    mutation_available = False


pytestmark = pytest.mark.skipif(
    not mutation_available or os.environ.get("MUTATION_DB") is None,
    reason="MUTATION_DB not set or mutationpp package not importable",
)


MIXTURE = "air_5"


@pytest.fixture(scope="module")
def mech():
    data_dir = os.environ.get("MUTATION_DB")

    if not data_dir:
        pytest.skip("MUTATION_DB not set")

    m = MutationMechanism(
        MIXTURE,
        data_dir=data_dir,
    )

    yield m

    if hasattr(m, "finalize"):
        m.finalize()


# --- Construction / setup ---


def test_constructs(mech):
    assert mech is not None


def test_namespace_has_mixture(mech):
    assert hasattr(mech, "namespace")
    assert hasattr(mech.namespace, "mix")


def test_mutation_data_directory_is_set(mech):
    assert mpp.GlobalOptions.dataDirectory() == os.environ["MUTATION_DB"]


def test_air5_xml_exists():
    data_dir = os.environ["MUTATION_DB"]
    mixture_file = os.path.join(data_dir, "mixtures", f"{MIXTURE}.xml")
    assert os.path.exists(mixture_file)


# --- Species metadata ---


def test_num_species(mech):
    assert mech.num_species == 5


def test_species_names(mech):
    assert mech.species_names == ["N", "O", "NO", "N2", "O2"]


def test_species_names_match_underlying_mix(mech):
    mix_species_names = [
        mech.namespace.mix.speciesName(i)
        for i in range(mech.namespace.mix.nSpecies())
    ]
    assert mech.species_names == mix_species_names


def test_species_index_name_roundtrip(mech):
    for name in mech.species_names:
        idx = mech.species_index(name)
        assert mech.species_name(idx) == name


def test_molecular_weights_shape(mech):
    assert mech.molecular_weights.shape == (mech.num_species,)


def test_molecular_weights_are_positive(mech):
    assert np.all(np.isfinite(mech.molecular_weights))
    assert np.all(mech.molecular_weights > 0.0)


# --- Reaction metadata currently exposed by Mutation++ ---


def test_num_reactions(mech):
    assert mech.num_reactions == 5


def test_num_reactions_matches_underlying_mix(mech):
    assert mech.num_reactions == mech.namespace.mix.nReactions()


def test_num_species_matches_underlying_mix(mech):
    assert mech.num_species == mech.namespace.mix.nSpecies()
