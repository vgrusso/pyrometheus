
################ TEST 1


# import os
# import pytest



# pytestmark = pytest.mark.skipif(
#     not mutation_available,
#     reason="Mutation++ database and or Mutation++ library not set o r package not importable",
# )

# DB = os.environ.get("MUTATION_DB", "")
# MIXTURE = "air_5"
# REACTION = "air_5"
# TRANSFER = "empty"

# @pytest.fixture(scope="module")
# def mech():
#     m = MutationMechanism(MIXTURE, REACTION, TRANSFER, DB)
#     yield m
#     m.finalize()

    
# #------------------#
# #     Species      #
# #------------------#

# def test_num_species(mech):
#     assert mech.num_species == 5   # N, O, N2, NO, O2




# ################ TEST 2

# import os
# import pytest

# #from pyrometheus.bandit.impl.mutation import MutationMechanism

# mutation_available = True
# try:
#     from pyrometheus.bandit.impl.mutation import MutationMechanism
# except ImportError:
#     mutation_available = False
        

# # try:
# #     from pyrometheus.bandit.impl.mutation import MutationMechanism
# #     mutation_available = True
# # except ImportError:
# #     mutation_available = False



# # Only skip if the required database is not available
# pytestmark = pytest.mark.skipif(
#     not mutation_available or os.environ.get("MUTATION_DB") is None,
#     reason="MUTATION_DB not set or mutationpp package not importable",
# )


# @pytest.fixture(scope="module")
# def mech():
#     db = os.environ.get("MUTATION_DB")

#     if not db:
#         pytest.skip("MUTATION_DB not set")

#     # m = MutationMechanism(
#     #     "air_5",
#     #     "air_5",
#     #     "empty",
#     #     db
#     # )

#     m = MutationMechanism(db)
    
#     yield m
#     m.finalize()



# def test_constructs(mech):
#     """Just verify mechanism builds successfully."""
#     assert mech is not None
    
    
# def test_num_species_positive(mech):
#     """Ensure mechanism actually loaded something physical."""
#     assert mech.num_species > 0
    
# def test_species_names(mech):
#     assert len(mech.species_names) == mech.num_species
    
    
# def test_molecular_weights_shape(mech):
#     assert mech.molecular_weights.shape == (mech.num_species,)
    

# def test_num_reactions_positive(mech):
#     assert mech.num_reactions > 0


# def test_mass_action_rates_length(mech):
#     assert len(mech.mass_action_rates) == mech.num_reactions


# def test_species_prod_rates_length(mech):
#     assert len(mech.species_prod_rates) == mech.num_species


# def test_participation_set_consistent(mech):
#     for sp in range(mech.num_species):
#         fwd, rev = mech.participation_set(sp)
#         for r in fwd:
#             assert sp in mech.reactants(r)
#         for r in rev:
#             assert sp in mech.products(r)


# def test_mass_action_rates_nonzero(mech):
#     for r, expr in enumerate(mech.mass_action_rates):
#         assert expr != 0, f"mass_action_rate[{r}] is zero"


# ## TEST 3 CLEANED UP

# import os
# import pytest

# mutation_available = True
# try:
#     from pyrometheus.bandit.impl.mutation import MutationMechanism
# except ImportError:
#     mutation_available = False


# pytestmark = pytest.mark.skipif(
#     not mutation_available or os.environ.get("MUTATION_DB") is None,
#     reason="MUTATION_DB not set or mutationpp package not importable",
# )


# @pytest.fixture(scope="module")
# def mech():
#     db = os.environ.get("MUTATION_DB")

#     if not db:
#         pytest.skip("MUTATION_DB not set")

#     m = MutationMechanism(
#         "air_5",
#         data_dir=db,
#     )

#     yield m

#     if hasattr(m, "finalize"):
#         m.finalize()


# def test_constructs(mech):
#     assert mech is not None

# # --- Species metadata ---    

# def test_num_species(mech):
#     assert mech.num_species == 5


# def test_species_names(mech):
#     assert mech.species_names ==  ["N", "O", "NO", "N2", "O2"]#mech.num_species


# def test_molecular_weights_shape(mech):
#     assert mech.molecular_weights.shape == (mech.num_species,)

# # --- Reaction list assembly ---    

# def test_num_reactions_positive(mech):
#     assert mech.num_reactions > 0



# def test_num_reactions_matches_types(mech):
#     print ("hello")
    
#     # ns = mech.namespace
#     # expected = (
#     #     ns.n_reactions_dh_cat_mt
#     #     + ns.n_reactions_exc
#     #     + ns.n_reactions_de
#     #     + ns.n_reactions_ie
#     #     + ns.n_reactions_ai
#     # )
#     # assert mech.num_reactions == expected
#     # print("\n===== DEBUG: mech =====")
#     # print("type(mech) =", type(mech))
#     # print("dir(mech) =")
#     # for name in dir(mech):
#     #     if not name.startswith("__"):
#     #         print("  ", name)

#     # print("\n===== DEBUG: mech.namespace =====")
#     # ns = mech.namespace
#     # print("type(ns) =", type(ns))
#     # print("dir(ns) =")
#     # for name in dir(ns):
#     #     if not name.startswith("__"):
#     #         print("  ", name)

#     print ("hello")
#     ns = mech.namespace
#     mix = ns.mix
#     assert mech.num_reactions == mix.nReactions()
#     assert True
    

# # def test_dh_cat_mt_come_first(mech):
# #     assert True
# #     ns = mech.namespace
# #     mix = ns.mix
# #     for name in dir(mix):
# #         if not name.startswith("__"):
# #             print (name)
# #     #n_dh = mech.namespace.n_reactions_dh_cat_mt
# #     #for r in range(n_dh):
# #     #    assert mech._reactions[r]["type"] == "dh_cat_mt"


# # --- Stoichiometry ---

# def test_dh_cat_mt_stoichiometry(mech):
#     # First dh_cat_mt: one reactant, two products
#     rxn = mech._reactions[0]
    
#     ns = mech.namespace
#     mix = ns.mix

#     for name in dir(mix):
#         if not name.startswith("__"):
#             print (name)
        

    
#     assert len(rxn["reactant_species"]) == 1
#     assert len(rxn["product_species"]) == 2

        
# def test_mass_action_rates_length(mech):
#     assert len(mech.mass_action_rates) == mech.num_reactions


# def test_species_prod_rates_length(mech):
#     assert len(mech.species_prod_rates) == mech.num_species


# def test_participation_set_consistent(mech):
#     for sp in range(mech.num_species):
#         fwd, rev = mech.participation_set(sp)

#         for r in fwd:
#             assert sp in mech.reactants(r)

#         for r in rev:
#             assert sp in mech.products(r)


# def test_mass_action_rates_nonzero(mech):
#     for r, expr in enumerate(mech.mass_action_rates):
#         assert expr != 0, f"mass_action_rate[{r}] is zero"

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
