import os
from typing import Union, List, Tuple

import numpy as np
import mutationpp as mpp

from pymbolic.primitives import Variable
from pymbolic import substitute

from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.kinetics import (
    RateCoefficient,
    make_arrhenius,
    species_production_rate_expr,
)


_temp_map = {
    "translational": Variable("temperature")[0],
    "electron": Variable("temperature")[1],
    "geometric_ttv": Variable("sqrt")(
        Variable("temperature")[0] * Variable("temperature")[1]
    ),
}

class Mutationpp(BaseNamespace):

    def __init__(self, file_name, data_dir=None):

        if data_dir is not None:
            os.environ["MPP_DATA_DIRECTORY"] = data_dir
        elif "MUTATION_DB" in os.environ:
            os.environ.setdefault("MPP_DATA_DIRECTORY", os.environ["MUTATION_DB"])

        print(
            "Mutation++ data directory:",
            os.environ.get("MPP_DATA_DIRECTORY", "<Mutation++ default>"),
        )

        self.one_atm = 101325.0
        self.standard_pressure = 101325.0
        self.gas_constant = 8.31446261815324

        mix_options = mpp.MixtureOptions(file_name)
        self.mix = mpp.Mixture(mix_options)
        
class MutationMechanism(BaseMechanism):

    num_temp = 2

    def __init__(
            self,
            file_name,
            data_dir=None,
            pyro_np=np,
            hardcode_params=True,
            reference_temperature=None,
    ):
        self.hardcode_params = hardcode_params
        self.pyro_np = pyro_np
        self.file_name = file_name
        self.data_dir = data_dir

        self.namespace = Mutationpp(file_name, data_dir=data_dir)

        self._reactions = list(self.namespace.mix.reactions)
        if len(self._reactions) != self.namespace.mix.num_reactions:
            raise RuntimeError(
                "Parsed reaction count does not match Mutation++ nReactions(): "
                f"parsed {len(self._reactions)}, "
                f"Mutation++ reports {self.namespace.mix.num_reactions}"
            )

        if reference_temperature is None:
            reference_temperature = float(
                os.environ.get("MUTATION_REFERENCE_T", "10000.0")
            )
        self.reference_temperature = float(reference_temperature)
            
        self.backward_rate_coeffs = self._compute_backward_rate_coefficients(
            self.reference_temperature
        )

        self.make_rates(hardcode_params)
        
    @property
    def one_atm(self):
        return 101325.0

    @property
    def standard_pressure(self):
        return 101325.0

    @property
    def gas_constant(self):
        return 8.31446261815324

    @property
    def num_temperatures(self):
        return self.num_temp

    @property
    def species_thermo_polynomials(self):
        return []

    @property
    def num_species(self):
        return self.namespace.mix.num_species

    @property
    def num_reactions(self):
        return len(self._reactions)

    @property
    def reactions(self):
        return self._reactions

    def reaction(self, reaction_index: int):
        return self._reactions[reaction_index]
    
    @property
    def molecular_weights(self):
        return np.array(
            [self.namespace.mix.speciesMw(i) for i in range(self.num_species)]
        )

    @property
    def species_names(self):
        return [
            self.namespace.mix.speciesName(i)
            for i in range(self.num_species)
        ]

    def species_index(self, species_name: str) -> int:
        return self.namespace.mix.speciesIndex(species_name)

    def species_name(self, species_index: int) -> str:
        return self.namespace.mix.speciesName(species_index)

    def reactants(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index).reactants

    def products(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index).products

    def stoichiometric_coefficients(self, reaction_index: int) -> List[int]:
        reactants = self.reactants(reaction_index)
        return [reactants.count(species_index) for species_index in reactants]

    def participation_set(self, species_id: Union[int, str]) -> Tuple[List[int]]:

        if isinstance(species_id, int):
            assert species_id < self.num_species
            species_index = species_id

        elif isinstance(species_id, str):
            species_index = self.species_index(species_id)

        else:
            raise ValueError(
                "species_id must be either str or int, "
                f"but received {type(species_id)}"
            )

        fwd_set = [
            i
            for i, r in enumerate(self.reactions)
            if species_index in r.reactants
        ]

        rev_set = [
            i
            for i, r in enumerate(self.reactions)
            if species_index in r.products
        ]

        return fwd_set, rev_set

    def is_reversible(self, reaction_index):
        return False

    def finalize(self):
        if hasattr(self.namespace, "finalize"):
            self.namespace.finalize()
        elif hasattr(self.namespace, "mix") and hasattr(
            self.namespace.mix,
            "finalize",
        ):
            self.namespace.mix.finalize()

    def make_rate_coefficient(self, reaction_index, hardcode_params):
        rate = self.reaction(reaction_index).rate_law()
        temp = self.reaction(reaction_index).fwd_rate_coeff_temperature

        if hardcode_params:
            params = {
                "a": rate.log_pre_exponential,
                "b": rate.exponent,
                "t_a": rate.activation_temperature,
            }

            k_fwd = make_arrhenius(
                reaction_index=reaction_index,
                params=params,
            )

        else:
            params = np.array(
                [
                    rate.log_pre_exponential,
                    rate.exponent,
                    rate.activation_temperature,
                ]
            )

            k_fwd = make_arrhenius(
                reaction_index=reaction_index,
            )

        if temp in _temp_map:
            k_fwd.expr = substitute(
                k_fwd.expr,
                {"temperature": _temp_map[temp]},
            )

        return k_fwd, params


    def _set_reference_state_for_temperature(self, temperature):

        mix = self.namespace.mix

        ns = self.num_species

        mole_fractions = np.full(ns, 1.0e-300, dtype=np.float64)
        mole_fractions[self.species_index("N2")] = 0.79
        mole_fractions[self.species_index("O2")] = 0.21
        mole_fractions /= np.sum(mole_fractions)

        molecular_weights = self.molecular_weights

        mass_fractions = (
            molecular_weights * mole_fractions
            / np.sum(molecular_weights * mole_fractions)
        )

        density = 1.0
        rhoi = np.ascontiguousarray(density * mass_fractions, dtype=np.float64)

        
        nT = int(getattr(mix, "num_energy_eqns", 1))

        temperature_vec = np.ascontiguousarray(
            temperature * np.ones(nT),
            dtype=np.float64,
        )

        mix.setState(rhoi, temperature_vec, 1)

    def _compute_backward_rate_coefficients(self, temperature):
        self._set_reference_state_for_temperature(temperature)
        return np.asarray(
            self.namespace.mix.backwardRateCoefficients(),
            dtype=np.float64,
        )

    def _concentration_product(self, species_indices):
        conc = Variable("concentrations")

        factor = 1.0

        for species_index in sorted(set(species_indices)):
            nu = species_indices.count(species_index)
            factor = factor * conc[species_index] ** nu

        return factor

    def _third_body_concentration(self, reaction):
        conc = Variable("concentrations")

        efficiencies = dict(reaction.efficiencies)

        third_body_concentration = 0.0

        for species_index in range(self.num_species):
            alpha = efficiencies.get(species_index, 1.0)
            third_body_concentration = (
                third_body_concentration + alpha * conc[species_index]
            )

        return third_body_concentration
    
    def make_mass_action_rate(self, reaction_index, hardcode_params=True):
        rate_coeff, param_vals = self.make_rate_coefficient(
            reaction_index,
            hardcode_params,
        )

        if not isinstance(rate_coeff, RateCoefficient):
            return 0

        if not hardcode_params:
            self.param_vals = (
                np.vstack((self.param_vals, param_vals))
                if self.param_vals.size
                else param_vals
            )

        reaction = self.reaction(reaction_index)

        reactant_factor = self._concentration_product(reaction.reactants)
        product_factor = self._concentration_product(reaction.products)

        q_fwd = rate_coeff.expr * reactant_factor

        k_bwd = float(self.backward_rate_coeffs[reaction_index])
        q_bwd = k_bwd * product_factor

        if reaction.is_third_body:
            m_eff = self._third_body_concentration(reaction)
            q_fwd = q_fwd * m_eff
            q_bwd = q_bwd * m_eff

        return q_fwd - q_bwd

    
    def make_species_production_rate(self, species_index):
        if isinstance(species_index, str):
            species_index = self.species_index(species_index)

        fwd_set, rev_set = self.participation_set(species_index)

        stoich_fwd = [
            self.reactants(r).count(species_index)
            for r in fwd_set
        ]

        stoich_rev = [
            self.products(r).count(species_index)
            for r in rev_set
        ]

        return species_production_rate_expr(
            species_index,
            fwd_set,
            rev_set,
            stoich_fwd,
            stoich_rev,
        )
