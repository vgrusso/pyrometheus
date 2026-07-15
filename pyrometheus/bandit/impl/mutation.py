import os
from typing import Union, List, Tuple
import numpy as np
import mutationpp as mpp

from pymbolic.primitives import Variable

from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism

from pyrometheus.bandit.chem_expr.kinetics import (
    RateCoefficient,
    make_arrhenius,
    species_production_rate_expr,
    conc,
    k_fwd,
    exp,
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
            hardcode_params=True
    ):
        self.hardcode_params = hardcode_params
        self.pyro_np = pyro_np
        self.file_name = file_name
        self.data_dir = data_dir

        self.namespace = Mutationpp(file_name, data_dir=data_dir)

        self._reactions = list(self.namespace.mix.reactions)
        if len(self._reactions) != self.namespace.mix.num_reactions:
            raise RuntimeError(
                "Reaction count does not match Mutation++ num_reactions: "
                f"stored {len(self._reactions)}, "
                f"Mutation++ reports {self.namespace.mix.num_reactions}"
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

    def stoichiometric_coefficients(
            self, reaction_index: int) -> Tuple[List[int], List[int]]:
        reactants = self.reactants(reaction_index)
        products = self.products(reaction_index)
        
        return (
            [reactants.count(species_index) for species_index in reactants],
            [products.count(species_index) for species_index in products],
        )

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
        return True

    def finalize(self):
        if hasattr(self.namespace, "finalize"):
            self.namespace.finalize()
        elif hasattr(self.namespace, "mix") and hasattr(
            self.namespace.mix,
            "finalize",
        ):
            self.namespace.mix.finalize()

    def _temperature_var(self):
        if self.num_temp == 1:
            return Variable("temperature")
        return Variable("temperature")[0]

    def _rrho_electronic_partition(self, rrho_data, temperature):
        z_e = 0
        e_e = 0

        for degeneracy, theta in rrho_data["electronic_levels"]:
            boltzmann = degeneracy * exp(-theta / temperature)
            z_e = z_e + boltzmann
            e_e = e_e + theta * boltzmann

        return z_e, e_e

    def _species_rrho_gibbs_over_rt(self, species_index):
        temperature = self._temperature_var()
        log = Variable("log")

        rrho_data = self.namespace.mix.rrhoThermoData(species_index)

        # Enthalpy contribution H/(R T)
        h = 2.5

        if rrho_data["has_rotational"]:
            h = h + rrho_data["linearity"]

        for theta in rrho_data["vibrational_temperatures"]:
            h = h + (theta / temperature) / (
                exp(theta / temperature) - 1.0
            )

        z_e, e_e = self._rrho_electronic_partition(rrho_data, temperature)
        h = h + (e_e / z_e) / temperature

        h = h + (
            rrho_data["hform_over_ru"] - rrho_data["part_sst"]
        ) / temperature

        # Entropy contribution S/R
        s = (
            2.5 * (1.0 + log(temperature))
            - log(self.one_atm)
            + rrho_data["ln_qt_mw"]
        )

        if rrho_data["has_rotational"]:
            s = s + rrho_data["linearity"] * (
                1.0 + log(temperature) - rrho_data["ln_omega_t"]
            )

        for theta in rrho_data["vibrational_temperatures"]:
            fac = exp(theta / temperature)
            s = s + theta / (temperature * (fac - 1.0)) - log(
                1.0 - 1.0 / fac
            )

        s = s + (e_e / z_e) / temperature + log(z_e)

        return h - s

    def _species_rrho_gibbs_concentration_corrected(self, species_index):
        temperature = self._temperature_var()
        log = Variable("log")

        g_over_rt = self._species_rrho_gibbs_over_rt(species_index)

        return g_over_rt - log(
            self.one_atm / (self.gas_constant * temperature)
        )

    def _reaction_delta_gibbs(self, reaction_index):
        reaction = self.reaction(reaction_index)

        delta_g = 0

        for species_index in reaction.products:
            delta_g = delta_g + self._species_rrho_gibbs_concentration_corrected(
                species_index
            )

        for species_index in reaction.reactants:
            delta_g = delta_g - self._species_rrho_gibbs_concentration_corrected(
                species_index
            )

        return delta_g
            

    def make_rate_coefficient(self, reaction_index, hardcode_params):
        reaction = self.reaction(reaction_index)
        rate = reaction.rate_law()

        temp = str(reaction.fwd_rate_coeff_temperature)
        
        if temp not in _temp_map:
            raise ValueError(
                f"Unknown Mutation++ rate temperature '{temp}' "
                f"for reaction {reaction_index}"
            )

        temp_var = _temp_map[temp]
        
        if hardcode_params:
            params = {
                "a": rate.log_pre_exponential,
                "b": rate.exponent,
                "t_a": rate.activation_temperature,
            }

            k_fwd = make_arrhenius(
                reaction_index=reaction_index,
                params=params,
                temperature=temp_var,
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
                temperature=temp_var,
            )
            
        return k_fwd, params
        
    def _concentration_product(self, species_indices):
        factor = 1.0
        
        for species_index in sorted(set(species_indices)):
            nu = species_indices.count(species_index)
            factor = factor * conc[species_index] ** nu

        return factor

    def _third_body_concentration(self, reaction):
        efficiencies = dict(reaction.efficiencies)
        third_body_concentration = 0.0
        
        for species_index in range(self.num_species):
            alpha = efficiencies.get(species_index, 1.0)
            third_body_concentration = (
                third_body_concentration + alpha * conc[species_index]
            )
            
        return third_body_concentration
    
    def make_mass_action_rate(self, reaction_index):
        reaction = self.reaction(reaction_index)

        reactant_factor = self._concentration_product(reaction.reactants)
        product_factor = self._concentration_product(reaction.products)

        q_fwd = k_fwd[reaction_index] * reactant_factor

        delta_g = self._reaction_delta_gibbs(reaction_index)
        k_bwd = k_fwd[reaction_index] * exp(delta_g)
        q_bwd = k_bwd * product_factor

        if reaction.is_third_body:
            third_body_concentration = self._third_body_concentration(reaction)
            q_fwd = third_body_concentration * q_fwd
            q_bwd = third_body_concentration * q_bwd

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
