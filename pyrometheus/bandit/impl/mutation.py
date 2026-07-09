import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
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

class _ParsedReaction:

    def __init__(
        self,
        formula,
        reactants,
        products,
        log_pre_exponential,
        exponent,
        activation_temperature,
        has_third_body=False,
        third_body_efficiencies=None,
        fwd_rate_coeff_temperature="translational",
    ):
        self.formula = formula
        self.reactants = reactants
        self.products = products
        self.has_third_body = has_third_body
        self.third_body_efficiencies = third_body_efficiencies or {}
        self.fwd_rate_coeff_temperature = fwd_rate_coeff_temperature

        self._rate_law = SimpleNamespace(
            log_pre_exponential=log_pre_exponential,
            exponent=exponent,
            activation_temperature=activation_temperature,
        )

    def rate_law(self):
        return self._rate_law


class Mutationpp(BaseNamespace):

    def __init__(self, file_name, data_dir=None):

        if data_dir is not None:
            mpp.GlobalOptions.dataDirectory(data_dir)

        print("Mutation++ data directory:", mpp.GlobalOptions.dataDirectory())

        self.one_atm = 101325.0
        self.standard_pressure = 101325.0
        self.gas_constant = 8.31446261815324

        mix_options = mpp.MixtureOptions(file_name)
        self.mix = mpp.Mixture(mix_options)

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.mix, name)(*args)
        else:
            return getattr(self.mix, name)


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

        self._reaction_list = self._parse_mutation_mechanism_xml()

        if len(self._reaction_list) != self.namespace.mix.nReactions():
            raise RuntimeError(
                "Parsed reaction count does not match Mutation++ nReactions(): "
                f"parsed {len(self._reaction_list)}, "
                f"Mutation++ reports {self.namespace.mix.nReactions()}"
            )

        if reference_temperature is None:
            reference_temperature = float(
                os.environ.get("MUTATION_REFERENCE_T", "10000.0")
            )
        self.reference_temperature = float(reference_temperature)
            
        self.backward_rate_coeffs = self._compute_backward_rate_coefficients(
            self.reference_temperature
        )

        self.forward_rate_coeffs_mutationpp = self._compute_forward_rate_coefficients(
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
        return self.namespace.mix.nSpecies()

    @property
    def num_reactions(self):
        return len(self._reaction_list)

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

    def _mixture_xml_path(self):
        data_dir = Path(self.data_dir or mpp.GlobalOptions.dataDirectory())
        return data_dir / "mixtures" / f"{self.file_name}.xml"

    def _mechanism_xml_path(self):
        mixture_xml = self._mixture_xml_path()
        tree = ET.parse(mixture_xml)
        root = tree.getroot()

        mechanism_name = root.attrib["mechanism"]
        data_dir = Path(self.data_dir or mpp.GlobalOptions.dataDirectory())

        return data_dir / "mechanisms" / f"{mechanism_name}.xml"

    def _parse_species_token(self, token):
        token = token.strip()

        if token == "M":
            return None, 1

        match = re.match(r"^([0-9]*\.?[0-9]*)([A-Za-z][A-Za-z0-9_]*)$", token)

        if match is None:
            raise ValueError(f"Could not parse reaction token: {token}")

        coeff_str, species_name = match.groups()

        if coeff_str == "":
            coeff = 1
        else:
            coeff_float = float(coeff_str)
            coeff = int(coeff_float)

            if abs(coeff_float - coeff) > 1.0e-14:
                raise ValueError(
                    "Non-integer stoichiometric coefficients are not supported "
                    f"by this simple parser: {token}"
                )

        species_index = self.species_index(species_name)

        return species_index, coeff

    def _parse_side(self, side):
        species = []
        has_third_body = False

        for token in side.split("+"):
            token = token.strip()

            if not token:
                continue

            species_index, coeff = self._parse_species_token(token)

            if species_index is None:
                has_third_body = True
                continue

            for _ in range(coeff):
                species.append(species_index)

        return species, has_third_body

    def _parse_third_body_efficiencies(self, text):
        efficiencies = {}

        if text is None:
            return efficiencies

        text = text.strip()

        if not text:
            return efficiencies

        for item in text.split(","):
            item = item.strip()

            if not item:
                continue

            species_name, value = item.split(":")
            species_name = species_name.strip()
            value = float(value.strip())

            efficiencies[self.species_index(species_name)] = value

        return efficiencies

    def _arrhenius_A_to_SI(self, A_cgs, reaction_order):
        """
        Mutation++ air5_Park.xml uses A units with cm, mol, s, K.

        For concentrations in mol/m^3:
            first-order:  no factor
            second-order: cm^3/mol/s -> m^3/mol/s, factor 1e-6
            third-order:  cm^6/mol^2/s -> m^6/mol^2/s, factor 1e-12
        """
        return A_cgs * (1.0e-6 ** (reaction_order - 1))

    def _parse_mutation_mechanism_xml(self):
        mechanism_xml = self._mechanism_xml_path()

        tree = ET.parse(mechanism_xml)
        root = tree.getroot()

        reaction_list = []

        for rxn_node in root.findall("reaction"):
            formula = rxn_node.attrib["formula"]

            lhs, rhs = formula.split("=")

            reactants, lhs_has_M = self._parse_side(lhs)
            products, rhs_has_M = self._parse_side(rhs)

            has_third_body = lhs_has_M or rhs_has_M

            arr_node = rxn_node.find("arrhenius")

            if arr_node is None:
                raise ValueError(f"Reaction has no Arrhenius node: {formula}")

            A_cgs = float(arr_node.attrib["A"])
            exponent = float(arr_node.attrib["n"])
            activation_temperature = float(arr_node.attrib["T"])

            third_body_efficiencies = {}

            if has_third_body:
                third_body_node = rxn_node.find("M")

                if third_body_node is not None:
                    third_body_efficiencies = self._parse_third_body_efficiencies(
                        third_body_node.text
                    )

            reaction_order = len(reactants) + (1 if has_third_body else 0)
            A_si = self._arrhenius_A_to_SI(A_cgs, reaction_order)

            reaction_list.append(
                _ParsedReaction(
                    formula=formula,
                    reactants=reactants,
                    products=products,
                    log_pre_exponential=np.log(A_si),
                    exponent=exponent,
                    activation_temperature=activation_temperature,
                    has_third_body=has_third_body,
                    third_body_efficiencies=third_body_efficiencies,
                )
            )

        return reaction_list

    def reactions(self):
        return self._reaction_list

    def reaction(self, reaction_index: int):
        return self._reaction_list[reaction_index]

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
            for i, r in enumerate(self.reactions())
            if species_index in r.reactants
        ]

        rev_set = [
            i
            for i, r in enumerate(self.reactions())
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

        try:
            nT = mix.nEnergyEqns()
        except AttributeError:
            nT = 1

        temperature_vec = np.ascontiguousarray(
            temperature * np.ones(nT),
            dtype=np.float64,
        )

        mix.setState(rhoi, temperature_vec, 1)

    def _compute_forward_rate_coefficients(self, temperature):
        self._set_reference_state_for_temperature(temperature)
        return np.asarray(
            self.namespace.mix.forwardRateCoefficients(),
            dtype=np.float64,
        )

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

        efficiencies = reaction.third_body_efficiencies

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

        if reaction.has_third_body:
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
