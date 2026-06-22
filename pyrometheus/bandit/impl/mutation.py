import numpy as np
import mutationpp as mpp
from pymbolic.primitives import Variable
from pymbolic import substitute
from typing import Union, List, Tuple
from pyrometheus.bandit.general_thermochem import BaseNamespace, BaseMechanism
from pyrometheus.bandit.chem_expr.kinetics import make_arrhenius,species_production_rate_expr

_temp_map = {
    'translational': Variable('temperature'),
    'electron': Variable('electron_temperature'),
    'geometric_ttv': Variable('sqrt')(
        Variable('temperature')[0] *
        Variable('temperature')[1]
    )
}


class Mutationpp(BaseNamespace):

    def __init__(self, file_name, data_dir=None):
        
        #Vincenzo: had to add this because xml path was breaking test
        if data_dir is not None:
            mpp.GlobalOptions.dataDirectory(data_dir)
            
        #Vincenzo: printing to screen for debugging
        print("Mutation++ data directory:", mpp.GlobalOptions.dataDirectory())

        mix_options = mpp.MixtureOptions(file_name)
        self.mix = mpp.Mixture(mix_options)

        

    def __getattr__(self, name, *args):
        if args:
            return getattr(self.mix, name)(*args)
        else:
            return getattr(self.mix, name)


class MutationMechanism(BaseMechanism):

    num_temp = 2

    def __init__(self, file_name, data_dir=None, pyro_np=np, hardcode_params=True):
        self.hardcode_params = hardcode_params
        self.namespace = Mutationpp(file_name, data_dir=data_dir)

        print("VINCENZO CHECKPOINT BEGINS")
        print("I have uncommented things below because I will fix simples tests first")
        #self.make_rates(hardcode_params)
        #self.make_pyro(pyro_np)
        print("VINCENZO CHECKPOINT ENDS")
        
    @property
    def num_species(self):
        #return self.namespace.__getattr__("num_species") #Vin: could not find num_species in mutation++
        #return self.namespace.__getattr__("nSpecies") #Vin: this is creating issues with nanobind
        return self.namespace.mix.nSpecies()
        

    @property
    def num_reactions(self):
        #return self.namespace.__getattr__("num_reactions")#Vin: could not find num_reactions in mutation++
        #return self.namespace.__getattr__("nReactions")
        return self.namespace.mix.nReactions()

    @property
    def molecular_weights(self):
        #value = self.namespace.__getattr__("speciesMw")
        #return value() if callable(value) else value
        return np.array([
            self.namespace.mix.speciesMw(i)
            for i in range(self.num_species)
        ])

    @property
    def species_names(self):
        #return [self.species_name(i) for i in range(self.num_species)]
        return [
            self.namespace.mix.speciesName(i)
            for i in range(self.num_species)
        ]

    def reactions(self): #-> List[mpp.Reaction]:
        value = self.namespace.__getattr__("reactions")
        return value() if callable(value) else value

    def reaction(self, reaction_index: int): # -> mpp.Reaction:
        return self.namespace.__getattr__("reaction", reaction_index)

    def species_index(self, species_name: str) -> int:
        #return self.namespace.__getattr__("speciesIndex", species_name)
        return self.namespace.mix.speciesIndex(species_name)

    def species_name(self, species_index: int) -> str:
        #return self.namespace.__getattr__("speciesName", species_index)
        return self.namespace.mix.speciesName(species_index)   

    def reactants(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index).reactants

    def products(self, reaction_index: int) -> List[int]:
        return self.reaction(reaction_index).products

    def stoichiometric_coefficients(self, reaction_index: int) -> List[int]:
        return [1 for i in self.reactants(reaction_index)]

    def participation_set(self,
                          species_id: Union[int, str]) -> Tuple[List[int]]:

        if isinstance(species_id, int):
            assert species_id < self.num_species
            species_index = species_id
        elif isinstance(species_id, str):
            species_index = self.species_index(species_id)
        else:
            raise ValueError("species_id must be either str or int, "
                             f"but received {type(species_id)}")

        fwd_set = [i for i, r in enumerate(self.reactions())
                   if species_index in r.reactants]
        rev_set = [i for i, r in enumerate(self.reactions())
                   if species_index in r.products]

        return (fwd_set, rev_set)

    def finalize(self):
        if hasattr(self.namespace, "finalize"):
            self.namespace.finalize()
        elif hasattr(self.namespace, "mix") and hasattr(self.namespace.mix, "finalize"):
            self.namespace.mix.finalize()
    

    def make_rate_coefficient(self, reaction_index, hardcode_params):
        rate = self.reaction(reaction_index).rate_law()
        temp = self.reaction(reaction_index).fwd_rate_coeff_temperature
        if hardcode_params:
            params = {
                'a': rate.log_pre_exponential,
                'b': rate.exponent,
                't_a': rate.activation_temperature
            }
            k_fwd = make_arrhenius(
                reaction_index=reaction_index, params=params
            )
        else:
            params = np.array([
                rate.log_pre_exponential,
                rate.exponent,
                rate.activation_temperature
            ])
            k_fwd = make_arrhenius(
                reaction_index=reaction_index
            )

        if temp == 'translational':
            return k_fwd, params
        else:
            k_fwd.expr = substitute(
                k_fwd.expr, {'temperature': _temp_map[temp]}
            )
            return k_fwd, params


    def make_species_production_rate(self, species_index):
        if isinstance(species_index, str):
            species_index = self.species_index(species_index)
        fwd_set, rev_set = self.participation_set(species_index)
        stoich_fwd = [self.reactants(r).count(species_index)
                      for r in fwd_set]
        stoich_rev = [self.products(r).count(species_index)
                      for r in rev_set]
        return species_production_rate_expr(
            species_index,
            fwd_set,
            rev_set,
            stoich_fwd,
            stoich_rev
        )
