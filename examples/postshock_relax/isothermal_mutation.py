import os
import time
import warnings

import mutationpp as mpp
import numpy as np
from matplotlib import pyplot as plt


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Conversion of an array with ndim > 0 to a scalar*",
)


def make_mixture():
    data_dir = os.environ.get("MUTATION_DB")
    if data_dir is None:
        raise RuntimeError("Please set MUTATION_DB to the Mutation++ data directory.")

    mpp.GlobalOptions.dataDirectory(data_dir)

    opts = mpp.MixtureOptions("air_5")
    mix = mpp.Mixture(opts)

    return mix


def rhs(mix, rhoi, temperature_vec, state_model):
    rhoi_ = np.maximum(rhoi, 1.0e-300)
    rhoi_ = np.ascontiguousarray(rhoi_, dtype=np.float64)

    temperature_vec = np.ascontiguousarray(temperature_vec, dtype=np.float64)

    mix.setState(rhoi_, temperature_vec, state_model)
    return mix.netProductionRates()


def finite_difference_jacobian(residual, state, state_prev, step_size):
    n = len(state)
    jac = np.zeros((n, n))

    f0 = residual(state, state_prev, step_size)

    rho_scale = max(np.sum(state), 1.0e-30)

    for j in range(n):
        h = 1.0e-8 * max(abs(state[j]), rho_scale)

        state_pert = state.copy()
        state_pert[j] += h

        fj = residual(state_pert, state_prev, step_size)

        jac[:, j] = (fj - f0) / h

    return jac


def crank_nicolson_step(mix, state, step_size, temperature_vec, state_model):
    state_prev = state.copy()

    rhs_prev = rhs(mix, state_prev, temperature_vec, state_model)

    def residual(state_current, state_old, dt):
        rhs_current = rhs(mix, state_current, temperature_vec, state_model)

        return (
            state_current
            - state_old
            - 0.5 * dt * (rhs_current + rhs_prev)
        )

    tol = 1.0e-10
    max_iter = 40

    state_new = state.copy()
    err = np.inf

    for it in range(max_iter):
        res = residual(state_new, state_prev, step_size)

        jac = finite_difference_jacobian(
            residual,
            state_new,
            state_prev,
            step_size,
        )

        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac, -res, rcond=None)[0]

        state_new = state_new + delta
        state_new = np.maximum(state_new, 1.0e-300)
        err = np.linalg.norm(delta)

        if err < tol:
            return state_new, it + 1, err

    return state_new, max_iter, err


def time_march(mix, num_steps, step_size, initial_state, temperature_vec):
    ns = mix.nSpecies()

    state_model = 1

    sol = np.empty((num_steps + 1, ns))
    sol[0, :] = initial_state.copy()

    state = initial_state.copy()

    print_every = max(1, int(os.environ.get("PRINT_EVERY", "100")))

    for step in range(num_steps):
        t0 = time.time()

        state, newton_it, newton_err = crank_nicolson_step(
            mix,
            state,
            step_size,
            temperature_vec,
            state_model,
        )

        sol[step + 1, :] = state

        elapsed = time.time() - t0

        if step % print_every == 0:
            total_rho = np.sum(state)
            min_rhoi = np.min(state)
            print(
                f"Step {step:6d}: "
                f"Newton it = {newton_it:2d}, "
                f"err = {newton_err:.3e}, "
                f"rho = {total_rho:.6e}, "
                f"min rhoi = {min_rhoi:.3e}, "
                f"cost = {elapsed:.4e} s"
            )

    return sol


def main():
    mix = make_mixture()

    ns = mix.nSpecies()
    nT = mix.nEnergyEqns()

    species_names = [mix.speciesName(i) for i in range(ns)]
    molecular_weights = np.array([mix.speciesMw(i) for i in range(ns)])

    print("Species:", species_names)
    print("Molecular weights:", molecular_weights)
    print("nEnergyEqns:", nT)

    cold_temp = 300.0
    bath_temp = 1.0e4
    pressure = 1.0e3

    temperature_vec = bath_temp * np.ones(nT)

    mole_fractions = np.zeros(ns)
    mole_fractions[mix.speciesIndex("O2")] = 0.21
    mole_fractions[mix.speciesIndex("N2")] = 0.79

    mass_fractions = (
        molecular_weights * mole_fractions
        / np.sum(molecular_weights * mole_fractions)
    )

    gas_constant = 8.31446261815324

    mix_molecular_weight = 1.0 / np.sum(
        mass_fractions / molecular_weights
    )

    density = pressure * mix_molecular_weight / (
        gas_constant * cold_temp
    )

    densities = density * mass_fractions

    densities = np.ascontiguousarray(densities, dtype=np.float64)

    print("Initial mixture molecular weight:", mix_molecular_weight)
    print("Initial density:", density)
    print("Initial mass fractions:", mass_fractions)
    print("Initial species densities:", densities)

    num_steps = int(os.environ.get("NUM_STEPS", "10000"))
    step_size = float(os.environ.get("STEP_SIZE", "1.0e-8"))

    print("num_steps:", num_steps)
    print("step_size:", step_size)

    sol = time_march(
        mix,
        num_steps,
        step_size,
        densities,
        temperature_vec,
    )

    sol_t = step_size * np.arange(num_steps + 1)
    sol_d = np.sum(sol, axis=1)
    sol_y = sol / sol_d[:, None]

    colors = [
        "k",
        "orangered",
        "mediumseagreen",
        "royalblue",
        "mediumpurple",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.spines[["top", "right"]].set_visible(False)

    for i in range(ns):
        ax.loglog(
            sol_t[1:],
            sol_y[1:, i],
            color=colors[i % len(colors)],
            linewidth=2,
            label=species_names[i],
        )

    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Mass Fractions", fontsize=16)
    ax.legend(
        frameon=False,
        labelcolor="linecolor",
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
        ncol=ns,
        fontsize=12,
    )

    plt.savefig("output_mutation.png", bbox_inches="tight")
    plt.close()

    np.savetxt(
        "solution_mutation.csv",
        np.column_stack((sol_t, sol_y)),
        delimiter=",",
        header="time," + ",".join(species_names),
        comments="",
    )

    print("Done.")
    print("Wrote output_mutation.png")
    print("Wrote solution_mutation.csv")


if __name__ == "__main__":
    main()
