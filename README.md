# Missing Term Recovery in Stellar Structure Equations using Scientific Machine Learning

**Author**: Vedaswaroop Kunamneni  

---

## Abstract

This repository contains the full Julia implementation for the project "Scientific Machine Learning for Missing Term Recovery in Stellar Structure Equations." The primary objective is to demonstrate the use of a Universal Differential Equation (UDE)—a form of Physics-Informed Neural Network (PINN)—to discover a missing physical term in a simplified stellar energy equation. This physics-informed approach is benchmarked against a standard black-box Neural ODE. The knowledge encoded in the trained UDE is subsequently distilled back into a human-readable, closed-form analytical expression using symbolic regression. The project serves as a comprehensive demonstration of the modern Scientific Machine Learning (SciML) workflow for automated scientific discovery.

---

## Results at a Glance

Running the pipeline will produce two key plots that summarize the project's findings:

1.  **Missing Term Recovery:** This plot shows how well the UDE's neural network (orange line) learned to approximate the true, underlying physical law (blue dashed line). The discrepancy highlights the scientific challenge of learning a physical relationship from indirect data.

    ![](outputs/3_ude_recovery.png)

2.  **Final Model Comparison:** This plot compares the final integrated luminosity profiles of all models. It shows that despite the imperfect physics recovery, the UDE's final prediction (orange dashed line) is nearly identical to the true physics, while the black-box Neural ODE (green dotted line) shows significant deviation.

    ![](outputs/4_final_comparison.png)

---

## Scientific Motivation

The structure and evolution of stars are governed by a set of coupled differential equations describing the balance between gravity, pressure, and energy transport. A central component of these models is the stellar energy equation, which connects local energy generation processes (e.g., nuclear fusion) and energy loss mechanisms (e.g., neutrino emission) to the star's observable luminosity.

This project investigates a scenario common in scientific modeling: a model that is known to be incomplete. We begin with a well-defined, simplified 1D stellar energy equation and then operate under the assumption that one of its constituent physical terms—an energy loss mechanism—is unknown. The goal is to employ modern machine learning techniques to discover the functional form of this missing physical law directly from synthetic observational data.

---

## Mathematical Formulation

The project is based on the following set of equations as defined in the official project document.

### The Stellar Energy Equation (Toy Model)

The system is governed by a single ordinary differential equation for the enclosed luminosity $L$ as a function of the radius $r$:

$$
\frac{dL}{dr} = 4\pi r^2 \rho(r) \left[ \epsilon_{\text{nuc}}(r) - \epsilon_{\text{loss}}(r) \right] \quad \text{(1)}
$$

### Surrogate Physical Profiles

To maintain a tractable system, the density $\rho(r)$ and temperature $T(r)$ are not solved for dynamically but are given by prescribed analytical forms:

$$
\rho(r) = \rho_c \left(1 - \frac{r^2}{R^2}\right)_+ \quad \text{and} \quad T(r) = T_c \left(1 - \frac{r^2}{R^2}\right)_+ \quad \text{(2)}
$$

where $(x)_+ = \max(x,0)$.

### Constitutive Physics

The known and unknown physics terms are defined as:
- **Known Nuclear Term**:
$$
\epsilon_{\text{nuc}}(r) = \epsilon_0 \rho(r) \left( \frac{T(r)}{T_*} \right)^{n_{\text{nuc}}} \quad \text{(3)}
$$
- **"Unknown" Loss Term (Ground Truth)**:
$$
\epsilon_{\text{loss}}(r) = \lambda_0 \left( \frac{T(r)}{T_*} \right)^{n_{\text{loss}}} \quad \text{(4)}
$$

### The Universal Differential Equation (UDE) Formulation

The UDE model retains the known physical terms and replaces only the unknown `ε_loss` term with a neural network `NN_θ(r, L)`:
$$
\frac{dL}{dr} = 4\pi r^2 \rho(r) \left[ \epsilon_{\text{nuc}}(r) - NN_{\theta}(r, L) \right] \quad \text{(5)}
$$

---

## Project Objectives

This project follows four explicit objectives as specified in the project roadmap:

1.  **Baseline ODE (Physics):** Implement and validate Eq. (1) with Eqs. (3)–(4) to generate the ground-truth data.
2.  **Neural ODE (Forecasting):** Replace the full right-hand side of the ODE with a neural network and fit it to the data to benchmark pure black-box performance.
3.  **UDE (Missing-Term Recovery):** Use Eq. (5) to learn `NN_θ(r,L)` while keeping `ε_nuc` analytic, and evaluate the recovery fidelity of the learned term against the ground truth.
4.  **Symbolic Discovery:** Apply symbolic regression on the input-output pairs of the trained `NN_θ` to obtain a closed-form, human-readable equation for `ε_loss`.

---

## Implementation

### Baseline Parameters

The code utilizes the following nondimensional parameters as suggested in the project document:

| Parameter | Description | Value |
| :--- | :--- | :--- |
| $R$ | Stellar Radius Scale | `1.0` |
| $\rho_c$ | Central Density Scale | `1.0` |
| $T_c$ | Central Temperature Scale| `1.0` |
| $T_*$ | Reference Temperature | `1.0` |
| $\epsilon_0$ | Nuclear Source Scale | `1.0` |
| $\lambda_0$ | True Loss Scale | `0.3` |
| $n_{\text{nuc}}$| Nuclear Term Exponent | `4` |
| $n_{\text{loss}}$| Loss Term Exponent | `9` |

### File Structure

-   `run_pipeline.jl`: The main executable script that runs the entire project pipeline.
-   `src/StellarModels.jl`: A Julia module defining the baseline physical model, equations, and parameters.
-   `Project.toml` / `Manifest.toml`: Julia environment files managing all project dependencies.
-   `data/`: Directory where the generated synthetic observational data is saved.
-   `outputs/`: Directory where all generated plots and figures are saved.
-   `run_log.md`: A markdown file that automatically logs the hyperparameters and final results of each execution.

---

## Installation and Execution

### Installation

This project is built in the Julia programming language.

1.  **Install Julia:** Download and install the latest stable version of Julia from the [official website](https://julialang.org/downloads/).
2.  **Activate Environment:** Navigate to the project's root directory in a terminal, start the Julia REPL (`julia`), and run the following commands to install the required packages:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

### Execution

To run the entire experimental pipeline, execute the main script from your **terminal** (not inside the Julia REPL):

```bash
julia run_pipeline.jl