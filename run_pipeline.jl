# Activate the project environment to ensure all dependencies are correctly loaded
using Pkg
Pkg.activate(".")

# Import all necessary libraries for the pipeline
using DifferentialEquations  # For solving ODEs
using Flux                 # For building neural networks
using DiffEqFlux           # For creating Neural ODEs and UDEs
using Optimization         # The optimization interface
using OptimizationOptimisers # The optimizers (e.g., ADAM)
using Plots                # For generating plots
using Random               # For generating noise
using CSV                  # For saving data to files
using DataFrames           # For handling data in tables
using SymbolicRegression   # For the symbolic discovery step
using Dates                # For logging run timestamps

# --- Hyperparameters & Configuration (as per user request) ---
const UDE_LEARNING_RATE = 0.001
const UDE_MAX_ITERS = 5000
const UDE_L2_LAMBDA = 1e-5
const NODE_LEARNING_RATE = 0.001
const NODE_MAX_ITERS = 5000
# ----------------------------------------------------------------

# Include the custom module containing the physics definitions
include("src/StellarModels.jl")
using .StellarModels

# Create output directories if they don't already exist
!ispath("data") && mkdir("data")
!ispath("outputs") && mkdir("outputs")

println("--- Starting Project: Missing Term Recovery in Stellar Equations ---")

println("Step 1: Solving baseline physics ODE to get ground truth...")
L₀ = [0.0]
r_span = (0.0, R)
prob_ode = ODEProblem(stellar_ode!, L₀, r_span)
sol_ode = solve(prob_ode, Tsit5(), saveat=0.01)

plt_base = plot(sol_ode, xlabel="Radius (r)", ylabel="Luminosity L(r)", label="Physics-Based Solution", lw=2)
title!("Baseline ODE Solution")
savefig(plt_base, "outputs/1_baseline_solution.png")
println("Saved baseline plot to outputs/1_baseline_solution.png")

println("\nStep 2: Generating synthetic observational data...")
radii = sol_ode.t
true_L_values = hcat(sol_ode.u...)'

df_baseline = DataFrame(radius=radii, luminosity=vec(true_L_values))
CSV.write("data/baseline_solution.csv", df_baseline)
println("Saved baseline solution data to data/baseline_solution.csv")

# Add Gaussian noise to mimic observational data
noise_level = 0.01 * maximum(true_L_values)
Random.seed!(123) # for reproducibility
noise = noise_level .* randn(length(radii))
noisy_L_data = true_L_values .+ noise
noisy_L_data[1] = 0.0 # The boundary condition L(0)=0 is known exactly

df_noisy = DataFrame(radius=radii, luminosity=vec(noisy_L_data))
CSV.write("data/synthetic_luminosity.csv", df_noisy)
println("Saved synthetic data to data/synthetic_luminosity.csv")

# --- Objective 2 / Action Item 3: Train a Neural ODE Baseline ---
println("\nStep 3: Training a Black-Box Neural ODE...")

# Define the neural network that represents the entire RHS
nn_node = Flux.Chain(
    Flux.Dense(1, 64, swish),
    Flux.Dense(64, 64, swish),
    Flux.Dense(64, 1)
) |> Flux.f64
p_node, re_node = Flux.destructure(nn_node)

# Define the Neural ODE function dL/dr = NN(L)
function neural_ode_func!(dL, L, p, r)
    dL[1] = re_node(p)(L)[1]
end

# Function to predict the ODE solution for a given set of NN parameters
function predict_node(p)
    prob_node = ODEProblem(neural_ode_func!, L₀, r_span, p)
    solve(prob_node, Tsit5(), saveat=radii, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
end

# Loss function: Mean squared error between prediction and noisy data
function loss_node(p)
    pred = predict_node(p)
    if pred.retcode != :Success
        return Inf
    end
    loss = sum(abs2, noisy_L_data .- hcat(pred.u...)')
    return loss
end

# Callback to show progress during training
iter_node = 0
callback_node = function (p, l)
    global iter_node
    iter_node += 1
    if iter_node % 100 == 0
        println("Iter: $(iter_node) | Current Neural ODE Loss: ", l)
    end
    return false
end
adtype_node = Optimization.AutoZygote()
optf_node = OptimizationFunction((x, p) -> loss_node(x), adtype_node)
optprob_node = Optimization.OptimizationProblem(optf_node, p_node)

result_node = Optimization.solve(optprob_node, ADAM(NODE_LEARNING_RATE), callback=callback_node, maxiters=NODE_MAX_ITERS)
p_trained_node = result_node.u
println("Neural ODE training complete.")

# --- Objective 3 / Action Item 4: Train the UDE for Missing-Term Recovery ---
println("\nStep 4: Training Physics-Informed Universal Differential Equation (UDE)...")

nn_ude = Flux.Chain(
    Flux.Dense(2, 128, swish),  # Wider
    Flux.Dense(128, 128, swish), # Wider and one extra layer
    Flux.Dense(128, 64, swish),
    Flux.Dense(64, 1, x -> max(0, x))
) |> Flux.f64
p_ude, re_ude = Flux.destructure(nn_ude)

# Define scaling factors for normalization
const r_scale = maximum(radii)
const L_scale = maximum(true_L_values)

# This function implements Equation (5) from the project roadmap EXACTLY
function ude_func!(dL, L, p, r)
    # The known physics term
    known_term = 4π * r^2 * ρ(r) * ϵ_nuc(r)
    
    # Normalize inputs to the NN: [r, L] to be roughly [0,1]
    r_norm = r / r_scale
    L_norm = L[1] / L_scale
    input_vector = [r_norm, L_norm]
    
    # The NN learns the entire unknown loss term, NN_theta(r, L)
    learned_loss_term = re_ude(p)(input_vector)[1]
    
    # The final UDE combines the known and learned parts
    dL[1] = known_term - 4π * r^2 * ρ(r) * learned_loss_term
end

function predict_ude(p)
    prob_ude = ODEProblem(ude_func!, L₀, r_span, p)
    solve(prob_ude, Tsit5(), saveat=radii, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
end

function l2_regularization(p)
    return sum(abs2, p)
end

function loss_ude(p)
    pred = predict_ude(p)
    if pred.retcode != :Success
        return Inf, pred
    end
    data_loss = sum(abs2, noisy_L_data .- hcat(pred.u...)')
    reg_loss = UDE_L2_LAMBDA * l2_regularization(p)
    return data_loss + reg_loss, pred
end

# Callback to show progress during training
iter_ude = 0
callback_ude = function (p, l, pred...)
    global iter_ude
    iter_ude += 1
    if iter_ude % 100 == 0
       println("Iter: $(iter_ude) | Current UDE Loss: ", l)
    end
    return false
end
adtype_ude = Optimization.AutoZygote()
optf_ude = OptimizationFunction((x, p) -> loss_ude(x)[1], adtype_ude)
optprob_ude = Optimization.OptimizationProblem(optf_ude, p_ude)
result_ude = Optimization.solve(optprob_ude, ADAM(UDE_LEARNING_RATE), callback=callback_ude, maxiters=UDE_MAX_ITERS)
p_trained_ude = result_ude.u
println("UDE training complete.")

# --- Evaluate Recovery Fidelity ---
println("\nStep 5: Evaluating UDE recovery fidelity...")
# The NN output *is* the learned loss term. We evaluate it at each true (r, L) point.
learned_loss_values = [re_ude(p_trained_ude)([r/r_scale, l_val/L_scale])[1] for (r, l_val) in zip(radii, vec(true_L_values))]
true_loss_values = ϵ_loss.(radii)

plt_recovery = plot(radii, true_loss_values, label="True ϵ_loss(r)", lw=3, ls=:dash)
plot!(plt_recovery, radii, learned_loss_values, label="Learned ϵ_loss(r) (from UDE)", lw=2)
title!("Missing Term Recovery (UDE)")
xlabel!("Radius (r)")
ylabel!("Loss Term")
savefig(plt_recovery, "outputs/3_ude_recovery.png")
println("Saved recovery plot to outputs/3_ude_recovery.png")

# --- Objective 4 / Action Item 4: Symbolic Discovery ---
println("\nStep 6: Performing symbolic regression on the learned term...")

# The symbolic regression task is to find a function of 'r' and 'L' that matches the output of the NN.
# Since the true physics only depends on r (via T(r)), a successful discovery should yield a function that ignores L.
X = vcat(radii', true_L_values')
y = learned_loss_values

options = SymbolicRegression.Options(
    binary_operators=[+, *, -, /, ^],
    unary_operators=[], # Keep it simple to encourage finding the polynomial form
    npopulations=32,
    complexity_of_constants=1,
    complexity_of_variables=1
)
hall_of_fame = SymbolicRegression.EquationSearch(
    X, y, niterations=100, options=options, variable_names=["r", "L"], parallelism=:multithreading
)

println("\n--- Symbolic Regression Results ---")
best_equations = SymbolicRegression.calculate_pareto_frontier(hall_of_fame)
equation_string = SymbolicRegression.string_tree(best_equations[end].tree, options)
println("The best discovered equation for the MISSING TERM is:")
println("ϵ_loss(r, L) ≈ ", equation_string)
println("The true equation is a function of r only: 0.3 * (1 - r^2)^9")
println("------------------------------------")

# --- Final Model Comparison ---
println("\nStep 7: Generating final comparison plot...")
sol_node_final = predict_node(p_trained_node)
sol_ude_final = predict_ude(p_trained_ude)

plt_final = plot(sol_ode, label="True Physics", lw=3)
plot!(plt_final, sol_ude_final, label="UDE Prediction", lw=2, ls=:dash)
plot!(plt_final, sol_node_final, label="Neural ODE Prediction", lw=2, ls=:dot)
title!("Model Comparison: Luminosity Profiles")
xlabel!("Radius (r)")
ylabel!("Luminosity L(r)")
savefig(plt_final, "outputs/4_final_comparison.png")
println("Saved final comparison plot to outputs/4_final_comparison.png")

# --- Action Item 5 & 6: Log Run Details ---
println("\nLogging run details...")

final_ude_loss, _ = loss_ude(p_trained_ude)
final_node_loss = loss_node(p_trained_node)

open("run_log.md", "a") do f
    write(f, "## Run Summary: $(now())\n\n")
    write(f, "**Hyperparameters:**\n")
    write(f, "- UDE Learning Rate: `$UDE_LEARNING_RATE`\n")
    write(f, "- UDE Max Iterations: `$UDE_MAX_ITERS`\n")
    write(f, "- UDE L2 Lambda: `$UDE_L2_LAMBDA`\n")
    write(f, "- Neural ODE Learning Rate: `$NODE_LEARNING_RATE`\n")
    write(f, "- Neural ODE Max Iterations: `$NODE_MAX_ITERS`\n\n")
    write(f, "**Final Results:**\n")
    write(f, "- Final UDE Loss: `$final_ude_loss`\n")
    write(f, "- Final Neural ODE Loss: `$final_node_loss`\n")
    write(f, "- Discovered Equation: `ϵ_loss(r, L) ≈ $equation_string`\n")
    write(f, "\n---\n\n")
end

println("Run details logged to run_log.md")

println("\n--- Project Pipeline Finished ---")