# Activate the project environment
using Pkg
Pkg.activate(".")

# Import necessary libraries
using DifferentialEquations
using Flux
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using Plots
using Random
using CSV
using DataFrames
using SymbolicRegression
using Dates

# --- Hyperparameters & Configuration ---
const UDE_LEARNING_RATE = 0.001
const UDE_MAX_ITERS = 5000
const UDE_L2_LAMBDA = 1e-5
const NODE_LEARNING_RATE = 0.01
const NODE_MAX_ITERS = 1000
# ------------------------------------

include("src/StellarModels.jl")
using .StellarModels

!ispath("data") && mkdir("data")
!ispath("plots") && mkdir("plots")

println("--- Starting Project: Missing Term Recovery in Stellar Equations ---")

println("Step 1: Solving baseline physics ODE...")
L₀ = [0.0]
r_span = (0.0, R)
prob_ode = ODEProblem(stellar_ode!, L₀, r_span)
sol_ode = solve(prob_ode, Tsit5(), saveat=0.01)

plt_base = plot(sol_ode, xlabel="Radius (r)", ylabel="Luminosity L(r)", label="Physics-Based Solution", lw=2)
title!("Baseline ODE Solution")
savefig(plt_base, "plots/1_baseline_solution.png")
println("Saved baseline plot to plots/1_baseline_solution.png")

println("\nStep 2: Generating synthetic observational data...")
radii = sol_ode.t
true_L_values = hcat(sol_ode.u...)'

df_baseline = DataFrame(radius=radii, luminosity=vec(true_L_values))
CSV.write("data/baseline_solution.csv", df_baseline)
println("Saved baseline solution data to data/baseline_solution.csv")

noise_level = 0.01 * maximum(true_L_values)
Random.seed!(123)
noise = noise_level .* randn(length(radii))
noisy_L_data = true_L_values .+ noise
noisy_L_data[1] = 0.0

df = DataFrame(radius=radii, luminosity=vec(noisy_L_data))
CSV.write("data/synthetic_luminosity.csv", df)
println("Saved synthetic data to data/synthetic_luminosity.csv")

println("\nStep 3: Training a Black-Box Neural ODE...")

nn_node = Flux.Chain(
    Flux.Dense(1, 64, swish),
    Flux.Dense(64, 64, swish),
    Flux.Dense(64, 1)
) |> Flux.f64
p_node, re_node = Flux.destructure(nn_node)

function neural_ode_func!(dL, L, p, r)
    dL[1] = re_node(p)(L)[1]
end

function predict_node(p)
    prob_node = ODEProblem(neural_ode_func!, L₀, r_span, p)
    solve(prob_node, Tsit5(), saveat=radii, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
end

function loss_node(p)
    pred = predict_node(p)
    if pred.retcode != :Success
        return Inf
    end
    loss = sum(abs2, noisy_L_data .- hcat(pred.u...)')
    return loss
end

callback_node = function (p, l)
    println("Current Neural ODE Loss: ", l)
    return false
end
adtype_node = Optimization.AutoZygote()
optf_node = OptimizationFunction((x, p) -> loss_node(x), adtype_node)
optprob_node = Optimization.OptimizationProblem(optf_node, p_node)

result_node = Optimization.solve(optprob_node, ADAM(NODE_LEARNING_RATE), callback=callback_node, maxiters=NODE_MAX_ITERS)
p_trained_node = result_node.u
println("Neural ODE training complete.")

println("\nStep 4: Training Physics-Informed Universal Differential Equation (UDE)...")

nn_ude = Flux.Chain(
    Flux.Dense(2, 64, swish),
    Flux.Dense(64, 64, swish),
    Flux.Dense(64, 1)
) |> Flux.f64
nn_ude[3].weight .= 1e-4 .* randn(Float64, size(nn_ude[3].weight))
nn_ude[3].bias .= [λ₀]
p_ude, re_ude = Flux.destructure(nn_ude)

const r_scale = maximum(radii)
const L_scale = maximum(true_L_values)

function ude_func!(dL, L, p, r)
    known_term = 4π * r^2 * ρ(r) * ϵ_nuc(r)
    r_norm = r / r_scale
    L_norm = L[1] / L_scale
    input_vector = [r_norm, L_norm]
    learned_factor = re_ude(p)(input_vector)[1]
    learned_loss_term = learned_factor * (T(r) / T_star)^9
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
    data_loss = sum(abs2, true_L_values .- hcat(pred.u...)')
    reg_loss = UDE_L2_LAMBDA * l2_regularization(p)
    return data_loss + reg_loss, pred
end

callback_ude = function (p, l, pred...)
    println("Current UDE Loss: ", l)
    return false
end
adtype_ude = Optimization.AutoZygote()
optf_ude = OptimizationFunction((x, p) -> loss_ude(x)[1], adtype_ude)
optprob_ude = Optimization.OptimizationProblem(optf_ude, p_ude)
result_ude = Optimization.solve(optprob_ude, ADAM(UDE_LEARNING_RATE), callback=callback_ude, maxiters=UDE_MAX_ITERS)
p_trained_ude = result_ude.u
println("UDE training complete.")

println("\nStep 5: Evaluating UDE recovery fidelity...")
learned_factors = [re_ude(p_trained_ude)([r/r_scale, sol_ode(r)[1]/L_scale])[1] for r in radii]
learned_loss_values = learned_factors .* (T.(radii) ./ T_star).^9
true_loss_values = ϵ_loss.(radii)

plt_recovery = plot(radii, true_loss_values, label="True ϵ_loss(r)", lw=3, ls=:dash)
plot!(plt_recovery, radii, learned_loss_values, label="Learned ϵ_loss(r) (from UDE)", lw=2)
title!("Missing Term Recovery (UDE)")
xlabel!("Radius (r)")
ylabel!("Loss Term")
savefig(plt_recovery, "plots/3_ude_recovery.png")
println("Saved recovery plot to plots/3_ude_recovery.png")

println("\nStep 6: Performing symbolic regression on the learned factor...")

X_r = radii'
X_L = hcat(sol_ode(radii).u...)[1,:]'
X = vcat(X_r, X_L)
y = learned_factors

options = SymbolicRegression.Options(binary_operators=[+, *, -, /, ^], unary_operators=[], npopulations=20)
hall_of_fame = SymbolicRegression.EquationSearch(X, y, niterations=50, options=options, variable_names=["r", "L"], parallelism=:multithreading)

println("\n--- Symbolic Regression Results ---")
best_equations = SymbolicRegression.calculate_pareto_frontier(hall_of_fame)
equation_string = SymbolicRegression.string_tree(best_equations[end].tree, options)
println("The best discovered equation for the CORRECTION FACTOR is:")
println(equation_string)
println("(This should ideally be a constant close to λ₀ = 0.3)")
println("------------------------------------")

println("\nStep 7: Generating final comparison plot...")
sol_node_final = predict_node(p_trained_node)
sol_ude_final = predict_ude(p_trained_ude)

plt_final = plot(sol_ode, label="True Physics", lw=3)
plot!(plt_final, sol_ude_final, label="UDE Prediction", lw=2, ls=:dash)
plot!(plt_final, sol_node_final, label="Neural ODE Prediction", lw=2, ls=:dot)
title!("Model Comparison: Luminosity Profiles")
xlabel!("Radius (r)")
ylabel!("Luminosity L(r)")
savefig(plt_final, "plots/4_final_comparison.png")
println("Saved final comparison plot to plots/4_final_comparison.png")

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
    write(f, "- Discovered Equation: `$equation_string`\n")
    write(f, "\n---\n\n")
end

println("Run details logged to run_log.md")

println("\n--- Project Pipeline Finished ---")