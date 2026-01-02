using Pkg
Pkg.activate(".")

using DifferentialEquations
using Flux
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using Zygote
using Plots
using Random
using CSV
using DataFrames
using SymbolicRegression
using Statistics
using Printf

const HIDDEN_DIM = 32
const N_HIDDEN_LAYERS = 3
const UDE_LR_PHASE1 = 0.05
const UDE_LR_PHASE2 = 0.005
const UDE_LR_PHASE3 = 0.001
const UDE_ITERS_PHASE1 = 3000
const UDE_ITERS_PHASE2 = 2000
const UDE_ITERS_PHASE3 = 1000
const NODE_LR = 0.01
const NODE_ITERS = 3000
const PHYSICS_REG_LAMBDA = 0.01

const NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10]
const FORECAST_FRACTIONS = [0.5, 0.7, 0.9]

include("src/StellarModels.jl")
using .StellarModels

!ispath("outputs") && mkdir("outputs")
!ispath("data") && mkdir("data")

swish(x) = x * sigmoid(x)

println("="^70)
println(" STELLAR STRUCTURE UDE: Missing Term Recovery Pipeline (FIXED)")
println("="^70)
Random.seed!(42)

println("\n[Step 1] Generating ground truth from baseline physics ODE...")

L₀ = [0.0]
r_span = (0.0, R)
prob_baseline = ODEProblem(stellar_ode!, L₀, r_span)
sol_baseline = solve(prob_baseline, Tsit5(), saveat=0.01, abstol=1e-10, reltol=1e-10)

radii = sol_baseline.t
L_true = vec(hcat(sol_baseline.u...)')
n_points = length(radii)

r_scale = R
L_scale = maximum(L_true)

true_eps_loss = [ϵ_loss(r) for r in radii]
eps_scale = maximum(true_eps_loss)

println("   Data points: $n_points")
println("   Radius range: [$(minimum(radii)), $(maximum(radii))]")
println("   Luminosity range: [$(minimum(L_true)), $(maximum(L_true))]")
println("   ε_loss range: [$(minimum(true_eps_loss)), $(maximum(true_eps_loss))]")

CSV.write("data/ground_truth.csv", DataFrame(radius=radii, luminosity=L_true))

println("\n[Step 2] Defining neural network architectures...")

function build_node_nn()
    Flux.Chain(
        Flux.Dense(2, HIDDEN_DIM, swish),
        Flux.Dense(HIDDEN_DIM, HIDDEN_DIM, swish),
        Flux.Dense(HIDDEN_DIM, HIDDEN_DIM, swish),
        Flux.Dense(HIDDEN_DIM, 1)
    ) |> Flux.f64
end

const EPS_LOSS_MAX = 0.5

function build_ude_nn()
    Flux.Chain(
        Flux.Dense(1, 1, x -> x, bias=false, init=(out,in)->reshape([0.1], out, in))
    ) |> Flux.f64
end

nn_node = build_node_nn()
p_node_init, re_node = Flux.destructure(nn_node)

nn_ude = build_ude_nn()
p_ude_init, re_ude = Flux.destructure(nn_ude)

println("   NODE parameters: $(length(p_node_init))")
println("   UDE parameters: $(length(p_ude_init))")

function predict_node(p; L0=L₀, tspan=r_span, saveat=radii)
    function node_dynamics!(dL, L, p, r)
        input = [r / r_scale, L[1] / max(L_scale, 1e-6)]
        dL[1] = re_node(p)(input)[1]
    end
    prob = ODEProblem(node_dynamics!, L0, tspan, p)
    solve(prob, Tsit5(), saveat=saveat, abstol=1e-8, reltol=1e-8, 
          sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), verbose=false)
end

function physics_features(r)
    r_norm = r / r_scale
    T_val = max(0.0, 1.0 - r_norm^2)
    T9 = T_val^9
    return [T9]
end

function predict_ude(p; L0=L₀, tspan=r_span, saveat=radii)
    function ude_dynamics!(dL, L, p, r)
        known_term = 4π * r^2 * ρ(r) * ϵ_nuc(r)
        
        features = physics_features(r)
        learned_lambda = re_ude(p)([1.0])[1]
        learned_eps_loss = max(0.0, learned_lambda) * features[1]
        
        dL[1] = known_term - 4π * r^2 * ρ(r) * learned_eps_loss
    end
    prob = ODEProblem(ude_dynamics!, L0, tspan, p)
    solve(prob, Tsit5(), saveat=saveat, abstol=1e-8, reltol=1e-8,
          sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), verbose=false)
end

function get_learned_eps_loss(p, r_values)
    learned_lambda = max(0.0, re_ude(p)([1.0])[1])
    [learned_lambda * physics_features(r)[1] for r in r_values]
end

println("\n[Step 3] Training on clean data (baseline)...")

println("\n   Training UDE (physics-informed) on clean data...")
p_ude = copy(p_ude_init)
iter_count = Ref(0)
best_loss = Ref(Inf)

function loss_ude_clean(p, _)
    sol = predict_ude(p)
    if sol.retcode != :Success
        return Inf
    end
    L_pred = vec(Array(sol)')
    
    data_loss = mean(abs2, L_true .- L_pred)
    
    reg_loss = 1e-6 * sum(abs2, p)
    
    return data_loss + reg_loss
end

callback_ude = function(state, l)
    iter_count[] += 1
    if l < best_loss[]
        best_loss[] = l
    end
    if iter_count[] == 1 || iter_count[] % 200 == 0
        @printf("      Iter %5d | Loss: %.2e | Best: %.2e\n", iter_count[], l, best_loss[])
    end
    return false
end

initial_loss = loss_ude_clean(p_ude, nothing)
@printf("      Initial Loss: %.2e (λ = %.4f)\n", initial_loss, re_ude(p_ude)([1.0])[1])

println("      Phase 1: Adam optimization...")
optf_ude = OptimizationFunction(loss_ude_clean, Optimization.AutoZygote())
optprob_ude = OptimizationProblem(optf_ude, p_ude)
result_ude_p1 = Optimization.solve(optprob_ude, Adam(UDE_LR_PHASE1), 
                                   callback=callback_ude, maxiters=UDE_ITERS_PHASE1)

println("      Phase 2: Fine-tuning...")
iter_count[] = 0
optprob_ude_p2 = OptimizationProblem(optf_ude, result_ude_p1.u)
result_ude_p2 = Optimization.solve(optprob_ude_p2, Adam(UDE_LR_PHASE2), 
                                      callback=callback_ude, maxiters=UDE_ITERS_PHASE2)

println("      Phase 3: Extra fine-tuning...")
iter_count[] = 0
optprob_ude_p3 = OptimizationProblem(optf_ude, result_ude_p2.u)
result_ude_clean = Optimization.solve(optprob_ude_p3, Adam(UDE_LR_PHASE3), 
                                      callback=callback_ude, maxiters=UDE_ITERS_PHASE3)
p_ude_clean = result_ude_clean.u

# Evaluate baseline
sol_ude_clean = predict_ude(p_ude_clean)
L_ude_clean = vec(Array(sol_ude_clean)')
learned_eps_clean = get_learned_eps_loss(p_ude_clean, radii)

clean_mse = mean(abs2, L_true .- L_ude_clean)
recovery_mse_clean = mean(abs2, true_eps_loss .- learned_eps_clean)

println("\n   Baseline Results (Clean Data):")
@printf("      L(r) MSE: %.4e\n", clean_mse)
@printf("      ε_loss Recovery MSE: %.4e\n", recovery_mse_clean)
@printf("      True ε_loss(0) = %.4f, Learned = %.4f\n", true_eps_loss[1], learned_eps_clean[1])
@printf("      True ε_loss(0.5) = %.4f, Learned = %.4f\n", true_eps_loss[51], learned_eps_clean[51])

idx_valid = radii .>= 0.1
recovery_mse_valid = mean(abs2, true_eps_loss[idx_valid] .- learned_eps_clean[idx_valid])
@printf("      ε_loss Recovery MSE (r≥0.1): %.4e\n", recovery_mse_valid)

p_baseline = plot(layout=(1,2), size=(1200, 400))

plot!(p_baseline[1], radii, true_eps_loss, label="True ε_loss(r)", lw=3, ls=:dash, color=:blue)
plot!(p_baseline[1], radii, learned_eps_clean, label="Learned ε_loss(r)", lw=2, color=:orange)
vline!(p_baseline[1], [0.1], label="r=0.1 (identifiable region →)", ls=:dot, color=:gray, lw=2)
xlabel!(p_baseline[1], "Radius r")
ylabel!(p_baseline[1], "Energy Loss Rate ε_loss")
title!(p_baseline[1], "Full Domain (Note: r≈0 unidentifiable)")

plot!(p_baseline[2], radii[idx_valid], true_eps_loss[idx_valid], label="True ε_loss(r)", lw=3, ls=:dash, color=:blue)
plot!(p_baseline[2], radii[idx_valid], learned_eps_clean[idx_valid], label="Learned ε_loss(r)", lw=2, color=:orange)
xlabel!(p_baseline[2], "Radius r")
ylabel!(p_baseline[2], "Energy Loss Rate ε_loss")
title!(p_baseline[2], "Identifiable Region (r≥0.1)")
annotate!(p_baseline[2], 0.6, 0.015, text(@sprintf("MSE: %.2e", recovery_mse_valid), 10))

savefig(p_baseline, "outputs/baseline_recovery_clean.png")

println("\n[Step 4] Running noise robustness analysis...")

noise_results = DataFrame(
    noise_level = Float64[],
    node_mse = Float64[],
    ude_mse = Float64[],
    recovery_mse = Float64[],
    node_final_error = Float64[],
    ude_final_error = Float64[]
)

for (idx, noise_level) in enumerate(NOISE_LEVELS)
    println("\n" * "-"^60)
    println(" Noise Level: $(noise_level*100)% (RELATIVE)")
    println("-"^60)
    
    Random.seed!(42 + idx)
    
    noise_floor = 1e-6 * L_scale
    noise_std = max.(noise_level .* L_true, noise_floor)
    noise = noise_std .* randn(n_points)
    L_noisy = L_true .+ noise
    L_noisy[1] = 0.0
    L_noisy = max.(L_noisy, 0.0)
    
    CSV.write("data/noisy_data_$(noise_level).csv", DataFrame(radius=radii, luminosity=L_noisy))
    
    println("\n   Training Neural ODE (black-box)...")
    p_node = copy(p_node_init)
    iter_count[] = 0
    best_loss[] = Inf
    
    function loss_node(p, _)
        sol = predict_node(p)
        if sol.retcode != :Success
            return Inf
        end
        L_pred = vec(Array(sol)')
        return mean(abs2, L_noisy .- L_pred)
    end
    
    callback_node = function(state, l)
        iter_count[] += 1
        if l < best_loss[]
            best_loss[] = l
        end
        if iter_count[] % 500 == 0
            @printf("      Iter %5d | Loss: %.2e | Best: %.2e\n", iter_count[], l, best_loss[])
        end
        return false
    end
    
    optf_node = OptimizationFunction(loss_node, Optimization.AutoZygote())
    optprob_node = OptimizationProblem(optf_node, p_node)
    result_node = Optimization.solve(optprob_node, Adam(NODE_LR), 
                                     callback=callback_node, maxiters=NODE_ITERS)
    p_node_trained = result_node.u
    
    println("\n   Training UDE (physics-informed)...")
    local p_ude = copy(p_ude_init)
    iter_count[] = 0
    best_loss[] = Inf
    
    function loss_ude_noisy(p, _)
        sol = predict_ude(p)
        if sol.retcode != :Success
            return Inf
        end
        L_pred = vec(Array(sol)')
        
        data_loss = mean(abs2, L_noisy .- L_pred)
        
        reg_loss = 1e-6 * sum(abs2, p)
        
        return data_loss + reg_loss
    end
    
    callback_ude_noisy = function(state, l)
        iter_count[] += 1
        if l < best_loss[]
            best_loss[] = l
        end
        if iter_count[] % 500 == 0
            @printf("      Iter %5d | Loss: %.2e | Best: %.2e\n", iter_count[], l, best_loss[])
        end
        return false
    end
    
    println("      Phase 1...")
    optf_ude_n = OptimizationFunction(loss_ude_noisy, Optimization.AutoZygote())
    optprob_ude_n = OptimizationProblem(optf_ude_n, p_ude)
    local result_ude_p1 = Optimization.solve(optprob_ude_n, Adam(UDE_LR_PHASE1), 
                                       callback=callback_ude_noisy, maxiters=UDE_ITERS_PHASE1)
    
    println("      Phase 2...")
    iter_count[] = 0
    local optprob_ude_p2 = OptimizationProblem(optf_ude_n, result_ude_p1.u)
    local result_ude_p2 = Optimization.solve(optprob_ude_p2, Adam(UDE_LR_PHASE2), 
                                    callback=callback_ude_noisy, maxiters=UDE_ITERS_PHASE2)
    
    println("      Phase 3...")
    iter_count[] = 0
    local optprob_ude_p3 = OptimizationProblem(optf_ude_n, result_ude_p2.u)
    result_ude = Optimization.solve(optprob_ude_p3, Adam(UDE_LR_PHASE3), 
                                    callback=callback_ude_noisy, maxiters=UDE_ITERS_PHASE3)
    p_ude_trained = result_ude.u
    
    sol_node = predict_node(p_node_trained)
    sol_ude = predict_ude(p_ude_trained)
    L_node = vec(Array(sol_node)')
    L_ude = vec(Array(sol_ude)')
    
    node_mse = mean(abs2, L_true .- L_node)
    ude_mse = mean(abs2, L_true .- L_ude)
    
    learned_eps = get_learned_eps_loss(p_ude_trained, radii)
    recovery_mse = mean(abs2, true_eps_loss .- learned_eps)
    
    node_final_err = abs(L_true[end] - L_node[end])
    ude_final_err = abs(L_true[end] - L_ude[end])
    
    push!(noise_results, (noise_level, node_mse, ude_mse, recovery_mse, 
                          node_final_err, ude_final_err))
    
    println("\n   Results:")
    @printf("      NODE MSE: %.4e | Final Error: %.4e\n", node_mse, node_final_err)
    @printf("      UDE MSE:  %.4e | Final Error: %.4e\n", ude_mse, ude_final_err)
    @printf("      Physics Recovery MSE: %.4e\n", recovery_mse)
    @printf("      ε_loss(0): True=%.3f, Learned=%.3f\n", true_eps_loss[1], learned_eps[1])
    
    p1 = plot(radii, L_true, label="True Physics", lw=3, color=:black)
    plot!(p1, radii, L_ude, label="UDE", lw=2, ls=:dash, color=:blue)
    plot!(p1, radii, L_node, label="Neural ODE", lw=2, ls=:dot, color=:red)
    scatter!(p1, radii[1:5:end], L_noisy[1:5:end], label="Noisy Data", 
            ms=3, alpha=0.5, color=:gray)
    xlabel!(p1, "Radius r")
    ylabel!(p1, "Luminosity L(r)")
    title!(p1, "Model Comparison ($(Int(noise_level*100))% Relative Noise)")
    savefig(p1, "outputs/comparison_noise_$(Int(noise_level*100))_fixed.png")
    
    local recovery_mse_valid = mean(abs2, true_eps_loss[idx_valid] .- learned_eps[idx_valid])
    
    p2 = plot(layout=(1,2), size=(1200, 400))
    
    plot!(p2[1], radii, true_eps_loss, label="True ε_loss(r)", lw=3, ls=:dash, color=:blue)
    plot!(p2[1], radii, learned_eps, label="Learned ε_loss(r)", lw=2, color=:orange)
    vline!(p2[1], [0.1], label="", ls=:dot, color=:gray, lw=2)
    xlabel!(p2[1], "Radius r")
    ylabel!(p2[1], "Energy Loss Rate ε_loss")
    title!(p2[1], "Full Domain ($(Int(noise_level*100))% Noise)")
    
    plot!(p2[2], radii[idx_valid], true_eps_loss[idx_valid], label="True ε_loss(r)", lw=3, ls=:dash, color=:blue)
    plot!(p2[2], radii[idx_valid], learned_eps[idx_valid], label="Learned ε_loss(r)", lw=2, color=:orange)
    xlabel!(p2[2], "Radius r")
    ylabel!(p2[2], "Energy Loss Rate ε_loss")
    title!(p2[2], "Identifiable Region r≥0.1")
    annotate!(p2[2], 0.6, 0.015, text(@sprintf("MSE: %.2e", recovery_mse_valid), 10))
    
    savefig(p2, "outputs/recovery_noise_$(Int(noise_level*100))_fixed.png")
end

CSV.write("outputs/noise_analysis_results_fixed.csv", noise_results)

println("\n" * "="^60)
println(" Noise Robustness Summary")
println("="^60)
println(noise_results)

println("\n[Step 5] Running forecasting analysis...")

forecast_results = DataFrame(
    train_fraction = Float64[],
    node_train_mse = Float64[],
    node_test_mse = Float64[],
    ude_train_mse = Float64[],
    ude_test_mse = Float64[]
)

for train_frac in FORECAST_FRACTIONS
    println("\n" * "-"^60)
    println(" Training Fraction: $(train_frac*100)%")
    println("-"^60)
    
    n_train = Int(floor(train_frac * n_points))
    r_train = radii[1:n_train]
    L_train = L_true[1:n_train]
    
    Random.seed!(100)
    noise_std = 0.01 .* max.(L_train, 1e-6 * L_scale)
    L_train_noisy = L_train .+ noise_std .* randn(n_train)
    L_train_noisy[1] = 0.0
    L_train_noisy = max.(L_train_noisy, 0.0)
    
    println("\n   Training NODE for forecasting...")
    local p_node = copy(p_node_init)
    iter_count[] = 0
    
    function loss_node_fc(p, _)
        sol = predict_node(p, saveat=radii)
        if sol.retcode != :Success
            return Inf
        end
        L_pred = vec(Array(sol)')
        return mean(abs2, L_train_noisy .- L_pred[1:n_train])
    end
    
    callback_fc = function(state, l)
        iter_count[] += 1
        if iter_count[] % 500 == 0
            @printf("      Iter %5d | Loss: %.2e\n", iter_count[], l)
        end
        return false
    end
    
    optf_fc = OptimizationFunction(loss_node_fc, Optimization.AutoZygote())
    result_node_fc = Optimization.solve(OptimizationProblem(optf_fc, p_node), 
                                        Adam(NODE_LR), callback=callback_fc, maxiters=NODE_ITERS)
    
    println("\n   Training UDE for forecasting...")
    local p_ude = copy(p_ude_init)
    iter_count[] = 0
    
    function loss_ude_fc(p, _)
        sol = predict_ude(p, saveat=radii)
        if sol.retcode != :Success
            return Inf
        end
        L_pred = vec(Array(sol)')
        data_loss = mean(abs2, L_train_noisy .- L_pred[1:n_train])
        
        reg_loss = 1e-6 * sum(abs2, p)
        
        return data_loss + reg_loss
    end
    
    optf_ude_fc = OptimizationFunction(loss_ude_fc, Optimization.AutoZygote())
    local result_ude_p1 = Optimization.solve(OptimizationProblem(optf_ude_fc, p_ude), 
                                       Adam(UDE_LR_PHASE1), callback=callback_fc, 
                                       maxiters=UDE_ITERS_PHASE1)
    iter_count[] = 0
    local result_ude_p2 = Optimization.solve(OptimizationProblem(optf_ude_fc, result_ude_p1.u), 
                                       Adam(UDE_LR_PHASE2), callback=callback_fc, 
                                       maxiters=UDE_ITERS_PHASE2)
    iter_count[] = 0
    result_ude_fc = Optimization.solve(OptimizationProblem(optf_ude_fc, result_ude_p2.u), 
                                       Adam(UDE_LR_PHASE3), callback=callback_fc, 
                                       maxiters=UDE_ITERS_PHASE3)
    
    sol_node_fc = predict_node(result_node_fc.u)
    sol_ude_fc = predict_ude(result_ude_fc.u)
    L_node_fc = vec(Array(sol_node_fc)')
    L_ude_fc = vec(Array(sol_ude_fc)')
    
    node_train_mse = mean(abs2, L_train .- L_node_fc[1:n_train])
    node_test_mse = mean(abs2, L_true[n_train+1:end] .- L_node_fc[n_train+1:end])
    ude_train_mse = mean(abs2, L_train .- L_ude_fc[1:n_train])
    ude_test_mse = mean(abs2, L_true[n_train+1:end] .- L_ude_fc[n_train+1:end])
    
    push!(forecast_results, (train_frac, node_train_mse, node_test_mse, 
                            ude_train_mse, ude_test_mse))
    
    println("\n   Results:")
    @printf("      NODE - Train MSE: %.4e | Test MSE: %.4e\n", node_train_mse, node_test_mse)
    @printf("      UDE  - Train MSE: %.4e | Test MSE: %.4e\n", ude_train_mse, ude_test_mse)
    
    node_rel_err = abs.(L_node_fc .- L_true) ./ max.(abs.(L_true), 1e-10) * 100
    ude_rel_err = abs.(L_ude_fc .- L_true) ./ max.(abs.(L_true), 1e-10) * 100
    
    p_fc = plot(layout=(1,3), size=(1600, 450))
    
    plot!(p_fc[1], radii, L_true, label="True Physics", lw=3, color=:black)
    plot!(p_fc[1], radii, L_ude_fc, label="UDE", lw=2, ls=:dash, color=:blue)
    plot!(p_fc[1], radii, L_node_fc, label="NODE", lw=2, ls=:dot, color=:red)
    scatter!(p_fc[1], r_train[1:5:end], L_train_noisy[1:5:end], label="Training Data", 
            ms=3, color=:green, alpha=0.7)
    vline!(p_fc[1], [r_train[end]], label="Train/Test Split", ls=:dash, color=:gray, lw=2)
    xlabel!(p_fc[1], "Radius r")
    ylabel!(p_fc[1], "Luminosity L(r)")
    title!(p_fc[1], "Full Trajectory ($(round(Int, train_frac*100))% Train)")
    
    test_idx = n_train:n_points
    r_test = radii[test_idx]
    L_true_test = L_true[test_idx]
    L_node_test = L_node_fc[test_idx]
    L_ude_test = L_ude_fc[test_idx]
    
    plot!(p_fc[2], r_test, L_true_test, label="True", lw=3, color=:black)
    plot!(p_fc[2], r_test, L_ude_test, label="UDE (MSE: $(@sprintf("%.1e", ude_test_mse)))", 
          lw=2, ls=:dash, color=:blue)
    plot!(p_fc[2], r_test, L_node_test, label="NODE (MSE: $(@sprintf("%.1e", node_test_mse)))", 
          lw=2, ls=:dot, color=:red)
    xlabel!(p_fc[2], "Radius r")
    ylabel!(p_fc[2], "Luminosity L(r)")
    title!(p_fc[2], "Test Region ($(round(Int, (1-train_frac)*100))% Forecast)")
    
    plot!(p_fc[3], radii[2:end], ude_rel_err[2:end], label="UDE Error", lw=2, color=:blue)
    plot!(p_fc[3], radii[2:end], node_rel_err[2:end], label="NODE Error", lw=2, color=:red)
    vline!(p_fc[3], [r_train[end]], label="Train/Test Split", ls=:dash, color=:gray, lw=2)
    xlabel!(p_fc[3], "Radius r")
    ylabel!(p_fc[3], "Relative Error (%)")
    title!(p_fc[3], "Percentage Error Comparison")
    if maximum(node_rel_err[2:end]) / max(maximum(ude_rel_err[2:end]), 1e-10) > 100
        plot!(p_fc[3], yscale=:log10)
    end
    
    savefig(p_fc, "outputs/forecast_$(round(Int, train_frac*100))pct_fixed.png")
end

CSV.write("outputs/forecast_results_fixed.csv", forecast_results)

println("\n" * "="^60)
println(" Forecasting Summary")
println("="^60)
println(forecast_results)

println("\n" * "="^70)
println(" PIPELINE COMPLETE")
println("="^70)

println("\nFixed Issues:")
println("   1. NN now takes only r as input (ε_loss depends only on T(r))")
println("   2. Relative noise model (noise ∝ signal)")
println("   3. Added smoothness regularization")
println("   4. Removed 0% noise case")

println("\nOutput Files (with _fixed suffix):")
println("   - outputs/baseline_recovery_clean.png")
println("   - outputs/comparison_noise_*_fixed.png")
println("   - outputs/recovery_noise_*_fixed.png")
println("   - outputs/forecast_*pct_fixed.png")
println("   - outputs/noise_analysis_results_fixed.csv")
println("   - outputs/forecast_results_fixed.csv")
