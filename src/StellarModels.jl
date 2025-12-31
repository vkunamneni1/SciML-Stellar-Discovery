module StellarModels

using DifferentialEquations

export R, ρ_c, T_c, T_star, ϵ₀, λ₀
export ρ, T, ϵ_nuc, ϵ_loss, stellar_ode!

const R = 1.0       # Stellar radius (nondimensional)
const ρ_c = 1.0     # Central density (nondimensional)
const T_c = 1.0     # Central temperature (nondimensional)
const T_star = 1.0  # Reference temperature T*
const ϵ₀ = 1.0      # Nuclear source scale
const λ₀ = 0.3      # Loss term scale (ground truth)

# Density profile, ρ(r) = ρ_c * (1 - r²/R²)_+
ρ(r) = ρ_c * max(0.0, 1.0 - (r/R)^2)

# Temperature profile, T(r) = T_c * (1 - r²/R²)_+
T(r) = T_c * max(0.0, 1.0 - (r/R)^2)

# Nuclear energy generation rate, ε_nuc
ϵ_nuc(r) = ϵ₀ * ρ(r) * (T(r) / T_star)^4

# True energy loss rate, ε_loss
ϵ_loss(r) = λ₀ * (T(r) / T_star)^9

# The complete stellar energy equation for the baseline model.
function stellar_ode!(dL, L, p, r)
    # dL/dr = 4π * r² * ρ(r) * [ε_nuc(r) - ε_loss(r)]
    if r < R
        dL[1] = 4π * r^2 * ρ(r) * (ϵ_nuc(r) - ϵ_loss(r))
    else
        dL[1] = 0.0
    end
end

end