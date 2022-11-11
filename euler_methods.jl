using Plots

"""
    my_forward_euler

Takes in an initial `t0` and `y0` as well as a `Tf` 
and `Δt` with a `A` matrix for a linear function and performs
explicit operations with solution vector `b` to solve for `y` 
and returns `y` and `t`

(t0, Tf, Δt, y0, A, b) -> (t, y)
"""
function my_forward_euler(t0, Tf, Δt, y0, A, b)
    N = size(y0)[1]

    M = Integer(Tf/Δt)  # M+1 total temporal nodes

    t = Vector{Float64}(undef, M+1)
    y = Matrix{Float64}(undef, N, M+1)

    # fill in the initial condition:
    t[1] = t0
    y[:, 1] = y0

    for n = 1:M # take N time steps
        y[:, n+1] = y[:, n] + Δt*(A*y[:, n] + b)
        t[n+1] = t[n] + Δt
    end
    
    return (t, y)
end

"""
    my_backward_euler
    
Takes in an initial `t0` and `y0` as well as a `Tf` 
and `Δt` with a `λ` for a linear function and performs
implicit operations to solve for `y` and returns `y` and `t`

(t0, Tf, Δt, y0, λ) -> (t, y)
"""
function my_backward_euler(t0, Tf, Δt, y0, λ)
    N = Integer(Tf/Δt)  # N+1 total temporal nodes

    t = Vector{Float64}(undef, N+1)
    y = Vector{Float64}(undef, N+1)

    # fill in the initial condition:
    t[1] = t0
    y[1] = y0

    for n = 1:N # take N time steps
        y[n+1] = y[n] / (1 - λ * Δt) #implicit y_(n+1)
        t[n+1] = t[n] + Δt
    end
    
    return (t, y)
end

#=
# example case
λ = -3
t0 = 0
Tf = 4
y0 = 17
Δt = 0.1*2/abs(λ)

(fT, fY) = my_forward_euler(t0, Tf, Δt, y0, λ)
(bT, bY) = my_backward_euler(t0, Tf, Δt, y0, λ)

# plotting example
plot(fT, fY, xlabel = "time (t)", 
    ylabel = "y approximation", title = "Forward & Backward Euler Approximations", 
    color = :magenta, linewidth = 2, label = "fY")
plot!(bT, bY, color = :green, linewidth = 2, label = "bY")
=#