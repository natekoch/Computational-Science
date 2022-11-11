using Plots
include("euler_methods.jl")

function F(t, x)
    return 2 * (π^2 - 1) * exp(-2 * t) * sin(π * x)
end

function f(x)
    return sin(π * x)
end

function b(t)
    return F.(t, x_int)
end

# Parameters
t0 = 0
Tf = 2
κ = 2
Δx = 0.1
Δt = 0.1 # Unstable
#Δt = 0.01 # Unstable
#Δt = 0.001 # Stable

M = Integer(Tf/Δt) # how many time steps to take

N = Integer(1/Δx) # N+1 total spatial nodes

x = collect(range(0, 1, step = Δx))
x_int = x[2:end-1] #interior spatial nodes 

t = collect(range(0, Tf, step = Δt))

# fill the A matrix diagonal
A = zeros(N-1, N-1)
for i = 1:N-1
    A[i, i] = -2κ*(1/Δx^2)
end

# fill the A matrix values surrounding the diagonal
for i = 1:N-2
    A[i+1, i] = 1κ*(1/Δx^2)
    A[i,i+1] = 1κ*(1/Δx^2)
end

# Solve y' = Ay + b(t) with y0 = f(x_int)
y0 = f.(x_int)
y = Matrix{Float64}(undef,N+1, M+1)  # my entire solution at all nodes and all time steps.

# fill in initial data:
y[:, 1] = f.(x)

#y_int = y[2:N, :]

y[1, :] .= 0
y[N+1, :] .= 0
for n = 1:M
    #global y_int
    (tt, y_int) = my_forward_euler(t0, Tf, Δt, y0, A, b(t[n]))
    y[2:N, n+1] = y_int[:, n+1]
end

# the exact solution
function exact(t, x)
    exp(-2*t) * sin(π*x)
end

xfine = collect(range(0, 1, step = Δx/100))

# Plot the exact solution against the numerical solution
for n = 1:M+1
    p = plot(x, y[:, n], label = ["numerical"], title = "Δt = 0.1")
    plot!(xfine, exact.(t[n], xfine), label = ["exact"])
    ylims!((0,1))
    display(p)

    sleep(0.7)
end
