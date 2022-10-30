"""
Nate Koch
CS 410: Computational Science
Homework 1 & 2
LUP Solver & Conjugate Gradient Method
"""

using Plots

"""
    computeLUP(A)

Computes and returns lower `L` and upper `U` triangular matricies 
and a permutation matrix `P` from a given square matrix `A`
via LUP factorization.

(A) -> (L, U, P)
"""
function computeLUP(A)

    N = size(A)[1]

    I = createIdentityMatrix(N)

    L = copy(I)
    P = copy(I)
    Ã = copy(A)

    for k = 1:N-1 # march across columns

        (j,p) = computePk(Ã, k)
        Ã .= p * Ã
        P .= p * P
        (Lk, Lk_inv) = computeLk(Ã, k)
        Ã .= Lk * Ã
        L .= L * Lk_inv

        if k > 1

            (L[k, 1:k-1], L[j, 1:k-1]) = (L[j, 1:k-1], L[k, 1:k-1])

        end

    end

    U = copy(Ã)

    return (L, U, P)

end

"""
    createIdentityMatrix(N)

Creates and returns an NxN identity matrix `I` from a given `N`

(N) -> I
"""
function createIdentityMatrix(N)

    I = Matrix{Float64}(undef, N, N)
    I .= 0

    for i = 1:N
        I[i,i] = 1
    end

    return I
end

"""
    computePk(A, k)

Computes and returns the jTh row of a given matrix `A` to be switched with
the given `k`th column and the permutation matrix `p`

(A, k) -> (jRow, p)
"""
function computePk(A, k)

    N = size(A)[1]
    p = createIdentityMatrix(N)
    pivot = 0
    jRow = 0

    for j = k:N
        temp = abs(A[j, k])
        if temp > pivot
            pivot = temp
            jRow = j
        end
    end

    p[[jRow,k], :] = p[[k,jRow], :]

    return (jRow, p)

end

"""
    computeLk(A, k)

Computes and returns the lower triangular matrix of the given matrix `A`
at the given pivot at column `k` "Lk" and its inverse `Lk_inv`

(A, k) -> (Lk, Lk_inv)
"""
function computeLk(A, k)

    N = size(A)[1]

    Lk = createIdentityMatrix(N)
    Lk_inv = createIdentityMatrix(N)

    for i = k+1:N
        Lk[i,k] = -A[i,k] / A[k,k] # - curr_elem / pivot_elem
        Lk_inv[i,k] = A[i,k] / A[k,k]
    end

    return (Lk, Lk_inv)
end

"""
    forwardSubtitution(L, b)

Performs forward substitution on a given matrix `L` and a solution
vector `b` and returns the vector `y`

(L, b) -> y
"""
function forwardSubtitution(L, b)
    
    N = size(L)[1]

    y = zeros(N)

    for i = 1:N
        f_sum = b[i]
        for j = 1:i-1
            f_sum -= L[i, j] * y[j]
        end
        y[i] = f_sum / L[i, i]
    end

    return y
end

"""
    backwardSubtitution(U, y)

Performs backward substitution on a given matrix `U` and a
vector `y` and returns the vector `x`

(U, y) -> x
"""
function backwardSubstitution(U, y)

    N = size(U)[1]

    x = zeros(N)

    for i = N:-1:1
        b_sum = y[i]
        for j = i+1:N
            b_sum -= U[i,j] * x[j]
        end
        x[i] = b_sum / U[i,i]
    end

    return x
end

"""
    LUPsolve(L, U, P, b)

Solves Ax = b for x given lower and upper triangular matricies `L` & `U`
with a permutation matrix `P` along with a solution vector `b`

(L, U, P, b) -> x
"""
function LUPsolve(L, U, P, b)

    N = size(L)[1]

    b .= P * b

    # forward substitution
    y = forwardSubtitution(L, b)

    # backward substitution
    x = backwardSubstitution(U, y)

    return x
end

"""
    conj_grad(A, b)

Computes the conjugate gradient of a given NxN matrix `A`
with a given N-element solution vector `b` that returns 
the computed vector `x` to solve the linear system Ax=b
for x. This is an iterative method. 

(A, b) -> x
"""
function conj_grad(A, b)
   
    N = size(b)
    max_iterations = 100 
    ϵ = 1.0e-6 # machine precision
    x = zeros(N) # guess for x just all zeros
    r = A * x - b # compute the residual from the initial guess

    p = r
    ρ_old = 1
    for i = 1:max_iterations
        ρ = (transpose(r) * r)[1]
        if i == 1
            p = r
        else
            β = ρ/ρ_old
            p = r + β * p
        end
        q = A * p
        δ = (ρ / ((transpose(p) * q))[1])[1]
        x = x - δ * p 
        r = r - δ * q
        ρ_old = ρ # store the ρ from this iteration

        if sqrt((transpose(r) * r)[1]) ≤ ϵ
            break
        end
    end

    return x
end

# ~~~ Tests and Timing at N = 10, 100, 1000 ~~~ #

# N = 1000
B_1000 = rand(1000, 1000)
b_1000 = rand(1000, 1)
I_1000 = createIdentityMatrix(1000)
A_1000 = transpose(B_1000) * B_1000 + I_1000

println("N = 1000")
#@time (L_1000, U_1000, P_1000) = computeLUP(A_1000)
#@time x_1000 = LUPsolve(L_1000, U_1000, P_1000, b_1000)

#@assert L_1000 * U_1000 * x_1000 ≈ P_1000 * b_1000
#@assert A_1000 * x_1000 ≈ b_1000

@time xx_1000 = conj_grad(A_1000, b_1000)

@assert isapprox((A_1000 * xx_1000), b_1000, atol = 1e-2) # need higher tolerance 1e-2

# N = 100
B_100 = rand(100, 100)
b_100 = rand(100, 1)
I_100 = createIdentityMatrix(100)
A_100 = transpose(B_100) * B_100 + I_100

println("N = 100")
#@time (L_100, U_100, P_100) = computeLUP(A_100)
#@time x_100 = LUPsolve(L_100, U_100, P_100, b_100)

#@assert L_100 * U_100 * x_100 ≈ P_100 * b_100
#@assert A_100 * x_100 ≈ b_100

@time xx_100 = conj_grad(A_100, b_100)

@assert isapprox((A_100 * xx_100), b_100, atol = 1e-3) # need higher tolerance 1e-3

# N = 10
B_10 = rand(10, 10)
b_10 = rand(10, 1)
I_10 = createIdentityMatrix(10)
A_10 = transpose(B_10) * B_10 + I_10

println("N = 10")
#@time (L_10, U_10, P_10) = computeLUP(A_10)
#@time x_10 = LUPsolve(L_10, U_10, P_10, b_10)

#@assert L_10 * U_10 * x_10 ≈ P_10 * b_10
#@assert A_10 * x_10 ≈ b_10

@time xx_10 = conj_grad(A_10, b_10)

@assert isapprox((A_10 * xx_10), b_10, atol = 1e-3) # need higher tolerance 1e-3

x_axis = [10, 100, 1000]
compute_times = [0.000040, 0.000412, 0.217529]
solve_times = [0.000016, 0.000043, 0.082035]

plot(x_axis, compute_times, xlabel = "N elements", 
ylabel = "execution time in seconds", title = "conj_grad Times", 
color = :red, linewidth = 6) 

"""
x_axis = [10, 100, 1000]
compute_times = [0.000045, 0.086714, 94.566791]
solve_times = [0.000016, 0.000043, 0.082035]

plot(x_axis, compute_times,xlabel = "N elements", 
ylabel = "execution time in seconds", title = "computeLUP Times", 
color = :magenta, linewidth = 6) 

plot(x_axis, solve_times,xlabel = "N elements", 
ylabel = "execution time in seconds", title = "LUPsolve Times", 
color = :blue, linewidth = 6)
"""