"""
Nate Koch
CS 410: Computational Science
Homework 1
LUP Solver
"""

using Plots

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

function createIdentityMatrix(N)

    I = Matrix{Float64}(undef, N, N)
    I .= 0

    for i = 1:N
        I[i,i] = 1
    end

    return I
end

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



function computeLk(A, k)

    N = size(A)[1]

    Lk = createIdentityMatrix(N)
    Lk_inv = createIdentityMatrix(N)

    for i = k+1:N
        Lk[i,k] = -A[i,k] / A[k,k] # curr_elem / pivot_elem
        Lk_inv[i,k] = A[i,k] / A[k,k]
    end

    return (Lk, Lk_inv)
end

function LUPsolve(A, b)

    N = size(A)[1]
    
    (L, U, P) = computeLUP(A)

    b .= P * b

    # forward substitution
    y = zeros(N)

    for i = 1:N
        f_sum = b[i]
        for j = 1:i-1
            f_sum -= L[i, j] * y[j]
        end
        y[i] = f_sum / L[i, i]
    end

    # backward substitution
    x = zeros(N)

    for i = N:-1:1
        b_sum = y[i]
        for j = i+1:N
            b_sum -= U[i,j] * x[j]
        end
        x[i] = b_sum / U[i,i]
    end

    round.(x)

    return x
end


tempA = Matrix{Float64}(undef, 3, 3)
tempA .= [6 -2 2;12 -8 6;3 -13 3]
b = [6,3,5]

(L, U, P) = computeLUP(tempA)

x = LUPsolve(tempA, b)