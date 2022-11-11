using LinearAlgebra
using SparseArrays

"""
    dense_solve

For use only in this file.
Take no arguments. Returns a computed x vector for a dense matrix.
LU factorization then solve for x in Ax=b.
Times each computation.

() -> x
"""
function dense_solve()
    A = zeros(N, N)

    b = Vector{Float64}(undef, N)
    b[:] .= 1

    for i = 1:N-1
        A[i, i] = -2
        A[i+1, i] = 1
        A[i, i+1] = 1
    end
    A[N, N] = -2

    println("DENSE: F = lu(A)")
    @time F = lu(A)

    println("DENSE: x = F \\ b")
    @time x = F \ b

    return x
end

"""
    sparse_solve

For use only in this file. 
Take no arguments. Returns a computed x vector for a sparse matrix.
LU factorization then solve for x in Ax=b.
Times each computation.

() -> x
"""
function sparse_solve()
    A = spzeros(N, N)

    b = Vector{Float64}(undef, N)
    b[:] .= 1

    for i = 1:N-1
        A[i, i] = -2
        A[i+1, i] = 1
        A[i, i+1] = 1
    end
    A[N, N] = -2

    println("SPARSE: F = lu(A)")
    @time F = lu(A)

    println("SPARSE: x = F \\ b")
    @time x = F \ b

    return x
end

N = 100
println("\nN = 100")
println("\n~~~DENSE~~~")
dense_solve()
dense_x = dense_solve()

println("\n~~~SPARSE~~~")
sparse_solve()
sparse_x = sparse_solve()

@assert dense_x ≈ sparse_x

N = 1000
println("\nN = 1000")
println("\n~~~DENSE~~~")
dense_solve()
dense_x = dense_solve()

println("\n~~~SPARSE~~~")
sparse_solve()
sparse_x = sparse_solve()

@assert dense_x ≈ sparse_x

N = 10000
println("\nN = 10000")
println("\n~~~DENSE~~~")
dense_solve()
dense_x = dense_solve()

println("\n~~~SPARSE~~~")
sparse_solve()
sparse_x = sparse_solve()

@assert dense_x ≈ sparse_x

#=
x_values = [100, 1000, 10000]

dense_lu = [0.000225, 0.036726, 7.089112]
sparse_lu = [0.000259, 0.001106, 0.011614]

dense_x_solve = [0.000010, 0.002436, 0.047155]
sparse_x_solve = [0.000042, 0.000080, 0.000561]


plot(x_values, dense_lu, label = "Dense LU", ylabel = "Computation Time (s)", xlabel = "N in NxN Matrix", title = "LU Factorization Time")
plot!(x_values, sparse_lu, label = "Sparse LU")

plot(x_values, dense_x_solve, label = "Dense x = F|b", ylabel = "Computation Time (s)", xlabel = "N in NxN Matrix", title = "x = F|b Solve Time")
plot!(x_values, sparse_x_solve, label = "Sparse x = F|b")
=#