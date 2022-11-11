using LinearAlgebra
using SparseArrays

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

println("IGNORE COMPILE TIME:")
@time 1 # to remove compilation time from results below

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