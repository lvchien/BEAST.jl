using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf
include("genmesh.jl")

# for meshsize in [0.055, 0.05, 0.045, 0.04]
# Computational mesh
meshsize = 0.1
Γ = meshcuboid(1.0, 1.0, 1.0, meshsize)

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    																		
N = NCross()

# Gram matrix
Nyx = assemble(N, Y, X)
Nyy = assemble(N, Y, Y)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

length_arr = 51
cond_cfie, cond_efie = zeros(length_arr), zeros(length_arr)
step = (√3 - √2)π/32
κ_arr = √3π .+ step * [9:1:12;]

# for i in 1:4
    κ = √3π
    # κ = κ_arr[i]
    γ = -im*κ
    η = κ^2

    # Operators
    S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
    K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
    Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                      # EFIO with pure imaginary wave number iκ 
    Ki = MWDoubleLayer3D(κ)                                                    # MFIO with pure imaginary wave number iκ

    # assembly of static operators
    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

    ℤ = assemble(S, X, X, quadstrat=nearstrat)
    𝕊x = assemble(Si, X, X, quadstrat=nearstrat) 
    𝕊y = assemble(Si, Y, Y, quadstrat=nearstrat) 
    𝕂x = assemble(K, X, X, quadstrat=nearstrat)
    𝕂ix = assemble(Ki, X, X, quadstrat=nearstrat)
    𝕂iy = assemble(Ki, Y, Y, quadstrat=nearstrat)

    lhs = im*η * iNxy * ℤ * iNyx * 𝕊y + κ^2 * iNxy * 𝕊x * iNyx * 𝕊y +  iNxy * (𝕂x - 𝕂ix) * iNyx * (0.5*Nyy + 𝕂iy)
    cond(lhs)

    open("cube_kappa.txt", "a") do io
        @printf(io, "%.3f %.10f %.10f\n", κ, cond(ℤ), cond(lhs))
    end;
# end




# Exact solution

using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf, StaticArrays
include("genmesh.jl")

# Computational mesh
meshsize = 0.1
# Γ = meshsphere(1.0, 0.15)
# Γ = meshcuboid(1.0, 1.0, 1.0, meshsize)

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    
N = NCross()

# Gram matrix
Nyx = assemble(N, Y, X)
Nxx = assemble(N, X, X)
Nyy = assemble(N, Y, Y)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

κ = π/2                                                                 
γ = -im*κ
η = κ^2

# Operators
S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                      # EFIO with pure imaginary wave number iκ 
Ki = MWDoubleLayer3D(κ)                                                    # MFIO with pure imaginary wave number iκ

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

ℤ = assemble(S, Y, Y, quadstrat=nearstrat)
𝕊x = assemble(Si, Y, Y, quadstrat=nearstrat) 
𝕊y = assemble(Si, X, X, quadstrat=nearstrat) 
𝕂x = assemble(K, Y, Y, quadstrat=nearstrat)
𝕂ix = assemble(Ki, Y, Y, quadstrat=nearstrat)
𝕂iy = assemble(Ki, X, X, quadstrat=nearstrat)

lhs = im*η * iNyx * ℤ * iNxy * 𝕊y + κ^2 * iNyx * 𝕊x * iNxy * 𝕊y +  iNyx * (𝕂x - 𝕂ix) * iNxy * (0.5*Nxx + 𝕂iy)

# Exact solution
function exct(x)
    r = norm(x)
    expn = exp(-im*κ*r)
    (-κ^2*r^2 + im*κ*r + 1)*expn/r^3*SVector(1.0, 0.0, 0.0) + (κ^2*r^2 - 3*im*κ*r - 3)*expn/r^5*x[1]*x
end
e = (n × exct) × n

@hilbertspace k
rhs = assemble(@discretise e[k] k∈Y)

xi, = solve(BEAST.GMRESSolver(lhs, tol=1e-12, restart=250), iNyx * rhs)

jc, mc = iNxy * 𝕊y * xi, iNxy * (0.5*Nxx + 𝕂iy) * xi

# Result visualization 
Θ = range(0, stop=2π, length=100)
pts = [SVector(2.0*cos(θ), 0.0, 2.0*sin(θ)) for θ in Θ]
Esct = im*η * potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, jc, Y) + potential(BEAST.MWDoubleLayerField3D(γ), pts, mc, Y)
# Hsct = -1/(im*κ) * (im*η * potential(BEAST.MWDoubleLayerField3D(γ), pts, jc, X) + κ^2 * potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, mc, X))

Θ1 = range(0, stop=2π, length=100)
pts1 = [SVector(2.0*cos(θ), 0.0, 2.0*sin(θ)) for θ in Θ1]
Eexct = exct.(pts)
# Hmie = scatteredfield(sp, ex, MagneticField(pts1))

