using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf
include("genmesh.jl")

# Computational mesh
meshsize = 0.15
Γ = meshsphere(1.0, meshsize)

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    																		
N = NCross()

# Gram matrix
Nyx = assemble(N, Y, X)
Nyy = assemble(N, Y, Y)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

length_arr = 101
cond_cfie, cond_efie = zeros(length_arr), zeros(length_arr)
κ_arr = 4π/3 .+ 2π/3/(length_arr-1) * [0:1:length_arr-1;]
for i in 1:length_arr
    # κ = 2π/3
    κ = κ_arr[i]
    γ = -im*κ
    η = κ^2

    # Operators
    S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
    K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
    Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                      # EFIO with pure imaginary wave number iκ 
    Ki = MWDoubleLayer3D(κ)                                                    # MFIO with pure imaginary wave number iκ

    # E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    # e = (n × E) × n
    # H = -1/γ*curl(E)
    # h = (n × H) × n

    # @hilbertspace j
    # @hilbertspace k

    # assembly of static operators
    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

    ℤ = assemble(S, X, X, quadstrat=nearstrat)
    𝕊x = assemble(Si, X, X, quadstrat=nearstrat) 
    𝕊y = assemble(Si, Y, Y, quadstrat=nearstrat) 
    𝕂x = assemble(K, X, X, quadstrat=nearstrat)
    𝕂ix = assemble(Ki, X, X, quadstrat=nearstrat)
    𝕂iy = assemble(Ki, Y, Y, quadstrat=nearstrat)

    lhs = im*η * iNxy * ℤ * iNyx * 𝕊y + κ^2 * iNxy * 𝕊x * iNyx * 𝕊y +  iNxy * (𝕂x - 𝕂ix) * iNyx * (0.5*Nyy + 𝕂iy)

    open("sphere_kappa.txt", "a") do io
        @printf(io, "%.3f %.10f %.10f\n", κ, cond(ℤ), cond(lhs))
    end;
end

Plots.plot(width = 600, height=400,
    grid = false,
    xscale = :identity, 
    yaxis = :log10, 
    xlims = (2.65, 3.05),
    xticks = [2.7; 2.8; 2.9; 3],
    ylims = (1e0, 1e1), 
    # yticks = [1e-15; 1e-10; 1e-5; 1e0;],
    xlabel = "Frequency (10^8 Hz)",
    ylabel = "Condition number")

scatter!(κ_arr*3/2π, cond_cfie, label = "CFIE")
scatter!(κ_arr*3/2π, cond_efie, label = "EFIE")
scatter!(κ_arr*3/2π, cond_mfie, label = "MFIE")



# Mie series

using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf
include("genmesh.jl")

# Computational mesh
meshsize = 0.2
Γ = meshsphere(1.0, meshsize)

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

κ = 1.43π                                                                 
γ = -im*κ
η = κ^2

# Operators
S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                      # EFIO with pure imaginary wave number iκ 
Ki = MWDoubleLayer3D(κ)                                                    # MFIO with pure imaginary wave number iκ

E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

ℤ = assemble(S, X, X, quadstrat=nearstrat)
𝕊x = assemble(Si, X, X, quadstrat=nearstrat) 
𝕊y = assemble(Si, Y, Y, quadstrat=nearstrat) 
𝕂x = assemble(K, X, X, quadstrat=nearstrat)
𝕂ix = assemble(Ki, X, X, quadstrat=nearstrat)
𝕂iy = assemble(Ki, Y, Y, quadstrat=nearstrat)

lhs = im*η * iNxy * ℤ * iNyx * 𝕊y + κ^2 * iNxy * 𝕊x * iNyx * 𝕊y +  iNxy * (𝕂x - 𝕂ix) * iNyx * (0.5*Nyy + 𝕂iy)

@hilbertspace k
rhs = assemble(@discretise -e[k] k∈X)

xi, = solve(BEAST.GMRESSolver(lhs, tol=1e-12, restart=250), iNxy * rhs)

jc, mc = iNyx * 𝕊y * xi, iNyx * (0.5*Nyy + 𝕂iy) * xi


using SphericalScattering, StaticArrays

# Result visualization 
Θ = range(0, stop=2π, length=100)
pts = [SVector(2.0*cos(θ), 0.0, 2.0*sin(θ)) for θ in Θ]
Esct = im*η * potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, jc, X) + potential(BEAST.MWDoubleLayerField3D(γ), pts, mc, X)
Hsct = -1/(im*κ) * (im*η * potential(BEAST.MWDoubleLayerField3D(γ), pts, jc, X) + κ^2 * potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, mc, X))

# Mie series
sp = PECSphere( 
    radius      = 1.0, 
    embedding   = Medium(1.0, 1.0),
)

ex = planeWave(
    embedding    = Medium(1.0, 1.0),
    frequency    = κ/(2π),
)

Θ1 = range(0, stop=2π, length=3600)
pts1 = [SVector(2.0*cos(θ), 0.0, 2.0*sin(θ)) for θ in Θ1]
Emie = scatteredfield(sp, ex, ElectricField(pts1))
Hmie = scatteredfield(sp, ex, MagneticField(pts1))
