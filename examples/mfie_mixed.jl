using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf, SphericalScattering, StaticArrays, IterativeSolvers
include("utils/genmesh.jl")

# Parameters
κ = 1.0
γ = -im*κ

# Setting
sp = PECSphere( 
    radius      = 1.0, 
    embedding   = Medium(1.0, 1.0),
)

ex = planeWave(
    embedding    = Medium(1.0, 1.0),
    frequency    = κ/(2π),
)

# Evaluation points on the external sphere of radius r = 2m
r = 2.0
Θ, φ = range(0, stop=π, length=50), range(0, stop=2π, length=101)
pts = [SVector(r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ)) for θ ∈ Θ for ϕ ∈ φ[1:end-1]]

# Exact solutions by Mie series
Emie = scatteredfield(sp, ex, ElectricField(pts))
Hmie = scatteredfield(sp, ex, MagneticField(pts))

# Computational mesh
# for meshsize in [0.45, 0.35, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.13, 0.12, 0.11, 0.1, 0.09, 0.075, 0.065, 0.0055]
for meshsize in [0.2, 0.18, 0.16, 0.15, 0.13, 0.12, 0.11, 0.1, 0.09, 0.075, 0.065, 0.0055]
# meshsize = 0.15
    Γ = meshsphere(1.0, meshsize)

    # RWG and BC function spaces
    X = raviartthomas(Γ)
    Y = buffachristiansen(Γ)
        
    N = NCross()

    # Gram matrix
    Nyx = assemble(N, Y, X)
    iNyx = inv(Matrix(Nyx))

#     length_arr = 51
#     step = π/150
#     κ_arr = 4π/3 .+ step * [0:1:50;]

# for κ ∈ κ_arr[14:end]
#     γ = -im*κ
#     κ′ = κ
#     η = -κ^2
    
    # Operators
    K = MWDoubleLayer3D(γ)                                                       # MFIO with wave number κ

    E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
    H = -1/(im)*curl(E)
    h = (n × H) × n

    # assembly of static operators
    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

    Kyx = assemble(K, Y, X, quadstrat=nearstrat)
    lhs = 0.5 * Nyx + Kyx

    @hilbertspace k
    rhs = assemble(@discretise h[k] k∈Y)

    xi, ch = IterativeSolvers.gmres(lhs, rhs, log=true, abstol=1e-12, reltol=1e-12)
    
    Esct = im*potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, xi, X)
    curlEsct = im*potential(BEAST.MWDoubleLayerField3D(γ), pts, xi, X)

    δe = Esct - conj.(Emie)
    δcurle = curlEsct - (im*κ).*conj.(Hmie)
    errorL2e = sqrt(real(dot(δe, δe)))/size(Emie)[1]
    errorHcurle = sqrt(real(dot(δe, δe)) + real(dot(δcurle, δcurle)))/size(Emie)[1]

    open("mfie_mixed_error.txt", "a") do io
        @printf(io, "%.3f %.10f %.3f %d %.10f %.10f \n", meshsize, κ, cond(lhs), ch.iters, errorL2e, errorHcurle)
    end; 
end
