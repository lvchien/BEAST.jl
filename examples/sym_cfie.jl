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

# Near fields

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


```
    Plot the E and H fields
```

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xtickfont = font(9, "Computer Modern"), 
    ylims = (0, 1.2), 
    yticks = ([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Observation angle",
    ylabel = "Scattered electric field",
    titlefont = font(10, "Computer Modern"), 
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"), 
    legendfont = font(10, "Computer Modern"))

Plots.plot(Θ1, norm.(Eexct), label="Mie series", linecolor=:black)
Plots.scatter!(Θ, norm.(Esct), label="CFIE", markershape=:x, markercolor=:black, markersize=3.5)
Plots.savefig("scatteredE.pdf")