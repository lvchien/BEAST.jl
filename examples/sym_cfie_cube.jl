using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, Printf, IterativeSolvers
include("utils/genmesh.jl")

κ = π/2
γ = -im*κ
κ′ = κ
η = -κ^2

```
    FOR CONDITION NUMBER W.R.T. h CALCULATION
```
### Different mesh sizes h for examining the condition number w.r.t. h
# for meshsize in [0.35, 0.25, 0.18, 0.12, 0.09, 0.07, 0.055, 0.004]
```
    end
```

### Computational mesh
meshsize = 0.18
Γ = meshcuboid(1.0, 1.0, 1.0, meshsize)

### RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
                                                                            
N = NCross()

### Gram matrix
Nyx = assemble(N, Y, X)
Nyy = assemble(N, Y, Y)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

```
    FOR CONDITION NUMBER W.R.T. κ CALCULATION
```
### Different wave numbers κ for examining the condition number w.r.t. κ
#     length_arr = 51
#     step = (√3 - √2)π/32
#     κ_arr = √3π .+ step * [-39:1:11;]

# for κ ∈ κ_arr
#     γ = -im*κ
#     κ′ = κ
#     η = -κ^2
```
    end
```

### Operators
S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
Si = MWSingleLayer3D(κ′, 1.0, 1.0/κ′^2)                                      # EFIO with pure imaginary wave number iκ 
Ki = MWDoubleLayer3D(κ′)                                                    # MFIO with pure imaginary wave number iκ

E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n

### Assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

ℤ = assemble(S, X, X, quadstrat=nearstrat)
𝕊x = assemble(Si, X, X, quadstrat=nearstrat) 
𝕊y = assemble(Si, Y, Y, quadstrat=nearstrat) 
𝕂x = assemble(K, X, X, quadstrat=nearstrat)
𝕂ix = assemble(Ki, X, X, quadstrat=nearstrat)
𝕂iy = assemble(Ki, Y, Y, quadstrat=nearstrat)

lhs = im*η * iNxy * ℤ * iNyx * 𝕊y + κ′^2 * iNxy * 𝕊x * iNyx * 𝕊y +  iNxy * (𝕂x - 𝕂ix) * iNyx * (0.5*Nyy + 𝕂iy)

@hilbertspace k
rhs = assemble(@discretise -e[k] k∈X)


### Solving cfie and efie by gmres (calculate iteration counts)
xi, ch = IterativeSolvers.gmres(lhs, iNxy * rhs, log=true, maxiter=100000000, abstol=1e-8, reltol=1e-8)
# xi2, ch2 = IterativeSolvers.gmres(im*κ*ℤ, rhs, log=true, maxiter=100000000, abstol=1e-8, reltol=1e-8)


### Print the condition number and iteration counts w.r.t. κ to file
# open("cube_kappa.txt", "a") do io
    # @printf(io, "%.3f %.3f %d %.3f %d\n", κ, cond(lhs), ch.iters, cond(im*κ*ℤ), ch2.iters)
# end;


### Print the condition number and iteration counts w.r.t. h to file
# open("cube_h.txt", "a") do io
#     @printf(io, "%.3f %.3f %d %.3f %d\n", meshsize, cond(lhs), ch.iters, cond(im*κ*ℤ), ch2.iters)
# end;

# end
