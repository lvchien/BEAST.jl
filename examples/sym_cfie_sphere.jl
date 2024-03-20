``` 
    This script file aims at calculating the scattering of an electromagnetic plane-wave by the unit sphere using the symmetrized CFIE formulation.
    Author: Van Chien Le, 2024
```

using BEAST, Printf, SphericalScattering, StaticArrays, IterativeSolvers
include("utils/genmesh.jl")

### Parameters
κ = 1.43π                                                   # resonant wave number of the unit sphere
# κ = 2π/3                                                  # non-resonant wave number               
γ = -im*κ
κ′ = κ
η = κ^2

```
    FOR ERROR CALCULATION ONLY
```
### Setting for SphericalScattering package
# sp = PECSphere( 
#     radius      = 1.0, 
#     embedding   = Medium(1.0, 1.0),
# )

# ex = planeWave(
#     embedding    = Medium(1.0, 1.0),
#     frequency    = κ/(2π),
# )

### Evaluation points on the external sphere of radius r = 2m 
# r = 2.0
# Θ, φ = range(0, stop=π, length=50), range(0, stop=2π, length=101)
# pts = [SVector(r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ)) for θ ∈ Θ for ϕ ∈ φ[1:end-1]]

### Exact solutions by Mie series
# Emie = scatteredfield(sp, ex, ElectricField(pts))
# Hmie = scatteredfield(sp, ex, MagneticField(pts))

```
    end
```


```
    FOR CONDITION NUMBER W.R.T. h CALCULATION
```
### Different mesh sizes h for examining the condition number w.r.t. h
# for meshsize in [0.45, 0.35, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.13, 0.12, 0.11, 0.1, 0.09, 0.075, 0.065, 0.0055]
```
    end
```

### Computational mesh
meshsize = 0.15
Γ = meshsphere(1.0, meshsize)

### RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

### Gram matrices
N = NCross()
Nyx = assemble(N, Y, X)
Nyy = assemble(N, Y, Y)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)


```
    FOR CONDITION NUMBER W.R.T. κ CALCULATION
```
### Different wave numbers κ for examining the condition number w.r.t. κ
#     length_arr = 51
#     step = π/150
#     κ_arr = 4π/3 .+ step * [0:1:50;]

# for κ ∈ κ_arr[14:end]
#     γ = -im*κ
#     κ′ = κ
#     η = -κ^2
```
    end
```

### Integral operators
S = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)                                      # EFIO with wave number κ 
K = MWDoubleLayer3D(γ)                                                     # MFIO with wave number κ
Si = MWSingleLayer3D(κ′, 1.0, 1.0/κ′^2)                                    # EFIO with pure imaginary wave number iκ′
Ki = MWDoubleLayer3D(κ′)                                                   # MFIO with pure imaginary wave number iκ′

### Incident electric field
E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n

### Assembly of integral operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

ℤ = assemble(S, X, X, quadstrat=nearstrat)
𝕊x = assemble(Si, X, X, quadstrat=nearstrat) 
𝕊y = assemble(Si, Y, Y, quadstrat=nearstrat) 
𝕂x = assemble(K, X, X, quadstrat=nearstrat)
𝕂ix = assemble(Ki, X, X, quadstrat=nearstrat)
𝕂iy = assemble(Ki, Y, Y, quadstrat=nearstrat)

lhs = im*η * iNxy * ℤ * iNyx * 𝕊y + κ′^2 * iNxy * 𝕊x * iNyx * 𝕊y + iNxy * (𝕂x - 𝕂ix) * iNyx * (0.5*Nyy + 𝕂iy)

@hilbertspace k
rhs = assemble(@discretise -e[k] k∈X)

### Solving the CFIE
xi, ch = IterativeSolvers.gmres(lhs, iNxy * rhs, log=true, abstol=1e-12, reltol=1e-12)

jc, mc = iNyx * 𝕊y * xi, iNyx * (0.5*Nyy + 𝕂iy) * xi
    

```
    FOR ERROR CALCULATION ONLY
```
# Esct = im*η*potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, jc, X) + potential(BEAST.MWDoubleLayerField3D(γ), pts, mc, X)
# curlEsct = im*η*potential(BEAST.MWDoubleLayerField3D(γ), pts, jc, X) + κ^2 * potential(MWSingleLayerField3D(γ, 1.0, -1.0/κ^2), pts, mc, X)

# δe = Esct - conj.(Emie)
# δcurle = curlEsct - (im*κ).*conj.(Hmie)
# errorL2e = sqrt(real(dot(δe, δe)))/size(Emie)[1]
# errorHcurle = sqrt(real(dot(δe, δe)) + real(dot(δcurle, δcurle)))/size(Emie)[1]
```
    end
```

### Print the error to file
# open("sym_cfie_sphere_error_h_kappa.txt", "a") do io
#     @printf(io, "%.3f %.10f %.3f %d %.10f %.10f \n", meshsize, κ′, cond(lhs), ch.iters, errorL2e, errorHcurle)
# end; 


### Print the condition number w.r.t. η to file
# open("sym_cfie_sphere_cond_eta.txt", "a") do io
#     @printf(io, "%.10f %.3f %.3f\n", η, cond(lhs_pos), cond(lhs_neg))
# end; 


### Solving cfie and efie by gmres (calculate iteration counts)
# xi, ch = IterativeSolvers.gmres(lhs, iNxy * rhs, log=true, maxiter=100000000, abstol=1e-8, reltol=1e-8)
# xi2, ch2 = IterativeSolvers.gmres(im*κ*ℤ, rhs, log=true, maxiter=100000000, abstol=1e-8, reltol=1e-8)


### Print the condition number and iteration counts w.r.t. h to file
# open("sym_cfie_sphere_cond_h.txt", "a") do io
#     @printf(io, "%.3f %.3f %d %.3f %d \n", meshsize, cond(lhs), ch.iters, cond(im*κ*ℤ), ch2.iters)
# end; 


### Print the condition number and iteration counts w.r.t. κ to file
# open("sphere_kappa.txt", "a") do io
#     @printf(io, "%.3f %.3f %d %.3f %d \n", κ, cond(lhs), ch.iters, cond(im*κ*ℤ), ch2.iters)
# end; 

# end
