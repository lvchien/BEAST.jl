``` 
    This script file aims at calculating the scattering of an electromagnetic plane-wave by a cuboid PEC with 1 hole using the mixed MFIE discretization.
    Author: Van Chien Le, 2024
```

using BEAST, IterativeSolvers, Printf, LinearAlgebra
include("utils/genmesh.jl")

### Parameters
κ = 1.0

### Incident fields
E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
H = curl(E)
e = (n × E) × n
h = (n × H) × n

### Operators
S = MWSingleLayer3D(-im*κ, 1.0, -1.0/κ^2)                                        # EFIO with wave number κ 
Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                            # EFIO with pure imaginary wave number iκ
K = MWDoubleLayer3D(-im*κ)                                                       # MFIO with wave number κ
N = NCross()

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)
@hilbertspace k

```
    REFERENCE SOLUTION COMPUTATION
``` 
ref_mesh_size = 0.0125
Γ_ref = meshcuboid1hole(1.0, 0.25, 0.5, ref_mesh_size)
X_ref = raviartthomas(Γ_ref)
Y_ref = buffachristiansen(Γ_ref)

### Assembly of integral operators
N_ref = assemble(N, Y_ref, X_ref)
# iN_ref = inv(Matrix(N_ref))
# S_ref = assemble(S, X_ref, X_ref, quadstrat=nearstrat)
Si_ref = assemble(Si, X_ref, X_ref, quadstrat=nearstrat)
K_ref = assemble(K, Y_ref, X_ref, quadstrat=nearstrat)
mfie_ref = 0.5 * N_ref + K_ref
# rhs_efie_ref = assemble(@discretise -e[k] k∈X_ref)
rhs_mfie_ref = assemble(@discretise -h[k] k∈Y_ref)

### Solving the reference MFIE problem
# j_efie_ref, ch_efie_ref = IterativeSolvers.gmres(S_ref, rhs_efie_ref, log=true, maxiter=10000000000, abstol=1e-8, reltol=1e-8)    
j_mfie_ref, ch_mfie_ref = IterativeSolvers.gmres(mfie_ref, rhs_mfie_ref, log=true, maxiter=10000000000, abstol=1e-8, reltol=1e-8)    
# norm_efie_ref = sqrt(real(dot(j_efie_ref, Si_ref * j_efie_ref)))
norm_mfie_ref = sqrt(real(dot(j_mfie_ref, Si_ref * j_mfie_ref)))

### Print to file
open("cuboid.txt", "a") do io
    @printf(io, "%.4f %.3f %d %.12f %.3f %d %.12f \n", ref_mesh_size, 0.0, 0, 0.0, 0.0, ch_mfie_ref.iters, norm_mfie_ref)
end; 
```
    end
```

### Computational mesh
for meshsize in [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.185, 0.18, 0.16, 0.15, 0.135, 0.13, 0.12, 0.11, 0.1, 0.09, 0.075, 0.07, 0.065, 0.055, 0.05, 0.04, 0.03, 0.025]
    # meshsize = 0.024
    Γ = meshcuboid1hole(1.0, 0.25, 0.5, meshsize)

    ### RWG and BC function spaces
    X = raviartthomas(Γ)
    Y = buffachristiansen(Γ)
        
    ### Gram matrix
    Nyx = assemble(N, Y, X)
    iNyx = inv(Matrix(Nyx))

    ### Assembly of integral operators
    Sxx = assemble(S, X, X, quadstrat=nearstrat)
    Kyx = assemble(K, Y, X, quadstrat=nearstrat)
    mfie = iNyx * (0.5 * Nyx + Kyx)

    rhs_efie = assemble(@discretise -e[k] k∈X)
    rhs_mfie = iNyx * assemble(@discretise -h[k] k∈Y)

    ### Solving efie and mfie
    j_efie, ch_efie = IterativeSolvers.gmres(Sxx, rhs_efie, log=true, maxiter=10000000000, abstol=1e-8, reltol=1e-8)
    j_mfie, ch_mfie = IterativeSolvers.gmres(mfie, rhs_mfie, log=true, maxiter=10000000000, abstol=1e-8, reltol=1e-8)    
    
    ### Error calculations
    Sixx = assemble(Si, X, X, quadstrat=nearstrat)

    norm_efie = sqrt(real(dot(j_efie, Sixx * j_efie)))
    norm_mfie = sqrt(real(dot(j_mfie, Sixx * j_mfie)))
    @show norm_efie, norm_mfie

    ### Print to file
    open("cuboid.txt", "a") do io
        @printf(io, "%.4f %.3f %d %.12f %.3f %d %.12f \n", meshsize, cond(Sxx), ch_efie.iters, norm_efie, cond(mfie), ch_mfie.iters, norm_mfie)
    end; 
end
