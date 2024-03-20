``` 
    This script file aims at calculating the scattering of an electromagnetic plane-wave by a sphere PEC using the mixed MFIE discretization.
    Author: Van Chien Le, 2024
```

using BEAST, IterativeSolvers, Printf, SphericalScattering, StaticArrays, LinearAlgebra
include("utils/genmesh.jl")

### Parameters
κ = 1.0
radius = 1.0

### Operators
S = MWSingleLayer3D(-im*κ, 1.0, -1.0/κ^2)                                        # EFIO with wave number κ 
K = MWDoubleLayer3D(-im*κ)                                                       # MFIO with wave number κ
N = NCross()

### Computational mesh
for meshsize in [0.45, 0.35, 0.25, 0.2, 0.185, 0.18, 0.16, 0.15, 0.135, 0.13, 0.12, 0.11, 0.1, 0.09, 0.075, 0.065, 0.055]
    # meshsize = 0.3
    Γ = meshsphere(radius, meshsize)
    # Γ = meshcuboid4holes(2.0, 0.5, 0.5, meshsize)
    # Γ = meshcone(0.5, 5.0, meshsize)

    ### RWG and BC function spaces
    X = raviartthomas(Γ)
    Y = buffachristiansen(Γ)
        
    ### Gram matrix
    Nyx = assemble(N, Y, X)
    iNyx = inv(Matrix(Nyx))

    ### Incident fields
    E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
    H = curl(E)
    e = (n × E) × n
    h = (n × H) × n

    ### Assembly of integral operators
    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

    Sxx = assemble(S, X, X, quadstrat=nearstrat)
    Kyx = assemble(K, Y, X, quadstrat=nearstrat)
    mfie = iNyx * (0.5 * Nyx + Kyx)

    @hilbertspace k
    rhs_efie = assemble(@discretise -e[k] k∈X)
    rhs_mfie = iNyx * assemble(@discretise -h[k] k∈Y)

    ### Solving efie and mfie
    j_efie, ch_efie = IterativeSolvers.gmres(Sxx, rhs_efie, log=true, abstol=1e-8, reltol=1e-8)
    j_mfie, ch_mfie = IterativeSolvers.gmres(mfie, rhs_mfie, log=true, abstol=1e-8, reltol=1e-8)    
    

    ```
       FOR ERROR CALCULATION ONLY
    ``` 
    mutable struct SurfaceCurrent <: BEAST.Functional
        sphere
        excitation
    end

    function (j::SurfaceCurrent)(p)
        ε = 1e-8
        sp = j.sphere
        ex = j.excitation
        cart = cartesian(p)
        x = [SVector(cart/(norm(cart)-ε))]
        n = normal(p)
        return im*κ*((n×field(sp, ex, MagneticField(x))[1])×n)
    end

    BEAST.integrand(f::SurfaceCurrent, tval, fval) = dot(fval, tval.value)

    sp = PECSphere( 
        radius      = radius, 
    )

    ex = planeWave(
        embedding    = Medium(1.0, 1.0),
        frequency    = κ/(2π),
    )

    ### Mie's solution
    j_mie = iNyx * assemble(SurfaceCurrent(sp, ex), Y)

    ### Error calculations
    Si = MWSingleLayer3D(κ, 1.0, 1.0/κ^2)                                        # EFIO with pure imaginary wave number iκ
    Sixx = assemble(Si, X, X, quadstrat=nearstrat)

    diff_efie = j_efie - j_mie
    diff_mfie = j_mfie - j_mie

    err_efie = sqrt(real(dot(diff_efie, Sixx * diff_efie)))
    err_mfie = sqrt(real(dot(diff_mfie, Sixx * diff_mfie)))

    ### Print to file
    open("mfie_mixed_sphere_error.txt", "a") do io
        @printf(io, "%.3f %.3f %d %.12f %.3f %d %.12f \n", meshsize, cond(Sxx), ch_efie.iters, err_efie, cond(mfie), ch_mfie.iters, err_mfie)
    end; 
    ```
        end
    ```
end
