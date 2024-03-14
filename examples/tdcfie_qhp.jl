using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0
T0 = 8.0                                                                         # D/c with D the diameter of scatterer [second]

# Computational mesh
Γ = meshsphere(1.0, 0.3)
# fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/torus.geo")
# Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=4.0)
∂Γ = boundary(Γ)

# Connectivity matrices
edges = setminus(skeleton(Γ, 1), ∂Γ)
verts = setminus(skeleton(Γ, 0), skeleton(∂Γ, 0))
cells = skeleton(Γ, 2)

Σ = Matrix(connectivity(cells, edges, sign))
Λ = Matrix(connectivity(verts, edges, sign))

# Projectors
Id = LinearAlgebra.I
PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = Id - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙΣH = Id - ℙΛ

# RWG and BC spatial function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    
Δt, Nt = 0.1, 1200
κ = 1/Δt

# Operators
I = Identity()																			
N = NCross()
Tis = MWSingleLayer3D(κ, -κ, 0.0)                                                # weakly-singular EFIO with imaginary wavenumber
Tih = MWSingleLayer3D(κ, 0.0, -1.0/κ)                                            # hyper-singular EFIO with imaginary wavenumber
T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO (numdiff=0)
Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0)                                       # weakly-singular TD-EFIO (numdiffs=0)
∂Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 2, 0)                                      # weakly-singular TD-EFIO (numdiffs=1)
∂Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                        # hyper-singular TD-EFIO (numdiffs=1)
Ki = MWDoubleLayer3D(κ)                                                          # MFIO with imaginary wavenumber
K0 = MWDoubleLayer3D(0.0)                                                        # static MFIO
K = TDMaxwell3D.doublelayer(speedoflight=c)                                      # TD-MFIO

@hilbertspace k
@hilbertspace j

# Gram matrix
Gxx = assemble(I, X, X)
Nyy = assemble(N, Y, Y)
Nyx = assemble(N, Y, X)
Nxy = -transpose(Nyx)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)

𝕋is = assemble(Tis, Y, Y, quadstrat=nearstrat)
𝕋ih = assemble(Tih, Y, Y, quadstrat=nearstrat)
𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄i = Matrix(-0.5 * Nyx + 𝕂i)

```
                MAIN PART 
```

# Plane wave incident fields
duration = 80 * Δt * c                                        
delay = 240 * Δt                                        
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
polarisation, direction = x̂, ẑ
E = planewave(polarisation, direction, gaussian, c)
iE = planewave(polarisation, direction, integrate(gaussian), c)
H = direction × E
iH = direction × iE

# Temporal function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = convolve(p, h)                                                                  # quadratic function space (*Δt)
∂q = derive(q)                                                                      # first order derivative of q (*Δt)


### FORM 1: standard TD-EFIE
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
ex = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Txx)[1:2])
ConvolutionOperators.timeslice!(Z01, Txx, 1)
iZ01 = inv(Z01)
je = marchonintime(iZ01, Txx, ex, Nt)


### FORM 3: qHP CP TD-EFIE
BEAST.@defaultquadstrat (∂Ts, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (∂Th, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

Ms_bilform_31 = @discretise ∂Ts[k, j] k∈X⊗δ j∈X⊗q
Ms_bilform_32 = @discretise ∂Th[k, j] k∈X⊗δ j∈X⊗q
Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗h

Ms_31 = T0/Δt * BEAST.td_assemble(Ms_bilform_31.bilform, Ms_bilform_31.test_space_dict, Ms_bilform_31.trial_space_dict)
Ms_32 = T0/Δt * BEAST.td_assemble(Ms_bilform_32.bilform, Ms_bilform_32.test_space_dict, Ms_bilform_32.trial_space_dict)
Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)

ECP = (Δt/T0 * ℙΣH * 𝕋is * ℙΣH + ℙΛ * 𝕋is * ℙΣH + ℙΣH * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋ih * ℙΛ) * (iNxy * PΛH + ℙΣH * iNxy * PΣ) * (Δt/T0 * PΛH + PΣ)

qhpefie = ECP * (Ml_3 * PΛH + Ms_31 * PΣ + PΣ * Ms_32 * PΣ)
rhs3 = ECP * ex


### FORM 4: standard TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
hy = BEAST.td_assemble(linform_4.linform, linform_4.test_space_dict)

Z04 = zeros(Float64, size(Kyx)[1:2])
ConvolutionOperators.timeslice!(Z04, Kyx, 1)
iZ04 = inv(Z04)
jm = marchonintime(iZ04, Kyx, hy, Nt)


### FORM 5: qHP symmetrized TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_5s = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

Ks = T0/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)

MCP = (Δt/T0 * ℙΣH + ℙΛ) * 𝕄i * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ)

qhpmfie = MCP * (Kyx * PΛH + Ks * PΣ)

rhs5 = MCP * hy


### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
cfie = Txx + (-1) * Gxx * iNyx * Kyx

Z06 = zeros(Float64, size(cfie)[1:2])
ConvolutionOperators.timeslice!(Z06, cfie, 1)
iZ06 = inv(Z06)
jc = marchonintime(iZ06, cfie, ex - Gxx * iNyx * hy, Nt)


### FORM 7: qHP CP TD-CFIE
qhpcfie = qhpefie + qhpmfie
rhs7 = rhs3 + rhs5

Z07 = zeros(Float64, size(qhpcfie)[1:2])
ConvolutionOperators.timeslice!(Z07, qhpcfie, 1)
# Z07 .-= Δt/T0 * ℙΣH * (-0.5 * Nyx + 𝕂0) * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ) * (0.5 * Nyx + 𝕂0) * PΛH
iZ07 = inv(Z07)
y7 = marchonintime(iZ07, qhpcfie, rhs7, Nt)

jqhpc = zeros(eltype(y7), size(y7)[1:2])
jqhpc[:, 1] = PΛH * y7[:, 1] + T0/Δt * PΣ * y7[:, 1]
for i in 2:Nt
    jqhpc[:, i] = PΛH * y7[:, i] + T0/Δt * PΣ * (y7[:, i] - y7[:, i-1])
end



```
                CONDITION NUMBERS WITH RESPECT TO MESH SIZES
```

using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0
T0 = 8.0                                                                            # D/c with D the diameter of scatterer

for meshsize in [0.9, 0.7, 0.55, 0.45, 0.4, 0.35, 0.3, 0.27, 0.24, 0.21, 0.18] 
    # Computational mesh
    # Γ = meshsphere(1.0, meshsize)
    fn = joinpath(dirname(pathof(CompScienceMeshes)),"geos/torus.geo")
    Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=meshsize)

    ∂Γ = boundary(Γ)

    # Connectivity matrices
    edges = setminus(skeleton(Γ,1), ∂Γ)
    verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))
    cells = skeleton(Γ,2)

    Σ = Matrix(connectivity(cells, edges, sign))
    Λ = Matrix(connectivity(verts, edges, sign))

    # Projectors
    Id = LinearAlgebra.I
    PΣ = Σ * pinv(Σ'*Σ) * Σ'
    PΛH = Id - PΣ

    ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
    ℙΣH = Id - ℙΛ

    # RWG and BC function spaces
    X = raviartthomas(Γ)
    Y = buffachristiansen(Γ)

    Δt, Nt = 8.0, 10
    κ = 1/Δt

    # Operators
    I = Identity()																			
    N = NCross()
    Tis = MWSingleLayer3D(κ, -κ, 0.0)                                                # weakly-singular EFIO with imaginary wavenumber
    Tih = MWSingleLayer3D(κ, 0.0, -1.0/κ)                                            # hyper-singular EFIO with imaginary wavenumber
    T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO (numdiff=0)
    Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0)                                       # weakly-singular TD-EFIO (numdiffs=0)
    ∂Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 2, 0)                                      # weakly-singular TD-EFIO (numdiffs=1)
    ∂Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                        # hyper-singular TD-EFIO (numdiffs=1)
    Ki = MWDoubleLayer3D(κ)                                                          # MFIO with imaginary wavenumber
    K0 = MWDoubleLayer3D(0.0)                                                        # static MFIO
    K = TDMaxwell3D.doublelayer(speedoflight=c)                                      # TD-MFIO
    
    @hilbertspace k
    @hilbertspace j

    # Gram matrix
    Gxx = assemble(I, X, X)
    Nyy = assemble(N, Y, Y)
    Nyx = assemble(N, Y, X)
    Nxy = -transpose(Nyx)
    iNyx = inv(Matrix(Nyx))
    iNxy = -transpose(iNyx)

    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)

    # assembly of static operators
    𝕋is = assemble(Tis, Y, Y, quadstrat=nearstrat)
    𝕋ih = assemble(Tih, Y, Y, quadstrat=nearstrat)
    𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
    𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
    𝕄i = Matrix(-0.5 * Nyx + 𝕂i)


    ```
                    MAIN PART 
    ```

    # Temporal function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = convolve(p, h)                                                                  # quadratic function space (*Δt)
    ∂q = derive(q)                                                                      # first order derivative of q (*Δt)


    ### FORM 1: standard TD-EFIE
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
    Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)
    
    Z01 = zeros(Float64, size(Txx)[1:2])
    ConvolutionOperators.timeslice!(Z01, Txx, 1)

    ### FORM 3: qHP CP TD-EFIE
    BEAST.@defaultquadstrat (∂Ts, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (∂Th, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
    
    Ms_bilform_31 = @discretise ∂Ts[k, j] k∈X⊗δ j∈X⊗q
    Ms_bilform_32 = @discretise ∂Th[k, j] k∈X⊗δ j∈X⊗q
    Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗h
    
    Ms_31 = T0/Δt * BEAST.td_assemble(Ms_bilform_31.bilform, Ms_bilform_31.test_space_dict, Ms_bilform_31.trial_space_dict)
    Ms_32 = T0/Δt * BEAST.td_assemble(Ms_bilform_32.bilform, Ms_bilform_32.test_space_dict, Ms_bilform_32.trial_space_dict)
    Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)
    
    ECP = (Δt/T0 * ℙΣH * 𝕋is * ℙΣH + ℙΛ * 𝕋is * ℙΣH + ℙΣH * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋ih * ℙΛ) * (iNxy * PΛH + ℙΣH * iNxy * PΣ) * (Δt/T0 * PΛH + PΣ)
    
    qhpefie = ECP * (Ml_3 * PΛH + Ms_31 * PΣ + PΣ * Ms_32 * PΣ)


    ### FORM 4: standard TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
    Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)
    
    Z04 = zeros(Float64, size(Kyx)[1:2])
    ConvolutionOperators.timeslice!(Z04, Kyx, 1)


    ### FORM 5: qHP symmetrized TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_5s = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q
    
    Ks = T0/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)
    
    MCP = (Δt/T0 * ℙΣH + ℙΛ) * 𝕄i * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ)
    
    qhpmfie = MCP * (Kyx * PΛH + Ks * PΣ)    


    ### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
    Z06 = Z01 - Gxx * iNyx * Z04


    ### FORM 7: qHP localized CP TD-CFIE
    qhpcfie = qhpefie + qhpmfie

    Z07 = zeros(Float64, size(qhpcfie)[1:2])
    ConvolutionOperators.timeslice!(Z07, qhpcfie, 1)
    Z07 .-= Δt/T0 * ℙΣH * (-0.5 * Nyx + 𝕂0) * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ) * Z04 * PΛH

    open("torus_cdt_8.txt", "a") do io
        @printf(io, "%.4f %.10f %.10f %.10f %.10f\n", meshsize, cond(Z01), cond(Z04), cond(Z06), cond(Z07))
    end; 
end







```
                CONDITION NUMBERS WITH RESPECT TO TIME STEPS
```

using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0
T0 = 8.0                                                                                # D/c with D the diameter of scatterer

# Computational mesh
# Γ = meshsphere(1.0, 0.3)
fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/torus.geo")
Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.6)
∂Γ = boundary(Γ)

# Connectivity matrices
edges = setminus(skeleton(Γ,1), ∂Γ)
verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))
cells = skeleton(Γ,2)

Σ = Matrix(connectivity(cells, edges, sign))
Λ = Matrix(connectivity(verts, edges, sign))

# Projectors
Id = LinearAlgebra.I
PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = Id - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙΣH = Id - ℙΛ

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
				
I = Identity()
N = NCross()

# Gram matrix
Gxx = assemble(I, X, X)
Nyy = assemble(N, Y, Y)
Nyx = assemble(N, Y, X)
Nxy = -transpose(Nyx)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)
K0 = MWDoubleLayer3D(0.0)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)

for Δt in [0.4375, 0.8750, 1.75, 3.5, 7.0, 14.0, 28.0, 56.0, 112.0, 224.0, 448.0, 896.0, 1702.0, 3404.0, 6808.0, 13616.0, 27232.0]
    Nt = 10
    κ = 1/Δt
   
    # Plane wave incident fields
    duration = 80 * Δt * c                                        
    delay = 240 * Δt                                        
    amplitude = 1.0
    gaussian = creategaussian(duration, delay, amplitude)
    fgaussian = fouriertransform(gaussian)
    polarisation, direction = x̂, ẑ
    E = planewave(polarisation, direction, gaussian, c)
    iE = planewave(polarisation, direction, integrate(gaussian), c)
    H = direction × E
    iH = direction × iE

    # Operators
    I = Identity()
    Tis = MWSingleLayer3D(κ, -κ, 0.0)                                                # weakly-singular EFIO with imaginary wavenumber
    Tih = MWSingleLayer3D(κ, 0.0, -1.0/κ)                                            # hyper-singular EFIO with imaginary wavenumber
    T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO (numdiff=0)
    Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0)                                       # weakly-singular TD-EFIO (numdiffs=0)
    ∂Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 2, 0)                                      # weakly-singular TD-EFIO (numdiffs=1)
    ∂Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                        # hyper-singular TD-EFIO (numdiffs=1)
    Ki = MWDoubleLayer3D(κ)                                                          # MFIO with imaginary wavenumber
    K = TDMaxwell3D.doublelayer(speedoflight=c)                                      # TD-MFIO
    
    @hilbertspace k
    @hilbertspace j

    # assembly of static operators
    𝕋is = assemble(Tis, Y, Y, quadstrat=nearstrat)
    𝕋ih = assemble(Tih, Y, Y, quadstrat=nearstrat)
    𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
    𝕄i = Matrix(-0.5 * Nyx + 𝕂i)

    ```
                    MAIN PART 
    ```

    # Temporal function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = convolve(p, h)                                                                  # quadratic function space (*Δt)
    ∂q = derive(q)                                                                      # first order derivative of q (*Δt)

    ### FORM 1: standard TD-EFIE
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
    Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)
    
    Z01 = zeros(Float64, size(Txx)[1:2])
    ConvolutionOperators.timeslice!(Z01, Txx, 1)


    ### FORM 3: qHP CP TD-EFIE
    BEAST.@defaultquadstrat (∂Ts, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (∂Th, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
    
    Ms_bilform_31 = @discretise ∂Ts[k, j] k∈X⊗δ j∈X⊗q
    Ms_bilform_32 = @discretise ∂Th[k, j] k∈X⊗δ j∈X⊗q
    Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗h
    
    Ms_31 = T0/Δt * BEAST.td_assemble(Ms_bilform_31.bilform, Ms_bilform_31.test_space_dict, Ms_bilform_31.trial_space_dict)
    Ms_32 = T0/Δt * BEAST.td_assemble(Ms_bilform_32.bilform, Ms_bilform_32.test_space_dict, Ms_bilform_32.trial_space_dict)
    Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)
    
    ECP = (Δt/T0 * ℙΣH * 𝕋is * ℙΣH + ℙΛ * 𝕋is * ℙΣH + ℙΣH * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋ih * ℙΛ) * (iNxy * PΛH + ℙΣH * iNxy * PΣ) * (Δt/T0 * PΛH + PΣ)
    
    qhpefie = ECP * (Ml_3 * PΛH + Ms_31 * PΣ + PΣ * Ms_32 * PΣ)

    
    ## FORM 4: standard TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
    Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

    Z04 = zeros(Float64, size(Kyx)[1:2])
    ConvolutionOperators.timeslice!(Z04, Kyx, 1)
    
    ### FORM 5: qHP symmetrized TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_5s = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q
    Ks = T0/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)
    MCP = (Δt/T0 * ℙΣH + ℙΛ) * 𝕄i * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ)
    qhpmfie = MCP * (Kyx * PΛH + Ks * PΣ)    


    ### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
    Z06 = Z01 - Gxx * iNyx * Z04


    ### FORM 7: qHP localized CP TD-CFIE
    qhpcfie = qhpefie + qhpmfie

    Z07 = zeros(Float64, size(qhpcfie)[1:2])
    ConvolutionOperators.timeslice!(Z07, qhpcfie, 1)
    
    if Δt < T0
        Z07 .-= Δt/T0 * ℙΣH * (-0.5 * Nyx + 𝕂0) * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ) * (0.5 * Nyx + 𝕂0) * PΛH
    else
        # Only for multiply-connected geometries
        Z07 .-= Δt/T0 * ℙΣH * (-0.5 * Nyx + 𝕂0) * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ) * Z04 * PΛH
    end

    open("torus_h_0.6m.txt", "a") do io
        @printf(io, "%.4f %.10f %.10f %.10f %.10f\n", Δt, cond(Z01), cond(Z04), cond(Z06), cond(Z07))
    end; 
end