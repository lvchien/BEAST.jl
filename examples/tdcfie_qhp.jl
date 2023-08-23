using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0

# Computational mesh
# Γ = meshsphere(1.0, 0.3)
# Γ = meshtorus(3.0, 1.0, 0.5)
# Γ = meshsquaretorus4holes(8.0, 2.0, 2.0, 0.8)
# Γ = meshsquaretorus(8.0, 2.0, 4.0, 0.5)
fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/torus.geo")
Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.6)
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

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    
Δt, Nt = 0.1, 3600
κ = 1/Δt

# Operators
I = Identity()																			
N = NCross()
Tis = MWSingleLayer3D(κ, κ, 0.0)                                                 # weakly-singular EFIO with imaginary wavenumber
Tih = MWSingleLayer3D(κ, 0.0, 1.0/κ)                                             # hyper-singular EFIO with imaginary wavenumber
T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO
Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0)                                       # weakly-singular TD-EFIO (numdiffs=1)
∂T = BEAST.derive(T)                                                             # time derivative of T
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
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

𝕋is = assemble(Tis, Y, Y, quadstrat=nearstrat)
𝕋ih = assemble(Tih, Y, Y, quadstrat=nearstrat)
𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
# 𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄i = Matrix(-0.5 * Nyx + 𝕂i)

```
                MAIN PART 
```

# Plane wave
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

# Time function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = timebasisshiftedlagrange(Δt, Nt, 2)                        		                # quadratic function space
cb = convolve(p, q)                                                                 # cubic function space (*Δt)
∂cb = derive(cb)                                                                    # first order derivative of cb (*Δt)

### FORM 1: standard TD-EFIE
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(10)

bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
ex = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Txx)[1:2])
ConvolutionOperators.timeslice!(Z01, Txx, 1)
iZ01 = inv(Z01)
je = marchonintime(iZ01, Txx, ex, Nt)


### FORM 3: qHP CP TD-EFIE
BEAST.@defaultquadstrat (∂T, X⊗δ, X⊗cb) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)

Ms_bilform_3 = @discretise ∂T[k, j] k∈X⊗δ j∈X⊗cb
Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗q

Ms_3 = 1/Δt * BEAST.td_assemble(Ms_bilform_3.bilform, Ms_bilform_3.test_space_dict, Ms_bilform_3.trial_space_dict)
Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)

ECP = (1/κ * ℙΣH * 𝕋is + ℙΛ * 𝕋is + ℙΛ * 𝕋ih) * (iNxy * PΛH + ℙΣH * iNxy * PΣ)

qhpefie = ECP * (Ml_3 * PΛH + Ms_3 * PΣ)
rhs3 = ECP * ex

# Z03 = zeros(Float64, size(qhpefie)[1:2])
# ConvolutionOperators.timeslice!(Z03, qhpefie, 1)
# iZ03 = inv(Z03)
# y3 = marchonintime(iZ03, qhpefie, rhs3, Nt)

# j3 = zeros(eltype(y3), size(y3)[1:2])
# j3[:, 1] = PΛH * y3[:, 1] + 1.0/Δt * PΣ * y3[:, 1]
# for i in 2:Nt
#     j3[:, i] = PΛH * y3[:, i] + 1.0/Δt * PΣ * (y3[:, i] - y3[:, i-1])
# end


### FORM 4: standard TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(10)

bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
hy = BEAST.td_assemble(linform_4.linform, linform_4.test_space_dict)

Z04 = zeros(Float64, size(Kyx)[1:2])
ConvolutionOperators.timeslice!(Z04, Kyx, 1)
iZ04 = inv(Z04)
jm = marchonintime(iZ04, Kyx, hy, Nt)


### FORM 5: qHP symmetrized TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂cb) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_5l = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗q
bilform_5s = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂cb

Kl = BEAST.td_assemble(bilform_5l.bilform, bilform_5l.test_space_dict, bilform_5l.trial_space_dict)
Ks = 1/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)

MCP = (1/κ * ℙΣH + ℙΛ) * 𝕄i * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ)

qhpmfie = MCP * (Kl * PΛH + Ks * PΣ)

rhs5 = MCP * hy

# Z05 = zeros(Float64, size(qhpmfie)[1:2])
# ConvolutionOperators.timeslice!(Z05, qhpmfie, 1)
# iZ05 = inv(Z05)
# y5 = marchonintime(iZ05, qhpmfie, rhs5, Nt)

# j5 = zeros(eltype(y5), size(y5)[1:2])
# j5[:, 1] = PΛH * y5[:, 1] + 1.0/Δt * PΣ * y5[:, 1]
# for i in 2:Nt
#     j5[:, i] = PΛH * y5[:, i] + 1.0/Δt * PΣ * (y5[:, i] - y5[:, i-1])
# end


### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
cfie = Txx + (-1) * Gxx * iNyx * Kyx

Z06 = zeros(Float64, size(cfie)[1:2])
ConvolutionOperators.timeslice!(Z06, cfie, 1)
iZ06 = inv(Z06)
jc = marchonintime(iZ06, cfie, ex - Gxx * iNyx * hy, Nt)


### FORM 7: qHP CP TD-CFIE
qhpcfie = qhpefie + (-1) * qhpmfie
rhs7 = rhs3 - rhs5

Z07 = zeros(Float64, size(qhpcfie)[1:2])
ConvolutionOperators.timeslice!(Z07, qhpcfie, 1)
iZ07 = inv(Z07)
y7 = marchonintime(iZ07, qhpcfie, rhs7, Nt)

jqhpc = zeros(eltype(y7), size(y7)[1:2])
jqhpc[:, 1] = PΛH * y7[:, 1] + 1.0/Δt * PΣ * y7[:, 1]
for i in 2:Nt
    jqhpc[:, i] = PΛH * y7[:, i] + 1.0/Δt * PΣ * (y7[:, i] - y7[:, i-1])
end



```
                CONDITION NUMBERS WITH RESPECT TO MESH SIZES
```

using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0

for meshsize in [0.9, 0.7, 0.55, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    # Computational mesh
    # Γ = meshsphere(1.0, meshsize)
    # Γ = meshcuboid(0.5, 2.0, 2.0, 0.3)
    # Γ = meshtorus(3.0, 1.0, 0.6)
    fn = joinpath(dirname(pathof(CompScienceMeshes)),"geos/torus.geo")
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

    Δt, Nt = 64.0, 4
    κ = 1/Δt

    # Operators
    I = Identity()																			
    N = NCross()
    Ti = MWSingleLayer3D(κ, κ, 1/κ)                                                  # EFIO with imaginary wavenumber
    T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO
    Ts = integrate(MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0))                            # weakly-singular TD-EFIO (numdiffs=0)
    Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                         # hypersingular TD-EFIO (numdiffs=0)
    Ki = MWDoubleLayer3D(κ)                                                          # MFIO with imaginary wavenumber
    K = TDMaxwell3D.doublelayer(speedoflight=c)                                      # TD-MFIO
    K0 = MWDoubleLayer3D(0.0)

    @hilbertspace k
    @hilbertspace j

    # Gram matrix
    Gxx = assemble(I, X, X)
    Nyy = assemble(N, Y, Y)
    Nyx = assemble(N, Y, X)
    Nxy = -transpose(Nyx)
    iNyx = inv(Matrix(Nyx))
    iNxy = -transpose(iNyx)

    nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

    # assembly of static operators
    𝕋i = assemble(Ti, Y, Y, quadstrat=nearstrat)
    𝕄i = Matrix(-0.5 * Nyx + 𝕂i)
    𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
    𝕄0 = Matrix(-0.5 * Nyx + 𝕂0)


    ```
                    MAIN PART 
    ```

    # Time function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
    ∂h = BEAST.derive(h)							                                    # derivative of h
    ∂q = BEAST.derive(q)					                                            # first order derivative of q (*Δt)
    ip = integrate(p) 	                			                                    # integral of p


    ### FORM 1: standard TD-EFIE
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
    Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

    Z01 = zeros(Float64, size(Txx)[1:2])
    ConvolutionOperators.timeslice!(Z01, Txx, 1)

    ### FORM 3: qHP CP TD-EFIE
    BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗∂h
    Ms_bilform_3 = @discretise T[k, j] k∈X⊗δ j∈X⊗∂q

    Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)
    Ms_3 = 1/Δt * BEAST.td_assemble(Ms_bilform_3.bilform, Ms_bilform_3.test_space_dict, Ms_bilform_3.trial_space_dict)

    qhpefie = (1/κ * ℙΣH + ℙΛ) * 𝕋i * (ℙΣH + κ * ℙΛ) * iNxy * (1/κ * PΛH + PΣ) * (Ml_3 * PΛH + Ms_3 * PΣ)

    Z03 = zeros(Float64, size(qhpefie)[1:2])
    ConvolutionOperators.timeslice!(Z03, qhpefie, 1)


    ### FORM 4: standard TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
    Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

    Z04 = zeros(Float64, size(Kyx)[1:2])
    ConvolutionOperators.timeslice!(Z04, Kyx, 1)


    ### FORM 5: qHP symmetrized TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_5s = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

    Kss = 1/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)

    qhpmfie = (1/κ * ℙΣH + ℙΛ) * 𝕄i * iNyx * (Kyx * PΛH + Kss * PΣ)

    Z05 = zeros(Float64, size(qhpmfie)[1:2])
    ConvolutionOperators.timeslice!(Z05, qhpmfie, 1)

    Z05 .-= 1/κ * ℙΣH * 𝕄0 * iNyx * (0.5 * Nyx + 𝕂0) * PΛH


    ### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
    Z06 = Z01 - Gxx * iNyx * Z04


    ### FORM 7: qHP localized CP TD-CFIE
    Z07 = Z03 - Z05


    open("torus_cdt_2.txt", "a") do io
        @printf(io, "%.4f %.10f %.10f %.10f %.10f %.10f %.10f\n", meshsize, cond(Z01), cond(Z03), cond(Z04), cond(Z05), cond(Z06), cond(Z07))
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

# Computational mesh
# Γ = meshsphere(1.0, 0.3)
# Γ = meshcuboid(0.5, 2.0, 2.0, 0.3)
Γ = meshtorus(3.0, 1.0, 0.6)
# fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/rectangular_torus.geo")
# Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.6)
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
																		
N = NCross()

# Gram matrix
Nyy = assemble(N, Y, Y)
Nyx = assemble(N, Y, X)
Nxy = -transpose(Nyx)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)
K0 = MWDoubleLayer3D(0.0)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄0 = Matrix(-0.5 * Nyx + 𝕂0)


Δt = 4096.0
    Nt = 4
    κ = 1/Δt

    # Operators
    Idn = Identity()
    Ti = MWSingleLayer3D(κ, κ, 1/κ)                                                  # EFIO with imaginary wavenumber
    T = TDMaxwell3D.singlelayer(speedoflight=c)                                      # TD-EFIO
    ∂T = BEAST.derive(T)                                                             # time derivative of T
    # Ts = integrate(MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0))                            # weakly-singular TD-EFIO (numdiffs=0)
    # Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                         # hypersingular TD-EFIO (numdiffs=0)
    Ki = MWDoubleLayer3D(κ)                                                          # MFIO with imaginary wavenumber
    K = TDMaxwell3D.doublelayer(speedoflight=c)                                      # TD-MFIO
    
    @hilbertspace k
    @hilbertspace j

    # assembly of static operators
    𝕋i = assemble(Ti, Y, Y, quadstrat=nearstrat)
    𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
    𝕄i = Matrix(-0.5 * Nyx + 𝕂i)

    ```
                    MAIN PART 
    ```

    # Time function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
    ∂h = BEAST.derive(h)							                                    # derivative of h
    ∂q = BEAST.derive(q)					                                            # first order derivative of q (*Δt)
    ip = integrate(p) 	                			                                    # integral of p


    ### FORM 1: standard TD-EFIE
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
    Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

    Z01 = zeros(Float64, size(Txx)[1:2])
    ConvolutionOperators.timeslice!(Z01, Txx, 1)

    ### FORM 3: qHP CP TD-EFIE
    BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(7)
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗∂h
    Ms_bilform_3 = @discretise T[k, j] k∈X⊗δ j∈X⊗∂q

    Ml_3 = BEAST.td_assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)
    Ms_3 = 1/Δt * BEAST.td_assemble(Ms_bilform_3.bilform, Ms_bilform_3.test_space_dict, Ms_bilform_3.trial_space_dict)

    qhpefie = (1/κ * ℙΣH + ℙΛ) * 𝕋i * (ℙΣH + κ * ℙΛ) * iNxy * (1/κ * PΛH + PΣ) * (Ml_3 * PΛH + Ms_3 * PΣ)

    Z03 = zeros(Float64, size(qhpefie)[1:2])
    ConvolutionOperators.timeslice!(Z03, qhpefie, 1)


    
    ### FORM 4: standard TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_4 = @discretise (0.5(N⊗Idn) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
    Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

    Z04 = zeros(Float64, size(Kyx)[1:2])
    ConvolutionOperators.timeslice!(Z04, Kyx, 1)


    ### FORM 5: qHP symmetrized TD-MFIE
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

    bilform_5s = @discretise (0.5(N⊗Idn) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

    Kss = 1/Δt * BEAST.td_assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)

    qhpmfie = (1/κ * ℙΣH + ℙΛ) * 𝕄i * iNyx * (Kyx * PΛH + Kss * PΣ)

    Z05 = zeros(Float64, size(qhpmfie)[1:2])
    ConvolutionOperators.timeslice!(Z05, qhpmfie, 1)

    if Δt < 8
        Z05 .-= 1/κ * ℙΣH * 𝕄0 * iNyx * (0.5 * Nyx + 𝕂0) * PΛH
    else
        Z05 .-= 1/κ * ℙΣH * 𝕄0 * iNyx * Z04 * PΛH
    end


    ### FORM 6: standard TD-CFIE (Beghein et. al., 2013)
    Z06 = Z01 - Gxx * iNyx * Z04


    ### FORM 7: qHP localized CP TD-CFIE
    Z07 = Z03 - Z05


    open("torus_h_0.6m.txt", "a") do io
        @printf(io, "%.4f %.10f %.10f %.10f %.10f %.10f %.10f\n", Δt, cond(Z01), cond(Z03), cond(Z04), cond(Z05), cond(Z06), cond(Z07))
    end; 
end