using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0

# static operators
I = Identity()																			
N = NCross()
T0s = MWSingleLayer3D(0.0, 1.0, 0.0)                                              # static weakly-singular EFIO 
T0h = MWSingleLayer3D(0.0, 0.0, 1.0)   

# time-domain operators 
T = TDMaxwell3D.singlelayer(speedoflight=c)                                     # TD-EFIO 
∂T = derive(T)                                                                  # differentiated TD-EFIE
Ts = MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0)                                      # weakly-singular TD-EFIO
∂Ts = derive(Ts)                                                                # differentiated weakly-singular TD-EFIO
iTs = integrate(Ts)                                                             # itegrated weakly-singular TD-EFIO  
∂Th = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                       # differentiated hyper-singular TD-EFIO                     
K = TDMaxwell3D.doublelayer(speedoflight=c)                                     # TD-MFIO
∂K = derive(K)                                                                  # differentiated TD-MFIO
iK = integrate(K)                                                               # integrated TD_MFIO

```
        SPATIAL DISCRETIZATION
```
# Diameter of the scatterer
T0 = 8.0                                                                        # D/c with D the diameter of scatterer [second]

# Computational mesh
# Γ = meshsphere(1.0, 0.55)
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

# RWG and BC spatial function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

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
𝕋0s = assemble(T0s, Y, Y, quadstrat=nearstrat)
𝕋0h = assemble(T0h, Y, Y, quadstrat=nearstrat)

# preconditioners
𝕄Λ = ℙΛ * 𝕋0h * ℙΛ * iNxy * PΛH
𝕄ΣH = ℙΣH * 𝕋0s * ℙΣH * iNxy 

```
        TEMPORAL DISCRETIZATION
```

Δt, Nt = 0.1, 3600

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
ip = integrate(p)                                                                   # integrate of p
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = convolve(p, h)                                                                  # quadratic function space (*Δt)                                                            
∂q = derive(q)                                                                      # first order derivative of q (*Δt)

```
                MAIN PART 
```
      
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

# FORM 2: qHP CP TD-CFIE
BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (∂T, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (∂K, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (iTs, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (iK, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)

Cll = @discretise (Ts + 0.5(N⊗I) + 1.0K)[k,j] k∈X⊗δ j∈X⊗h
Cls = @discretise (∂T + ∂K)[k,j] k∈X⊗δ j∈X⊗q
ClsI = @discretise (0.5(N⊗I))[k, j] k∈X⊗δ j∈X⊗∂q
Csl = @discretise (iTs + iK)[k,j] k∈X⊗δ j∈X⊗p

Zll = BEAST.td_assemble(Cll.bilform, Cll.test_space_dict, Cll.trial_space_dict)
Zls = T0/Δt * (BEAST.td_assemble(Cls.bilform, Cls.test_space_dict, Cls.trial_space_dict) + BEAST.td_assemble(ClsI.bilform, ClsI.test_space_dict, ClsI.trial_space_dict))
Zsl = 1/T0 * (BEAST.td_assemble(Csl.bilform, Csl.test_space_dict, Csl.trial_space_dict))

Zqhp = 𝕄ΣH * Zll * PΛH + 𝕄ΣH * Zls * PΣ + 𝕄Λ * Zsl * PΛH + 𝕄Λ * Zll * PΣ
Zqhp = ConvolutionOperators.truncate(Zqhp, ConvolutionOperators.tailindex(Zqhp))



### FORM 4: standard TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
Kyx = BEAST.td_assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)
