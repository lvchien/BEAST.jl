using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("genmesh.jl")

# Physical coefficients
c = 1.0

# Computational mesh
Γ = meshsphere(1.0, 0.2)
# Γ = meshcuboid(0.5, 2.0, 2.0, 0.15)
# fn = joinpath(dirname(pathof(CompScienceMeshes)),"geos/torus.geo")
# h = 0.6
# Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.21)


# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
    
# Time step size and number of time steps
Δt, Nt = 0.1, 1200

### Imaginary wave number iκ
κ = 1/(c*Δt)

# Operators
I = Identity()																			
N = NCross()
Ti = MWSingleLayer3D(κ, -κ, -1/κ)                                               # EFIO with imaginary wave number
T = TDMaxwell3D.singlelayer(speedoflight=c)                                     # TD-EFIO (numdiffs=0)
Ki = MWDoubleLayer3D(κ)                                                         # MFIO with imaginary wave number
K = TDMaxwell3D.doublelayer(speedoflight=c)                                     # TD-MFIO

@hilbertspace k
@hilbertspace j

# Gram matrix
Gxx = assemble(I, X, X)
Nyx = assemble(N, Y, X)
Nxy = -transpose(Nyx)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7) 

𝕋i = assemble(Ti, Y, Y, quadstrat=nearstrat)
𝕂i = assemble(Ki, Y, X, quadstrat=nearstrat)
𝕄i = 0.5 * Matrix(Nyx) - 𝕂i


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
∂E = planewave(polarisation, direction, derive(gaussian), c)
H = direction × E
iH = direction × iE
∂H = direction × ∂E

# Time function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space


BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
Txx = BEAST.td_assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

bilform_2 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
Kyx = BEAST.td_assemble(bilform_2.bilform, bilform_2.test_space_dict, bilform_2.trial_space_dict)

# right-hand side
linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
ex = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

linform_2 = @discretise(-1.0H[k], k∈Y⊗δ)
my = BEAST.td_assemble(linform_2.linform, linform_2.test_space_dict)


### Standard EFIE
Z01 = zeros(Float64, size(Txx)[1:2])
ConvolutionOperators.timeslice!(Z01, Txx, 1)
iZ01 = inv(Z01)
je = marchonintime(iZ01, Txx, ex, Nt)


### Standard MFIE
Z02 = zeros(Float64, size(Kyx)[1:2])
ConvolutionOperators.timeslice!(Z02, Kyx, 1)
iZ02 = inv(Z02)
jm = marchonintime(iZ02, Kyx, my, Nt)


### Mixed TD-CFIE (Beghein et. al. 2013)
cfie = Txx + (-1) * Gxx * iNyx * Kyx

Z03 = zeros(Float64, size(cfie)[1:2])
ConvolutionOperators.timeslice!(Z03, cfie, 1)
iZ03 = inv(Z03)
jc = marchonintime(iZ03, cfie, ex - Gxx * iNyx * my, Nt)


### Yukawa-Calderon TD-CFIE
scfie = 𝕋i * iNxy * Txx + (-1) * 𝕄i * iNyx * Kyx

Z04 = zeros(Float64, size(scfie)[1:2])
ConvolutionOperators.timeslice!(Z04, scfie, 1)
iZ04 = inv(Z04)
jsc = marchonintime(iZ04, scfie, 𝕋i * iNxy * ex - 𝕄i * iNyx * my, Nt)
