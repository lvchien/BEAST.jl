using BEAST, LinearAlgebra, ConvolutionOperators, Printf
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
ϵ, μ = 1.0, 1.0
ϵ′, μ′ = 1.0, 1.0
η, η′ = √(μ/ϵ), √(μ′/ϵ′)
sol = 1/√(ϵ*μ)
sol′ = 1/√(ϵ′*μ′)

# Operators
I = BEAST.Identity()																			
N = NCross()
T = BEAST.LaplaceDomainOperator(s::ComplexF64 -> MWSingleLayer3D(s/sol, -s^2/sol, ComplexF64(sol)))
K = BEAST.LaplaceDomainOperator(s::ComplexF64 -> MWDoubleLayer3D(s))

T1 = TDMaxwell3D.singlelayer(speedoflight=sol, numdiffs=1)                                    # TD-EFIO (numdiff=0)
T2 = TDMaxwell3D.singlelayer(speedoflight=sol, numdiffs=1)

Γ = meshsphere(1.0, 0.5)

# RWG and BC spatial function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

@hilbertspace k
@hilbertspace j

# Gram matrix
Nxy = assemble(N, X, Y)
iNxy = inv(Matrix(Nxy))
iNyx = transpose(iNxy)

```
                MAIN PART 
```

Nt, Δt = 500, 3.0

δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = timebasisshiftedlagrange(Δt, Nt, 3)

(A, b, c) = butcher_tableau_radau_3stages()
CQ = StagedTimeStep(Δt, Nt, c, A, b, 50, 1.0001)

# Plane wave incident fields
duration = 120 * Δt * sol                                       
delay = 240 * Δt                                        
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
fgaussian = fouriertransform(gaussian)
polarisation, direction = x̂, ẑ
E = planewave(polarisation, direction, gaussian, sol)
∂E = planewave(polarisation, direction, derive(gaussian), sol)
iE = planewave(polarisation, direction, integrate(gaussian), sol)
H = direction × E
∂H = direction × ∂E
iH = direction × iE


### FORM 1: standard TD-EFIE
BEAST.@defaultquadstrat (T1, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_1 = @discretise T1[k,j] k∈X⊗δ j∈X⊗q
Txx = BEAST.assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

linform_1 = @discretise(-1.0∂E[k], k∈X⊗δ)
ex = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Txx)[1:2])
ConvolutionOperators.timeslice!(Z01, Txx, 1)
iZ01 = inv(Z01)
je = marchonintime(iZ01, Txx, ex, Nt)


### FORM 2: CQ TD-EFIE with 13 quadrature points
BEAST.@defaultquadstrat (T, X⊗CQ, X⊗CQ) BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)

efie13 = @discretise T[k,j] k∈X⊗CQ j∈X⊗CQ
Zxx1 = BEAST.assemble(efie13.bilform, efie13.test_space_dict, efie13.trial_space_dict)

linform_2 = @discretise(-1.0∂E[k], k∈X⊗CQ)
rhs = BEAST.td_assemble(linform_2.linform, linform_2.test_space_dict)

Z02 = zeros(Float64, size(Zxx1)[1:2])
ConvolutionOperators.timeslice!(Z02, Zxx1, 1)
iZ02 = inv(Z02)
jpmchwt13 = marchonintime(iZ02, Zxx1, rhs, Nt)


### FORM 3: CQ TD-EFIE 
tdefie_irk = @discretise T2[k,j] == -1.0∂E[k]  k∈X⊗CQ j∈X⊗CQ
xefie_irk = solve(tdefie_irk)
