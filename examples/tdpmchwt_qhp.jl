using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
ϵ, μ = 1.0, 1.0
ϵ′, μ′ = 2.0, 1.0
η, η′ = √(μ/ϵ), √(μ′/ϵ′)
sol = 1/√(ϵ*μ)
sol′ = 1/√(ϵ′*μ′)

# Diameter of the scatterer
T0 = 2.0                                                                         # D/c with D the diameter of scatterer [second]

# for meshsize ∈ [0.7, 0.0675, 0.06]
meshsize = 0.15
# Computational mesh
# Γ = meshsphere(1.0, 0.3)
# fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/torus.geo")
# Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.6)
Γ = meshtorus(0.75, 0.25, meshsize)
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

# Operators
I = Identity()																			
N = NCross()
T = TDMaxwell3D.singlelayer(speedoflight=sol)                                    # TD-EFIO (numdiff=0)
Ts = MWSingleLayerTDIO(sol, -1/sol, 0.0, 1, 0)                                   # weakly-singular TD-EFIO (numdiffs=0)
iTs = integrate(Ts)                                                              # weakly-singular TD-EFIO (numdiffs=1)
∂T = derive(T)
K = TDMaxwell3D.doublelayer(speedoflight=sol)                                    # TD-MFIO
∂K = TDMaxwell3D.doublelayer(speedoflight=sol, numdiffs=1)
iK = integrate(K)

T′ = TDMaxwell3D.singlelayer(speedoflight=sol′)
Ts′ = MWSingleLayerTDIO(sol′, -1/sol′, 0.0, 1, 0)                               
iTs′ = integrate(Ts′)                              
∂T′ = derive(T′)             
K′ = TDMaxwell3D.doublelayer(speedoflight=sol′)
∂K′ = TDMaxwell3D.doublelayer(speedoflight=sol′, numdiffs=1)
iK′ = integrate(K′)

T0s = MWSingleLayer3D(0.0, 1, 0)
T0h = MWSingleLayer3D(0.0, 0, 1)

@hilbertspace k l
@hilbertspace j m

# Gram matrix
Nxy = assemble(N, X, Y)
iNxy = inv(Matrix(Nxy))
ℤ0s = assemble(T0s, Y, Y)
ℤ0h = assemble(T0h, Y, Y)

Zr = zeros(size(ℤ0s))

𝕄l = [ℙΣH * ℤ0s * ℙΣH * iNxy Zr; Zr ℙΣH * ℤ0s * ℙΣH * iNxy]
𝕄s = [ℙΛ * ℤ0h * ℙΛ * iNxy * PΛH Zr; Zr ℙΛ * ℤ0h * ℙΛ * iNxy * PΛH]
Ml = [PΛH Zr; Zr PΛH]
Ms = [PΣ Zr; Zr PΣ]

```
                MAIN PART 
```

Nt = 5
for Δt in 0.36*[1, 2, 4, 8, 16, 32, 64]
# Plane wave incident fields
duration = 120 * Δt * sol                                       
delay = 240 * Δt                                        
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
fgaussian = fouriertransform(gaussian)
polarisation, direction = x̂, ẑ
E = planewave(polarisation, direction, gaussian, sol)
iE = planewave(polarisation, direction, integrate(gaussian), sol)
H = direction × E
iH = direction × iE

# Temporal function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space


### FORM 1: standard TD-PMCHWT with 13 quadrature points
# BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (T′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (K′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

pmchwt13 = @discretise(η*T[k,j] + η′*T′[k,j] + K[l,j] + K′[l,j]
                    - K[k,m] - K′[k,m] + 1/η*T[l,m] + 1/η′*T′[l,m],
                    k∈X⊗δ, l∈X⊗δ, j∈X⊗h, m∈X⊗h)

Zxx1 = BEAST.assemble(pmchwt13.bilform, pmchwt13.test_space_dict, pmchwt13.trial_space_dict)

# linform_1 = @discretise(E[k] + H[l], k∈X⊗δ, l∈X⊗δ)
# rhs = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Zxx1)[1:2])
ConvolutionOperators.timeslice!(Z01, Zxx1, 1)
# iZ01 = inv(Z01)
# jpmchwt13 = marchonintime(iZ01, Zxx1, rhs, Nt)


# ### FORM 2: standard TD-PMCHWT with 78 quadrature points
# BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)
# BEAST.@defaultquadstrat (T′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)
# BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)
# BEAST.@defaultquadstrat (K′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

# pmchwt78 = @discretise(η*T[k,j] + η′*T′[k,j] + K[l,j] + K′[l,j]
#                     - K[k,m] - K′[k,m] + 1/η*T[l,m] + 1/η′*T′[l,m],
#                     k∈X⊗δ, l∈X⊗δ, j∈X⊗h, m∈X⊗h)

# Zxx2 = BEAST.assemble(pmchwt78.bilform, pmchwt78.test_space_dict, pmchwt78.trial_space_dict)

# Z02 = zeros(Float64, size(Zxx2)[1:2])
# ConvolutionOperators.timeslice!(Z02, Zxx2, 1)
# iZ02 = inv(Z02)
# jpmchwt78 = marchonintime(iZ02, Zxx2, rhs, Nt)


# ### FORM 3: standard TD-PMCHWT with 400 quadrature points
# BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(12)
# BEAST.@defaultquadstrat (T′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(12)
# BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(12)
# BEAST.@defaultquadstrat (K′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(12)

# pmchwt400 = @discretise(η*T[k,j] + η′*T′[k,j] + K[l,j] + K′[l,j]
#                     - K[k,m] - K′[k,m] + 1/η*T[l,m] + 1/η′*T′[l,m],
#                     k∈X⊗δ, l∈X⊗δ, j∈X⊗h, m∈X⊗h)

# Zxx3 = BEAST.assemble(pmchwt400.bilform, pmchwt400.test_space_dict, pmchwt400.trial_space_dict)

# Z03 = zeros(Float64, size(Zxx3)[1:2])
# ConvolutionOperators.timeslice!(Z03, Zxx3, 1)
# iZ03 = inv(Z03)
# jpmchwt400 = marchonintime(iZ03, Zxx3, rhs, Nt)


### FORM 4: qHP TD-PMCHWT with 13 quadrature points

# BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (Ts′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (K′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (∂T, X⊗p, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (∂T′, X⊗p, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (∂K, X⊗p, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (∂K′, X⊗p, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (iTs, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (iTs′, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (iK, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
# BEAST.@defaultquadstrat (iK′, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)

qhp_ll = @discretise(η*Ts[k,j] + η′*Ts′[k,j] - K[k,m] - K′[k,m] 
                    + K[l,j] + K′[l,j] + 1/η*Ts[l,m] + 1/η′*Ts′[l,m],
                    k∈X⊗δ, l∈X⊗δ, j∈X⊗h, m∈X⊗h)

qhp_ls = @discretise(η*∂T[k,j] + η′*∂T′[k,j] - ∂K[k,m] - ∂K′[k,m] 
                    + ∂K[l,j] + ∂K′[l,j] + 1/η*∂T[l,m] + 1/η′*∂T′[l,m],
                    k∈X⊗p, l∈X⊗p, j∈X⊗h, m∈X⊗h)

if Δt < T0
    qhp_sl = @discretise(η*iTs[k,j] + η′*iTs′[k,j] - iK[k,m] - iK′[k,m]
                    + iK[l,j] + iK′[l,j] + 1/η*iTs[l,m] + 1/η′*iTs′[l,m],
                    k∈X⊗δ, l∈X⊗δ, j∈X⊗p, m∈X⊗p)
else
    qhp_sl = @discretise(η*iTs[k,j] + η′*iTs′[k,j] + 1/η*iTs[l,m] + 1/η′*iTs′[l,m],
                    k∈X⊗δ, l∈X⊗δ, j∈X⊗p, m∈X⊗p)
end
                    
Zll = BEAST.assemble(qhp_ll.bilform, qhp_ll.test_space_dict, qhp_ll.trial_space_dict)
Zls = T0/Δt * BEAST.assemble(qhp_ls.bilform, qhp_ls.test_space_dict, qhp_ls.trial_space_dict)
Zsl = 1/T0 * BEAST.assemble(qhp_sl.bilform, qhp_sl.test_space_dict, qhp_sl.trial_space_dict)

ZqHP =  𝕄l * Zll * Ml + 𝕄s * Zll * Ms + 𝕄l * Zls * Ms + 𝕄s * Zsl * Ml
# ZqHP = ConvolutionOperators.truncate(ZqHP, ConvolutionOperators.tailindex(ZqHP))

# linform_2s = @discretise(iE[k] + iH[l], k∈X⊗δ, l∈X⊗δ)
# linform_2l = @discretise(E[k] + H[l], k∈X⊗p, l∈X⊗p)

# rhs2s = 1/T0 * BEAST.td_assemble(linform_2s.linform, linform_2s.test_space_dict)
# rhs2l = 1/Δt * BEAST.td_assemble(linform_2l.linform, linform_2l.test_space_dict)

# rhs_qhp = 𝕄l * rhs2l + 𝕄s * rhs2s

Z04 = zeros(Float64, size(ZqHP)[1:2])
ConvolutionOperators.timeslice!(Z04, ZqHP, 1)
# iZ04 = inv(Z04)
# y = marchonintime(iZ04, ZqHP, rhs_qhp, Nt)

# jqhp = zeros(eltype(y), size(y)[1:2])
# jqhp[:, 1] = Ml * y[:, 1] + T0/Δt * Ms * y[:, 1]
# for i in 2:Nt
#     jqhp[:, i] = Ml * y[:, i] + T0/Δt * Ms * (y[:, i] - y[:, i-1])
# end

# for i ∈ 1:Nt
#     open("torus_current_h_0.15_cdt_1.5.txt", "a") do io
#         @printf(io, "%.4f %.20f %.20f %.20f %.20f\n", i*Δt, abs(jpmchwt13[1, i]), abs(jpmchwt78[1, i]), abs(jpmchwt400[1, i]), abs(jqhp[1, i]))
#     end;
# end 

open("torus_cond_h_0.15m.txt", "a") do io
    @printf(io, "%.4f %.4f %.4f \n", Δt, cond(Z01), cond(Z04))
end;
end
