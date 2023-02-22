using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf
include("genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
μ, ε = 1.0, 1.0
c = 1.0
η = 1.0
# Computational mesh
mesh_size = 0.3
# 0.55, 0.45, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09
Γ = meshsphere2(radius=1.0, h=mesh_size)
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
    
# Operators
I = Identity()																			
N = NCross()
T̂0s = MWSingleLayer3D(0.0, -1.0/c, 0.0)                                      # static weakly-singular TD-EFIO (numdiffs=0)
T̂0h = MWSingleLayer3D(0.0, 0.0, -c)                                          # static hypersingular TD-EFIO	(numdiffs=0)
T = TDMaxwell3D.singlelayer(speedoflight=c)                                  # TD-EFIE
T̂s = integrate(MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0))             # weakly-singular TD-EFIO (numdiffs=0)
T̂h = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                          # hypersingular TD-EFIO (numdiffs=0)
K0 = Maxwell3D.doublelayer(gamma=0.0)                                                   # static MFIO
K = TDMaxwell3D.doublelayer(speedoflight=c)                                  # TD-MFIO

@hilbertspace k 
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
Gxx = assemble(I, X, X)
iNyx = inv(Matrix(Nyx))
iNxy = transpose(iNyx)

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

𝕋0s = assemble(T̂0s, Y, Y, quadstrat=nearstrat)
𝕋0h = assemble(T̂0h, Y, Y, quadstrat=nearstrat)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄0 = Matrix(0.5 * Nyx - 𝕂0)

llm_efie = Matrix((μ * ℙΣH * 𝕋0s * ℙΣH + ε * ℙΛ * 𝕋0h * ℙΛ) * (ℙΛ * iNxy * PΛH + ℙΣH * iNxy * PΣ + 0.5 * ℙΣH * iNxy * PΛH))

#=
                MAIN PART 
=#

#= 
    FORM 1: standard TD-EFIE
=#

Nt = 20
for Δt in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    # Time function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
    ∂h = BEAST.derive(h)							                                    # derivative of h
    ∂q = BEAST.derive(q)					                                            # first order derivative of q (*Δt)
    ip = integrate(p) 	                			                                    # integral of p

    ### standard TD-EFIE
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

    lhs_bilform_1 = @discretise η*T[k,j] k∈X⊗δ j∈X⊗h
    lhs1 = BEAST.td_assemble(lhs_bilform_1.bilform, lhs_bilform_1.test_space_dict, lhs_bilform_1.trial_space_dict)

    Z01 = zeros(Float64, size(lhs1)[1:2])
    ConvolutionOperators.timeslice!(Z01, lhs1, 1)

    #=
        FORM 3: qHP CP TD-EFIE
    =#

    BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(9)
    BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(9)
    BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(9)

    Mll_bilform_3 = @discretise η*T̂s[k, j] k∈X⊗δ j∈X⊗p
    Mls_bilform_3 = @discretise η*T̂s[k, j] k∈X⊗δ j∈X⊗∂h
    Mss_bilform_3 = @discretise η*T[k, j] k∈X⊗δ j∈X⊗∂q

    Mll_3 = BEAST.td_assemble(Mll_bilform_3.bilform, Mll_bilform_3.test_space_dict, Mll_bilform_3.trial_space_dict)
    Mls_3 = BEAST.td_assemble(Mls_bilform_3.bilform, Mls_bilform_3.test_space_dict, Mls_bilform_3.trial_space_dict)
    Mss_3 = 1/Δt * BEAST.td_assemble(Mss_bilform_3.bilform, Mss_bilform_3.test_space_dict, Mss_bilform_3.trial_space_dict)

    lhs3 = η * llm_efie * (PΛH * Mll_3 * PΛH + PΛH * Mls_3 * PΣ + PΣ * Mls_3 * PΛH + PΣ * Mss_3 * PΣ)

    Z03 = zeros(Float64, size(lhs3)[1:2])
    ConvolutionOperators.timeslice!(Z03, lhs3, 1)

    #=
        FORM 4: standard TD-MFIE
    =#

    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

    lhs_bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
    lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)

    Z04 = zeros(Float64, size(lhs4)[1:2])
    ConvolutionOperators.timeslice!(Z04, lhs4, 1)

    #=
        FORM 5: qHP symmetrized TD-MFIE
    =#

    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(9)
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(9)
    BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(9)

    Msl_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗p
    Mss_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂h
    Mls_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

    Msl_5 = BEAST.td_assemble(Msl_bilform_5.bilform, Msl_bilform_5.test_space_dict, Msl_bilform_5.trial_space_dict)
    Mss_5 = BEAST.td_assemble(Mss_bilform_5.bilform, Mss_bilform_5.test_space_dict, Mss_bilform_5.trial_space_dict)
    Mls_5 = 1/Δt * BEAST.td_assemble(Mls_bilform_5.bilform, Mls_bilform_5.test_space_dict, Mls_bilform_5.trial_space_dict)

    # inverse of time-domain Gram matrix
    iGmx = PΛH * iNyx * ℙΛ + PΣ * iNyx * ℙΣH + 2 * PΛH * iNyx * ℙΣH
    lhs_sl = ℙΛ * Msl_5 * PΛH
    lhs_dg = ℙΛ * Mss_5 * PΣ + ℙΣH * Mls_5 * PΣ
    lhs_ll = ℙΣH * lhs4 * PΛH

    lhs5 = ℙΛ * 𝕄0 * iGmx * lhs_sl + ℙΛ * 𝕄0 * iGmx * lhs_dg +  ℙΛ * 𝕄0 * iGmx * lhs_ll + Δt * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_sl + Δt * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_dg + Δt * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_ll

    Z05 = zeros(Float64, size(lhs5)[1:2])
    ConvolutionOperators.timeslice!(Z05, lhs5, 1)
 
    #=
        FORM 6: standard TD-CFIE (Beghein et. al., 2013)
    =#
    lhs6 = lhs1 + (-η) * Gxx * iNyx * lhs4

    Z06 = zeros(Float64, size(lhs6)[1:2])
    ConvolutionOperators.timeslice!(Z06, lhs6, 1)

    #=
    FORM 7: qHP localized CP TD-CFIE
    =#
    Z07 = Z03 + η^2 * Z05

    # Save condition numbers to file
    open("cond-sphere-h_0.3m.txt", "a") do io
        @printf(io, "%.3f %.10f %.10f %.10f %.10f %.10f\n", Δt, cond(Z01), cond(Z03), cond(Z04), cond(Z06), cond(Z07))
    end;
end