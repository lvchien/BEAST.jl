using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators

setminus(A,B) = submesh(!in(B), A)

function q2h(j)
    h = zeros(eltype(j), size(j))
    for i in axes(j, 1)
        h[i, 1] = j[i, 1]
        for k in 2:size(j)[2]
            h[i, k] = 0.5 * (j[i, k] + j[i, k-1])
        end
    end
    return h
end

# Parameters
c0 = 1/√(μ0*ε0)
η0 = √(μ0/ε0)
μ, ε = 1.0, 1.0
c = 1.0
η = 1.0

# Computational mesh
radius, mesh_size = 1.0, 0.3
Γ = CompScienceMeshes.meshsphere2(radius=radius, h=mesh_size)
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
T̂0s = MWSingleLayer3D(0.0, -1.0, 0.0)                                      # static weakly-singular TD-EFIO (numdiffs=0)
T̂0h = MWSingleLayer3D(0.0, 0.0, -1.0)                                          # static hypersingular TD-EFIO	(numdiffs=0)
T = TDMaxwell3D.singlelayer(speedoflight=c)                                  # TD-EFIE
T̂s = integrate(MWSingleLayerTDIO(c, -1/c, 0.0, 1, 0))                        # weakly-singular TD-EFIO (numdiffs=0)
T̂h = MWSingleLayerTDIO(c, 0.0, -c, 0, 0)                                     # hypersingular TD-EFIO (numdiffs=0)
K0 = Maxwell3D.doublelayer(gamma=0.0)                                        # static MFIO
K = TDMaxwell3D.doublelayer(speedoflight=c)                                  # TD-MFIO

@hilbertspace k
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
Gxx = assemble(I, X, X)
iNyx = inv(Matrix(Nyx))
iNxy = transpose(iNyx)

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)

𝕋0s = assemble(T̂0s, Y, Y, quadstrat=nearstrat)
𝕋0h = assemble(T̂0h, Y, Y, quadstrat=nearstrat)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄0 = Matrix(0.5 * Nyx - 𝕂0)
llm_mfie = 𝕄0 * iNyx
llm_efie = Matrix((μ * ℙΣH * 𝕋0s * ℙΣH + ε * ℙΛ * 𝕋0h * ℙΛ) * iNxy)

```
                MAIN PART 
```

Δt, Nt = 0.1, 2000
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
q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
∂h = BEAST.derive(h)							                                    # derivative of h
∂q = BEAST.derive(q)					                                            # first order derivative of q (*Δt)
ip = integrate(p) 	                			                                    # integral of p

### FORM 1: TD-EFIE
BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

lhs_bilform_1 = @discretise η*T[k,j] k∈X⊗δ j∈X⊗∂q
lhs1 = 1/Δt * BEAST.td_assemble(lhs_bilform_1.bilform, lhs_bilform_1.test_space_dict, lhs_bilform_1.trial_space_dict)

rhs_linform_1 = @discretise(-1.0∂E[k], k∈X⊗δ)
rhs1 = BEAST.td_assemble(rhs_linform_1.linform, rhs_linform_1.test_space_dict)

Z01 = zeros(Float64, size(lhs1)[1:2])
ConvolutionOperators.timeslice!(Z01, lhs1, 1)
iZ01 = inv(Z01)
jq1 = marchonintime(iZ01, lhs1, rhs1, Nt)
j1 = q2h(jq1)

# ### FORM 2: CP TD-EFIE (preconditioned by the low-frequency limit of the qHP TD-EFIE operator)
# lhs2 = llm_efie * lhs1
# rhs2 = llm_efie * rhs1

# Z02 = Matrix(llm_efie * Z01)
# iZ02 = inv(Z02)
# j2 = marchonintime(iZ02, lhs2, rhs2, Nt)

### FORM 3: qHP CP TD-EFIE
BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

Mll_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗p
Mls_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂h
Mss_bilform_3 = @discretise T[k, j] k∈X⊗δ j∈X⊗∂q

Mll_3 = BEAST.td_assemble(Mll_bilform_3.bilform, Mll_bilform_3.test_space_dict, Mll_bilform_3.trial_space_dict)
Mls_3 = BEAST.td_assemble(Mls_bilform_3.bilform, Mls_bilform_3.test_space_dict, Mls_bilform_3.trial_space_dict)
Mss_3 = 1/Δt * BEAST.td_assemble(Mss_bilform_3.bilform, Mss_bilform_3.test_space_dict, Mss_bilform_3.trial_space_dict)

lhs3 = η^2 * llm_efie * (PΛH * Mll_3 * PΛH + PΛH * Mls_3 * PΣ + PΣ * Mls_3 * PΛH + PΣ * Mss_3 * PΣ)

el_linform_3 = @discretise(-1.0iE[k], k∈X⊗δ)
es_linform_3 = @discretise(-1.0E[k], k∈X⊗p)

el_3 = BEAST.td_assemble(el_linform_3.linform, el_linform_3.test_space_dict)
es_3 = 1/Δt * BEAST.td_assemble(es_linform_3.linform, es_linform_3.test_space_dict)

rhs3 = llm_efie * (PΛH * el_3 + PΣ * es_3)

Z03 = zeros(Float64, size(lhs3)[1:2])
ConvolutionOperators.timeslice!(Z03, lhs3, 1)
iZ03 = inv(Z03)
y3 = marchonintime(iZ03, lhs3, rhs3, Nt)

j3 = zeros(eltype(y3), size(y3)[1:2])
j3[:, 1] = PΛH * y3[:, 1] + 1.0/Δt * PΣ * y3[:, 1]
for i in 2:Nt
    j3[:, i] = PΛH * y3[:, i] + 1.0/Δt * PΣ * (y3[:, i] - y3[:, i-1])
end

### FORM 4: TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

lhs_bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)

rhs_linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
rhs4 = BEAST.td_assemble(rhs_linform_4.linform, rhs_linform_4.test_space_dict)

Z04 = zeros(Float64, size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
iZ04 = inv(Z04)
j4 = marchonintime(iZ04, lhs4, rhs4, Nt)

### FORM 5: qHP static TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗ip) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

Mll_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗ip
Msl_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗h
Mss_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

Mll_5f = BEAST.td_assemble(Mll_bilform_5.bilform, Mll_bilform_5.test_space_dict, Mll_bilform_5.trial_space_dict)
Msl_5 = BEAST.td_assemble(Msl_bilform_5.bilform, Msl_bilform_5.test_space_dict, Msl_bilform_5.trial_space_dict)
Mss_5 = 1/Δt * BEAST.td_assemble(Mss_bilform_5.bilform, Mss_bilform_5.test_space_dict, Mss_bilform_5.trial_space_dict)

# # Truncate the long tail of the loop-loop component
Mll_5 = ConvolutionOperators.truncate(Mll_5f, ConvolutionOperators.tailindex(Mll_5f))

lhs5 = ℙΛ * llm_mfie * (Mss_5 * PΣ + Msl_5 * PΛH) + ℙΣH * 𝕄0 * PΣ * iNyx * ℙΣH * (Msl_5 * PΣ + Mll_5 * PΛH)
# lhs5 = ℙΛ * llm_mfie * (Mss_5 * PΣ + Msl_5 * PΛH) + ℙΣH * llm_mfie * Msl_5 * PΣ

el_linform_5 = @discretise(-1.0iH[k], k∈Y⊗δ)
es_linform_5 = @discretise(-1.0H[k], k∈Y⊗p)

el_5 = BEAST.td_assemble(el_linform_5.linform, el_linform_5.test_space_dict)
es_5 = 1/Δt * BEAST.td_assemble(es_linform_5.linform, es_linform_5.test_space_dict)

rhs5 = ℙΣH * 𝕄0 * PΣ * iNyx * ℙΣH * el_5 + ℙΛ * llm_mfie * es_5

Z05 = zeros(Float64, size(lhs5)[1:2])
ConvolutionOperators.timeslice!(Z05, lhs5, 1)
iZ05 = inv(Z05)
y5 = marchonintime(iZ05, lhs5, rhs5, Nt)

j5 = zeros(eltype(y5), size(y5)[1:2])
j5[:, 1] = PΛH * y5[:, 1] + 1.0/Δt * PΣ * y5[:, 1]
for i in 2:Nt
    j5[:, i] = PΛH * y5[:, i] + 1.0/Δt * PΣ * (y5[:, i] - y5[:, i-1])
end

#=
    FORM 6: TD-CFIE (Beghein et. al., 2013)
=#
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

mfio_bilform_6 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗∂q
mfio6 = -1/Δt * Gxx * iNyx * BEAST.td_assemble(mfio_bilform_6.bilform, mfio_bilform_6.test_space_dict, mfio_bilform_6.trial_space_dict) 
lhs6 = lhs1 + η * mfio6

mlinform_6 = @discretise(-1.0∂H[k], k∈Y⊗δ)
rhs6 = rhs1 - η * Gxx * iNyx * BEAST.td_assemble(mlinform_6.linform, mlinform_6.test_space_dict)

Z06 = zeros(Float64, size(lhs6)[1:2])
ConvolutionOperators.timeslice!(Z06, lhs6, 1)
iZ06 = inv(Z06)
jq6 = marchonintime(iZ06, lhs6, rhs6, Nt)
j6 = q2h(jq6)

#=
    FORM 7: qHP localized CP TD-CFIE
=#
lhs7 = lhs3 + η^2 * lhs5
rhs7 = rhs3 + η^2 * rhs5

Z07 = Z03 + η^2 * Z05
iZ07 = inv(Z07)
y7 = marchonintime(iZ07, lhs7, rhs7, Nt)

j7 = zeros(eltype(y7), size(y7)[1:2])
j7[:, 1] = PΛH * y7[:, 1] + 1.0/Δt * PΣ * y7[:, 1]
for i in 2:Nt
    j7[:, i] = PΛH * y7[:, i] + 1.0/Δt * PΣ * (y7[:, i] - y7[:, i-1])
end

using Printf
open("current-sphere-h_0.3-tau_0.1-width_8.txt", "a") do io
    for i in 1:Nt
        @printf(io, "%.10f %.10f %.10f %.10f %.10f %.10f\n", i*Δt, log10.(abs.(j1[1, i])), log10.(abs.(j3[1, i])), log10.(abs.(j4[1, i])), log10.(abs.(j6[1, i])), log10.(abs.(j7[1, i])))
    end
end;

## Plot results
using Plots
plotly()
plt = Plots.plot(
    width = 600, height=400,
    grid = false,
    xscale = :identity, 
    yaxis = :log10, 
    xlims = (0, 204),
    xticks = [0; 50; 100; 150; 200],
    # xtickfont = font(9, "Times"),
    ylims = (1e-16, 2), 
    yticks = [1e-15; 1e-10; 1e-5; 1e0;],
    # ytickfont = font(9),
    xlabel = "c t (m)",
    ylabel = "j(t) (A/m)")

x = Δt * [1:1:Nt;]
plot!(x, abs.(j1[1, :]), label="TD-EFIE")
plot!(x, abs.(j3[1,:]), label="CP qHP TD-EFIE")
plot!(x, abs.(j4[1,:]), label="TD-MFIE")
plot!(x, abs.(j5[1,:]), label="qHP TD-MFIE")
plot!(x, abs.(j6[1,:]), label=" standard TD-CFIE")
plot!(x, abs.(j7[1,:]), label="qHP TD-CFIE")

savefig("symmetrized_CFIE.pdf")