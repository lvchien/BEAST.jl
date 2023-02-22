using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
include("genmesh.jl")

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
end;

# Physical coefficients
μ, ε = 1.0, 1.0
c = 1.0
η = 1.0

# Computational mesh
# radius, mesh_size = 1.0, 0.55
# innerradius, outerradius, mesh_size = 0.5, 1.0, 0.45
# Γ = meshtorus(innerradius, outerradius, mesh_size)
Γ = meshsphere2(1.0, 0.3)
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
T̂0s = MWSingleLayer3D(0.0, -1.0, 0.0)                                        # static weakly-singular TD-EFIO (numdiffs=0)
T̂0h = MWSingleLayer3D(0.0, 0.0, -1.0)                                        # static hypersingular TD-EFIO (numdiffs=0)
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
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

𝕋0s = assemble(T̂0s, Y, Y, quadstrat=nearstrat)
𝕋0h = assemble(T̂0h, Y, Y, quadstrat=nearstrat)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄0 = Matrix(0.5 * Nyx - 𝕂0)

# llm_efie = Matrix((μ * ℙΣH * 𝕋0s * ℙΣH + 0.5 * ε * ℙΛ * 𝕋0h * ℙΛ) * (ℙΛ * iNxy * PΛH + ℙΣH * iNxy * PΣ + 0.5 * ℙΣH * iNxy * PΛH))

# ```
#                 MAIN PART 
# ```

Δt, Nt = 0.1, 1000

# Plane wave
duration = 80 * Δt * c                                        
delay = 120 * Δt                                        
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
∂2q = BEAST.derive(∂q)					                                            # second order derivative of q (*Δt)
ip = integrate(p) 	                			                                    # integral of p
ih = integrate(h) 	                			                                    # integral of h

### standard TD-EFIE
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

lhs_bilform_1 = @discretise η*T[k,j] k∈X⊗δ j∈X⊗h
lhs1 = BEAST.td_assemble(lhs_bilform_1.bilform, lhs_bilform_1.test_space_dict, lhs_bilform_1.trial_space_dict)

rhs_linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
rhs1 = BEAST.td_assemble(rhs_linform_1.linform, rhs_linform_1.test_space_dict)

Z01 = zeros(Float64, size(lhs1)[1:2])
ConvolutionOperators.timeslice!(Z01, lhs1, 1)
iZ01 = inv(Z01)
j1 = marchonintime(iZ01, lhs1, rhs1, Nt)

####

BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

lhs_bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)

rhs_linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
rhs4 = BEAST.td_assemble(rhs_linform_4.linform, rhs_linform_4.test_space_dict)

Z04 = zeros(Float64, size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
iZ04 = inv(Z04)
j4 = marchonintime(iZ04, lhs4, rhs4, Nt)

BEAST.@defaultquadstrat (K, Y⊗δ, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

Msl_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗p
Mss_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂h
Mls_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

Msl_5 = BEAST.td_assemble(Msl_bilform_5.bilform, Msl_bilform_5.test_space_dict, Msl_bilform_5.trial_space_dict)
Mss_5 = BEAST.td_assemble(Mss_bilform_5.bilform, Mss_bilform_5.test_space_dict, Mss_bilform_5.trial_space_dict)
Mls_5 = 1/Δt * BEAST.td_assemble(Mls_bilform_5.bilform, Mls_bilform_5.test_space_dict, Mls_bilform_5.trial_space_dict)

# inverse of time-domain Gram matrix
# iGmx = PΛH * iNyx * ℙΛ + PΣ * iNyx * ℙΣH + 2 * PΛH * iNyx * ℙΣH
lhs_sl = ℙΛ * lhs4 * PΛH
lhs_dg = ℙΛ * Mls_5 * PΣ + ℙΣH * Mss_5 * PΣ
lhs_ll = ℙΣH * Msl_5 * PΛH

lhs5 = lhs_sl + lhs_dg + lhs_ll

el_linform_5 = @discretise(-1.0H[k], k∈Y⊗p)
es_linform_5 = @discretise(-1.0H[k], k∈Y⊗δ)

el_5 = 1/Δt * BEAST.td_assemble(el_linform_5.linform, el_linform_5.test_space_dict)
es_5 = BEAST.td_assemble(es_linform_5.linform, es_linform_5.test_space_dict)

rhs5 = ℙΛ * el_5 + ℙΣH * es_5

Z05 = zeros(Float64, size(lhs5)[1:2])
ConvolutionOperators.timeslice!(Z05, lhs5, 1)
iZ05 = inv(Z05)
y5 = marchonintime(iZ05, lhs5, rhs5, Nt)

j5 = zeros(eltype(y5), size(y5)[1:2])
j5[:, 1] = PΛH * y5[:, 1] + 1.0/Δt * PΣ * y5[:, 1]
for i in 2:Nt
    j5[:, i] = PΛH * y5[:, i] + 1.0/Δt * PΣ * (y5[:, i] - y5[:, i-1])
end

### TD-EFIE
BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(9)

bf_1 = @discretise η*T̂s[k,j] k∈X⊗δ j∈X⊗∂h
bf_2 = @discretise η*T[k,j] k∈X⊗δ j∈X⊗∂q

lhs1 = BEAST.td_assemble(bf_1.bilform, bf_1.test_space_dict, bf_1.trial_space_dict)
lhs2 = 1/Δt * BEAST.td_assemble(bf_2.bilform, bf_2.test_space_dict, bf_2.trial_space_dict)

lhs_efie = Δt * ℙΣH * 𝕋0s * ℙΣH * iNxy * lhs1 * PΛH + Δt * ℙΣH * 𝕋0s * ℙΣH * iNxy * lhs2 * PΣ + ℙΛ * 𝕋0h * ℙΛ * iNxy * lhs1 * PΛH + ℙΛ * 𝕋0h * ℙΛ * iNxy * lhs2 * PΣ

lf_1 = @discretise(-1.0E[k], k∈X⊗p)

rhs_1 = 1/Δt * BEAST.td_assemble(lf_1.linform, lf_1.test_space_dict)

rhs_efie = Δt * ℙΣH * 𝕋0s * ℙΣH * iNxy * rhs_1 + ℙΛ * 𝕋0h * ℙΛ * iNxy * rhs_1

Z0 = zeros(Float64, size(lhs_efie)[1:2])
ConvolutionOperators.timeslice!(Z0, lhs_efie, 1)
iZ0 = inv(Z0)
y = marchonintime(iZ0, lhs_efie, rhs_efie, Nt)

jefie = zeros(eltype(y), size(y)[1:2])
jefie[:, 1] = PΛH * y[:, 1] + 1.0/Δt * PΣ * y[:, 1]
for i in 2:Nt
    jefie[:, i] = PΛH * y[:, i] + 1.0/Δt * PΣ * (y[:, i] - y[:, i-1])
end

### TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(9)

bf_5 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
bf_6 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗∂q

lhs5 = BEAST.td_assemble(bf_5.bilform, bf_5.test_space_dict, bf_5.trial_space_dict)
lhs6 = 1/Δt * BEAST.td_assemble(bf_6.bilform, bf_6.test_space_dict, bf_6.trial_space_dict)

lhs_mfie = ℙΛ * 𝕄0 * iNyx * lhs5 * PΛH + ℙΛ * 𝕄0 * iNyx * lhs6 * PΣ + Δt * ℙΣH * 𝕄0 * iNyx * lhs5 * PΛH + Δt * ℙΣH * 𝕄0 * iNyx * lhs6 * PΣ

lf_3 = @discretise(-1.0H[k], k∈Y⊗p)

rhs3 = 1/Δt * BEAST.td_assemble(lf_3.linform, lf_3.test_space_dict)

rhs_mfie = ℙΛ * 𝕄0 * iNyx * rhs3 + Δt * ℙΣH * 𝕄0 * iNyx * rhs3 

Z0 = zeros(Float64, size(lhs_mfie)[1:2])
ConvolutionOperators.timeslice!(Z0, lhs_mfie, 1)
iZ0 = inv(Z0)
y = marchonintime(iZ0, lhs_mfie, rhs_mfie, Nt)

jmfie = zeros(eltype(y), size(y)[1:2])
jmfie[:, 1] = PΛH * y[:, 1] + 1.0/Δt * PΣ * y[:, 1]
for i in 2:Nt
    jmfie[:, i] = PΛH * y[:, i] + 1.0/Δt * PΣ * (y[:, i] - y[:, i-1])
end

lhs = lhs_efie + η^2 * lhs_mfie
rhs = rhs_efie + η^2 * rhs_mfie

Z0 = zeros(Float64, size(lhs)[1:2])
ConvolutionOperators.timeslice!(Z0, lhs, 1)
iZ0 = inv(Z0)
y = marchonintime(iZ0, lhs, rhs, Nt)

jcfie = zeros(eltype(y), size(y)[1:2])
jcfie[:, 1] = PΛH * y[:, 1] + 1.0/Δt * PΣ * y[:, 1]
for i in 2:Nt
    jcfie[:, i] = PΛH * y[:, i] + 1.0/Δt * PΣ * (y[:, i] - y[:, i-1])
end

using Plots
plotly()
plt = Plots.plot(
    width = 600, height=400,
    grid = false,
    xscale = :identity, 
    yaxis = :log10, 
    xlims = (0, 104),
    xticks = [0; 50; 100],
    xtickfont = font(9, "Times"),
    ylims = (1e-22, 2), 
    yticks = [ 1e-15; 1e-10; 1e-5; 1e0;],
    ytickfont = font(9),
    xlabel = "c t (m)",
    ylabel = "j(t) (A/m)")

x = Δt * [1:1:Nt;]

Plots.plot!(x, abs.(jefie[1, :]), label="qHP TD-EFIE")
Plots.plot!(x, abs.(jmfie[1, :]), label="qHP TD-MFIE")
Plots.plot!(x, abs.(jcfie[1, :]), label="qHP TD-CFIE")
Plots.plot!(x, abs.(j1[1, :]), label="standard TD-EFIE")
Plots.plot!(x, abs.(j4[1, :]), label="standard TD-MFIE")
Plots.plot!(x, abs.(j5[1, :]), label="modified TD-MFIE")

savefig("qHP_TD-CFIE_current.png")

w = ConvolutionOperators.polyvals(lhs)
using Plots
# plotly()
plot(exp.(im*range(0,2pi,length=1000)))
scatter!(w)
savefig("qHP_TD-CFIE_torus_polyvals.pdf")