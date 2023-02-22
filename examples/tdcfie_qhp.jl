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
end

# Physical coefficients
μ, ε = 1.0, 1.0
c = 1.0
η = 1.0

# Computational mesh
radius, mesh_size = 1.0, 0.55
Γ = meshsphere2(radius=radius, h=mesh_size)
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
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

𝕋0s = assemble(T̂0s, Y, Y, quadstrat=nearstrat)
𝕋0h = assemble(T̂0h, Y, Y, quadstrat=nearstrat)
𝕂0 = assemble(K0, Y, X, quadstrat=nearstrat)
𝕄0 = Matrix(0.5 * Nyx - 𝕂0)

llm_efie = Matrix((μ * ℙΣH * 𝕋0s * ℙΣH + ε * ℙΛ * 𝕋0h * ℙΛ) * (ℙΛ * iNxy * PΛH + ℙΣH * iNxy * PΣ + 0.5 * ℙΣH * iNxy * PΛH))

```
                MAIN PART 
```

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
ip = integrate(p) 	                			                                    # integral of p

### FORM 1: standard TD-EFIE
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

lhs_bilform_1 = @discretise η*T[k,j] k∈X⊗δ j∈X⊗h
lhs1 = BEAST.td_assemble(lhs_bilform_1.bilform, lhs_bilform_1.test_space_dict, lhs_bilform_1.trial_space_dict)

rhs_linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
rhs1 = BEAST.td_assemble(rhs_linform_1.linform, rhs_linform_1.test_space_dict)

Z01 = zeros(Float64, size(lhs1)[1:2])
ConvolutionOperators.timeslice!(Z01, lhs1, 1)
iZ01 = inv(Z01)
j1 = marchonintime(iZ01, lhs1, rhs1, Nt)

# ### FORM 2: CP TD-EFIE (preconditioned by the low-frequency limit of the qHP TD-EFIE operator)
# lhs2 = llm_efie * lhs1
# rhs2 = llm_efie * rhs1

# Z02 = Matrix(llm_efie * Z01)
# iZ02 = inv(Z02)
# j2 = marchonintime(iZ02, lhs2, rhs2, Nt)

### FORM 3: qHP CP TD-EFIE
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

### FORM 4: standard TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)

lhs_bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)

rhs_linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
rhs4 = BEAST.td_assemble(rhs_linform_4.linform, rhs_linform_4.test_space_dict)

Z04 = zeros(Float64, size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
iZ04 = inv(Z04)
j4 = marchonintime(iZ04, lhs4, rhs4, Nt)

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

lhs5 = ℙΛ * 𝕄0 * iGmx * lhs_sl + ℙΛ * 𝕄0 * iGmx * lhs_dg +  ℙΛ * 𝕄0 * iGmx * lhs_ll + Δt/2 * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_sl + Δt/2 * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_dg + Δt/2 * ℙΣH * 𝕄0 * PΣ * iGmx * lhs_ll

el_linform_5 = @discretise(-1.0H[k], k∈Y⊗p)
es_linform_5 = @discretise(-1.0H[k], k∈Y⊗δ)

el_5 = 1/Δt * BEAST.td_assemble(el_linform_5.linform, el_linform_5.test_space_dict)
es_5 = BEAST.td_assemble(es_linform_5.linform, es_linform_5.test_space_dict)

rhs5 = ℙΛ * 𝕄0 * PΛH * iGmx * ℙΛ * es_5 + ℙΛ * 𝕄0 * PΛH * iGmx * ℙΣH * el_5 + Δt * ℙΣH * 𝕄0 * PΣ * iGmx * ℙΛ * es_5 + Δt * ℙΣH * 𝕄0 * PΣ * iGmx * ℙΣH * el_5

Z05 = zeros(Float64, size(lhs5)[1:2])
ConvolutionOperators.timeslice!(Z05, lhs5, 1)
# iZ05 = inv(Z05)
# y5 = marchonintime(iZ05, lhs5, rhs5, Nt)

# j5 = zeros(eltype(y5), size(y5)[1:2])
# j5[:, 1] = PΛH * y5[:, 1] + 1.0/Δt * PΣ * y5[:, 1]
# for i in 2:Nt
#     j5[:, i] = PΛH * y5[:, i] + 1.0/Δt * PΣ * (y5[:, i] - y5[:, i-1])
# end

#=
    FORM 6: standard TD-CFIE (Beghein et. al., 2013)
=#
lhs6 = lhs1 + (-η) * Gxx * iNyx * lhs4
rhs6 = rhs1 + (-η) * Gxx * iNyx * rhs4

Z06 = zeros(Float64, size(lhs6)[1:2])
ConvolutionOperators.timeslice!(Z06, lhs6, 1)
iZ06 = inv(Z06)
j6 = marchonintime(iZ06, lhs6, rhs6, Nt)

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

# using Printf
# open("qHP-TD_CFIE_current-sphere-h_0.2-tau_0.1-width_8.txt", "a") do io
#     for i in 1:Nt
#         @printf(io, "%.10f %.10f %.10f %.10f %.10f %.10f\n", i*Δt, log10.(abs.(j1[1, i])), log10.(abs.(j3[1, i])), log10.(abs.(j4[1, i])), log10.(abs.(j6[1, i])), log10.(abs.(j7[1, i])))
#     end
# end;

## Plot results
# using Plots
# plotly()
# plt = Plots.plot(
#     width = 600, height=400,
#     grid = false,
#     xscale = :identity, 
#     yaxis = :log10, 
#     xlims = (0, 102),
#     xticks = [0; 50; 100],
#     # xtickfont = font(9, "Times"),
#     ylims = (1e-18, 2), 
#     yticks = [1e-15; 1e-10; 1e-5; 1e0;],
#     # ytickfont = font(9),
#     xlabel = "c t (m)",
#     ylabel = "j(t) (A/m)")

# x = Δt * [1:1:Nt;]
# plot!(x, abs.(j1[1, :]), label="standard TD-EFIE")
# plot!(x, abs.(j3[1,:]), label="CP qHP TD-EFIE")
# plot!(x, abs.(j4[1,:]), label="standard TD-MFIE")
# # plot!(x, abs.(j5[1,:]), label="qHP TD-MFIE")
# plot!(x, abs.(j6[1,:]), label=" standard TD-CFIE")
# plot!(x, abs.(j7[1,:]), label="qHP TD-CFIE")

# savefig("qHP-TD_CFIE_current.pdf")


using SphericalScattering, LinearAlgebra, StaticArrays, FastGaussQuadrature

function Hdivnorm(j, ω)
    𝕋0 = ω .* 𝕋0s + 1/ω .* 𝕋0h
    real(dot(j, 𝕋0 * j))
end

function L2norm(j)
    real(dot(j, j))
end

jω1, Δω, ω0 = fouriertransform(j1, Δt, 0.0, 2)
jω3, _, _ = fouriertransform(j3, Δt, 0.0, 2)
jω4, _, _ = fouriertransform(j4, Δt, 0.0, 2)
jω6, _, _ = fouriertransform(j6, Δt, 0.0, 2)
jω7, _, _ = fouriertransform(j7, Δt, 0.0, 2)

ω = collect(ω0 .+ (0:Nt-1)*Δω)
err1 = zeros(Nt)
err3 = zeros(Nt)
err4 = zeros(Nt)
err6 = zeros(Nt)
err7 = zeros(Nt)

sp = PECSphere(radius = radius, embedding =  Medium(ε, μ))

for i in Nt/2+2:Nt
    @show i
    κ = ω[i]

    eexc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    hexc = -1/(im*κ*c)*curl(eexc)

    ex = planeWave(embedding =  Medium(ε, μ), frequency=ω[i]/2π, direction=ẑ)
    Httf = r -> scatteredfield(sp, ex, MagneticField([SVector(r/norm(r))]))[1] + hexc(r)
    httf = (n × Httf) × n
    jref = iNyx * assemble(@discretise(httf[k], k∈Y))

    err1[i] = L2norm(jref - jω1[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err1[i]    
    err3[i] = L2norm(jref - jω3[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err3[i]   
    err4[i] = L2norm(jref - jω4[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err4[i]   
    err6[i] = L2norm(jref - jω6[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err6[i]   
    err7[i] = L2norm(jref - jω7[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err7[i]   
end