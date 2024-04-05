using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots, SphericalScattering
include("utils/genmesh.jl")

setminus(A,B) = submesh(!in(B), A)

# Physical coefficients
c = 1.0
T0 = 2.0                                                                         # D/c with D the diameter of scatterer [second]

# Computational mesh
Γ = meshsphere(1.0, 0.3)
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
    
Δt, Nt = 1e5, 1000
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
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 9, 9, 9, 9)

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
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(2)

bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗h
Txx = BEAST.assemble(bilform_1.bilform, bilform_1.test_space_dict, bilform_1.trial_space_dict)

linform_1 = @discretise(-1.0E[k], k∈X⊗δ)
ex = BEAST.assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Txx)[1:2])
ConvolutionOperators.timeslice!(Z01, Txx, 1)
iZ01 = inv(Z01)
je = marchonintime(iZ01, Txx, ex, Nt)


### FORM 3: qHP CP TD-EFIE
BEAST.@defaultquadstrat (∂Ts, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(2)
BEAST.@defaultquadstrat (∂Th, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(2)
BEAST.@defaultquadstrat (Ts, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(2)

Ms_bilform_31 = @discretise ∂Ts[k, j] k∈X⊗δ j∈X⊗q
Ms_bilform_32 = @discretise ∂Th[k, j] k∈X⊗δ j∈X⊗q
Ml_bilform_3 = @discretise Ts[k, j] k∈X⊗δ j∈X⊗h

Ms_31 = T0/Δt * BEAST.assemble(Ms_bilform_31.bilform, Ms_bilform_31.test_space_dict, Ms_bilform_31.trial_space_dict)
Ms_32 = T0/Δt * BEAST.assemble(Ms_bilform_32.bilform, Ms_bilform_32.test_space_dict, Ms_bilform_32.trial_space_dict)
Ml_3 = BEAST.assemble(Ml_bilform_3.bilform, Ml_bilform_3.test_space_dict, Ml_bilform_3.trial_space_dict)

ECP = (Δt/T0 * ℙΣH * 𝕋is * ℙΣH + ℙΛ * 𝕋is * ℙΣH + ℙΣH * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋is * ℙΛ + T0/Δt * ℙΛ * 𝕋ih * ℙΛ) * (iNxy * PΛH + ℙΣH * iNxy * PΣ) * (Δt/T0 * PΛH + PΣ)

qhpefie = ECP * (Ml_3 * PΛH + Ms_31 * PΣ + PΣ * Ms_32 * PΣ)
rhs3 = ECP * ex


### FORM 4: standard TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_4 = @discretise 0.5(N⊗I)[k, j] + 1.0K[k,j] k∈Y⊗δ j∈X⊗h
Kyx = BEAST.assemble(bilform_4.bilform, bilform_4.test_space_dict, bilform_4.trial_space_dict)

linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
hy = BEAST.assemble(linform_4.linform, linform_4.test_space_dict)

Z04 = zeros(Float64, size(Kyx)[1:2])
ConvolutionOperators.timeslice!(Z04, Kyx, 1)
iZ04 = inv(Z04)
jm = marchonintime(iZ04, Kyx, hy, Nt)


### FORM 5: qHP symmetrized TD-MFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

bilform_5s = @discretise 0.5(N⊗I)[k, j] + 1.0K[k, j] k∈Y⊗δ j∈X⊗∂q

Ks = T0/Δt * BEAST.assemble(bilform_5s.bilform, bilform_5s.test_space_dict, bilform_5s.trial_space_dict)

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
Z07 .-= Δt/T0 * ℙΣH * (-0.5 * Nyx + 𝕂0) * (iNyx * ℙΣH + PΛH * iNyx * ℙΛ) * Z04 * PΛH
iZ07 = inv(Z07)
y7 = marchonintime(iZ07, qhpcfie, rhs7, Nt)

jqhpc = zeros(eltype(y7), size(y7)[1:2])
jqhpc[:, 1] = PΛH * y7[:, 1] + T0/Δt * PΣ * y7[:, 1]
for i in 2:Nt
    jqhpc[:, i] = PΛH * y7[:, i] + T0/Δt * PΣ * (y7[:, i] - y7[:, i-1])
end

### Fourier transform
Xefie, Δω, ω0 = fouriertransform(je, Δt, 0.0, 2)
Xmfie, _, _ = fouriertransform(jm, Δt, 0.0, 2)
Xcfie, _, _ = fouriertransform(jc, Δt, 0.0, 2)
Xcfieqhp, _, _ = fouriertransform(jqhpc, Δt, 0.0, 2)
ω = collect(ω0 .+ (0:Nt-1)*Δω)

i1 = div(Nt, 2) + 2
ω1 = ω[i1]
ue = Xefie[:,i1] / fouriertransform(gaussian)(ω1)
um = Xmfie[:,i1] / fouriertransform(gaussian)(ω1)
uc = Xcfie[:,i1] / fouriertransform(gaussian)(ω1)
uqhp = Xcfieqhp[:,i1] / fouriertransform(gaussian)(ω1)

```
    Mie series
```
### Setting for SphericalScattering package
sp = PECSphere( 
    radius      = 1.0, 
)

ex = planeWave(
    embedding    = Medium(1.0, 1.0),
    frequency    = ω1/(2π),
)

Θ1, Φ = range(0.0,stop=2π,length=1000), 0.0
dirs1 = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for θ in Θ1 for ϕ in Φ]

### Farfield by Mie series
ff_mie = scatteredfield(sp, ex, FarField(dirs1))

ff_norm = norm.(ff_mie)
_, i2 = findmin(ff_norm[1:500])
_, i3 = findmin(ff_norm[500:end])

### Farfields by different schemes
Θ = range(0.0,stop=2π,length=35)
dirs = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for θ in Θ for ϕ in Φ]
ff_efie = potential(MWFarField3D(wavenumber=ω1,amplitude=ω1/4π), dirs, ue, X)
ff_mfie = potential(MWFarField3D(wavenumber=ω1,amplitude=ω1/4π), dirs, um, X)

Θ2 = vcat(Θ[1:3], range(Θ[4],stop=Θ1[i2],length=5), range(Θ1[i2],stop=Θ[9],length=5), Θ[10:26], range(Θ[27],stop=Θ1[499+i3],length=5), range(Θ1[499+i3],stop=Θ[32],length=5), Θ[33:end])
Θ2 = vcat(Θ2[1:6], range(Θ2[7],stop=Θ2[8],length=3), range(Θ2[9],stop=Θ2[10],length=3), Θ2[11:33], range(Θ2[34],stop=Θ2[35],length=3), range(Θ2[36],stop=Θ2[37],length=3), Θ2[38:end])
dirs2 = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for θ in Θ2 for ϕ in Φ]
ff_cfie = potential(MWFarField3D(wavenumber=ω1,amplitude=ω1/4π), dirs2, uc, X)
ff_qhp = potential(MWFarField3D(wavenumber=ω1,amplitude=ω1/4π), dirs2, uqhp, X)

```
    Plotting
```
plt = Plots.plot(
    size = (600, 400),
    grid = true,
    xscale = :identity, 
    xlims = (0, 2π+0.07),
    xticks = ([0, π/2, π, 3π/2, 2π], [0, 45, 90, 135, 180]),
    xtickfont = Plots.font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e-18, 1e-12), 
    yticks = [1e-18, 1e-16, 1e-14, 1e-12],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Angle (degree)",
    ylabel = "Scattered far field",
    titlefont = Plots.font(10, "Computer Modern"),
    guidefont = Plots.font(11, "Computer Modern"),
    colorbar_titlefont = Plots.font(10, "Computer Modern"),
    legendfont = Plots.font(11, "Computer Modern"),
    legend = :bottom,
    dpi = 300)

Plots.plot!(Θ1, norm.(ff_mie), label="Mie series", linecolor=:black, lw=1.6)
Plots.plot!(Θ, norm.(ff_efie), label="TD-EFIE", linecolor=1, lw=1.6, markershape=:square, markercolor=1, markersize=4.2)
Plots.plot!(Θ, norm.(ff_mfie), label="TD-MFIE", linecolor=2, lw=1.6, markershape=:utriangle, markercolor=2, markersize=5)
Plots.plot!(Θ2, norm.(ff_cfie), label="TD-CFIE", linecolor=4, lw=1.6, markershape=:circle, markercolor=4, markersize=5)
Plots.plot!(Θ2, norm.(ff_qhp), label="qHP TD-CFIE", linecolor=3, lw=1.6, markershape=:dtriangle, markercolor=3, markersize=5)

savefig("farfield.pdf")