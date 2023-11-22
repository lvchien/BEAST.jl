using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
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

# Computational mesh
Γ = meshsphere(1.0, 0.3)
# fn = joinpath(dirname(pathof(CompScienceMeshes)), "geos/torus.geo")
# Γ = CompScienceMeshes.meshgeo(fn; dim=2, h=0.6)
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
    
Δt, Nt = 0.1, 600
κ = 1/Δt

# Operators
I = Identity()																			
N = NCross()
T = TDMaxwell3D.singlelayer(speedoflight=sol)                                    # TD-EFIO (numdiff=0)
T′ = TDMaxwell3D.singlelayer(speedoflight=sol′)
Ts = MWSingleLayerTDIO(sol, -1/sol, 0.0, 1, 0)                                   # weakly-singular TD-EFIO (numdiffs=0)
∂Ts = MWSingleLayerTDIO(sol, -1/sol, 0.0, 2, 0)                                  # weakly-singular TD-EFIO (numdiffs=1)
Ts′ = MWSingleLayerTDIO(sol′, -1/sol′, 0.0, 1, 0)                               
∂Ts′ = MWSingleLayerTDIO(sol′, -1/sol′, 0.0, 2, 0)                              
∂Th = MWSingleLayerTDIO(sol, 0.0, -sol, 0, 0)                                    # hyper-singular TD-EFIO (numdiffs=1)
∂Th′ = MWSingleLayerTDIO(sol′, 0.0, -sol′, 0, 0)                                 
K = TDMaxwell3D.doublelayer(speedoflight=sol)                                    # TD-MFIO
K′ = TDMaxwell3D.doublelayer(speedoflight=sol′)

@hilbertspace k l
@hilbertspace j m

# Gram matrix
Nyy = assemble(N, Y, Y)
Nyx = assemble(N, Y, X)
Nxy = -transpose(Nyx)
iNyx = inv(Matrix(Nyx))
iNxy = -transpose(iNyx)

```
                MAIN PART 
```

# Plane wave incident fields
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

# Temporal function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = convolve(p, h)                                                                  # quadratic function space (*Δt)
∂q = derive(q)                                                                      # first order derivative of q (*Δt)


### FORM 1: standard TD-PMCHWT
BEAST.@defaultquadstrat (T, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (T′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K′, X⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

pmchwt = @discretise(η*T[k,j] + η′*T′[k,j]  
                    - K[k,m] - K′[k,m] 
                    + K[l,j] + K′[l,j]
                    + 1/η*T[l,m] + 1/η′*T′[l,m],
                    k∈X⊗δ, l∈X⊗δ, j∈X⊗h, m∈X⊗h)

Zxx = BEAST.td_assemble(pmchwt.bilform, pmchwt.test_space_dict, pmchwt.trial_space_dict)

linform_1 = @discretise(E[k] + H[l], k∈X⊗δ, l∈X⊗δ)
rhs = BEAST.td_assemble(linform_1.linform, linform_1.test_space_dict)

Z01 = zeros(Float64, size(Zxx)[1:2])
ConvolutionOperators.timeslice!(Z01, Zxx, 1)
iZ01 = inv(Z01)
jpmchwt = marchonintime(iZ01, Zxx, rhs, Nt)

using Plots
x = 10 * Δt/3 * [1:1:Nt;]

plt = Plots.plot(
    size = (600, 400),
    grid = false,
    xscale = :identity, 
    xlims = (0, 200),
    # xticks = [400, 800, 1200, 1600, 2000],
    xtickfont = Plots.font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e-20, 1e0), 
    yticks = [1e-50, 1e-45, 1e-40, 1e-35, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Time (ns)",
    ylabel = "Current density intensity (A/m)",
    titlefont = Plots.font(10, "Computer Modern"),
    guidefont = Plots.font(11, "Times"),
    colorbar_titlefont = Plots.font(10, "Times"),
    legendfont = Plots.font(11, "Computer Modern"),
    legend = :bottomleft,
    dpi = 300)

Plots.plot!(x, abs.(jpmchwt[1, :]), label="TD-PMCHWT", linecolor=1, lw=1.3)