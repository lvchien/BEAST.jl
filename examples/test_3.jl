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

### FORM 1: standard TD-EFIE
BEAST.@defaultquadstrat (K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(7)

lhs_bilform_4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗δ j∈X⊗h
lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)

rhs_linform_4 = @discretise(-1.0H[k], k∈Y⊗δ)
rhs4 = BEAST.td_assemble(rhs_linform_4.linform, rhs_linform_4.test_space_dict)

Z04 = zeros(Float64, size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
iZ04 = inv(Z04)
j4 = marchonintime(iZ04, lhs4, rhs4, Nt)

# MODIFIED MFIE
BEAST.@defaultquadstrat (K, Y⊗p, X⊗p) BEAST.OuterNumInnerAnalyticQStrat(7)
BEAST.@defaultquadstrat (K, Y⊗p, X⊗∂h) BEAST.OuterNumInnerAnalyticQStrat(7)

bf_5 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗p j∈X⊗p
bf_6 = @discretise (0.5(N⊗I) + 1.0K)[k,j] k∈Y⊗p j∈X⊗∂h

lhs5 = BEAST.td_assemble(bf_5.bilform, bf_5.test_space_dict, bf_5.trial_space_dict)
lhs6 = BEAST.td_assemble(bf_6.bilform, bf_6.test_space_dict, bf_6.trial_space_dict)

lhs9 = lhs5 * PΛH + lhs6 * PΣ

lf_3 = @discretise(-1.0H[k], k∈Y⊗p)

rhs9 = BEAST.td_assemble(lf_3.linform, lf_3.test_space_dict)

Z09 = zeros(Float64, size(lhs9)[1:2])
ConvolutionOperators.timeslice!(Z09, lhs9, 1)
iZ09 = inv(Z09)
y9 = marchonintime(iZ09, lhs9, rhs9, Nt)

j9 = zeros(eltype(y9), size(y9)[1:2])
j9[:, 1] = PΛH * y9[:, 1] + 1.0/Δt * PΣ * y9[:, 1]
for i in 2:Nt
    j9[:, i] = PΛH * y9[:, i] + 1.0/Δt * PΣ * (y9[:, i] - y9[:, i-1])
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
    ylims = (1e-17, 2), 
    yticks = [ 1e-15; 1e-10; 1e-5; 1e0;],
    ytickfont = font(9),
    xlabel = "c t (m)",
    ylabel = "j(t) (A/m)")

x = Δt * [1:1:Nt;]

Plots.plot!(x, abs.(j4[1, :]), label="TD-MFIE")
Plots.plot!(x, abs.(j9[1, :]), label="midified TD-MFIE")


using SphericalScattering

sp = PECSphere(radius = radius)

function L2norm(j)
    real(dot(j, j))
end