using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Printf

setminus(A,B) = submesh(!in(B), A)

# Parameters
speedoflight = 1.0
Δt1, Nt = 4.0, 20

mesh_size = 0.1
# Computational domain
Γ = CompScienceMeshes.meshtorus(innerradius=0.6, outerradius=1.0, h=mesh_size)
∂Γ = boundary(Γ)
Γ = CompScienceMeshes.meshsquaretorus4(width=2.0, height=0.5, holewidth=0.5, h=mesh_size)


using Plotly
Plotly.plot([patch(Γ, opacity=0.2), wireframe(Γ)])

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
K0 = Maxwell3D.doublelayer(gamma=0.0)                                               # static MFIO
K = TDMaxwell3D.doublelayer(speedoflight=speedoflight)                              # TD-MFIO

@hilbertspace k
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))
iNxy = transpose(iNyx)

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(10, 10, 10, 10, 10, 10, 10, 10)
dmat(op,tfs,bfs) = BEAST.assemble(op,tfs,bfs; quadstrat=nearstrat)
mat = dmat

𝕂0 = assemble(@discretise(K0[k, j], k∈Y, j∈X), materialize=mat)
𝕄0 = Matrix(0.5* Nyx - 𝕂0)

δ1 = timebasisdelta(Δt1, Nt)	                			                            # delta distribution space
h1 = timebasisc0d1(Δt1, Nt) 

BEAST.@defaultquadstrat (K, Y⊗δ1, X⊗h1) BEAST.OuterNumInnerAnalyticQStrat(7)
lhs_bilform_4 = @discretise K[k,j] k∈Y⊗δ1 j∈X⊗h1
lhs4 = BEAST.td_assemble(lhs_bilform_4.bilform, lhs_bilform_4.test_space_dict, lhs_bilform_4.trial_space_dict)
Z04 = zeros(eltype(𝕂0), size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
a = Matrix(ℙΣH * (0.5 * Nyx - 𝕂0) * iNyx * (0.5 * Nyx + 𝕂0) * PΛH)

for Δt in [0.25; 0.5; 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0; 128.0; 256.0; 512.0]
    # Time function spaces
    δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
    p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
    h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
    q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
    ∂q = BEAST.derive(q)					                                            # first order derivative of q
    ip = integrate(p) 	

    Mll_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗ip
    Msl_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗h
    Mss_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

    Mll_5f = BEAST.td_assemble(Mll_bilform_5.bilform, Mll_bilform_5.test_space_dict, Mll_bilform_5.trial_space_dict)
    Msl_5 = BEAST.td_assemble(Msl_bilform_5.bilform, Msl_bilform_5.test_space_dict, Msl_bilform_5.trial_space_dict)
    Mss_5 = 1/Δt * BEAST.td_assemble(Mss_bilform_5.bilform, Mss_bilform_5.test_space_dict, Mss_bilform_5.trial_space_dict)

    lhs5 = ℙΛ * Z04 * iNyx * (Mss_5 * PΣ + Msl_5 * PΛH) + ℙΣH * Z04 * iNyx * Msl_5 * PΣ

    Z05 = zeros(eltype(𝕄0), size(lhs5)[1:2])
    ConvolutionOperators.timeslice!(Z05, lhs5, 1)

    open("cond_timestep_torus.txt", "a") do io
        # @printf(io, "%.2f %.10f %.10f %.10f %.10f %.10f %.10f\n", Δt, cond(Z01), cond(Z02), cond(Z03), cond(Z04), cond(Z05), cond(Z06))
        @printf(io, "%.2f %.10f\n", Δt, cond(Z05))
    end;
end
