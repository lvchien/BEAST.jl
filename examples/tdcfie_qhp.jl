using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators
using Plots

setminus(A,B) = submesh(!in(B), A)

# Computational domain
Γ = readmesh(joinpath(dirname(pathof(BEAST)),"../examples/sphere.in"))
∂Γ = boundary(Γ)

# Parameters
Δt, Nt = 0.1, 200

# Connectivity matrices
edges = setminus(skeleton(Γ,1), ∂Γ)
verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))
cells = skeleton(Γ,2)

Σ = Matrix(connectivity(cells, edges, sign))
Λ = Matrix(connectivity(verts, edges, sign))

# Projectors
I = LinearAlgebra.I
PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = I - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙΣH = I - ℙΛ

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

# Time function spaces
T0 = timebasisshiftedlagrange(Δt, Nt, 0) 	# pulse function space
T1 = timebasisshiftedlagrange(Δt, Nt, 1)	# hat function space
δ = timebasisdelta(Δt, Nt)					# delta distribution space
iT0 = integrate(T0)
dT1 = derive(T1)

# Weakly singular TD single-layer operator
function weaklysingularTDIO(;speedoflight, numdiffs=0)
	@assert numdiffs >= 1
	numdiffs == 1 && return BEAST.integrate(BEAST.MWSingleLayerTDIO(speedoflight, -1/speedoflight, 0.0, 2, 0))
	return BEAST.MWSingleLayerTDIO(speedoflight, -1/speedoflight, 0.0, numdiffs, 0)
end

# Hyper singular TD single-layer operator
function hypersingularTDIO(;speedoflight)
	return BEAST.integrate(BEAST.MWSingleLayerTDIO(speedoflight, 0.0, -speedoflight, 0, 0))
end

# TD single-layer and double-layer operators
∂T = TDMaxwell3D.singlelayer(speedoflight=1.0, numdiffs=1)
Ts = weaklysingularTDIO(speedoflight=1.0, numdiffs=1)
dTs = weaklysingularTDIO(speedoflight=1.0, numdiffs=2)
iTs = BEAST.integrate(Ts)
Th = hypersingularTDIO(speedoflight=1.0)
K = TDMaxwell3D.doublelayer(speedoflight=1.0)
K0 = Maxwell3D.doublelayer(wavenumber=0.0)                  # localized double-layer operator
I = Identity()
N = NCross()

# Plane wave
duration = 20 * Δt * 2
delay = 1.5 * duration
amplitude = 1.0
gaussian = BEAST.creategaussian(duration, delay, amplitude)
polarisation, direction = x̂, ẑ
E = BEAST.planewave(polarisation, direction, gaussian, 1.0)
iE = BEAST.planewave(polarisation, direction, BEAST.integrate(gaussian), 1.0)
H = direction × E
iH = direction × iE

@hilbertspace k
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))

M0 = assemble(@discretise((0.5N - 1.0K0)[k, j], k∈Y, j∈X))

Mss_bilform = @discretise((0.5(N⊗I) + 1.0K)[k,j], k∈(Y⊗T0), j∈(X⊗dT1))
Msl_bilform = @discretise((0.5(N⊗I) + 1.0K)[k,j], k∈(Y⊗δ), j∈(X⊗T1))
Mll_bilform = @discretise((0.5(N⊗I) + 1.0K)[k,j], k∈(Y⊗δ), j∈(X⊗iT0))
# Mll_bilform = @discretise((1.0K - 1.0(K0⊗I))[k,j], k∈(Y⊗δ), j∈(X⊗iT0))

Mss = 1/Δt * BEAST.td_assemble(Mss_bilform.bilform, Mss_bilform.test_space_dict, Mss_bilform.trial_space_dict)
Msl = BEAST.td_assemble(Msl_bilform.bilform, Msl_bilform.test_space_dict, Msl_bilform.trial_space_dict)
Mll = BEAST.td_assemble(Mll_bilform.bilform, Mll_bilform.test_space_dict, Mll_bilform.trial_space_dict)

# Truncate the long tail
tail = Mll.v[2].convop.tail 
Mll.v[2].convop.tail .= zeros(eltype(tail), size(tail))

lhs = ℙΛ * M0 * iNyx * Mss * PΣ + ℙΛ * M0 * iNyx * Msl * PΛH +  ℙΣH * M0 * iNyx * Msl * PΣ + ℙΣH * M0 * iNyx * Mll * PΛH
Z0 = zeros(eltype(M0), size(lhs)[1:2])
ConvolutionOperators.timeslice!(Z0, lhs, 1)
@show cond(Z0)
iZ0 = inv(Z0)

rhs_loop_linform = @discretise(-1.0iH[k], k∈(Y⊗δ))
rhs_star_linform = @discretise(-1.0H[k], k∈(Y⊗T0))
rhs_loop = BEAST.td_assemble(rhs_loop_linform.linform, rhs_loop_linform.test_space_dict)
rhs_star = 1.0/Δt * BEAST.td_assemble(rhs_star_linform.linform, rhs_star_linform.test_space_dict)
rhs = ℙΣH * M0 * iNyx * rhs_loop + ℙΛ * M0 * iNyx * rhs_star

y = marchonintime(iZ0, lhs, rhs, Nt)
j = zeros(eltype(y), size(y)[1:2])
j[:, 1] = PΛH * y[:, 1] + 1.0/Δt * PΣ * y[:, 1]
for i in 2:Nt
	j[:, i] = PΛH * y[:, i] + 1.0/Δt * PΣ * (y[:, i] - y[:, i-1])
end

Plots.plot(j[1,:])