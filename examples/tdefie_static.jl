using BEAST, CompScienceMeshes, LinearAlgebra, ConvolutionOperators, Plots

# Computational domain
Γ = readmesh(joinpath(dirname(pathof(BEAST)),"../examples/sphere.in"))

# Parameters
Δt, Nt = 0.1, 200
speedoflight = 1.0

# RWG and BC function spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

# Time function spaces
T0 = timebasisshiftedlagrange(Δt, Nt, 0) 	# pulse function space
T1 = timebasisshiftedlagrange(Δt, Nt, 1) 	# hat function space
δ = timebasisdelta(Δt, Nt)					# delta distribution space

# TD single- and double-layer operators
dTs = BEAST.MWSingleLayerTDIO(speedoflight, -1/speedoflight, 0.0, 2, 0)
iTs = BEAST.integrate(BEAST.integrate(BEAST.MWSingleLayerTDIO(speedoflight, -1/speedoflight, 0.0, 2, 0)))
dTh = MWSingleLayerTDIO(speedoflight, 0.0, -speedoflight, 0, 0)
iTs_static = MWSingleLayer3D(0.0, -1.0/speedoflight, 0.0)
dTh_static = MWSingleLayer3D(0.0, 0.0, -speedoflight)
I = Identity()
N = NCross()

# Plane wave
duration = 20 * Δt * 2
delay = 1.5 * duration
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
polarisation, direction = x̂, ẑ
E = BEAST.planewave(polarisation, direction, gaussian, 1.0)

@hilbertspace k
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))

efie = @discretise(T[k,j] == -1.0E[k], k∈X⊗δ, j∈X⊗T1)
xefie = solve(efie)

mfie = @discretise((0.5(N⊗I) + 1.0K)[k,j] == -1.0H[k], k∈(Y⊗δ), j∈(X⊗T0))
lhs = BEAST.td_assemble(mfie.equation.lhs, mfie.test_space_dict, mfie.trial_space_dict)
rhs = BEAST.td_assemble(mfie.equation.rhs, mfie.test_space_dict)

M0 = assemble(@discretise((0.5N - 1.0K0)[k, j], k∈Y, j∈X))

lhs′ = M0 * iNyx * lhs
rhs′ = M0 * iNyx * rhs

xmfie = solve(mfie)

Z0 = zeros(eltype(lhs), size(lhs′)[1:2])
ConvolutionOperators.timeslice!(Z0, lhs′, 1)
iZ0 = inv(Z0)
j = marchonintime(iZ0, lhs′, rhs′, Nt)

Plots.plot(xmfie[1, :])
Plots.plot!(xefie[1, :])
Plots.plot!(j[1,:])