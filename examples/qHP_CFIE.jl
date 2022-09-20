using BEAST, CompScienceMeshes, LinearAlgebra
using Plots

setminus(A,B) = submesh(!in(B), A)

# Computational domain
Γ = meshsphere(;radius=1.0, h=0.3)
∂Γ = boundary(Γ)

# Connectivity matrices
edges = setminus(skeleton(Γ,1), ∂Γ)
verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))
cells = skeleton(Γ,2)

Σ = Matrix(connectivity(cells, edges, sign))
Λ = Matrix(connectivity(verts, edges, sign))

# New quaddata for integral approximation 
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)
dmat(op,tfs,bfs) = assemble(op,tfs,bfs; quadstrat=nearstrat)
mat = dmat

# Material coefficients
ϵ, μ, ω = 8.8541878128e-12, 4π*1e-7, 2π*1e3; κ, η = ω * √(ϵ*μ), √(μ/ϵ)

# EFIOs and MFIOs
T = Maxwell3D.singlelayer(wavenumber=κ)
Ts = Maxwell3D.weaklysingular(wavenumber=κ)
K = Maxwell3D.doublelayer(wavenumber=κ)
𝕋 = Maxwell3D.singlelayer(wavenumber=-im*κ)
𝕋s = Maxwell3D.weaklysingular(wavenumber=-im*κ)
𝕂 = Maxwell3D.doublelayer(wavenumber=-im*κ)
K0 = Maxwell3D.doublelayer(wavenumber=0.0)
N = NCross()

# Incident wave
E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n
H = -1/(im*μ*ω)*curl(E)
h = (n × H) × n

# Projectors
PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = I - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙΣH = I - ℙΛ

M = im * √(κ) * PΣ + 1/√(κ) * PΛH
𝕄 = im * √(κ) * ℙΛ + 1/√(κ) * ℙΣH

# BE spaces
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

@hilbertspace p
@hilbertspace q

# Gram matrix
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))

# Assembling operators
Txx = η * assemble(@discretise(T[p,q], p∈X, q∈X), materialize=mat)
Tsxx = η * assemble(@discretise(Ts[p,q], p∈X, q∈X), materialize=mat)
Kyx = assemble(@discretise(K[p,q], p∈Y, q∈X), materialize=mat)
K0yx = assemble(@discretise(K0[p,q], p∈Y, q∈X), materialize=mat)
𝕋yy = η * assemble(@discretise(𝕋[p,q], p∈Y, q∈Y), materialize=mat)
𝕋syy = η * assemble(@discretise(𝕋s[p,q], p∈Y, q∈Y), materialize=mat)
𝕂yx = assemble(@discretise(𝕂[p,q], p∈Y, q∈X), materialize=mat)

Dyx = Matrix(0.5 * Nyx + Kyx)
𝔻yx = Matrix(0.5 * Nyx - 𝕂yx)

ex = assemble(@discretise(e[p], p∈X))
hy = assemble(@discretise(h[p], p∈Y))

# Matrix systems
sys1 = Txx                                                                                                  # standard EFIE 
sys2 = -κ * PΣ * Txx * PΣ + im * (PΣ * Tsxx * PΛH + PΛH * Tsxx * PΣ) + 1/κ * PΛH * Tsxx * PΛH               # qHP-EFIE
sys′ = - κ * ℙΛ * 𝕋yy * ℙΛ + im * (ℙΛ * 𝕋syy * ℙΣH + ℙΣH * 𝕋syy * ℙΛ) + 1/κ * ℙΣH * 𝕋syy * ℙΣH             # auxiliary term
sys3 = sys′ * iNyx' * sys2                                                                                  # qHP Calderon preconditioned EFIE
sys4 = Dyx                      
sys5 = -κ * ℙΛ * 𝔻yx * iNyx * Dyx * PΣ + im * (ℙΛ * 𝔻yx * iNyx * Dyx * PΛH + ℙΣH * 𝔻yx * iNyx * Dyx * PΣ) + 1/κ * ℙΣH * 𝔻yx * iNyx * Dyx * PΛH - 1/κ * ℙΣH * Matrix(0.5 * Nyx - K0yx) * iNyx * Matrix(0.5 * Nyx + K0yx) * PΛH                                                                        # qHP-MFIE
sys6 = η^2 * 𝔻yx * iNyx * Dyx + 𝕋yy * iNyx' * Txx                                                          # Yukawa-Calderon preconditioned CFIE
sys7 = η^2 * sys5 + sys3                                                                                    # qHP-YC-CFIE

@show cond(Matrix(sys1))
@show cond(Matrix(sys2))
@show cond(Matrix(sys3))
@show cond(Matrix(sys4))
@show cond(Matrix(sys5))
@show cond(Matrix(sys6))
@show cond(Matrix(sys7))

rhs1 = ex
rhs2 = M * ex
rhs3 = sys′ * iNyx' * M * ex
rhs4 = hy
rhs5 = 𝕄 * 𝔻yx * iNyx * hy
rhs6 = η^2 * 𝔻yx * iNyx * hy + 𝕋yy * iNyx' * ex
rhs7 = η^2 * rhs5 + rhs3

# Solving the matrix systems
u1, ch1 = solve(BEAST.GMRESSolver(sys1, tol=2e-5, restart=250), rhs1)
v2, ch2 = solve(BEAST.GMRESSolver(sys2, tol=2e-5, restart=250), rhs2)
v3, ch3 = solve(BEAST.GMRESSolver(sys3, tol=2e-5, restart=250), rhs3)
u4, ch4 = solve(BEAST.GMRESSolver(sys4, tol=2e-5, restart=250), rhs4)
v5, ch5 = solve(BEAST.GMRESSolver(sys5, tol=2e-5, restart=250), rhs5)
u6, ch6 = solve(BEAST.GMRESSolver(sys6, tol=2e-5, restart=250), rhs6)
v7, ch7 = solve(BEAST.GMRESSolver(sys7, tol=2e-5, restart=250), rhs7)

u2 = M * v2
u3 = M * v3
u5 = M * v5
u7 = M * v7

# Result visualization 
Φ, Θ = [0.0], range(0,stop=π,length=50)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

near1 = potential(MWFarField3D(wavenumber=κ), pts, u1, X)
near2 = potential(MWFarField3D(wavenumber=κ), pts, u2, X)
near3 = potential(MWFarField3D(wavenumber=κ), pts, u3, X)
near4 = potential(MWFarField3D(wavenumber=κ), pts, u4, X)
near5 = potential(MWFarField3D(wavenumber=κ), pts, u5, X)
near6 = potential(MWFarField3D(wavenumber=κ), pts, u6, X)
near7 = potential(MWFarField3D(wavenumber=κ), pts, u7, X)

plot();
plot!(Θ, norm.(near1), label = "EFIE");
scatter!(Θ, norm.(near2), label = "qHP-EFIE")
scatter!(Θ, norm.(near3), label = "qHP-CP-EFIE")
scatter!(Θ, norm.(near4), label = "MFIE")
scatter!(Θ, norm.(near5), label = "qHP-MFIE")
scatter!(Θ, norm.(near6), label = "YC-CFIE")
scatter!(Θ, norm.(near7), label = "qHP-YC-CFIE")