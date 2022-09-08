using BEAST, CompScienceMeshes, LinearAlgebra
using Plots

setminus(A,B) = submesh(!in(B), A)

Γ = meshsphere(;radius=1.0, h=0.3)
∂Γ = boundary(Γ)

edges = setminus(skeleton(Γ,1), ∂Γ)
verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))

Σ = Matrix(connectivity(Γ, edges, sign))
Λ = Matrix(connectivity(Γ, edges, sign))

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)
dmat(op,tfs,bfs) = BEAST.assemble(op,tfs,bfs; quadstrat=nearstrat)
mat = dmat

ϵ, μ, ω = 1.0, 1.0, 0.001; κ, η = ω * √(ϵ*μ), √(μ/ϵ)
γ = 

T = Maxwell3D.singlelayer(wavenumber=κ)
Ts = Maxwell3D.weaklysingular(wavenumber=κ)
𝕋 = Maxwell3D.singlelayer(wavenumber=-im*κ)
𝕋s = Maxwell3D.weaklysingular(wavenumber=-im*κ)

K = Maxwell3D.doublelayer(wavenumber=κ)
𝕂 = Maxwell3D.doublelayer(wavenumber=-im*κ)
N = NCross()

E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n;
H = -1/(im*μ*ω)*curl(E)
h = (n × H) × n

PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = I - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙHΣ = I - ℙΛ

M = im * √(κ) * PΣ + 1/√(κ) * PΛH
𝕄 = im * √(κ) * ℙΛ + 1/√(κ) * ℙHΣ

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

@hilbertspace p
@hilbertspace q

Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))
Nxy = assemble(N, X, Y)
iNxy = inv(Matrix(Nxy))

Txx = assemble(@discretise(T[p,q], p∈X, q∈X), materialize=mat)
Tsxx = assemble(@discretise(Ts[p,q], p∈X, q∈X), materialize=mat)
𝕋yy = assemble(@discretise(𝕋[p,q], p∈Y, q∈Y), materialize=mat)
𝕋syy = assemble(@discretise(𝕋s[p,q], p∈Y, q∈Y), materialize=mat)

Kyx = BEAST.assemble(@discretise(K[p,q], p∈Y, q∈X), materialize=mat)
𝕂yx = BEAST.assemble(@discretise(𝕂[p,q], p∈Y, q∈X), materialize=mat)

Dyx = Matrix(0.5 * Nyx + Kyx)
𝔻yx = Matrix(0.5 * Nyx - 𝕂yx)

ex = assemble(@discretise(e[p], p∈X))
hy = assemble(@discretise(h[p], p∈Y))

sys0 = Txx
sys1 = -κ * PΣ * Txx * PΣ + im * (PΣ * Tsxx * PΛH + PΛH * Tsxx * PΣ) + 1/κ * PΛH * Tsxx * PΛH
sys1′ = -κ * ℙΛ * 𝕋yy * ℙHΣ + im * (ℙΛ * 𝕋syy * ℙHΣ + ℙHΣ * 𝕋syy * ℙΛ) + 1/κ * ℙHΣ * 𝕋syy * ℙHΣ
sys2 = Dyx 
sys3 = 𝕄 * 𝔻yx * iNyx * Dyx * M
sys4 = η^2 * 𝕄 * 𝔻yx * iNyx * Dyx * M + sys1′ * iNxy * sys1

rhs0 = ex
rhs1 = M * ex
rhs2 = hy
rhs3 = 𝕄 * 𝔻yx * iNyx * hy
rhs4 = η^2 * 𝕄 * 𝔻yx * iNyx * hy + sys1′ * iNxy * M * ex 

u0, ch0 = solve(BEAST.GMRESSolver(sys0, tol=2e-5, restart=250), rhs0)
v1, ch1 = solve(BEAST.GMRESSolver(sys1, tol=2e-5, restart=250), rhs1)
u2, ch2 = solve(BEAST.GMRESSolver(sys2, tol=2e-5, restart=250), rhs2)
v3, ch3 = solve(BEAST.GMRESSolver(sys3, tol=2e-5, restart=250), rhs3)
v4, ch4 = solve(BEAST.GMRESSolver(sys4, tol=2e-5, restart=250), rhs4)

u1 = M * v1
u3 = M * v3
u4 = M * v4

Φ, Θ = [0.0], range(0,stop=π,length=50)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

near0 = potential(MWFarField3D(wavenumber=κ), pts, u0, X)
near1 = potential(MWFarField3D(wavenumber=κ), pts, u1, X)
near2 = potential(MWFarField3D(wavenumber=κ), pts, u2, X)
near3 = potential(MWFarField3D(wavenumber=κ), pts, u3, X)
near4 = potential(MWFarField3D(wavenumber=κ), pts, u4, X)

plot();
plot!(Θ, norm.(near0));
scatter!(Θ, norm.(near1))
scatter!(Θ, norm.(near2))
scatter!(Θ, norm.(near3))
scatter!(Θ, norm.(near4))
