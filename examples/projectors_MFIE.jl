using BEAST, CompScienceMeshes, LinearAlgebra
using Plots

setminus(A,B) = submesh(!in(B), A)

radius = 1.0
h = 0.1
Γ = meshsphere(;radius, h)
∂Γ = boundary(Γ)

ϵ, μ, ω = 1.0, 1.0, 0.0001; κ, η = ω * √(ϵ*μ), √(μ/ϵ)
γ = im*κ

SL = Maxwell3D.singlelayer(wavenumber=κ)
WS = Maxwell3D.weaklysingular(wavenumber=κ)

K = Maxwell3D.doublelayer(wavenumber=κ)
K′ = Maxwell3D.doublelayer(wavenumber=-im*κ)
N = NCross()

E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n;
H = -1/(im*μ*ω)*curl(E)
h′ = (n × H) × n

edges = setminus(skeleton(Γ,1), ∂Γ)
verts = setminus(skeleton(Γ,0), skeleton(∂Γ,0))

Σ = Matrix(connectivity(Γ, edges, sign))
Λ = Matrix(connectivity(Γ, edges, sign))

PΣ = Σ * pinv(Σ'*Σ) * Σ'
PΛH = I - PΣ

ℙΛ = Λ * pinv(Λ'*Λ) * Λ'
ℙHΣ = I - ℙΛ

M = im * √(κ) * PΣ + 1 / √(κ) * PΛH
𝕄 = im * √(κ) * ℙΛ + 1 / √(κ) * ℙHΣ

MR = γ * PΣ + PΛH
ML = PΣ + 1/γ * PΛH

MRΣ = γ * PΣ
MRΛH = PΛH
MLΣ = PΣ
MLΛH = 1/γ * PΛH

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

@hilbertspace p
@hilbertspace q

Kyx = assemble(@discretise((0.5N + K)[p,q], p∈Y, q∈X))
K′yx = assemble(@discretise((0.5N - K′)[p,q], p∈Y, q∈X))
Nxy = Matrix(assemble(N, Y, X))
iNxy = inv(Nxy)

SLxx = assemble(@discretise(SL[p,q], p∈X, q∈X))
WSxx = assemble(@discretise(WS[p,q], p∈X, q∈X))

hy = assemble(@discretise(h′[p], p∈Y))
ex = assemble(@discretise(e[p], p∈X))

sys0 = Kyx
sys1 = 𝕄 * K′yx * iNxy * Kyx * M
sys2 = MLΣ * SLxx * MRΣ + MLΛH * WSxx * MRΣ + MLΣ * WSxx * MRΛH + MLΛH * WSxx * MRΛH
sys3 = SLxx

rhs0 = hy
rhs1 = 𝕄 * K′yx * iNxy * hy
rhs2 = ML * ex
rhs3 = ex

u0, ch0 = solve(BEAST.GMRESSolver(sys0, tol=2e-6, restart=250), rhs0)
v1, ch1 = solve(BEAST.GMRESSolver(sys1, tol=2e-6, restart=250), rhs1)
v2, ch2 = solve(BEAST.GMRESSolver(sys2, tol=2e-6, restart=250), rhs2)
u3, ch3 = solve(BEAST.GMRESSolver(sys3, tol=2e-6, restart=250), rhs3)

u1 = M * v1
u2 = MR * v2

Φ, Θ = [0.0], range(0,stop=π,length=40)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

near0 = potential(MWFarField3D(wavenumber=κ), pts, u0, X)
near1 = potential(MWFarField3D(wavenumber=κ), pts, u1, X)
near2 = potential(MWFarField3D(wavenumber=κ), pts, u2, X)
near3 = potential(MWFarField3D(wavenumber=κ), pts, u3, X)

plot();
plot!(Θ, norm.(near0));
scatter!(Θ, norm.(near1))
scatter!(Θ, norm.(near2))
scatter!(Θ, norm.(near3))
