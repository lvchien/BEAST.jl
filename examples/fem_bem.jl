using CompScienceMeshes, BEAST

T = CompScienceMeshes.tetmeshsphere(1.0,0.2)
X = nedelecc3d(T)
Γ = boundary(T)
Y = raviartthomas(Γ)
Z = buffachristiansen(Γ)

trc = X -> ttrace(X, Γ)

κ = 1.0
μ_r, ϵ_r = 1.0, 4.0
γ = -im*κ

N = NCross()
Id = BEAST.Identity()

T = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)              
K = MWDoubleLayer3D(γ)                             

E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
H = curl(E)

e, h = (n × E) × n, (n × H) × n

A = μ_r*assemble(Id, curl(X), curl(X)) - ϵ_r*κ^2*assemble(Id, X, X) - κ^2*assemble(T, trc(X), trc(X))
C1 = -assemble(0.5+K, trc(X), Y)
C2 = assemble(-0.5+K, Y, trc(X))
B = assemble(T, Y, Y)

lhs = [A C1
       C2 B]

D1 = assemble(T, trc(X), Y)
D2 = assemble(-0.5+K, Y, Y)

Nyx = assemble(N, Z, Y)
iNyx = inv(Matrix(Nyx))

ez = assemble(e, Z) 
hy = κ*assemble(h, trc(X))

rhs = [-κ^2*D1*iNyx*ez + hy
        D2*iNyx*ez]

u = lhs \ rhs

Z1 = range(-1,1,length=200)
Y1 = range(-1,1,length=200)
nfpoints = [point(0,y,z) for y in Y1, z in Z1]

Enear = BEAST.grideval(nfpoints,u,X)
Enear = reshape(Enear,200,200)

using Plots
heatmap(Z1, Y1, real.(getindex.(Enear,1)))

using SphericalScattering, StaticArrays

sp = DielectricSphere(
    radius      = 1.0, 
    embedding   = Medium(1.0, 1.0), 
    filling     = Medium(4.0, 1.0)
)

ex = planeWave(
    embedding    = Medium(1.0, 1.0),
    frequency    = κ/(2π),
)


Emie = field(sp, ex, ElectricField(nfpoints[:]))
heatmap(Z1, Y1, reshape(real.(getindex.(Emie,1)),200,200))