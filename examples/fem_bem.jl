using CompScienceMeshes, BEAST, Plots

function nearEfield(eD,eN,eincD,XD,XN,Xinc,κ,points,Einc)

    K = BEAST.MWDoubleLayerField3D(gamma=-im*κ)
    T = BEAST.MWSingleLayerField3D(-im*κ, 1.0, -1.0/κ^2)

    eDL = potential(K, points, eD, XD) - potential(K, points, eincD, Xinc)
    eSL = potential(T, points, eN, XN)
    E = eDL + eSL + Einc.(points)
    return E
end


Ω = CompScienceMeshes.tetmeshsphere(1.0,0.3)
X = nedelecc3d(Ω)
Γ = boundary(Ω)
Y = raviartthomas(Γ)
Z = buffachristiansen(Γ)

trc = X -> ttrace(X, Γ)

κ, η = 1.0, 1.0
μ_r, ϵ_r = 1.0, 4.0
γ = -im*κ

N = NCross()
Id = BEAST.Identity()

T = MWSingleLayer3D(γ, 1.0, -1.0/κ^2)              
K = MWDoubleLayer3D(γ)                             

E = Maxwell3D.planewave(direction=-ẑ, polarization=x̂, wavenumber=κ)
H = curl(E)

e, h = (n × E) × n, (n × H) × n

A = μ_r*assemble(Id, curl(X), curl(X)) - κ^2*assemble(Id, X, X) - κ^2*assemble(T, trc(X), trc(X))
C1 = assemble(-0.5N+K, trc(X), Y)
C2 = assemble(-0.5N-K, Y, trc(X))
B = assemble(T, Y, Y)

lhs = [A C1
       C2 B]

D1 = assemble(T, trc(X), Y)
D2 = assemble(-0.5N-K, Y, Y)

Nzy = assemble(N, Z, Y)
iNzy = inv(Matrix(Nzy))

ez = assemble(e, Z) 
hy = assemble(h, trc(X))

rhs = [-κ^2*D1*iNzy*ez + hy
        D2*iNzy*ez]

u, = solve(BEAST.GMRESSolver(lhs,tol=2e-5, restart=2500), rhs)

u = lhs\rhs

aZ = range(-1,1,length=200)
aY = range(-1,1,length=200)
nfpoints = [point(0,y,z) for z in aZ, y in aY]

E_in = BEAST.grideval(nfpoints,u[1:numfunctions(X)],X)
E_ex = nearEfield(u[1:numfunctions(X)],u[numfunctions(X)+1:end],iNzy*ez,trc(X),Y,Y,κ,nfpoints,E)
E_tot = E_in + E_ex

heatmap(aZ, aY, reshape(real.(getindex.(E.(nfpoints),1)),200,200))

# Mie series
using SphericalScattering, StaticArrays

sp = DielectricSphere(
    radius      = 1.0, 
    embedding   = Medium(1.0, 1.0), 
    filling     = Medium(1.0, 1.0)
)

ex = planeWave(
    embedding    = Medium(1.0, 1.0),
    frequency    = κ/(2π),
)

Emie = field(sp, ex, ElectricField(nfpoints[:]))

heatmap(aZ, aY, reshape(real.(getindex.(Emie,1)),200,200))
