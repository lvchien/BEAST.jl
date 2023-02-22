using BEAST, CompScienceMeshes, SphericalScattering, StaticArrays, ConvolutionOperators
using LinearAlgebra, StaticArrays
using FastGaussQuadrature

include("genmesh.jl")

function q2h(j)
    h = zeros(eltype(j), size(j))
    for i in axes(j, 1)
        h[i, 1] = j[i, 1]
        for k in 2:size(j)[2]
            h[i, k] = 0.5 * (j[i, k] + j[i, k-1])
        end
    end
    return h
end

radius = 1.0
c = 1.0
# η = √(μ0/ε0)
# speedoflight = 1.0
sp = PECSphere(radius = radius)

## Solve the scatering problem using BEM
Γ = meshtorus(0.5, 1.0, 0.3)
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)
T̂0s = MWSingleLayer3D(0.0, 1/speedoflight, 0.0)                                      
T̂0h = MWSingleLayer3D(0.0, 0.0, speedoflight)
𝕋0s = assemble(T̂0s, X, X, quadstrat=nearstrat)                                          
𝕋0h = assemble(T̂0h, X, X, quadstrat=nearstrat)  

N = NCross()
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))

function Hdivnorm(j, ω)
    𝕋0 = ω .* 𝕋0s + 1/ω .* 𝕋0h
    real(dot(j, 𝕋0 * j))
end

function L2norm(j)
    real(dot(j, j))
end

Δt, Nt = 0.1, 1000

duration = 3.0
delay = 18.0
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
fgaussian = ω -> amplitude * exp(-im*ω*delay - (duration*ω/(8c))^2) / sqrt(2π)
polarisation, direction = x̂, ẑ
E = planewave(polarisation, direction, gaussian, c)
∂E = planewave(polarisation, direction, derive(gaussian), c)

# Time function spaces
δ = timebasisdelta(Δt, Nt)	                			                            # delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			                            # pulse function space
h = timebasisc0d1(Δt, Nt) 	                			                            # hat function space
q = BEAST.convolve(p, h)                        		                            # quadratic function space (*Δt)
∂q = BEAST.derive(q)					                                            # first order derivative of q (*Δt)

T = TDMaxwell3D.singlelayer(speedoflight=c)                                         # TD-EFIE

@hilbertspace j
@hilbertspace k

BEAST.@defaultquadstrat (T, X⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(7)

lhs_bilform_1 = @discretise T[k,j] k∈X⊗δ j∈X⊗∂q
lhs1 = 1.0/Δt * BEAST.td_assemble(lhs_bilform_1.bilform, lhs_bilform_1.test_space_dict, lhs_bilform_1.trial_space_dict)

rhs_linform_1 = @discretise(-1.0∂E[k], k∈X⊗δ)
rhs1 = BEAST.td_assemble(rhs_linform_1.linform, rhs_linform_1.test_space_dict)

Z01 = zeros(Float64, size(lhs1)[1:2])
ConvolutionOperators.timeslice!(Z01, lhs1, 1)
iZ01 = inv(Z01)
jq1 = marchonintime(iZ01, lhs1, rhs1, Nt)
j1 = q2h(jq1)

jω, Δω, ω0 = fouriertransform(j1, Δt, 0.0, 2)
ω = collect(ω0 .+ (0:Nt-1)*Δω)


i = 550
κ = ω[i]

t = Maxwell3D.singlelayer(wavenumber=κ, alpha=-im * ω[i], beta=1 / (-im * ω[i]))
E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n
H = -1/(im*κ*c)*curl(E)

efie = @discretise t[j,k]==e[k] j∈X k∈X
j = gmres(efie; restart=1500)

Hdivnorm(jω[:, i]/fgaussian(κ) - j, κ) / Hdivnorm(j, κ)

err = zeros(Nt)

# for i in 1650:1700
i=550
    @show i
    κ = ω[i]
    μ = 1.0
    eexc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    hexc = -1/(im*μ0*ω[i]/√(μ0*ε0))*curl(eexc)

    ex = planeWave(frequency=ω[i]/(2π*√(μ0*ε0)), direction=ẑ)
    Httf = r -> scatteredfield(sp, ex, MagneticField([SVector(r/norm(r))]))[1] + hexc(r)
    httf = (n × Httf) × n
    jref = iNyx * assemble(@discretise(httf[k], k∈Y))

    err[i] = L2norm(jref - jω[:, i]/fgaussian(ω[i])) / L2norm(jref)
    @show err[i]
# end

err = zeros(800)
for i in 671:672
    @show i
    f = 1e6 * i / 2
    κ = 2π*f/speedoflight                                             
    t = Maxwell3D.singlelayer(wavenumber=κ, alpha=-im * μ0 * (2π * f), beta=1 / (-im * ε0 * (2π * f)))
    E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    e = (n × E) × n
    H = -1/(im*μ0*κ*speedoflight)*curl(E)

    @hilbertspace j
    @hilbertspace k
    efie = @discretise t[j,k]==e[k] j∈X k∈X
    j = gmres(efie; restart=1500)

    # Solve the scattering problem by computing the Mie series
    ex = planeWave(frequency=f, direction=ẑ)
    Httf = r -> scatteredfield(sp, ex, MagneticField([SVector(r/norm(r))]))[1] + H(r)
    httf = (n × Httf) × n
    jref = iNyx * assemble(@discretise(httf[k], k∈Y))

    err[i] = Hdivnorm(j - jref, 2π*f) / Hdivnorm(jref, 2π*f)
    @show err[i]
end



using Plots 
plot(1e6.*[660:672]./2, err[660:672])
plot!(yaxis=:log10)