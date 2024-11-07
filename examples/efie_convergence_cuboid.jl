using CompScienceMeshes
using BEAST

using Makeitso
using BakerStreet
using DataFrames

@target solutions () -> begin
    function payload(;h)
        d = 1.0
        Γ = meshcuboid(d, d, d/2, h)
        X = raviartthomas(Γ)

        κ, η = 1.0, 1.0
        t = Maxwell3D.singlelayer(wavenumber=κ)
        s = -Maxwell3D.singlelayer(gamma=κ)
        E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
        e = (n × E) × n

        @hilbertspace j
        @hilbertspace k

        A = assemble(t[k,j], X, X)
        S = assemble(s[k,j], X, X)
        b = assemble(e[k], X)

        Ai = BEAST.GMRESSolver(A; restart=1500, abstol=1e-8, maxiter=10_000)
        u = Ai * b
        u_Snorm = real(sqrt(dot(u, S*u)))
        return (;u, X, u_Snorm)
    end

    α = 0.8
    h = collect(0.4 * α.^(0:15))
    runsims(payload, "solutions"; h)
end

@make solutions
df = loadsims("solutions")
error()

using LinearAlgebra
using Plots
plot(df.h, real.(df.u_Snorm), marker=:.)

# α = df.h[1] / df.h[2]
W1 = reverse(df.u_Snorm)
# W1 = reverse((df.h).^2.231)
W2 = Iterators.drop(W1,1)
W3 = Iterators.drop(W1,2)
p = [log(abs((w2-w1)/(w3-w2))) / log(1/α) for (w1,w2,w3) in zip(W1,W2,W3)]
plot(collect(Iterators.drop(reverse(df.h),2)), p, marker=:.)

refsol = df.u_Snorm[1]
plot(log.(df.h), log.(abs.(df.u_Snorm .- refsol)), marker=:.)
plot!(log.(df.h), 1.5 * log.(df.h))

fcr, geo = facecurrents(df.u[1], df.X[1])
import Plotly
Plotly.plot(patch(geo, log10.(norm.(fcr))))

uref = df.u[1]
Xref = df.X[1]
Γref = geometry(Xref)
errs = zeros(length(df.h))
s = -Maxwell3D.singlelayer(gamma=1.0)
@hilbertspace k
@hilbertspace j
S11 = assemble(s, Xref, Xref)
for i in reverse(2:length(df.h))
    @show i, df.h[i]
    u = df.u[i]
    X = df.X[i]
    Γ = geometry(X)

    qstrat12 = BEAST.NonConformingIntegralOpQStrat(BEAST.DoubleNumSauterQstrat(3, 4, 6, 6, 6, 6))
    S12 = assemble(s, Xref, X; quadstrat=[qstrat12])
    S22 = assemble(s, X, X)
    errs[i] = sqrt(real(dot(uref, S11*uref) - 2*real(dot(uref, S12*u)) + real(dot(u, S22*u))))
    @show errs[i]
end
