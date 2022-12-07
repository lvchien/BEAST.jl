using BEAST, CompScienceMeshes, LinearAlgebra, Plots, ConvolutionOperators

setminus(A,B) = submesh(!in(B), A)

# Computational domain
Γ = readmesh(joinpath(dirname(pathof(BEAST)),"../examples/sphere.in"))
∂Γ = boundary(Γ)

# Parameters
Δt, Nt = 0.25, 1200
speedoflight = 1.0

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

# Time function spaces
δ = timebasisdelta(Δt, Nt)	                			# delta distribution space
p = timebasiscxd0(Δt, Nt) 	                			# pulse function space
h = timebasisc0d1(Δt, Nt) 	                			# hat function space
q = BEAST.convolve(p, h)                        		        # quadratic function space (*Δt)
c = BEAST.convolve(p, q)                        		        # cubic function space (*Δt^2)
∂h = BEAST.derive(h)							# derivative of h
∂q = BEAST.derive(q)					                # first order derivative of q
∂c = BEAST.derive(c)							# derivative of c
∂2q = BEAST.derive(BEAST.derive(q))					# second order derivative of q
∂3c = BEAST.derive(BEAST.derive(BEAST.derive(c)))		        # third order derivative of c
ip = BEAST.integrate(p) 	                			# integral of p
ih = BEAST.integrate(h) 	                			# integral of h

# single basis function spaces for calculating Z0 only
δ1 = timebasisdelta(Δt, 1)
p1 = timebasiscxd0(Δt, 1)
h1 = timebasisc0d1(Δt, 1)

# Operators
I = Identity()																			
N = NCross()
T̂0s = MWSingleLayer3D(0.0, -1.0/speedoflight, 0.0)							        # static weakly-singular TD-EFIO (numdiffs=0)
T̂0h = MWSingleLayer3D(0.0, 0.0, -speedoflight)									# static hypersingular TD-EFIO	(numdiffs=0)
T = TDMaxwell3D.singlelayer(speedoflight=speedoflight)							       # TD-EFIE
T̂s = BEAST.integrate(MWSingleLayerTDIO(speedoflight, -1/speedoflight, 0.0, 1, 0))			   	# weakly-singular TD-EFIO (numdiffs=0)
T̂h = MWSingleLayerTDIO(speedoflight, 0.0, -speedoflight, 0, 0)							# hypersingular TD-EFIO (numdiffs=0)
K0 = Maxwell3D.doublelayer(wavenumber=0.0)    								       # static MFIO
K = TDMaxwell3D.doublelayer(speedoflight=speedoflight)							       # TD-MFIO

# Plane wave
duration = 80 * Δt
delay = 3 * duration
amplitude = 1.0
gaussian = creategaussian(duration, delay, amplitude)
polarisation, direction = x̂, ẑ
E = BEAST.planewave(polarisation, direction, gaussian, 1.0)
H = direction × E
∂E = BEAST.planewave(polarisation, direction, BEAST.derive(gaussian), 1.0)
iE = BEAST.planewave(polarisation, direction, BEAST.integrate(gaussian), 1.0)
iH = direction × iE

@hilbertspace k
@hilbertspace j

# Gram matrix
Nyx = assemble(N, Y, X)
iNyx = inv(Matrix(Nyx))
iNxy = transpose(iNyx)

# assembly of static operators
nearstrat = BEAST.DoubleNumWiltonSauterQStrat(6, 7, 6, 7, 7, 7, 7, 7)
farstrat  = BEAST.DoubleNumQStrat(1,2)

dmat(op,tfs,bfs) = BEAST.assemble(op,tfs,bfs; quadstrat=nearstrat)
mat = dmat

𝕋0s = assemble(@discretise(T̂0s[k, j], k∈Y, j∈Y), materialize=mat)
𝕋0h = assemble(@discretise(T̂0h[k, j], k∈Y, j∈Y), materialize=mat)
𝕂0 = assemble(@discretise(K0[k,j], k∈Y, j∈X), materialize=mat)
𝕄0 = Matrix(0.5 * Nyx - 𝕂0)
left_linear_map = 𝕄0 * iNyx

"""
    Truncate the tail from kmax = maximum(Z.k1)
"""
function marchonintime_trunc(W0,Z,B,I)

        T = eltype(W0)
        M,N = size(W0)
        @assert M == size(B,1)
    
        x = zeros(T,N,I)
        y = zeros(T,N)
        csx = zeros(T,N,I)
    
        for i in 1:I
            R = B[:,i]
            k_start = 2
            k_stop = I
    
            fill!(y,0)
            ConvolutionOperators.convolve!(y,Z,x,csx,i,k_start,k_stop)
            b = R - y
            x[:,i] .+= W0 * b
            if (i > 1 && i < ConvolutionOperators.tailindex(Z))
                csx[:,i] .= csx[:,i-1] .+ x[:,i]
            else
                csx[:,i] .= x[:,i]
            end
    
            (i % 10 == 0) && print(i, "[", I, "] - ")
        end
        return x
end

#=
				MAIN PART 
=#

### FORM 1: TD-EFIE (DONE!)
eq1 = @discretise T[k,j] == -1.0E[k] k∈X⊗δ j∈X⊗h
lhs1 = BEAST.td_assemble(eq1.equation.lhs, eq1.test_space_dict, eq1.trial_space_dict)
rhs1 = BEAST.td_assemble(eq1.equation.rhs, eq1.test_space_dict)

Z01 = zeros(eltype(𝕄0), size(lhs1)[1:2])
ConvolutionOperators.timeslice!(Z01, lhs1, 1)
iZ01 = inv(Z01)

j1 = marchonintime(iZ01, lhs1, rhs1, Nt)

### FORM 2: localized CP TD-EFIE (preconditioned by the static operator) (DONE!)
BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗∂2q) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (T̂s, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (T̂h, X⊗δ, X⊗q) BEAST.OuterNumInnerAnalyticQStrat(9)

Ts1_bilform_2 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂2q
Ts2_bilform_2 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗q
Th_bilform_2 = @discretise T̂h[k, j] k∈X⊗δ j∈X⊗q

Ts1_2 = BEAST.td_assemble(Ts1_bilform_2.bilform, Ts1_bilform_2.test_space_dict, Ts1_bilform_2.trial_space_dict)
Ts2_2 = BEAST.td_assemble(Ts2_bilform_2.bilform, Ts2_bilform_2.test_space_dict, Ts2_bilform_2.trial_space_dict)
Th_2 = BEAST.td_assemble(Th_bilform_2.bilform, Th_bilform_2.test_space_dict, Th_bilform_2.trial_space_dict)

lhs2 = 1/Δt * (𝕋0s * iNxy * Ts1_2 + 𝕋0h * iNxy * Ts2_2 + 𝕋0s * iNxy * Th_2)

e1_linform_2 = @discretise(-1.0∂E[k], k∈X⊗p)
e2_linform_2 = @discretise(-1.0iE[k], k∈X⊗p)

e1_2 = 1/Δt * BEAST.td_assemble(e1_linform_2.linform, e1_linform_2.test_space_dict)
e2_2 = 1/Δt * BEAST.td_assemble(e2_linform_2.linform, e2_linform_2.test_space_dict)

rhs2 = 𝕋0s * iNxy * e1_2 + 𝕋0h * iNxy * e2_2

Z02 = zeros(eltype(Ts1_2), size(lhs2)[1:2])
ConvolutionOperators.timeslice!(Z02, lhs2, 1)
iZ02 = inv(Z02)
j2 = marchonintime(iZ02, lhs2, rhs2, Nt)

### FORM 3:  qHP localized CP TD-EFIE (DONE!)
Mll_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂h
Mls1_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂2q
Mls2_bilform_3 = @discretise T̂h[k, j] k∈X⊗δ j∈X⊗q
Msl2_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗q
Mss1_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂3c
Mss2_bilform_3 = @discretise T̂h[k, j] k∈X⊗δ j∈X⊗∂c
Mss3_bilform_3 = @discretise T̂s[k, j] k∈X⊗δ j∈X⊗∂c

Mll_3 = BEAST.td_assemble(Mll_bilform_3.bilform, Mll_bilform_3.test_space_dict, Mll_bilform_3.trial_space_dict)
Mls1_3 = 1/Δt * BEAST.td_assemble(Mls1_bilform_3.bilform, Mls1_bilform_3.test_space_dict, Mls1_bilform_3.trial_space_dict)
Mls2_3 = 1/Δt * BEAST.td_assemble(Mls2_bilform_3.bilform, Mls2_bilform_3.test_space_dict, Mls2_bilform_3.trial_space_dict)
Msl2_3 = 1/Δt * BEAST.td_assemble(Msl2_bilform_3.bilform, Msl2_bilform_3.test_space_dict, Msl2_bilform_3.trial_space_dict)
Mss1_3 = 1/Δt^2 * BEAST.td_assemble(Mss1_bilform_3.bilform, Mss1_bilform_3.test_space_dict, Mss1_bilform_3.trial_space_dict)
Mss2_3 = 1/Δt^2 * BEAST.td_assemble(Mss2_bilform_3.bilform, Mss2_bilform_3.test_space_dict, Mss2_bilform_3.trial_space_dict)
Mss3_3 = 1/Δt^2 * BEAST.td_assemble(Mss3_bilform_3.bilform, Mss3_bilform_3.test_space_dict, Mss3_bilform_3.trial_space_dict)

lhs3 = ℙΣH * 𝕋0s * iNxy * Mll_3 * PΛH + ℙΣH * 𝕋0s * iNxy * (Mls1_3 + Mls2_3) * PΣ + ℙΛ * 𝕋0s * iNxy * Mls1_3 * PΛH + ℙΛ * 𝕋0h * iNxy * Msl2_3 * PΛH + ℙΛ * 𝕋0s * iNxy * (Mss1_3 + Mss2_3) * PΣ + ℙΛ * 𝕋0h * iNxy * Mss3_3 * PΣ 

el_linform_3 = @discretise(-1.0E[k], k∈X⊗p)
es1_linform_3 = @discretise(-1.0∂E[k], k∈X⊗h)
es2_linform_3 = @discretise(-1.0iE[k], k∈X⊗h)

el_3 = 1/Δt * BEAST.td_assemble(el_linform_3.linform, el_linform_3.test_space_dict)
es1_3 = 1/Δt * BEAST.td_assemble(es1_linform_3.linform, es1_linform_3.test_space_dict)
es2_3 = 1/Δt * BEAST.td_assemble(es2_linform_3.linform, es2_linform_3.test_space_dict)

rhs3 = ℙΣH * 𝕋0s * iNxy * el_3 + ℙΛ * 𝕋0s * iNxy * es1_3 + ℙΛ * 𝕋0h * iNxy * es2_3

Z03 = zeros(eltype(Mll_3), size(lhs3)[1:2])
ConvolutionOperators.timeslice!(Z03, lhs3, 1)
iZ03 = inv(Z03)
y3 = marchonintime(iZ03, lhs3, rhs3, Nt)

j3 = zeros(eltype(y3), size(y3)[1:2])
j3[:, 1] = PΛH * y3[:, 1] + 1.0/Δt * PΣ * y3[:, 1]
for i in 2:Nt
        j3[:, i] = PΛH * y3[:, i] + 1.0/Δt * PΣ * (y3[:, i] - y3[:, i-1])
end

### FORM 4: localized CP TD-MFIE (DONE!)
eq4 = @discretise (0.5(N⊗I) + 1.0K)[k,j] == -1.0H[k] k∈Y⊗δ j∈X⊗h
lhs4 = BEAST.td_assemble(eq4.equation.lhs, eq4.test_space_dict, eq4.trial_space_dict)
rhs4 = BEAST.td_assemble(eq4.equation.rhs, eq4.test_space_dict)

Z04 = zeros(eltype(𝕄0), size(lhs4)[1:2])
ConvolutionOperators.timeslice!(Z04, lhs4, 1)
iZ04 = inv(Z04)

j4 = marchonintime(iZ04, lhs4, rhs4, Nt)

### FORM 5: qHP localized CP TD-MFIE
BEAST.@defaultquadstrat (0.5(N⊗I) + 1.0K, Y⊗δ, X⊗ip) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (0.5(N⊗I) + 1.0K, Y⊗δ, X⊗h) BEAST.OuterNumInnerAnalyticQStrat(9)
BEAST.@defaultquadstrat (0.5(N⊗I) + 1.0K, Y⊗δ, X⊗∂q) BEAST.OuterNumInnerAnalyticQStrat(9)

Mll_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗ip
Msl_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗h
Mss_bilform_5 = @discretise (0.5(N⊗I) + 1.0K)[k, j] k∈Y⊗δ j∈X⊗∂q

Mll_5 = BEAST.td_assemble(Mll_bilform_5.bilform, Mll_bilform_5.test_space_dict, Mll_bilform_5.trial_space_dict)
Msl_5 = BEAST.td_assemble(Msl_bilform_5.bilform, Msl_bilform_5.test_space_dict, Msl_bilform_5.trial_space_dict)
Mss_5 = 1/Δt * BEAST.td_assemble(Mss_bilform_5.bilform, Mss_bilform_5.test_space_dict, Mss_bilform_5.trial_space_dict)

# Truncate the long tail of the loop-loop component (explicitly set to 0)

lhs5 = ℙΛ * left_linear_map * (Mss_5 * PΣ + Msl_5 * PΛH) + ℙΣH * left_linear_map * (Mll_5 * PΛH + Msl_5 * PΣ)

el_linform_5 = @discretise(-1.0iH[k], k∈Y⊗δ)
es_linform_5 = @discretise(-1.0H[k], k∈Y⊗p)

el_5 = BEAST.td_assemble(el_linform_5.linform, el_linform_5.test_space_dict)
es_5 = 1/Δt * BEAST.td_assemble(es_linform_5.linform, es_linform_5.test_space_dict)

rhs5 = ℙΣH * left_linear_map * el_5 + ℙΛ * left_linear_map * es_5

Z05 = zeros(eltype(𝕄0), size(lhs5)[1:2])
ConvolutionOperators.timeslice!(Z05, lhs5, 1)
iZ05 = inv(Z05)

y5 = marchonintime_trunc(iZ05, lhs5, rhs5, Nt)

j5 = zeros(eltype(y5), size(y5)[1:2])
j5[:, 1] = PΛH * y5[:, 1] + 1.0/Δt * PΣ * y5[:, 1]
for i in 2:Nt
        j5[:, i] = PΛH * y5[:, i] + 1.0/Δt * PΣ * (y5[:, i] - y5[:, i-1])
end

### FORM 6: qHP localized CP TD-CFIE
lhs6 = lhs3 + lhs5
rhs6 = rhs3 + rhs5

Z06 = zeros(eltype(𝕄0), size(𝕄0)[1:2])
ConvolutionOperators.timeslice!(Z06, lhs6, 1)
iZ06 = inv(Z06)

y6 = marchonintime(iZ06, lhs6, rhs6, Nt)

j6 = zeros(eltype(y6), size(y6)[1:2])
j6[:, 1] = PΛH * y6[:, 1] + 1.0/Δt * PΣ * y6[:, 1]
for i in 2:Nt
        j6[:, i] = PΛH * y6[:, i] + 1.0/Δt * PΣ * (y6[:, i] - y6[:, i-1])
end

# Show condition numbers
@show cond(Z01)
@show cond(Z02)
@show cond(Z03)
@show cond(Z04)
@show cond(Z05)
@show cond(Z06)

# Plot
plotly()
plot(xscale=:identity, yscale=:log10)
ylims!(1e-20, 0)
xlabel!("ct (m)")
ylabel!("j(t) (A/m)")
x = Δt * [1:1:Nt;]

plot!(x, abs.(j1[1,:]), label="TD-EFIE")
plot!(x, abs.(j2[1,:]), label="CP TD-EFIE")
plot!(x, abs.(j3[1,:]), label="qHP CP TD-EFIE")
plot!(x, abs.(j4[1,:]), label="TD-MFIE")
plot!(x, abs.(j5[1,:]), label="qHP CP TD-MFIE")
plot!(x, abs.(j6[1,:]), label="qHP CP TD-CFIE")