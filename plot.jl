using Plots, DelimitedFiles, Plotly


# Server 31: torus with dt = 0.1, Nt=3600, h = 0.6m
# Server 32: sphere 


```
    Plot the triangle mesh
```

using PlotlyJS

pt1 = PlotlyJS.plot(
    [patch(Γ, opacity=1.0, color="#c0c0c4"), CompScienceMeshes.wireframe(Γ)],
    Layout(
        height=400, width=400,
        scene=attr(
            xaxis=attr(
                color="#FFFFFF",
                showbackground=false
            ),
            yaxis=attr(
                color="#FFFFFF",
                showbackground=false
            ),
            zaxis=attr(
                color="#FFFFFF",
                showbackground=false
            )
        )
    )
)





```
    Plot the current density
```

x = 10 * Δt/3 * [1:1:Nt;]

plt = Plots.plot(
    size = (600, 400),
    grid = false,
    xscale = :identity, 
    xlims = (0, 1200),
    # xticks = [450, 900, 1350],
    xtickfont = font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e-19, 1e0), 
    yticks = [1e-35, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5],
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Time (ns)",
    ylabel = "Current density intensity (A/m)",
    titlefont = font(10, "Computer Modern"),
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"),
    legendfont = font(11, "Computer Modern"),
    legend = :topright)

Plots.plot!(x, abs.(je[1, :]), label="TD-EFIE", linecolor=1, lw=1.3)
# Plots.plot!(x[1:1200], abs.(j3[1, 1:1200]), label="qHP CP TD-EFIE", linecolor=2)
Plots.plot!(x, abs.(jm[1, :]), label="", linecolor=1, lw=1.3)
# Plots.plot!(x[1:1200], abs.(j5[1, 1:1200]), label="qHP TD-MFIE", linecolor=4)
Plots.plot!(x, abs.(jc[1, :]), label="mixed TD-CFIE", linecolor=3)
Plots.plot!(x, abs.(jqhpc[1, :]), label="TD-CFIE", linecolor=3, lw=1.4)

Plots.savefig("static.pdf")


```
    Plot the polynomial eigenvalues
```

we = ConvolutionOperators.polyvals(Txx)
wm = ConvolutionOperators.polyvals(Kyx)
wc = ConvolutionOperators.polyvals(cfie)
wsc = ConvolutionOperators.polyvals(qhpcfie)

pe = Plots.plot(exp.(im*range(0,2pi,length=1000)), label="")
scatter!(we, label="")
pm = Plots.plot(exp.(im*range(0,2pi,length=1000)), label="")
scatter!(wm, label="")
pc = Plots.plot(width = 400, height=400)
plot!(exp.(im*range(0,2pi,length=1000)), label="")
scatter!(wc, label="")
psc = Plots.plot(exp.(im*range(0,2pi,length=1000)), label="")
scatter!(wsc, label="")
plot(size=(600,600),
    grid = false,
    pe, pm, pc, psc, 
    layout = 4, 
    markercolor=:black,
    markersize=2.5,
    linecolor=:red,
    xlabel = "Real part",
    ylabel = "Imaginary part",
    title=["(a) standard TD-EFIE" "(b) standard TD-MFIE" "(c) mixed TD-CFIE" "(d) symmetrized TD-CFIE"],
    xtickfont = font(9, "Computer Modern"), 
    ytickfont = font(9, "Computer Modern"), 
    titlefont = font(11, "Computer Modern"),
    guidefont = font(9, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"),
    legendfont = font(9, "Computer Modern"))

savefig("polyvals_sphere_h_0.3m_cdt_0.25m.pdf")





```
    Plot the condition numbers respect to wave numbers
```

cond_num = readdlm("cube_kappa.txt", ' ', Float64, '\n')

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    # xscale = :log10, 
    xlims = (4.17, 5.83),
    xticks = ([4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8], [4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8]),
    xtickfont = font(9, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1, 1e5), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Wave number (1/m)",
    ylabel = "Condition number",
    titlefont = font(10, "Computer Modern"), 
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"), 
    legendfont = font(10, "Computer Modern"))

Plots.plot!(cond_num[4:54,1], cond_num[4:54,2], label="EFIE", linecolor=:black, linestyle=:dash, markershape=:x, markercolor=:black, markersize=4.5)
Plots.plot!(cond_num[4:54,1], cond_num[4:54,3], label="CFIE", linecolor=:black, linestyle=:dash, markershape=:circle, markercolor=:black, markersize=4)
Plots.savefig("cube_kappa.pdf")



```
    Plot the condition numbers with respect to mesh sizes
```

cond_num = readdlm("sphere_cdt_1.txt", ' ', Float64, '\n')

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xscale = :log10, 
    xlims = (0.06, 0.6),
    xticks = ([0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    xtickfont = font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e1, 1e4), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4],
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Mesh size (m)",
    ylabel = "Condition number",
    titlefont = font(10, "Computer Modern"), 
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"), 
    legendfont = font(10, "Computer Modern"))

Plots.plot!(cond_num[:,1], cond_num[:,2], label="", linecolor=1, lw=1.5, markershape=:square, markercolor=1, markersize=4)
Plots.plot!(cond_num[:,1], cond_num[:,4], label="", linecolor=2, lw=1.5, markershape=:diamond, markercolor=2, markersize=5)
Plots.plot!(cond_num[:,1], cond_num[:,6], label="mixed TD-CFIE", linecolor=3, markershape=:circle, markercolor=3, markersize=4)
Plots.plot!(cond_num[:,1], cond_num[:,7], label="", linecolor=3, lw=1.5, markershape=:circle, markercolor=3, markersize=5)
Plots.savefig("dense_discretization.pdf")





```
    Plot the condition numbers respect to time step size
```

cond_num = readdlm("sphere_h_0.3m.txt", ' ', Float64, '\n')

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xscale = :log10, 
    xlims = (0.6e0, 1e3),
    xticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    xtickfont = font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1, 2e7), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Time step size (ns)",
    ylabel = "Condition number",
    titlefont = font(10, "Computer Modern"), 
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"), 
    legendfont = font(11, "Computer Modern"),
    legend = :topleft)

Plots.plot!(cond_num[2:12,1]/0.3, cond_num[2:12,2], label="", linecolor=1, lw=1.5, markershape=:square, markercolor=1, markersize=4)
# Plots.plot!(cond_num[:,1], cond_num[:,3], label="qHP CP TD-EFIE", linecolor=2, markershape=:diamond, markercolor=2, markersize=4.5)
Plots.plot!(cond_num[3:12,1]/0.3, cond_num[3:12,4], label="TD-MFIE", linecolor=2, lw=1.5, markershape=:diamond, markercolor=2, markersize=5)
# Plots.plot!(cond_num[:,1], cond_num[:,5], label="qHP TD-MFIE", linecolor=4, markershape=:dtriangle, markercolor=4, markersize=4.5)
Plots.plot!(cond_num[3:12,1]/0.3, cond_num[3:12,6], label="mixed TD-CFIE", linecolor=3, markershape=:circle, markercolor=3, markersize=4)
Plots.plot!(cond_num[3:12,1]/0.3, cond_num[3:12,7], label="TD-CFIE", linecolor=3, lw=1.5, markershape=:circle, markercolor=3, markersize=5)

Plots.savefig("large_time_step.pdf")





```
    Plot the E and H fields
```

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xtickfont = font(9, "Computer Modern"), 
    ylims = (0, 1.2), 
    yticks = ([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
    ytickfont = font(10, "Computer Modern"),
    xlabel = "Observation angle",
    ylabel = "Scattered electric field",
    titlefont = font(10, "Computer Modern"), 
    guidefont = font(11, "Computer Modern"),
    colorbar_titlefont = font(10, "Computer Modern"), 
    legendfont = font(10, "Computer Modern"))

Plots.plot(Θ1, norm.(Eexct), label="Mie series", linecolor=:black)
Plots.scatter!(Θ, norm.(Esct), label="CFIE", markershape=:x, markercolor=:black, markersize=3.5)
Plots.savefig("scatteredE.pdf")