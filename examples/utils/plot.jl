import Plots, DelimitedFiles, Plotly

# Server 31: 
# Server 32: 
# Server 33: cuboid

```
    Plot the triangle mesh
```

using PlotlyJS

pt1 = PlotlyJS.plot(
    [patch(Γ, opacity=1.0, color="#bcbcbc"), CompScienceMeshes.wireframe(Γ), 
    CompScienceMeshes.normals(Γ)],
    Layout(
        height=400, width=600,
        scene=attr(
            xaxis=attr(
                visible=false,
                showbackground=false
            ),
            yaxis=attr(
                visible=false,
                showbackground=false
            ),
            zaxis=attr(
                visible=false,
                showbackground=false
            ),
            camera=attr(
                up=attr(x=0, y=0, z=1),
                center=attr(x=0, y=0, z=0),
                eye=attr(x=0.8, y=1, z=1)
            )
        )
    )
)

savefig(pt1, "starpyramid.pdf", width=600, height=400, scale=8)


```
    Plot the current density
```

x = 1e-2 * Δt/3 * [1:1:Nt;]

# Plots.plotly()
plt = Plots.plot(
    size = (600, 400),
    grid = false,
    xscale = :identity, 
    # xlims = (0, 16),
    # xticks = [0, 1, 2, 3, 4],
    xtickfont = Plots.font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e-20, 1e0), 
    yticks = [1e-50, 1e-45, 1e-40, 1e-35, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e0, 1e5],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Time ({\\mu}s)",
    ylabel = "Current density intensity (A/m)",
    titlefont = Plots.font(10, "Computer Modern"),
    guidefont = Plots.font(11, "Computer Modern"),
    colorbar_titlefont = Plots.font(10, "Computer Modern"),
    legendfont = Plots.font(11, "Computer Modern"),
    legend = :left,
    dpi = 300)

Plots.plot!(x, abs.(je[1, :]), label="Exact solution", linecolor=2, lw=1.5)
# Plots.plot!(x, abs.(jex4[1, :]), label="qHP TD-CFIE", linecolor=4, lw=1.3)
Plots.plot!(x, abs.(jpmchwt13[1, :]), label="TD-EFIE", linecolor=1, lw=1.5)
Plots.plot!(x, abs.(xefie_irk[1, :]), label="TD-MFIE", linecolor=4, lw=1.5)
Plots.plot!(x, abs.(jc[1, :]), label="TD-CFIE", linecolor=4, lw=1.5)
Plots.plot!(x, abs.(jqhp[1, :]), label="qHP PMCHWT", linecolor=3, lw=1.5)

Plots.savefig("current_sqtorus_cdt_1.5_h_0.3m.pdf")


```
    Plot the polynomial eigenvalues
```

we = ConvolutionOperators.polyvals(Txx)
wm = ConvolutionOperators.polyvals(Kyx)
wc = ConvolutionOperators.polyvals(cfie)
wsc = ConvolutionOperators.polyvals(qhpcfie)

Plots.plotly()
Plots.plot(size=(400,400),
    exp.(im*range(0, 2pi,length=3000)),
    aspect_ratio = :equal,
    xlims=(-1.2, 1.2),
    ylims=(-1.2, 1.2),
    grid = false,
    linecolor=:red,
    lw=1.5,
    label="diameter = 8, c.dt = 3",
    xlabel="",
    ylabel="")

Plots.scatter!(wm,
    label="", 
    markercolor=:blue, 
    markersize=3.5, 
    xlabel = "Real part", 
    ylabel = "Imaginary part",
    xtickfont = Plots.font(10, "Computer Modern"), 
    ytickfont = Plots.font(10, "Computer Modern"), 
    titlefont = Plots.font(12, "Computer Modern"),
    guidefont = Plots.font(10, "Times"),
    colorbar_titlefont = Plots.font(11, "Times"),
    legendfont = Plots.font(10, "Computer Modern"),
    dpi = 600)

Plots.savefig("polyvals_sqtorus_qHP_PMCHWT_13_h_0.4m_cdt_1.5m.pdf")

# Polyvals with small window for zoom-in data
Plots.plot!(
    exp.(im*range(-pi/5,pi/5,length=10000)), 
    xlims = (0.85, 1.15),
    ylims = (-0.15, 0.15),
    xticks = [1.0],
    yticks = [0.0],
    xlabel="",
    ylabel="",
    xtickfont = Plots.font(10, "Computer Modern"), 
    ytickfont = Plots.font(10, "Computer Modern"), 
    aspect_ratio=:equal,
    label="", 
    lw=1.5,
    grid=false,
    linecolor=:red,
    inset = Plots.bbox(0.01, 0.04, 0.24, 0.27, :top, :right),
    subplot = 2,
    framestyle=:box)

Plots.scatter!(wm, 
    xlabel="", 
    ylabel="",
    markercolor=:blue, 
    markersize=3.5, 
    label="", 
    subplot=2)

# Plots.savefig("polyvals_torus_CFIE_h_1.0m_cdt_0.1m.pdf")

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
    xtickfont = Plots.font(9, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1, 1e5), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Wave number (1/m)",
    ylabel = "Condition number",
    titlefont = Plots.font(10, "Computer Modern"), 
    guidefont = Plots.font(11, "Times"),
    colorbar_titlefont = Plots.font(10, "Times"), 
    legendfont = Plots.font(10, "Computer Modern"))

Plots.plot!(cond_num[4:54,1], cond_num[4:54,2], label="EFIE", linecolor=:black, linestyle=:dash, markershape=:x, markercolor=:black, markersize=4.5)
Plots.plot!(cond_num[4:54,1], cond_num[4:54,3], label="CFIE", linecolor=:black, linestyle=:dash, markershape=:circle, markercolor=:black, markersize=4)
Plots.savefig("cube_kappa.pdf")



```
    Plot the condition numbers with respect to mesh sizes
```

cond_num = DelimitedFiles.readdlm("sphere_cdt_1.txt", ' ', Float64, '\n')

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xscale = :log10, 
    xlims = (0.05, 0.6),
    xticks = ([0.06, 0.1, 0.16, 0.2, 0.3, 0.5, 0.7, 1.0], [0.06, 0.1, 0.16, 0.2, 0.3, 0.5, 0.7, 1.0]),
    xtickfont = Plots.font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1e0, 1e4), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Mesh size (m)",
    ylabel = "Condition number",
    titlefont = Plots.font(10, "Computer Modern"), 
    guidefont = Plots.font(11, "Times"),
    colorbar_titlefont = Plots.font(10, "Times"), 
    legendfont = Plots.font(10, "Computer Modern"),
    dpi = 300)

Plots.plot!(cond_num[:,1], cond_num[:,2], label="TD-EFIE", linecolor=1, lw=1.6, markershape=:square, markercolor=1, markersize=4.2)
Plots.plot!(cond_num[:,1], cond_num[:,3], label="TD-MFIE", linecolor=2, lw=1.6, markershape=:utriangle, markercolor=2, markersize=5)
Plots.plot!(cond_num[:,1], cond_num[:,4], label="mixed TD-CFIE", linecolor=4, lw=1.6, markershape=:circle, markercolor=4, markersize=5)
Plots.plot!(cond_num[:,1], cond_num[:,5], label="qHP TD-CFIE", linecolor=3, lw=1.6, markershape=:dtriangle, markercolor=3, markersize=5)

Plots.savefig("cond_torus_cdt_8.pdf")





```
    Plot the condition numbers respect to time step size
```

cond_num = DelimitedFiles.readdlm("sqtorus4holes_h_0.3m.txt", ' ', Float64, '\n')

plt = Plots.plot(
    width = 600, height=400,
    grid = true,
    xscale = :log10, 
    xlims = (0.3e0, 1e4),
    xticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    xtickfont = Plots.font(10, "Computer Modern"), 
    yscale = :log10, 
    ylims = (1, 1e9), 
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11],
    ytickfont = Plots.font(10, "Computer Modern"),
    xlabel = "Time step size (ns)",
    ylabel = "Condition number",
    titlefont = Plots.font(10, "Computer Modern"), 
    guidefont = Plots.font(11, "Times"),
    colorbar_titlefont = Plots.font(10, "Times"), 
    legendfont = Plots.font(11, "Computer Modern"),
    legend = :topleft, 
    dpi = 600)

Plots.plot!(cond_num[:,1]/0.3, cond_num[:,2], label="TD-EFIE", linecolor=1, lw=1.6, markershape=:square, markercolor=1, markersize=4.2)
Plots.plot!(cond_num[:,1]/0.3, cond_num[:,3], label="TD-MFIE", linecolor=2, lw=1.6, markershape=:utriangle, markercolor=2, markersize=5)
Plots.plot!(cond_num[:,1]/0.3, cond_num[:,4], label="mixed TD-CFIE", linecolor=4, lw=1.6, markershape=:circle, markercolor=4, markersize=5)
Plots.plot!(cond_num[:,1]/0.3, cond_num[:,5], label="qHP TD-CFIE", linecolor=3, lw=1.6, markershape=:dtriangle, markercolor=3, markersize=5)

Plots.savefig("cond_sphere_h_0.3m.pdf")





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
    guidefont = font(11, "Times"),
    colorbar_titlefont = font(10, "Times"), 
    legendfont = font(10, "Computer Modern"))

Plots.plot(Θ1, norm.(Eexct), label="Mie series", linecolor=:black)
Plots.scatter!(Θ, norm.(Esct), label="CFIE", markershape=:x, markercolor=:black, markersize=3.5)
Plots.savefig("scatteredE.pdf")