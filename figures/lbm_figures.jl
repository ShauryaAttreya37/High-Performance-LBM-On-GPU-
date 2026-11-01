# Makie figures for the LBM project, called from Python (figures/make_figures.py) via juliacall.
#
# Python owns the data: it runs the cases (or reads cached results) and passes
# plain NumPy arrays; these functions only draw. Text is typeset with
# MathTeXEngine: L"..." strings mix text and $...$ math, and theme_latexfonts()
# sets Computer Modern for everything, so no TeX installation is needed.
# Each figure is written as PNG (for viewing) and PDF (vector, for reports).
#
# Dark mode: a dark surface with the palette's dark steps (validated for
# contrast and colour-blind separation against #1a1a19), not inverted light
# colours. Sequential maps run dark (small) to light (large) so that small
# values recede into the background; the diverging map's neutral midpoint is
# a dark grey.

using CairoMakie, LaTeXStrings

const SURFACE = colorant"#1a1a19"
const INK = colorant"#ffffff"
const INK2 = colorant"#c3c2b7"
const GRID = colorant"#383835"
const BAND = colorant"#2e2e2c"      # benchmark intervals, a step above the surface
const BODY = colorant"#6b6a65"      # solid bodies drawn inside field maps
const BLUE, ORANGE, AQUA = colorant"#3987e5", colorant"#d95926", colorant"#199e70"
# Fixed categorical order: the colour follows the method, not its position in a plot.
const METHOD_STYLE = Dict(
    "Bouzidi" => (BLUE, :circle),
    "staircase" => (ORANGE, :rect),
    "IBM" => (AQUA, :utriangle),
)
const DIVERGING = cgrad([colorant"#86b6ef", colorant"#3987e5", colorant"#383835",
                         colorant"#e66767", colorant"#f2a3a3"])
const SEQUENTIAL = cgrad([colorant"#232322", colorant"#184f95", colorant"#3987e5",
                          colorant"#86b6ef", colorant"#cde2fb"])

function lbm_theme()
    merge(theme_latexfonts(), Theme(
        fontsize = 15,
        backgroundcolor = SURFACE,
        textcolor = INK,
        Axis = (backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
                topspinevisible = false, rightspinevisible = false,
                xminorticksvisible = false, yminorticksvisible = false,
                leftspinecolor = INK2, bottomspinecolor = INK2,
                xtickcolor = INK2, ytickcolor = INK2, titlesize = 16, titlefont = :regular,
                xticklabelcolor = INK2, yticklabelcolor = INK2),
        Colorbar = (tickcolor = INK2, ticklabelcolor = INK2, spinewidth = 0),
        Legend = (framevisible = false, labelsize = 13),
    ))
end

function save_both(fig, stem)
    save(stem * ".png", fig; px_per_unit = 2)
    save(stem * ".pdf", fig)
    return stem
end

"Series with markers ringed in the surface colour, plus a direct label at its last point."
function method_series!(ax, x, y, name)
    color, marker = METHOD_STYLE[name]
    scatterlines!(ax, x, y; color, marker, markersize = 13, linewidth = 2.5,
                  strokecolor = SURFACE, strokewidth = 1.5, label = name)
    text!(ax, x[end], y[end]; text = name, align = (:left, :center), offset = (10, 0),
          color = INK2, fontsize = 13)
end

"Reference line y = y0 (x0 / x)^p through the first point, labelled at its end."
function slope_guide!(ax, x, y0, p, label)
    xs = [first(x), last(x)]
    ys = y0 .* (first(x) ./ xs) .^ p
    lines!(ax, xs, ys; color = INK2, linestyle = :dash, linewidth = 1.2)
    text!(ax, xs[end], ys[end]; text = label, align = (:left, :top), offset = (4, -2),
          color = INK2, fontsize = 12)
end

# Extra room on the right for direct labels (text is clipped at the axis edge).
loglog_axis(pos; kw...) = Axis(pos; xscale = log10, yscale = log10,
                               xautolimitmargin = (0.05, 0.22), kw...)

function curved_wall_convergence(stem, cyl, pipe, sph)
    with_theme(lbm_theme()) do
        fig = Figure(size = (1500, 520))
        ax1 = loglog_axis(fig[1, 1]; xlabel = L"cylinder radius $R$ [nodes]",
            ylabel = L"$|C_D / C_D^{\mathrm{SA}} - 1|$ [%]",
            title = L"2D cylinder array, Stokes drag ($c = 0.05$)",
            xticks = [4, 6, 8, 12, 16, 24])
        for (name, (r, e)) in cyl
            method_series!(ax1, r, e, name)
        end
        slope_guide!(ax1, [4.04, 24.2], cyl["Bouzidi"][2][1], 2, L"\propto R^{-2}")
        slope_guide!(ax1, [4.04, 24.2], cyl["IBM"][2][1], 1, L"\propto R^{-1}")

        ax2 = loglog_axis(fig[1, 2]; xlabel = L"pipe radius $R$ [nodes]",
            ylabel = L"$\Vert \rho u - \rho u_{\mathrm{exact}} \Vert_2 / \Vert \rho u_{\mathrm{exact}} \Vert_2$",
            title = "3D pipe flow (Hagen–Poiseuille)", xticks = [4, 6, 8, 12, 16, 24])
        for (name, (r, e)) in pipe
            method_series!(ax2, r, e, name)
        end
        slope_guide!(ax2, [4.3, 24.3], pipe["Bouzidi"][2][1], 2, L"\propto R^{-2}")

        ax3 = loglog_axis(fig[1, 3]; xlabel = L"sphere radius $R$ [nodes]",
            ylabel = L"$|C_D / C_D^{\mathrm{H}} - 1|$ [%]",
            title = L"3D sphere array, Stokes drag ($c = 0.01$)", xticks = [4, 5, 6, 7, 8],
            yticks = [0.03, 0.1, 0.3, 1, 3])
        for (name, (r, e)) in sph
            method_series!(ax3, r, e, name)
        end
        slope_guide!(ax3, sph["Bouzidi"][1], sph["Bouzidi"][2][1], 2, L"\propto R^{-2}")
        text!(ax3, 6.42, 0.026; text = "staircase error\ncrosses zero", align = (:left, :bottom),
              offset = (8, 4), color = INK2, fontsize = 12)
        for ax in (ax1, ax2, ax3)
            axislegend(ax; position = :lb)
        end
        Label(fig[0, :], "Curved walls: staircase vs interpolated bounce-back (Bouzidi) vs immersed boundary (IBM)";
              fontsize = 18, font = :bold)
        save_both(fig, stem)
    end
end

function compressibility(stem, u_max, cd, lo, hi)
    with_theme(lbm_theme()) do
        fig = Figure(size = (760, 520))
        ax = Axis(fig[1, 1]; xlabel = L"$u_{\max}^2$ (lattice units) $\;\propto\; \mathrm{Ma}^2$",
                  xautolimitmargin = (0.05, 0.12),
                  ylabel = L"C_D",
                  title = L"Schäfer–Turek 2D-1, $N = 40$: the drag excess is $O(\mathrm{Ma}^2)$")
        x = u_max .^ 2
        hspan!(ax, lo, hi; color = (BAND, 1.0))
        text!(ax, 0.98 * maximum(x), (lo + hi) / 2; text = L"benchmark interval $[%$lo, %$hi]$",
              align = (:right, :center), color = INK2, fontsize = 13)
        a, b = hcat(ones(length(x)), x) \ cd  # least-squares line C_D = a + b u^2
        xs = range(0, 1.05 * maximum(x), length = 50)
        lines!(ax, xs, a .+ b .* xs; color = INK2, linestyle = :dash, linewidth = 1.2)
        scatter!(ax, x, cd; color = BLUE, markersize = 15, strokecolor = SURFACE, strokewidth = 1.5)
        for (xi, ui, c) in zip(x, u_max, cd)
            text!(ax, xi, c; text = L"u_{\max} = %$ui", offset = (10, -4), align = (:left, :top),
                  color = INK2, fontsize = 13)
        end
        text!(ax, 0.55 * xs[end], a + 0.25 * b * xs[end];
              text = L"least-squares line: $C_D(\mathrm{Ma} \to 0) = %$(round(a, digits = 3))$",
              align = (:left, :center), color = INK2, fontsize = 13)
        save_both(fig, stem)
    end
end

"Decade-and-3x ticks between 1e-5 and 1e-1 with LaTeX labels (avoids 10^{-2.5})."
const ERROR_TICKS = (
    [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    [L"10^{-5}", L"3 \times 10^{-5}", L"10^{-4}", L"3 \times 10^{-4}", L"10^{-3}",
     L"3 \times 10^{-3}", L"10^{-2}", L"3 \times 10^{-2}", L"10^{-1}"],
)

function d3q19_validation(stem, n_abc, e_abc, h_duct, e_duct)
    with_theme(lbm_theme()) do
        fig = Figure(size = (1100, 480))
        ax1 = loglog_axis(fig[1, 1]; xlabel = L"N \; (N^3 \text{ nodes})",
            ylabel = L"relative $L_2$ velocity error",
            title = "3D ABC (Beltrami) flow: exact viscous decay", xticks = [16, 32, 64, 128],
            yticks = ERROR_TICKS)
        scatterlines!(ax1, n_abc, e_abc; color = BLUE, markersize = 12, linewidth = 2.5,
                      strokecolor = SURFACE, strokewidth = 1.5)
        slope_guide!(ax1, n_abc, e_abc[1], 2, L"\propto N^{-2}")
        ax2 = loglog_axis(fig[1, 2]; xlabel = L"duct height $H$ [nodes]",
            ylabel = L"relative $L_2$ error vs Fourier series",
            title = L"3D square duct (TRT, $\tau = 1$)", xticks = [8, 16, 32, 64], yticks = ERROR_TICKS)
        scatterlines!(ax2, h_duct, e_duct; color = BLUE, markersize = 12, linewidth = 2.5,
                      strokecolor = SURFACE, strokewidth = 1.5)
        slope_guide!(ax2, h_duct, e_duct[1], 2, L"\propto H^{-2}")
        save_both(fig, stem)
    end
end

"Vorticity map (x along the first matrix axis) above C_D(t) and C_L(t)."
function cylinder_wake(stem, x, y, omega, t, cd, cl, cd_band, cl_max)
    with_theme(lbm_theme()) do
        fig = Figure(size = (1300, 820))
        ax = Axis(fig[1, 1]; xlabel = L"x \; [\mathrm{m}]", ylabel = L"y \; [\mathrm{m}]",
                  xgridvisible = false, ygridvisible = false,
                  title = L"Schäfer–Turek 2D-2 ($\mathrm{Re} = 100$), Bouzidi cylinder, $N = 40$")
        hm = heatmap!(ax, x, y, omega; colormap = DIVERGING, colorrange = (-8, 8),
                      nan_color = BODY)
        Colorbar(fig[1, 2], hm; label = L"\omega D / \bar{U}")
        axd = Axis(fig[2, 1]; ylabel = L"C_D")
        hspan!(axd, cd_band...; color = (BAND, 1.0))
        lines!(axd, t, cd; color = BLUE, linewidth = 2)
        text!(axd, t[1], cd_band[2]; text = L"benchmark $C_{D,\max} \in [%$(cd_band[1]), %$(cd_band[2])]$",
              offset = (6, 4), align = (:left, :bottom), color = INK2, fontsize = 12)
        axl = Axis(fig[3, 1]; ylabel = L"C_L", xlabel = L"t \, \bar{U} / D")
        hlines!(axl, [-cl_max, cl_max]; color = INK2, linestyle = :dash, linewidth = 1.2)
        lines!(axl, t, cl; color = BLUE, linewidth = 2)
        linkxaxes!(axd, axl)
        hidexdecorations!(axd; grid = false, ticks = false)
        # Row height from the column width, so the channel keeps its true 2.2 : 0.41 shape
        # and the colorbar is exactly as tall as the map.
        rowsize!(fig.layout, 1, Aspect(1, (last(y) - first(y)) / (last(x) - first(x))))
        save_both(fig, stem)
    end
end

"Speed in the mid-plane (x along the first matrix axis) with the u_x = 0 contour."
function sphere_wake(stem, x, y, speed, ux)
    with_theme(lbm_theme()) do
        fig = Figure(size = (1200, 560))
        ax = Axis(fig[1, 1]; xlabel = L"x / D", ylabel = L"y / D", aspect = DataAspect(),
                  xgridvisible = false, ygridvisible = false,
                  title = L"Sphere, $\mathrm{Re} = 100$ (D3Q19, Bouzidi, $N = 16$): $|\mathbf{u}| / U$ in the mid-plane")
        hm = heatmap!(ax, x, y, speed; colormap = SEQUENTIAL, colorrange = (0, 1.15), nan_color = BODY)
        contour!(ax, x, y, ux; levels = [0.0], color = ORANGE, linewidth = 2.5)
        Colorbar(fig[1, 2], hm; label = L"|\mathbf{u}| / U")
        Label(fig[2, 1], L"orange contour: $u_x = 0$, the edge of the recirculation bubble";
              color = INK2, fontsize = 13, tellwidth = false)
        save_both(fig, stem)
    end
end

# ---------------------------------------------------------------- social media card

"Integer with thousands separators: 2130 -> \"2,130\"."
thousands(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

"A large number over a short caption, centred in its cell."
function stat!(pos, value, caption)
    g = GridLayout(pos)
    Label(g[1, 1], value; fontsize = 62, font = :bold, color = INK)
    Label(g[2, 1], caption; fontsize = 19, color = INK2, justification = :center, lineheight = 1.2)
    rowgap!(g, 8)
end

"""
Portrait 1080 x 1350 card (LinkedIn's 4:5 feed format): headline, the
vorticity map edge to edge, three headline numbers. `omega` is an Observable
so that the same card can be animated.
"""
function social_card(x, y, omega, stats)
    pad = 64
    left(row, text; kw...) = Label(row, text; halign = :left, justification = :left,
                                   padding = (pad, pad, 0, 0), tellwidth = false, kw...)
    fig = Figure(size = (1080, 1350), figure_padding = (0, 0, 72, 72))
    left(fig[1, 1], "LATTICE BOLTZMANN CFD  ·  PYTHON + CUDA"; color = BLUE, font = :bold,
         fontsize = 19)
    left(fig[2, 1], "Vortex shedding behind a cylinder,\nsimulated at the speed of memory";
         fontsize = 46, font = :bold, lineheight = 1.1)
    left(fig[3, 1], L"Schäfer–Turek benchmark 2D-2, $\mathrm{Re} = 100$, on a laptop RTX 3050 (6 GB)";
         fontsize = 22, color = INK2)
    ax = Axis(fig[4, 1]; xautolimitmargin = (0, 0), yautolimitmargin = (0, 0))
    hidedecorations!(ax)
    hidespines!(ax)
    heatmap!(ax, x, y, omega; colormap = DIVERGING, colorrange = (-8, 8), nan_color = BODY,
             interpolate = true)
    # The exact cylinder over its lattice nodes (the NaN mask shows the grid steps).
    poly!(ax, Circle(Point2f(stats["cylinder"][1:2]...), stats["cylinder"][3]); color = BODY)
    # Row height from the full figure width, so the channel keeps its true shape.
    rowsize!(fig.layout, 4, Aspect(1, (last(y) - first(y)) / (last(x) - first(x))))
    left(fig[5, 1], "vorticity: red turns counter-clockwise, blue clockwise";
         fontsize = 16, color = INK2)
    s = GridLayout(fig[6, 1]; alignmode = Outside(pad - 16, pad - 16, 0, 0), tellwidth = false)
    stat!(s[1, 1], thousands(stats["mlups"]), "million lattice updates\nper second")
    stat!(s[1, 2], "$(stats["bandwidth"])%", "of the GPU's measured\nmemory bandwidth")
    stat!(s[1, 3], string(round(stats["strouhal"]; digits = 3)),
          "Strouhal number; the\nbenchmark accepts 0.295–0.305")
    nx, ny = stats["grid"]
    left(fig[7, 1], "Fused CUDA kernels  ·  D2Q9, TRT collision  ·  interpolated bounce-back cylinder  ·  $nx × $ny nodes";
         fontsize = 16, color = INK2)
    # Fill the width: an Auto column would shrink to the widest content that
    # reports a width, and the map row takes its height from this column.
    colsize!(fig.layout, 1, Relative(1))
    for (row, gap) in enumerate((20, 18, 56, 14, 72, 64))
        rowgap!(fig.layout, row, gap)
    end
    return fig
end

"""
Hero PNG (2160 x 2700) from frame `hero`, and an MP4 loop of all frames at
24 fps (1080 x 1350, H.264). frames[k, :, :] is the vorticity at frame k.
"""
function social_figures(stem, x, y, frames, stats; hero = 1)
    with_theme(lbm_theme()) do
        omega = Observable(frames[hero, :, :])
        fig = social_card(x, y, omega, stats)
        save(stem * "_hero.png", fig; px_per_unit = 2)
        record(fig, stem * "_loop.mp4", axes(frames, 1); framerate = 24, px_per_unit = 1) do k
            omega[] = frames[k, :, :]
        end
    end
    return stem
end
